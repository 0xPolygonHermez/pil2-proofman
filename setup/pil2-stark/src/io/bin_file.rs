use anyhow::Result;

use crate::io::bin_file_writer::BinFileWriter;
use crate::io::parser_args::{get_parser_args, ExpsInfo};
use crate::types::stark_info::{ExpressionsInfo, Hint, OpType, StarkInfo, VerifierInfo};

const CHELPERS_NSECTIONS: u32 = 3;
const CHELPERS_EXPRESSIONS_SECTION: u32 = 1;
const CHELPERS_CONSTRAINTS_DEBUG_SECTION: u32 = 2;
const CHELPERS_HINTS_SECTION: u32 = 3;

/// Prepared expression binary data ready for writing.
struct PreparedExpressionsBin {
    exps_info: Vec<ExpsInfo>,
    constraints_info: Vec<ExpsInfo>,
    hints_info: Vec<Hint>,
    numbers_exps: Vec<String>,
    numbers_constraints: Vec<String>,
    max_tmp1: u64,
    max_tmp3: u64,
    max_args: u64,
    max_ops: u64,
}

/// Prepared verifier binary data.
struct PreparedVerifierBin {
    q_code: ExpsInfo,
    query_code: ExpsInfo,
    numbers_exps: Vec<String>,
    max_tmp1: u64,
    max_tmp3: u64,
    max_args: u64,
    max_ops: u64,
}

fn prepare_expressions_bin(
    stark_info: &StarkInfo,
    expressions_info: &ExpressionsInfo,
) -> Result<PreparedExpressionsBin> {
    let mut exps_info: Vec<ExpsInfo> = Vec::new();
    let mut constraints_info: Vec<ExpsInfo> = Vec::new();
    let mut numbers_exps: Vec<String> = Vec::new();
    let mut numbers_constraints: Vec<String> = Vec::new();

    let n = 1u64 << stark_info.stark_struct.n_bits;

    let mut max_tmp1: u64 = 0;
    let mut max_tmp3: u64 = 0;
    let mut max_args: u64 = 0;
    let mut max_ops: u64 = 0;

    // Process constraints
    for constraint_code in &expressions_info.constraints {
        let (first_row, last_row) = match constraint_code.boundary.as_str() {
            "everyRow" => (0, n),
            "firstRow" | "finalProof" => (0, 1),
            "lastRow" => (n - 1, n),
            "everyFrame" => (constraint_code.offset_min, n - constraint_code.offset_max),
            other => anyhow::bail!("Invalid boundary: {}", other),
        };

        let result =
            get_parser_args(stark_info, &constraint_code.code, &mut numbers_constraints, false, false, None, "")?;

        let mut info = result.exps_info;
        info.stage = constraint_code.stage;
        info.first_row = first_row;
        info.last_row = last_row;
        info.line = constraint_code.line.clone();
        info.im_pol = constraint_code.im_pol;

        if info.n_temp1 > max_tmp1 {
            max_tmp1 = info.n_temp1;
        }
        if info.n_temp3 > max_tmp3 {
            max_tmp3 = info.n_temp3;
        }
        if info.args.len() as u64 > max_args {
            max_args = info.args.len() as u64;
        }
        if info.ops.len() as u64 > max_ops {
            max_ops = info.ops.len() as u64;
        }

        constraints_info.push(info);
    }

    // Process expressions
    for exp_code_orig in &expressions_info.expressions_code {
        let mut exp_code = exp_code_orig.clone();

        // Check if dest should be redirected to tmp
        let is_special = exp_code.exp_id == stark_info.c_exp_id
            || exp_code.exp_id == stark_info.deep_exp_id
            || stark_info.cm_pols_map.iter().any(|c| c.exp_id == exp_code.exp_id);

        if is_special {
            if let Some(last) = exp_code.code.last_mut() {
                last.dest.op_type = OpType::Tmp;
                last.dest.id = exp_code.tmp_used;
                exp_code.tmp_used += 1;
            }
        }

        let result = get_parser_args(stark_info, &exp_code.code, &mut numbers_exps, false, false, None, "")?;

        let mut info = result.exps_info;
        info.exp_id = exp_code.exp_id;
        info.stage = exp_code.stage;
        info.line = exp_code.line.clone();

        if info.n_temp1 > max_tmp1 {
            max_tmp1 = info.n_temp1;
        }
        if info.n_temp3 > max_tmp3 {
            max_tmp3 = info.n_temp3;
        }
        if info.args.len() as u64 > max_args {
            max_args = info.args.len() as u64;
        }
        if info.ops.len() as u64 > max_ops {
            max_ops = info.ops.len() as u64;
        }

        exps_info.push(info);
    }

    Ok(PreparedExpressionsBin {
        exps_info,
        constraints_info,
        hints_info: expressions_info.hints_info.clone(),
        numbers_exps,
        numbers_constraints,
        max_tmp1,
        max_tmp3,
        max_args,
        max_ops,
    })
}

fn prepare_verifier_expressions_bin(
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
) -> Result<PreparedVerifierBin> {
    let mut max_tmp1: u64 = 0;
    let mut max_tmp3: u64 = 0;
    let mut max_args: u64 = 0;
    let mut max_ops: u64 = 0;
    let mut numbers_exps: Vec<String> = Vec::new();

    let q_result =
        get_parser_args(stark_info, &verifier_info.q_verifier.code, &mut numbers_exps, false, true, None, "")?;
    let mut q_code = q_result.exps_info;
    q_code.exp_id = stark_info.c_exp_id;
    q_code.stage = stark_info.n_stages + 1;
    q_code.line = String::new();

    if q_code.n_temp1 > max_tmp1 {
        max_tmp1 = q_code.n_temp1;
    }
    if q_code.n_temp3 > max_tmp3 {
        max_tmp3 = q_code.n_temp3;
    }
    if q_code.args.len() as u64 > max_args {
        max_args = q_code.args.len() as u64;
    }
    if q_code.ops.len() as u64 > max_ops {
        max_ops = q_code.ops.len() as u64;
    }

    let query_result =
        get_parser_args(stark_info, &verifier_info.query_verifier.code, &mut numbers_exps, false, true, None, "")?;
    let mut query_code = query_result.exps_info;
    query_code.exp_id = stark_info.deep_exp_id;
    query_code.stage = stark_info.n_stages + 2;
    query_code.line = String::new();

    if query_code.n_temp1 > max_tmp1 {
        max_tmp1 = query_code.n_temp1;
    }
    if query_code.n_temp3 > max_tmp3 {
        max_tmp3 = query_code.n_temp3;
    }
    if query_code.args.len() as u64 > max_args {
        max_args = query_code.args.len() as u64;
    }
    if query_code.ops.len() as u64 > max_ops {
        max_ops = query_code.ops.len() as u64;
    }

    Ok(PreparedVerifierBin { q_code, query_code, numbers_exps, max_tmp1, max_tmp3, max_args, max_ops })
}

#[allow(clippy::too_many_arguments)]
fn write_expressions_section(
    writer: &mut BinFileWriter,
    expressions_info: &[ExpsInfo],
    numbers_exps: &[String],
    max_tmp1: u64,
    max_tmp3: u64,
    max_args: u64,
    max_ops: u64,
    section: u32,
) -> Result<()> {
    writer.start_write_section(section)?;

    let mut ops_all: Vec<u8> = Vec::new();
    let mut args_all: Vec<u16> = Vec::new();
    let mut ops_offsets: Vec<u32> = Vec::new();
    let mut args_offsets: Vec<u32> = Vec::new();

    for (i, exp) in expressions_info.iter().enumerate() {
        if i == 0 {
            ops_offsets.push(0);
            args_offsets.push(0);
        } else {
            ops_offsets.push(ops_offsets[i - 1] + expressions_info[i - 1].ops.len() as u32);
            args_offsets.push(args_offsets[i - 1] + expressions_info[i - 1].args.len() as u32);
        }
        for op in &exp.ops {
            ops_all.push(*op as u8);
        }
        for arg in &exp.args {
            args_all.push(*arg as u16);
        }
    }

    writer.write_u32(max_tmp1 as u32)?;
    writer.write_u32(max_tmp3 as u32)?;
    writer.write_u32(max_args as u32)?;
    writer.write_u32(max_ops as u32)?;
    writer.write_u32(ops_all.len() as u32)?;
    writer.write_u32(args_all.len() as u32)?;
    writer.write_u32(numbers_exps.len() as u32)?;

    let n_expressions = expressions_info.len() as u32;
    writer.write_u32(n_expressions)?;

    for (i, exp) in expressions_info.iter().enumerate() {
        writer.write_u32(exp.exp_id as u32)?;
        writer.write_u32(exp.dest_dim as u32)?;
        writer.write_u32(exp.dest_id as u32)?;
        writer.write_u32(exp.stage as u32)?;
        writer.write_u32(exp.n_temp1 as u32)?;
        writer.write_u32(exp.n_temp3 as u32)?;

        writer.write_u32(exp.ops.len() as u32)?;
        writer.write_u32(ops_offsets[i])?;

        writer.write_u32(exp.args.len() as u32)?;
        writer.write_u32(args_offsets[i])?;

        writer.write_string(&exp.line)?;
    }

    // Write ops as u8
    for op in &ops_all {
        writer.write_u8(*op)?;
    }

    // Write args as u16 LE
    for arg in &args_all {
        writer.write_u16(*arg)?;
    }

    // Write numbers as u64 LE
    for num_str in numbers_exps {
        let num: u64 = num_str.parse().unwrap_or(0);
        writer.write_u64(num)?;
    }

    writer.end_write_section()?;
    Ok(())
}

fn write_constraints_section(
    writer: &mut BinFileWriter,
    constraints_info: &[ExpsInfo],
    numbers_constraints: &[String],
    section: u32,
) -> Result<()> {
    writer.start_write_section(section)?;

    let mut ops_all: Vec<u8> = Vec::new();
    let mut args_all: Vec<u16> = Vec::new();
    let mut ops_offsets: Vec<u32> = Vec::new();
    let mut args_offsets: Vec<u32> = Vec::new();

    for (i, constraint) in constraints_info.iter().enumerate() {
        if i == 0 {
            ops_offsets.push(0);
            args_offsets.push(0);
        } else {
            ops_offsets.push(ops_offsets[i - 1] + constraints_info[i - 1].ops.len() as u32);
            args_offsets.push(args_offsets[i - 1] + constraints_info[i - 1].args.len() as u32);
        }
        for op in &constraint.ops {
            ops_all.push(*op as u8);
        }
        for arg in &constraint.args {
            args_all.push(*arg as u16);
        }
    }

    writer.write_u32(ops_all.len() as u32)?;
    writer.write_u32(args_all.len() as u32)?;
    writer.write_u32(numbers_constraints.len() as u32)?;

    let n_constraints = constraints_info.len() as u32;
    writer.write_u32(n_constraints)?;

    for (i, constraint) in constraints_info.iter().enumerate() {
        writer.write_u32(constraint.stage as u32)?;
        writer.write_u32(constraint.dest_dim as u32)?;
        writer.write_u32(constraint.dest_id as u32)?;
        writer.write_u32(constraint.first_row as u32)?;
        writer.write_u32(constraint.last_row as u32)?;
        writer.write_u32(constraint.n_temp1 as u32)?;
        writer.write_u32(constraint.n_temp3 as u32)?;

        writer.write_u32(constraint.ops.len() as u32)?;
        writer.write_u32(ops_offsets[i])?;

        writer.write_u32(constraint.args.len() as u32)?;
        writer.write_u32(args_offsets[i])?;

        writer.write_u32(constraint.im_pol as u32)?;
        writer.write_string(&constraint.line)?;
    }

    for op in &ops_all {
        writer.write_u8(*op)?;
    }
    for arg in &args_all {
        writer.write_u16(*arg)?;
    }
    for num_str in numbers_constraints {
        let num: u64 = num_str.parse().unwrap_or(0);
        writer.write_u64(num)?;
    }

    writer.end_write_section()?;
    Ok(())
}

fn write_hints_section(writer: &mut BinFileWriter, hints_info: &[Hint], section: u32) -> Result<()> {
    writer.start_write_section(section)?;

    let n_hints = hints_info.len() as u32;
    writer.write_u32(n_hints)?;

    for hint in hints_info {
        writer.write_string(&hint.name)?;
        let n_fields = hint.fields.len() as u32;
        writer.write_u32(n_fields)?;

        for field in &hint.fields {
            writer.write_string(&field.name)?;
            let n_values = field.values.len() as u32;
            writer.write_u32(n_values)?;

            for value in &field.values {
                writer.write_string(&value.op)?;

                if value.op == "number" {
                    writer.write_u64(value.value)?;
                } else if value.op == "string" {
                    writer.write_string(&value.string_value)?;
                } else {
                    writer.write_u32(value.id as u32)?;
                }

                if value.op == "custom" || value.op == "const" || value.op == "cm" {
                    writer.write_u32(value.row_offset_index as u32)?;
                }
                if value.op == "tmp" {
                    writer.write_u32(value.dim as u32)?;
                }
                if value.op == "custom" {
                    writer.write_u32(value.commit_id as u32)?;
                }

                writer.write_u32(value.pos.len() as u32)?;
                for p in &value.pos {
                    writer.write_u32(*p as u32)?;
                }
            }
        }
    }

    writer.end_write_section()?;
    Ok(())
}

/// Writes the prover expressions binary file (3 sections: expressions, constraints, hints).
pub fn write_expressions_bin_file(
    path: &str,
    stark_info: &StarkInfo,
    expressions_info: &ExpressionsInfo,
) -> Result<()> {
    println!("> Writing the chelpers file");

    let bin = prepare_expressions_bin(stark_info, expressions_info)?;

    let mut writer = BinFileWriter::new(path, "chps", 1, CHELPERS_NSECTIONS)?;

    write_expressions_section(
        &mut writer,
        &bin.exps_info,
        &bin.numbers_exps,
        bin.max_tmp1,
        bin.max_tmp3,
        bin.max_args,
        bin.max_ops,
        CHELPERS_EXPRESSIONS_SECTION,
    )?;

    write_constraints_section(
        &mut writer,
        &bin.constraints_info,
        &bin.numbers_constraints,
        CHELPERS_CONSTRAINTS_DEBUG_SECTION,
    )?;

    write_hints_section(&mut writer, &bin.hints_info, CHELPERS_HINTS_SECTION)?;

    println!("> Writing the chelpers file finished");
    println!("---------------------------------------------");

    writer.close()?;
    Ok(())
}

/// Writes the verifier expressions binary file (1 section: expressions for q and query).
pub fn write_verifier_expressions_bin_file(
    path: &str,
    stark_info: &StarkInfo,
    verifier_info: &VerifierInfo,
) -> Result<()> {
    println!("> Writing chelpers verifier file");

    let bin = prepare_verifier_expressions_bin(stark_info, verifier_info)?;

    let ver_exps = vec![bin.q_code, bin.query_code];

    let mut writer = BinFileWriter::new(path, "chps", 1, 1)?;

    write_expressions_section(
        &mut writer,
        &ver_exps,
        &bin.numbers_exps,
        bin.max_tmp1,
        bin.max_tmp3,
        bin.max_args,
        bin.max_ops,
        CHELPERS_EXPRESSIONS_SECTION,
    )?;

    println!("> Writing the chelpers file finished");
    println!("---------------------------------------------");

    writer.close()?;
    Ok(())
}
