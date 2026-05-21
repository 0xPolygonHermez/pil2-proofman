use anyhow::Result;

use crate::io::bin_file_writer::BinFileWriter;
use crate::io::parser_args::{get_parser_args, ExpsInfo, GlobalInfo};
use crate::types::stark_info::{GlobalConstraintsInfo, Hint, StarkInfo};

const GLOBAL_CONSTRAINTS_NSECTIONS: u32 = 2;
const GLOBAL_CONSTRAINTS_SECTION: u32 = 1;
const GLOBAL_HINTS_SECTION: u32 = 2;

/// Writes the global constraints binary file.
///
/// The file has 2 sections:
///   1. Global constraints (ops, args, numbers for each constraint)
///   2. Global hints
pub fn write_global_constraints_bin_file(
    global_info: &GlobalInfo,
    global_constraints_info: &GlobalConstraintsInfo,
    path: &str,
) -> Result<()> {
    let mut writer = BinFileWriter::new(path, "chps", 1, GLOBAL_CONSTRAINTS_NSECTIONS)?;

    let mut constraints_info: Vec<ExpsInfo> = Vec::new();
    let mut numbers: Vec<String> = Vec::new();

    // Empty StarkInfo for global mode (global constraints don't reference starkInfo fields)
    let empty_stark_info = StarkInfo::default();

    // Process each global constraint
    for constraint in &global_constraints_info.constraints {
        let result =
            get_parser_args(&empty_stark_info, &constraint.code, &mut numbers, true, false, Some(global_info), "")?;
        let mut info = result.exps_info;
        info.line = constraint.line.clone();
        constraints_info.push(info);
    }

    write_global_constraints_section(&mut writer, &constraints_info, &numbers, GLOBAL_CONSTRAINTS_SECTION)?;
    write_global_hints_section(&mut writer, &global_constraints_info.hints, GLOBAL_HINTS_SECTION)?;

    println!("> Writing the global constraints file finished");
    println!("---------------------------------------------");

    writer.close()?;
    Ok(())
}

fn write_global_constraints_section(
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
        // Global constraints section omits stage, firstRow, lastRow, imPol compared to
        // the non-global constraints section
        writer.write_u32(constraint.dest_dim as u32)?;
        writer.write_u32(constraint.dest_id as u32)?;
        writer.write_u32(constraint.n_temp1 as u32)?;
        writer.write_u32(constraint.n_temp3 as u32)?;

        writer.write_u32(constraint.ops.len() as u32)?;
        writer.write_u32(ops_offsets[i])?;

        writer.write_u32(constraint.args.len() as u32)?;
        writer.write_u32(args_offsets[i])?;

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

fn write_global_hints_section(writer: &mut BinFileWriter, hints_info: &[Hint], section: u32) -> Result<()> {
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

                match value.op.as_str() {
                    "number" => {
                        writer.write_u64(value.value)?;
                    }
                    "string" => {
                        writer.write_string(&value.string_value)?;
                    }
                    "airgroupvalue" => {
                        writer.write_u32(value.airgroup_id as u32)?;
                        writer.write_u32(value.id as u32)?;
                    }
                    "tmp" | "public" | "proofvalue" => {
                        writer.write_u32(value.id as u32)?;
                    }
                    _ => {
                        anyhow::bail!("Unknown operand type in global hint: {}", value.op);
                    }
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
