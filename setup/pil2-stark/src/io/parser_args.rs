use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::types::stark_info::{CodeOperation, CodeType, OpType, StarkInfo};
use pil_info::io::temporaries::get_id_maps;

const FIELD_EXTENSION: u64 = crate::types::FIELD_EXTENSION as u64;

/// Maps operation names to numeric codes used in the binary encoding.
fn operation_type_code(op: &str) -> u64 {
    match op {
        "add" | "copy" => 0, // copy maps to 0 (same as JS undefined -> 0)
        "sub" => 1,
        "mul" => 2,
        "sub_swap" => 3,
        _ => panic!("Unknown operation type: {}", op),
    }
}

/// Maps operand type strings to numeric codes for operation sorting/lookup.
fn operations_map_value(key: &str) -> u64 {
    match key {
        "commit1" | "Zi" | "const" | "custom1" => 0,
        "tmp1" => 1,
        "public" => 2,
        "number" => 3,
        "airvalue1" => 4,
        "proofvalue1" => 5,
        "custom3" | "commit3" => 6,
        "tmp3" => 7,
        "airvalue3" => 8,
        "airgroupvalue" => 9,
        "proofvalue" | "proofvalue3" => 10,
        "challenge" => 11,
        "eval" => 12,
        _ => panic!("Unknown operations map key: {}", key),
    }
}

/// Parsed expression info produced by `get_parser_args`.
#[derive(Debug, Clone, Default)]
pub struct ExpsInfo {
    pub n_temp1: u64,
    pub n_temp3: u64,
    pub ops: Vec<u64>,
    pub args: Vec<u64>,
    pub dest_dim: u64,
    pub dest_id: u64,
    /// Populated by callers after get_parser_args returns.
    pub exp_id: u64,
    pub stage: u64,
    pub line: String,
    pub first_row: u64,
    pub last_row: u64,
    pub im_pol: u64,
}

/// The three allowed dimension-combination operations.
struct OperationEntry {
    dest_type: &'static str,
    src0_type: &'static str,
    src1_type: &'static str,
}

const OPERATIONS: [OperationEntry; 3] = [
    OperationEntry { dest_type: "dim1", src0_type: "dim1", src1_type: "dim1" },
    OperationEntry { dest_type: "dim3", src0_type: "dim3", src1_type: "dim1" },
    OperationEntry { dest_type: "dim3", src0_type: "dim3", src1_type: "dim3" },
];

/// Returns the type key string used in the operations map for a given CodeType.
fn get_type_key(r: &CodeType) -> String {
    match r.op_type {
        OpType::Cm => format!("commit{}", r.dim),
        OpType::Const | OpType::Zi => "commit1".to_string(),
        OpType::Custom if r.dim == 1 => "custom1".to_string(),
        OpType::Custom if r.dim == FIELD_EXTENSION => "custom3".to_string(),
        OpType::Custom => format!("custom{}", r.dim),
        OpType::Tmp => format!("tmp{}", r.dim),
        OpType::Airvalue => format!("airvalue{}", r.dim),
        OpType::Proofvalue => format!("proofvalue{}", r.dim),
        _ => r.op_type.to_str().to_string(),
    }
}

/// Returns the dim-string for the operations table lookup.
fn dim_str(dim: u64) -> String {
    format!("dim{}", dim)
}

/// Sorts sources and determines the operation, potentially swapping to sub_swap.
fn get_operation(r: &CodeOperation) -> (String, u64, Vec<CodeType>, String, String) {
    let op = r.op.clone();
    let _dest_dim = dim_str(r.dest.dim);

    let mut srcs = r.src.clone();

    // copy operations have only 1 source - use same dim for both lookup keys
    if srcs.len() < 2 {
        let src0_dim = if !srcs.is_empty() { dim_str(srcs[0].dim) } else { "dim1".to_string() };
        let src1_dim = src0_dim.clone();
        return (op, 0, srcs, src0_dim, src1_dim);
    }

    let op_a_key = get_type_key(&srcs[0]);
    let op_b_key = get_type_key(&srcs[1]);
    let op_a = operations_map_value(&op_a_key);
    let op_b = operations_map_value(&op_b_key);

    let swap_val = if srcs[0].dim != srcs[1].dim {
        (srcs[1].dim as i64) - (srcs[0].dim as i64)
    } else {
        (op_a as i64) - (op_b as i64)
    };

    let mut final_op = op;
    if swap_val > 0 {
        srcs.swap(0, 1);
        if final_op == "sub" {
            final_op = "sub_swap".to_string();
        }
    }

    let src0_dim = dim_str(srcs[0].dim);
    let src1_dim = dim_str(srcs[1].dim);

    (final_op, r.dest.dim, srcs, src0_dim, src1_dim)
}

/// Port of getParserArgs from JS. Converts code blocks into flat arrays
/// (ops, args, numbers) suitable for the binary file format.
pub fn get_parser_args(
    stark_info: &StarkInfo,
    code_info_code: &[CodeOperation],
    numbers: &mut Vec<String>,
    global: bool,
    global_info: Option<&GlobalInfo>,
) -> Result<ExpsInfo> {
    let mut numbers_index: HashMap<String, u32> =
        numbers.iter().enumerate().map(|(i, s)| (s.clone(), i as u32)).collect();

    let mut ops: Vec<u64> = Vec::new();
    let mut args: Vec<u64> = Vec::new();

    let custom_commits = if !global { &stark_info.custom_commits } else { &vec![] };

    let maxid = code_info_code
        .iter()
        .flat_map(|r| std::iter::once(&r.dest).chain(r.src.iter()))
        .filter(|t| t.op_type == OpType::Tmp)
        .map(|t| t.id as usize + 1)
        .max()
        .unwrap_or(0);
    let mut id1d = vec![-1i64; maxid];
    let mut id3d = vec![-1i64; maxid];
    let (count1d, count3d) = get_id_maps(maxid, &mut id1d, &mut id3d, code_info_code, FIELD_EXTENSION)?;

    for r in code_info_code {
        let (op, _dest_dim_val, mut sorted_src, src0_dim, mut src1_dim) = get_operation(r);
        // The C++ reads 8 args per op (6 global): a copy is `src + 0`.
        let pads_copy = r.op == "copy";
        if pads_copy {
            sorted_src.push(CodeType { op_type: OpType::Number, dim: 1, ..Default::default() });
            src1_dim = "dim1".to_string();
        }
        args.push(operation_type_code(&op));

        push_args(
            &mut args,
            &r.dest,
            &r.dest.op_type,
            true,
            stark_info,
            custom_commits,
            &id1d,
            &id3d,
            numbers,
            &mut numbers_index,
            global,
            global_info,
        )?;
        for s in &sorted_src {
            push_args(
                &mut args,
                s,
                &s.op_type,
                false,
                stark_info,
                custom_commits,
                &id1d,
                &id3d,
                numbers,
                &mut numbers_index,
                global,
                global_info,
            )?;
        }

        let dest_dim_str = dim_str(r.dest.dim);
        let ops_index = OPERATIONS.iter().position(|entry| {
            entry.dest_type == dest_dim_str && entry.src0_type == src0_dim && entry.src1_type == src1_dim
        });
        match ops_index {
            Some(idx) => ops.push(idx as u64),
            None => bail!("Operation not considered: dest={} src0={} src1={}", dest_dim_str, src0_dim, src1_dim),
        }
    }

    let dest_tmp = &code_info_code[code_info_code.len() - 1].dest;
    let (dest_dim, dest_id) = if dest_tmp.dim == 1 {
        (1, id1d[dest_tmp.id as usize] as u64)
    } else if dest_tmp.dim == FIELD_EXTENSION {
        (FIELD_EXTENSION, id3d[dest_tmp.id as usize] as u64)
    } else {
        bail!("Unknown destination dimension");
    };

    Ok(ExpsInfo { n_temp1: count1d, n_temp3: count3d, ops, args, dest_dim, dest_id, ..Default::default() })
}

/// Information about global proof structure needed by pushArgs in global mode.
#[derive(Debug, Clone, Default)]
pub struct GlobalInfo {
    pub proof_values_map: Vec<ProofValueEntry>,
    pub agg_types: Vec<Vec<u64>>,
}

/// Entry in the proof values map or air values map.
#[derive(Debug, Clone)]
pub struct ProofValueEntry {
    pub stage: u64,
}

/// Pushes arguments for a single operand reference into the args array.
#[allow(clippy::too_many_arguments)]
fn push_args(
    args: &mut Vec<u64>,
    r: &CodeType,
    op_type: &OpType,
    dest: bool,
    stark_info: &StarkInfo,
    custom_commits: &[crate::types::stark_info::CustomCommit],
    id1d: &[i64],
    id3d: &[i64],
    numbers: &mut Vec<String>,
    numbers_index: &mut HashMap<String, u32>,
    global: bool,
    global_info: Option<&GlobalInfo>,
) -> Result<()> {
    if dest && r.op_type != OpType::Tmp {
        bail!("Invalid reference type set: {:?}", r.op_type);
    }

    let buffer_size = 1 + stark_info.n_stages + 3 + custom_commits.len() as u64;

    match op_type {
        OpType::Tmp => {
            if r.dim == 1 {
                if !dest {
                    if !global {
                        args.push(buffer_size);
                    } else {
                        args.push(0);
                    }
                }
                args.push(id1d[r.id as usize] as u64);
            } else {
                assert_eq!(r.dim, FIELD_EXTENSION);
                if !dest {
                    if !global {
                        args.push(buffer_size + 1);
                    } else {
                        args.push(4);
                    }
                }
                args.push(FIELD_EXTENSION * id3d[r.id as usize] as u64);
            }
            if !global && !dest {
                args.push(0);
            }
        }
        OpType::Const => {
            if global {
                bail!("const pols should not appear in a global constraint");
            }
            let prime_index = stark_info.opening_points.iter().position(|&p| p == r.prime).ok_or_else(|| {
                anyhow::anyhow!(
                    "opening point {} not found for const id={}, opening_points={:?}",
                    r.prime,
                    r.id,
                    stark_info.opening_points
                )
            })?;
            args.push(0);
            args.push(r.id);
            args.push(prime_index as u64);
        }
        OpType::Custom => {
            if global {
                bail!("custom pols should not appear in a global constraint");
            }
            let prime_index = stark_info
                .opening_points
                .iter()
                .position(|&p| p == r.prime)
                .ok_or_else(|| anyhow::anyhow!("opening point not found for custom"))?;
            args.push(stark_info.n_stages + 4 + r.commit_id);
            args.push(r.id);
            args.push(prime_index as u64);
        }
        OpType::Cm => {
            if global {
                bail!("witness pols should not appear in a global constraint");
            }
            let prime_index = stark_info
                .opening_points
                .iter()
                .position(|&p| p == r.prime)
                .ok_or_else(|| anyhow::anyhow!("opening point not found for cm"))?;
            args.push(stark_info.cm_pols_map[r.id as usize].stage);
            args.push(stark_info.cm_pols_map[r.id as usize].stage_pos);
            args.push(prime_index as u64);
        }
        OpType::Number => {
            // r.value is already a u64 parsed from the JSON string representation.
            // Values are already in Goldilocks field representation (e.g. p-1 = 0xFFFFFFFF00000000).
            // Do NOT apply modular reduction - the JSON stores the final u64 value directly.
            let num_string = format!("{}", r.value);
            let idx = match numbers_index.get(&num_string) {
                Some(&i) => i,
                None => {
                    let i = numbers.len() as u32;
                    numbers.push(num_string.clone());
                    numbers_index.insert(num_string, i);
                    i
                }
            };
            if !global {
                args.push(buffer_size + 3);
            } else {
                args.push(2);
            }
            args.push(idx as u64);
            if !global {
                args.push(0);
            }
        }
        OpType::Public => {
            if !global {
                args.push(buffer_size + 2);
            } else {
                args.push(1);
            }
            args.push(r.id);
            if !global {
                args.push(0);
            }
        }
        OpType::Eval => {
            if global {
                bail!("evals and airvalues should not appear in a global constraint");
            }
            args.push(buffer_size + 8);
            args.push(FIELD_EXTENSION * r.id);
            if !global {
                args.push(0);
            }
        }
        OpType::Airvalue => {
            if global {
                bail!("evals and airvalues should not appear in a global constraint");
            }
            args.push(buffer_size + 4);
            let mut air_value_pos: u64 = 0;
            for air_value in &stark_info.air_values_map[..r.id as usize] {
                air_value_pos += if air_value.stage == 1 { 1 } else { FIELD_EXTENSION };
            }
            args.push(air_value_pos);
            if !global {
                args.push(0);
            }
        }
        OpType::Proofvalue => {
            if !global {
                args.push(buffer_size + 5);
            } else {
                args.push(3);
            }
            let mut proof_value_pos: u64 = 0;
            for i in 0..r.id as usize {
                if !global {
                    proof_value_pos += if stark_info.proof_values_map[i].stage == 1 { 1 } else { FIELD_EXTENSION };
                } else if let Some(gi) = global_info {
                    proof_value_pos += if gi.proof_values_map[i].stage == 1 { 1 } else { FIELD_EXTENSION };
                }
            }
            args.push(proof_value_pos);
            if !global {
                args.push(0);
            }
        }
        OpType::Challenge => {
            if !global {
                args.push(buffer_size + 7);
            } else {
                args.push(6);
            }
            args.push(FIELD_EXTENSION * r.id);
            if !global {
                args.push(0);
            }
        }
        OpType::Airgroupvalue => {
            if !global {
                args.push(buffer_size + 6);
            } else {
                args.push(5);
            }
            if !global {
                let mut airgroup_value_pos: u64 = 0;
                for airgroup_value in &stark_info.airgroup_values_map[..r.id as usize] {
                    airgroup_value_pos += if airgroup_value.stage == 1 { 1 } else { FIELD_EXTENSION };
                }
                args.push(airgroup_value_pos);
            } else if let Some(gi) = global_info {
                let mut offset: u64 = 0;
                for agg_types in &gi.agg_types[..r.airgroup_id as usize] {
                    offset += FIELD_EXTENSION * agg_types.len() as u64;
                }
                args.push(offset + FIELD_EXTENSION * r.id);
            }
            if !global {
                args.push(0);
            }
        }
        OpType::Zi => {
            if global {
                bail!("Zerofier polynomial should not appear in a global constraint");
            }
            args.push(stark_info.n_stages + 2);
            args.push(1 + r.boundary_id);
            args.push(0);
        }
        _ => bail!("Unknown type {:?}", op_type),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::stark_info::PolMap;

    #[test]
    fn test_operation_type_code() {
        assert_eq!(operation_type_code("add"), 0);
        assert_eq!(operation_type_code("sub"), 1);
        assert_eq!(operation_type_code("mul"), 2);
        assert_eq!(operation_type_code("sub_swap"), 3);
    }

    fn copy_op(dest_dim: u64, src: CodeType) -> CodeOperation {
        let dest = CodeType { op_type: OpType::Tmp, id: 0, dim: dest_dim, ..Default::default() };
        CodeOperation { op: "copy".to_string(), dest, src: vec![src] }
    }

    fn parse(code: &[CodeOperation], global: bool) -> (ExpsInfo, Vec<String>) {
        let info = StarkInfo {
            n_stages: 2,
            opening_points: vec![0],
            cm_pols_map: vec![PolMap { stage: 1, stage_pos: 0, dim: 1, ..Default::default() }],
            ..Default::default()
        };
        let mut numbers = Vec::new();
        (get_parser_args(&info, code, &mut numbers, global, None).unwrap(), numbers)
    }

    // A copy is `src + 0`: the C++ strides 8 args per op (6 global), so a 1-operand copy would desync it.
    #[test]
    fn copy_serializes_as_src_plus_zero() {
        let cm = CodeType { op_type: OpType::Cm, id: 0, dim: 1, ..Default::default() };
        let (exps, numbers) = parse(&[copy_op(1, cm)], false);
        assert_eq!((exps.ops.as_slice(), exps.args.len()), (&[0][..], 8));
        assert_eq!(numbers[exps.args[6] as usize], "0");

        // dim3 source: the (dim3, dim3, dim1) op
        let cm3 = CodeType { op_type: OpType::Cm, id: 0, dim: 3, ..Default::default() };
        let (exps, numbers) = parse(&[copy_op(3, cm3)], false);
        assert_eq!((exps.ops.as_slice(), exps.args.len()), (&[1][..], 8));
        assert_eq!(numbers[exps.args[6] as usize], "0");

        let five = CodeType { op_type: OpType::Number, dim: 1, value: 5, ..Default::default() };
        let (exps, numbers) = parse(&[copy_op(1, five)], true);
        assert_eq!((exps.ops.as_slice(), exps.args.len()), (&[0][..], 6));
        assert_eq!((numbers[exps.args[3] as usize].as_str(), numbers[exps.args[5] as usize].as_str()), ("5", "0"));
    }
}
