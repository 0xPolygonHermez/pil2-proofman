use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::types::stark_info::{CodeOperation, CodeType, OpType, StarkInfo};

const FIELD_EXTENSION: u64 = 3;

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
        "custom3" | "commit3" | "xDivXSubXi" => 6,
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

/// Rust-source-code lines for the verifier, produced when `verify == true`.
#[derive(Debug, Clone, Default)]
pub struct ParserArgsResult {
    pub exps_info: ExpsInfo,
    /// Top-level `#[inline(never)]` helper functions emitted before the main verify function
    /// when the operation count exceeds `RUST_VERIFIER_CHUNK_SIZE`.
    pub verify_rust_helpers: Vec<String>,
    /// Lines for the body of the verify function (tmp declarations + chunk calls or inline ops).
    pub verify_rust: Vec<String>,
}

/// Number of verifier operations per `#[inline(never)]` helper chunk (matches JS).
const RUST_VERIFIER_CHUNK_SIZE: usize = 100;

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

/// Determines if two lifetime segments overlap (open intervals on the right).
fn is_intersecting(seg1: &[i64; 3], seg2: &[i64; 3]) -> bool {
    seg2[0] < seg1[1] && seg1[0] < seg2[1]
}

/// Packs lifetime segments into non-overlapping subsets using a greedy
/// closest-fit algorithm (matching the JS `temporalsSubsets`).
fn temporals_subsets(segments: &mut [[i64; 3]]) -> Vec<Vec<[i64; 3]>> {
    segments.sort_by_key(|s| s[1]);
    let mut subsets: Vec<Vec<[i64; 3]>> = Vec::new();

    for segment in segments.iter() {
        let mut closest_idx: Option<usize> = None;
        let mut min_distance = i64::MAX;

        for (i, subset) in subsets.iter().enumerate() {
            let last = subset.last().unwrap();
            if is_intersecting(segment, last) {
                continue;
            }
            let distance = (last[1] - segment[0]).abs();
            if distance < min_distance {
                min_distance = distance;
                closest_idx = Some(i);
            }
        }

        if let Some(idx) = closest_idx {
            subsets[idx].push(*segment);
        } else {
            subsets.push(vec![*segment]);
        }
    }
    subsets
}

/// Analyses tmp variable lifetimes and assigns compacted IDs.
fn get_id_maps(maxid: usize, id1d: &mut [i64], id3d: &mut [i64], code: &[CodeOperation]) -> (u64, u64) {
    let mut ini1d = vec![-1i64; maxid];
    let mut end1d = vec![-1i64; maxid];
    let mut ini3d = vec![-1i64; maxid];
    let mut end3d = vec![-1i64; maxid];

    for (j, r) in code.iter().enumerate() {
        let j = j as i64;
        // Check dest
        if r.dest.op_type == OpType::Tmp {
            let id = r.dest.id as usize;
            assert!(id < maxid, "Id exceeds maxid");
            if r.dest.dim == 1 {
                if ini1d[id] == -1 {
                    ini1d[id] = j;
                    end1d[id] = j;
                } else {
                    end1d[id] = j;
                }
            } else {
                assert_eq!(r.dest.dim, FIELD_EXTENSION);
                if ini3d[id] == -1 {
                    ini3d[id] = j;
                    end3d[id] = j;
                } else {
                    end3d[id] = j;
                }
            }
        }
        // Check sources
        for src in &r.src {
            if src.op_type == OpType::Tmp {
                let id = src.id as usize;
                assert!(id < maxid, "Id exceeds maxid");
                if src.dim == 1 {
                    if ini1d[id] == -1 {
                        ini1d[id] = j;
                        end1d[id] = j;
                    } else {
                        end1d[id] = j;
                    }
                } else {
                    assert_eq!(src.dim, FIELD_EXTENSION);
                    if ini3d[id] == -1 {
                        ini3d[id] = j;
                        end3d[id] = j;
                    } else {
                        end3d[id] = j;
                    }
                }
            }
        }
    }

    let mut segments1d: Vec<[i64; 3]> = Vec::new();
    let mut segments3d: Vec<[i64; 3]> = Vec::new();
    for j in 0..maxid {
        if ini1d[j] >= 0 {
            segments1d.push([ini1d[j], end1d[j], j as i64]);
        }
        if ini3d[j] >= 0 {
            segments3d.push([ini3d[j], end3d[j], j as i64]);
        }
    }

    let subsets1d = temporals_subsets(&mut segments1d);
    let subsets3d = temporals_subsets(&mut segments3d);

    let mut count1d: u64 = 0;
    for s in &subsets1d {
        for a in s {
            id1d[a[2] as usize] = count1d as i64;
        }
        count1d += 1;
    }
    let mut count3d: u64 = 0;
    for s in &subsets3d {
        for a in s {
            id3d[a[2] as usize] = count3d as i64;
        }
        count3d += 1;
    }
    (count1d, count3d)
}

/// Returns the type key string used in the operations map for a given CodeType.
fn get_type_key(r: &CodeType, verify: bool) -> String {
    match r.op_type {
        OpType::Cm => format!("commit{}", r.dim),
        OpType::Const | OpType::Zi if !verify => "commit1".to_string(),
        OpType::Zi if verify => "commit3".to_string(),
        OpType::Const => "commit1".to_string(),
        OpType::Custom if r.dim == 1 => "custom1".to_string(),
        OpType::Custom if r.dim == FIELD_EXTENSION => "custom3".to_string(),
        OpType::Custom => format!("custom{}", r.dim),
        OpType::XDivXSubXi => "commit3".to_string(),
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
fn get_operation(r: &CodeOperation, verify: bool) -> (String, u64, Vec<CodeType>, String, String) {
    let op = r.op.clone();
    let _dest_dim = dim_str(r.dest.dim);

    let mut srcs = r.src.clone();

    // copy operations have only 1 source - use same dim for both lookup keys
    if srcs.len() < 2 {
        let src0_dim = if !srcs.is_empty() { dim_str(srcs[0].dim) } else { "dim1".to_string() };
        let src1_dim = src0_dim.clone();
        return (op, 0, srcs, src0_dim, src1_dim);
    }

    let op_a_key = get_type_key(&srcs[0], verify);
    let op_b_key = get_type_key(&srcs[1], verify);
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

/// Generates a Rust expression string for the verifier code generator.
fn get_operation_verify(r: &CodeType, id1d: &[i64], id3d: &[i64], stark_info: &StarkInfo) -> String {
    match r.op_type {
        OpType::Tmp => {
            if r.dim == 1 {
                format!("tmp_1[{}]", id1d[r.id as usize])
            } else {
                format!("tmp_3[{}]", id3d[r.id as usize])
            }
        }
        OpType::Const => format!("vals[0][{}]", r.id),
        OpType::Cm => {
            let pol = &stark_info.cm_pols_map[r.id as usize];
            let stage = pol.stage;
            let stage_pos = pol.stage_pos;
            if r.dim == 1 {
                format!("vals[{}][{}]", stage, stage_pos)
            } else {
                format!(
                    "CubicExtensionField {{ value: [vals[{}][{}], vals[{}][{}], vals[{}][{}]] }}",
                    stage,
                    stage_pos,
                    stage,
                    stage_pos + 1,
                    stage,
                    stage_pos + 2
                )
            }
        }
        OpType::Custom => {
            format!("vals[{}][{}]", stark_info.n_stages + 1 + r.commit_id, r.id)
        }
        OpType::Public => format!("_publics[{}]", r.id),
        OpType::Eval => format!("evals[{}]", r.id),
        OpType::Challenge => format!("challenges[{}]", r.id),
        OpType::Number => {
            // r.value is already a canonical Goldilocks u64 field element;
            // use it directly (no sign-extension needed).
            format!("Goldilocks::new({})", r.value)
        }
        OpType::Airvalue => format!("air_values[{}]", r.id),
        OpType::Proofvalue => format!("proof_values[{}]", r.id),
        OpType::Airgroupvalue => format!("airgroup_values[{}]", r.id),
        OpType::Zi => format!("zi[{}]", r.boundary_id),
        OpType::XDivXSubXi => format!("xdivxsub[{}]", r.id),
        OpType::X => "xi_challenge".to_string(),
        _ => panic!("Invalid type for verify: {:?}", r.op_type),
    }
}

/// Port of getParserArgs from JS. Converts code blocks into flat arrays
/// (ops, args, numbers) suitable for the binary file format.
///
/// When `verify` is true, also generates Rust source lines for the verifier.
/// `function_name` is used as a prefix for the chunk helper names (e.g. `"q_verify"` → `"q_verify_chunk_0"`).
pub fn get_parser_args(
    stark_info: &StarkInfo,
    code_info_code: &[CodeOperation],
    numbers: &mut Vec<String>,
    global: bool,
    verify: bool,
    global_info: Option<&GlobalInfo>,
    function_name: &str,
) -> Result<ParserArgsResult> {
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
    let (count1d, count3d) = get_id_maps(maxid, &mut id1d, &mut id3d, code_info_code);

    for r in code_info_code {
        let (op, _dest_dim_val, sorted_src, src0_dim, src1_dim) = get_operation(r, verify);
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

        // Copy ops always use ops_index 0 (matching JS where operationsTypeMap["copy"]
        // is undefined -> 0, and the src1_dim lookup would also fall through to 0).
        if r.op == "copy" {
            ops.push(0);
        } else {
            let dest_dim_str = dim_str(r.dest.dim);
            let ops_index = OPERATIONS.iter().position(|entry| {
                entry.dest_type == dest_dim_str && entry.src0_type == src0_dim && entry.src1_type == src1_dim
            });
            match ops_index {
                Some(idx) => ops.push(idx as u64),
                None => bail!("Operation not considered: dest={} src0={} src1={}", dest_dim_str, src0_dim, src1_dim),
            }
        }
    }

    // Generate verifier Rust code if requested
    let mut verify_rust: Vec<String> = Vec::new();
    let mut verify_rust_helpers: Vec<String> = Vec::new();
    if verify {
        // --- Step 1: build one operation line per code instruction, tracking
        //     which parameter categories each line references (mirrors JS operationRefs).
        #[derive(Default, Clone)]
        struct OpRefs {
            challenges: bool,
            evals: bool,
            vals: bool,
            publics: bool,
            zi: bool,
            xdivxsub: bool,
            air_values: bool,
            proof_values: bool,
            airgroup_values: bool,
        }

        let mut operation_lines: Vec<String> = Vec::new();
        let mut operation_refs: Vec<OpRefs> = Vec::new();

        for r in code_info_code {
            let mut refs = OpRefs::default();

            // Track every operand's category.
            let track = |t: &CodeType, refs: &mut OpRefs| match t.op_type {
                OpType::Challenge => refs.challenges = true,
                OpType::Eval => refs.evals = true,
                OpType::Const | OpType::Cm | OpType::Custom => refs.vals = true,
                OpType::Public => refs.publics = true,
                OpType::Zi => refs.zi = true,
                OpType::XDivXSubXi => refs.xdivxsub = true,
                OpType::Airvalue => refs.air_values = true,
                OpType::Proofvalue => refs.proof_values = true,
                OpType::Airgroupvalue => refs.airgroup_values = true,
                _ => {}
            };
            if r.op != "copy" {
                for s in &r.src {
                    track(s, &mut refs);
                }
            } else {
                track(&r.src[0], &mut refs);
            }

            let line = if r.op == "copy" {
                format!(
                    "    {} = {};",
                    get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                    get_operation_verify(&r.src[0], &id1d, &id3d, stark_info)
                )
            } else if r.op == "mul" {
                if r.src[0].dim == 1 && r.src[1].dim == FIELD_EXTENSION {
                    format!(
                        "    {} = {} * {};",
                        get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[1], &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[0], &id1d, &id3d, stark_info)
                    )
                } else {
                    format!(
                        "    {} = {} * {};",
                        get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[0], &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[1], &id1d, &id3d, stark_info)
                    )
                }
            } else if r.op == "add" {
                if r.src[0].dim == 1 && r.src[1].dim == FIELD_EXTENSION {
                    format!(
                        "    {} = {} + {};",
                        get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[1], &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[0], &id1d, &id3d, stark_info)
                    )
                } else {
                    format!(
                        "    {} = {} + {};",
                        get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[0], &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[1], &id1d, &id3d, stark_info)
                    )
                }
            } else {
                // sub
                if r.src[0].dim == 1 && r.src[1].dim == FIELD_EXTENSION {
                    format!(
                        "    {} = {}.sub_from_scalar({});",
                        get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[1], &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[0], &id1d, &id3d, stark_info)
                    )
                } else {
                    format!(
                        "    {} = {} - {};",
                        get_operation_verify(&r.dest, &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[0], &id1d, &id3d, stark_info),
                        get_operation_verify(&r.src[1], &id1d, &id3d, stark_info)
                    )
                }
            };
            operation_lines.push(line);
            operation_refs.push(refs);
        }

        // --- Step 2: decide whether to chunk.
        let should_chunk = operation_lines.len() > RUST_VERIFIER_CHUNK_SIZE;

        // --- Step 3: emit helper functions when chunking.
        if should_chunk {
            let num_chunks = operation_lines.len().div_ceil(RUST_VERIFIER_CHUNK_SIZE);

            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * RUST_VERIFIER_CHUNK_SIZE;
                let chunk_end = (chunk_start + RUST_VERIFIER_CHUNK_SIZE).min(operation_lines.len());

                // Collect which parameter categories this chunk touches.
                let mut used = OpRefs::default();
                for r in &operation_refs[chunk_start..chunk_end] {
                    used.challenges |= r.challenges;
                    used.evals |= r.evals;
                    used.vals |= r.vals;
                    used.publics |= r.publics;
                    used.zi |= r.zi;
                    used.xdivxsub |= r.xdivxsub;
                    used.air_values |= r.air_values;
                    used.proof_values |= r.proof_values;
                    used.airgroup_values |= r.airgroup_values;
                }

                // Build parameter list. `count1d`/`count3d` size the tmp slices for the WHOLE
                // expression, so a chunk gets them even when it touches neither -- blake3's
                // recursive2 verifier has ~90 chunks that never index tmp_1. Test the chunk's own
                // body and underscore what it does not reference, the same treatment `_publics`
                // already gets below; the caller passes positionally, so only the name changes.
                let body = &operation_lines[chunk_start..chunk_end];
                let touches = |name: &str| body.iter().any(|l| l.contains(name));
                let mut params: Vec<&'static str> = Vec::new();
                if count1d > 0 {
                    params.push(if touches("tmp_1[") {
                        "tmp_1: &mut [Goldilocks]"
                    } else {
                        "_tmp_1: &mut [Goldilocks]"
                    });
                }
                if count3d > 0 {
                    params.push(if touches("tmp_3[") {
                        "tmp_3: &mut [CubicExtensionField<Goldilocks>]"
                    } else {
                        "_tmp_3: &mut [CubicExtensionField<Goldilocks>]"
                    });
                }

                if !global {
                    if used.challenges {
                        params.push("challenges: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.evals {
                        params.push("evals: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.vals {
                        params.push("vals: &[Vec<Goldilocks>]");
                    }
                    if used.publics {
                        params.push("_publics: &[Goldilocks]");
                    }
                    if used.zi {
                        params.push("zi: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.xdivxsub {
                        params.push("xdivxsub: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.air_values {
                        params.push("air_values: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.proof_values {
                        params.push("proof_values: &[CubicExtensionField<Goldilocks>]");
                    }
                } else {
                    if used.challenges {
                        params.push("challenges: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.publics {
                        params.push("_publics: &[Goldilocks]");
                    }
                    if used.proof_values {
                        params.push("proof_values: &[CubicExtensionField<Goldilocks>]");
                    }
                    if used.airgroup_values {
                        params.push("airgroup_values: &[CubicExtensionField<Goldilocks>]");
                    }
                }

                let chunk_name = if function_name.is_empty() {
                    format!("chunk_{}", chunk_idx)
                } else {
                    format!("{}_chunk_{}", function_name, chunk_idx)
                };

                verify_rust_helpers.push("#[inline(never)]".to_string());
                verify_rust_helpers.push("#[rustfmt::skip]".to_string());
                verify_rust_helpers.push("#[allow(clippy::all)]".to_string());
                verify_rust_helpers.push(format!("fn {}({}) {{", chunk_name, params.join(", ")));
                for line in &operation_lines[chunk_start..chunk_end] {
                    verify_rust_helpers.push(line.clone());
                }
                verify_rust_helpers.push("}\n".to_string());
            }
        }

        // --- Step 4: build main body (tmp declarations + calls or inline ops).
        if count1d > 0 {
            verify_rust.push(format!("    let mut tmp_1 = vec![Goldilocks::ZERO; {}];", count1d));
        }
        if count3d > 0 {
            verify_rust.push(format!(
                "    let mut tmp_3 = vec![CubicExtensionField {{ value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] }}; {}];",
                count3d
            ));
        }

        if should_chunk {
            let num_chunks = operation_lines.len().div_ceil(RUST_VERIFIER_CHUNK_SIZE);

            for chunk_idx in 0..num_chunks {
                let chunk_start = chunk_idx * RUST_VERIFIER_CHUNK_SIZE;
                let chunk_end = (chunk_start + RUST_VERIFIER_CHUNK_SIZE).min(operation_lines.len());

                let mut used = OpRefs::default();
                for r in &operation_refs[chunk_start..chunk_end] {
                    used.challenges |= r.challenges;
                    used.evals |= r.evals;
                    used.vals |= r.vals;
                    used.publics |= r.publics;
                    used.zi |= r.zi;
                    used.xdivxsub |= r.xdivxsub;
                    used.air_values |= r.air_values;
                    used.proof_values |= r.proof_values;
                    used.airgroup_values |= r.airgroup_values;
                }

                let mut call_args: Vec<String> = Vec::new();
                if count1d > 0 {
                    call_args.push("&mut tmp_1".to_string());
                }
                if count3d > 0 {
                    call_args.push("&mut tmp_3".to_string());
                }

                if !global {
                    if used.challenges {
                        call_args.push("challenges".to_string());
                    }
                    if used.evals {
                        call_args.push("evals".to_string());
                    }
                    if used.vals {
                        call_args.push("vals".to_string());
                    }
                    if used.publics {
                        call_args.push("_publics".to_string());
                    }
                    if used.zi {
                        call_args.push("zi".to_string());
                    }
                    if used.xdivxsub {
                        call_args.push("xdivxsub".to_string());
                    }
                    if used.air_values {
                        call_args.push("air_values".to_string());
                    }
                    if used.proof_values {
                        call_args.push("proof_values".to_string());
                    }
                } else {
                    if used.challenges {
                        call_args.push("challenges".to_string());
                    }
                    if used.publics {
                        call_args.push("_publics".to_string());
                    }
                    if used.proof_values {
                        call_args.push("proof_values".to_string());
                    }
                    if used.airgroup_values {
                        call_args.push("airgroup_values".to_string());
                    }
                }

                let chunk_name = if function_name.is_empty() {
                    format!("chunk_{}", chunk_idx)
                } else {
                    format!("{}_chunk_{}", function_name, chunk_idx)
                };
                verify_rust.push(format!("    {}({});", chunk_name, call_args.join(", ")));
            }
        } else {
            // No chunking — inline all operations directly.
            verify_rust.extend(operation_lines);
        }

        let dest_tmp = &code_info_code[code_info_code.len() - 1].dest;
        if dest_tmp.dim == 1 {
            verify_rust.push(format!("    return tmp_1[{}];", id1d[dest_tmp.id as usize]));
        } else if dest_tmp.dim == FIELD_EXTENSION {
            verify_rust.push(format!("    return tmp_3[{}];", id3d[dest_tmp.id as usize]));
        } else {
            bail!("Unknown destination dimension");
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

    Ok(ParserArgsResult {
        exps_info: ExpsInfo { n_temp1: count1d, n_temp3: count3d, ops, args, dest_dim, dest_id, ..Default::default() },
        verify_rust_helpers,
        verify_rust,
    })
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
            for i in 0..r.id as usize {
                air_value_pos += if stark_info.air_values_map[i].stage == 1 { 1 } else { FIELD_EXTENSION };
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
                for i in 0..r.id as usize {
                    airgroup_value_pos +=
                        if stark_info.airgroup_values_map[i].stage == 1 { 1 } else { FIELD_EXTENSION };
                }
                args.push(airgroup_value_pos);
            } else if let Some(gi) = global_info {
                let mut offset: u64 = 0;
                for i in 0..r.airgroup_id as usize {
                    offset += FIELD_EXTENSION * gi.agg_types[i].len() as u64;
                }
                args.push(offset + FIELD_EXTENSION * r.id);
            }
            if !global {
                args.push(0);
            }
        }
        OpType::XDivXSubXi => {
            if global {
                bail!("xDivXSub should not appear in a global constraint");
            }
            args.push(stark_info.n_stages + 3);
            args.push(r.id);
            args.push(0);
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

    #[test]
    fn test_temporals_subsets_empty() {
        let mut segments: Vec<[i64; 3]> = Vec::new();
        let result = temporals_subsets(&mut segments);
        assert!(result.is_empty());
    }

    #[test]
    fn test_temporals_subsets_non_overlapping() {
        let mut segments = vec![[0, 2, 0], [3, 5, 1], [6, 8, 2]];
        let result = temporals_subsets(&mut segments);
        // All fit in one subset since none overlap
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 3);
    }

    #[test]
    fn test_temporals_subsets_overlapping() {
        let mut segments = vec![[0, 3, 0], [1, 4, 1], [5, 7, 2]];
        let result = temporals_subsets(&mut segments);
        // First two overlap, third fits in first subset
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_is_intersecting() {
        assert!(is_intersecting(&[0, 3, 0], &[1, 4, 1]));
        assert!(!is_intersecting(&[0, 2, 0], &[2, 4, 1]));
        assert!(!is_intersecting(&[0, 2, 0], &[3, 5, 1]));
    }

    #[test]
    fn test_operation_type_code() {
        assert_eq!(operation_type_code("add"), 0);
        assert_eq!(operation_type_code("sub"), 1);
        assert_eq!(operation_type_code("mul"), 2);
        assert_eq!(operation_type_code("sub_swap"), 3);
    }
}
