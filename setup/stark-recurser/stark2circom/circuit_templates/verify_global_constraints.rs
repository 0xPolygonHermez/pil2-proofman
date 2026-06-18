//! Rust port of `vadcop/helpers/templates/verify_global_constraints.circom.ejs`.
//!
//! Generates the `VerifyGlobalConstraints` circom template and its helper
//! `GlobalConstraint{c}_chunk{i}` templates.  The JS template splits each
//! global constraint's instruction list into chunks of ≤50 instructions to
//! keep circom compile times manageable.

use serde_json::Value;

use crate::stark2circom::transcript::Transcript;

fn parse_num_proof_values(v: &Value) -> usize {
    if let Some(arr) = v.as_array() {
        arr.iter().map(|e| e.as_u64().unwrap_or(0)).sum::<u64>() as usize
    } else {
        v.as_u64().unwrap_or(0) as usize
    }
}

const MIN_CHUNK_SIZE: usize = 50;

// ── ref helper ────────────────────────────────────────────────────────────────

/// Convert a ref object to a circom expression string (mirrors JS `ref()` function).
fn ref_expr(r: &Value, dest: bool, initialized: &[u64]) -> String {
    match r["type"].as_str().unwrap_or("") {
        "public" => {
            let id = r["id"].as_u64().unwrap_or(0);
            format!("publics[{id}]")
        }
        "proofvalue" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            if dim == 1 {
                format!("proofValues[{id}][0]")
            } else {
                format!("proofValues[{id}]")
            }
        }
        "tmp" => {
            let id = r["id"].as_u64().unwrap_or(0);
            let dim = r["dim"].as_u64().unwrap_or(1);
            if dest && !initialized.contains(&id) {
                if dim == 1 {
                    format!("signal tmp_{id}")
                } else {
                    format!("signal tmp_{id}[3]")
                }
            } else {
                format!("tmp_{id}")
            }
        }
        "number" => r["value"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| r["value"].as_u64().map(|n| n.to_string()).unwrap_or_else(|| "0".to_string())),
        "challenge" => {
            let id = r["id"].as_u64().unwrap_or(0);
            format!("challenges[{id}]")
        }
        "airgroupvalue" => {
            let airgroup_id = r["airgroupId"].as_u64().unwrap_or(0);
            let id = r["id"].as_u64().unwrap_or(0);
            format!("s{airgroup_id}_airgroupvalues[{id}]")
        }
        other => panic!("verify_global_constraints: invalid ref type '{other}'"),
    }
}

// ── unroll_code ───────────────────────────────────────────────────────────────

/// Emit circom signal assignments for a slice of constraint instructions.
fn unroll_code(code: &[Value], initialized: &[u64]) -> String {
    let mut out = String::new();
    let mut local_init: Vec<u64> = initialized.to_vec();

    for inst in code {
        let op = inst["op"].as_str().unwrap_or("");
        let dest = &inst["dest"];
        let src0 = &inst["src"][0];
        let src1 = &inst["src"][1];

        let _dest_dim = dest["dim"].as_u64().unwrap_or(1);
        let s0_dim = src0["dim"].as_u64().unwrap_or(1);
        let s1_dim = src1["dim"].as_u64().unwrap_or(1);

        let dest_str = ref_expr(dest, true, &local_init);
        // After first use, mark as initialized
        if dest["type"].as_str() == Some("tmp") {
            let id = dest["id"].as_u64().unwrap_or(0);
            if !local_init.contains(&id) {
                local_init.push(id);
            }
        }
        let s0_str = ref_expr(src0, false, &local_init);
        let s1_str = ref_expr(src1, false, &local_init);

        let line = match op {
            "add" => match (s0_dim, s1_dim) {
                (1, 1) => format!("    {dest_str} <== {s0_str} + {s1_str};\n"),
                (1, 3) => format!("    {dest_str} <== [{s0_str} + {s1_str}[0], {s1_str}[1], {s1_str}[2]];\n"),
                (3, 1) => format!("    {dest_str} <== [{s0_str}[0] + {s1_str}, {s0_str}[1], {s0_str}[2]];\n"),
                (3, 3) => format!("    {dest_str} <== [{s0_str}[0] + {s1_str}[0], {s0_str}[1] + {s1_str}[1], {s0_str}[2] + {s1_str}[2]];\n"),
                _ => panic!("unroll_code: invalid add dims {s0_dim}x{s1_dim}"),
            },
            "sub" => match (s0_dim, s1_dim) {
                (1, 1) => format!("    {dest_str} <== {s0_str} - {s1_str};\n"),
                (1, 3) => format!("    {dest_str} <== [{s0_str} - {s1_str}[0], -{s1_str}[1], -{s1_str}[2]];\n"),
                (3, 1) => format!("    {dest_str} <== [{s0_str}[0] - {s1_str}, {s0_str}[1], {s0_str}[2]];\n"),
                (3, 3) => format!("    {dest_str} <== [{s0_str}[0] - {s1_str}[0], {s0_str}[1] - {s1_str}[1], {s0_str}[2] - {s1_str}[2]];\n"),
                _ => panic!("unroll_code: invalid sub dims {s0_dim}x{s1_dim}"),
            },
            "mul" => match (s0_dim, s1_dim) {
                (1, 1) => format!("    {dest_str} <== {s0_str} * {s1_str};\n"),
                (1, 3) => format!("    {dest_str} <== [{s0_str} * {s1_str}[0], {s0_str} * {s1_str}[1], {s0_str} * {s1_str}[2]];\n"),
                (3, 1) => format!("    {dest_str} <== [{s0_str}[0] * {s1_str}, {s0_str}[1] * {s1_str}, {s0_str}[2] * {s1_str}];\n"),
                (3, 3) => format!("    {dest_str} <== CMul()({s0_str}, {s1_str});\n"),
                _ => panic!("unroll_code: invalid mul dims {s0_dim}x{s1_dim}"),
            },
            "copy" => format!("    {dest_str} <== {s0_str};\n"),
            other => panic!("unroll_code: unknown op '{other}'"),
        };
        out.push_str(&line);
    }
    out
}

// ── chunk splitting ────────────────────────────────────────────────────────────

struct Chunk {
    code: Vec<Value>,
    inputs: Vec<u64>,
    outputs: Vec<u64>,
}

/// Split a constraint's code array into chunks of ≤ MIN_CHUNK_SIZE instructions,
/// tracking which tmp IDs cross chunk boundaries as inputs/outputs.
fn split_chunks(code: &[Value]) -> Vec<Chunk> {
    // First pass: find last position of every tmp ref
    let mut last_pos: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut tmp_dims: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();

    for (i, inst) in code.iter().enumerate() {
        let dest = &inst["dest"];
        let src0 = &inst["src"][0];
        let src1 = &inst["src"][1];

        if dest["type"].as_str() == Some("tmp") {
            let id = dest["id"].as_u64().unwrap_or(0);
            last_pos.insert(id, i);
            tmp_dims.insert(id, dest["dim"].as_u64().unwrap_or(1));
        }
        if src0["type"].as_str() == Some("tmp") {
            let id = src0["id"].as_u64().unwrap_or(0);
            last_pos.insert(id, i);
        }
        if src1["type"].as_str() == Some("tmp") {
            let id = src1["id"].as_u64().unwrap_or(0);
            last_pos.insert(id, i);
        }
    }

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut current: Vec<Value> = Vec::new();

    let mut live_tmps: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut prev_live: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut outputs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut inputs: std::collections::HashSet<u64> = std::collections::HashSet::new();

    for (i, inst) in code.iter().enumerate() {
        current.push(inst.clone());

        let dest = &inst["dest"];
        let src0 = &inst["src"][0];
        let src1 = &inst["src"][1];

        if dest["type"].as_str() == Some("tmp") {
            let id = dest["id"].as_u64().unwrap_or(0);
            live_tmps.insert(id);
            outputs.insert(id);
        }
        for src in &[src0, src1] {
            if src["type"].as_str() == Some("tmp") {
                let id = src["id"].as_u64().unwrap_or(0);
                if last_pos.get(&id) == Some(&i) {
                    live_tmps.remove(&id);
                    outputs.remove(&id);
                }
                if prev_live.contains(&id) {
                    inputs.insert(id);
                    if last_pos.get(&id) == Some(&i) {
                        prev_live.remove(&id);
                    }
                }
            }
        }

        if current.len() + 1 >= MIN_CHUNK_SIZE {
            let mut ins: Vec<u64> = inputs.iter().copied().collect();
            ins.sort();
            let mut outs: Vec<u64> = outputs.iter().copied().collect();
            outs.sort();
            chunks.push(Chunk { code: current.clone(), inputs: ins, outputs: outs });

            current.clear();
            for &t in &live_tmps {
                prev_live.insert(t);
            }
            live_tmps.clear();
            outputs.clear();
            inputs.clear();
        }
    }

    if !current.is_empty() {
        let mut ins: Vec<u64> = inputs.iter().copied().collect();
        ins.sort();
        let mut outs: Vec<u64> = outputs.iter().copied().collect();
        outs.sort();
        chunks.push(Chunk { code: current, inputs: ins, outputs: outs });
    }

    chunks
}

// ── main generator ────────────────────────────────────────────────────────────

/// Port of `verify_global_constraints.circom.ejs`.
///
/// Returns the complete circom text: all `GlobalConstraint{c}_chunk{i}` helper
/// templates followed by `template VerifyGlobalConstraints() { ... }`.
pub fn gen_verify_global_constraints(stark_info: &Value, vadcop_info: &Value) -> String {
    let mut out = String::new();

    let arity = stark_info["starkStruct"]["merkleTreeArity"].as_u64().unwrap_or(2) as usize;
    let agg_types = vadcop_info["aggTypes"].as_array().cloned().unwrap_or_default();
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);
    let num_challenges_1 =
        vadcop_info["numChallenges"].as_array().and_then(|a| a.get(1)).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

    let global_constraints = vadcop_info["globalConstraints"].as_array().cloned().unwrap_or_default();

    // Pre-compute chunks for each constraint
    let all_chunks: Vec<Vec<Chunk>> = global_constraints
        .iter()
        .map(|gc| {
            let code = gc["code"].as_array().cloned().unwrap_or_default();
            split_chunks(&code)
        })
        .collect();

    // We need a tmp_dims map for declaring signals
    // We can re-derive dims from the flat code
    let get_tmp_dim = |code: &[Value], id: u64| -> u64 {
        for inst in code {
            if inst["dest"]["type"].as_str() == Some("tmp") && inst["dest"]["id"].as_u64() == Some(id) {
                return inst["dest"]["dim"].as_u64().unwrap_or(1);
            }
        }
        1
    };

    // ── Emit chunk helper templates ───────────────────────────────────────────
    for (c, gc) in global_constraints.iter().enumerate() {
        let flat_code: Vec<Value> = gc["code"].as_array().cloned().unwrap_or_default();
        let chunks = &all_chunks[c];

        for (ci, chunk) in chunks.iter().enumerate() {
            out.push_str(&format!("template GlobalConstraint{c}_chunk{ci}() {{\n"));

            // airgroupvalues inputs
            for (i, ag) in agg_types.iter().enumerate() {
                let len = ag.as_array().map(|a| a.len()).unwrap_or(0);
                out.push_str(&format!("    signal input s{i}_airgroupvalues[{len}][3];\n"));
            }

            // publics
            if n_publics > 0 {
                out.push_str(&format!("    signal input publics[{n_publics}];\n"));
            }

            // proofValues
            if num_proof_values > 0 {
                out.push_str(&format!("    signal input proofValues[{num_proof_values}][3];\n"));
            }

            // challenges
            out.push_str(&format!("    signal input challenges[{num_challenges_1}][3];\n\n"));

            // chunk inputs (tmp signals from previous chunks)
            for &id in &chunk.inputs {
                let dim = get_tmp_dim(&flat_code, id);
                if dim == 1 {
                    out.push_str(&format!("    signal input tmp_{id};\n"));
                } else {
                    out.push_str(&format!("    signal input tmp_{id}[3];\n"));
                }
            }
            out.push('\n');

            // chunk outputs
            for &id in &chunk.outputs {
                let dim = get_tmp_dim(&flat_code, id);
                if dim == 1 {
                    out.push_str(&format!("    signal output tmp_{id};\n"));
                } else {
                    out.push_str(&format!("    signal output tmp_{id}[3];\n"));
                }
            }
            out.push('\n');

            // All initialized from inputs + outputs
            let initialized: Vec<u64> = chunk.inputs.iter().chain(chunk.outputs.iter()).copied().collect();
            out.push_str(&unroll_code(&chunk.code, &initialized));

            out.push_str("}\n");
        }
    }

    // ── Emit VerifyGlobalConstraints template ─────────────────────────────────
    out.push_str("template VerifyGlobalConstraints() {\n\n");

    // Collect input signal names
    let mut inputs_list: Vec<String> = Vec::new();

    // airgroupvalues
    for (i, ag) in agg_types.iter().enumerate() {
        let len = ag.as_array().map(|a| a.len()).unwrap_or(0);
        out.push_str(&format!("    signal input s{i}_airgroupvalues[{len}][3];\n"));
        inputs_list.push(format!("s{i}_airgroupvalues"));
    }

    // publics
    if n_publics > 0 {
        out.push_str(&format!("    signal input publics[{n_publics}];\n"));
        inputs_list.push("publics".into());
    }

    // proofValues
    if num_proof_values > 0 {
        out.push_str(&format!("    signal input proofValues[{num_proof_values}][3];\n"));
        inputs_list.push("proofValues".into());
    }

    out.push_str("\n    signal input globalChallenge[3];\n\n");
    out.push_str(&format!("    signal challenges[{num_challenges_1}][3];\n"));

    let poseidon2 = vadcop_info.get("hash").and_then(|v| v.as_str()).map(|s| s == "Poseidon2").unwrap_or(true);

    // Transcript to derive challenges from globalChallenge
    let mut transcript = Transcript::new(arity, None);
    transcript.set_poseidon2(poseidon2);
    transcript.put("globalChallenge", 3);
    for i in 0..num_challenges_1 {
        transcript.get_field(&format!("challenges[{i}]"));
    }
    out.push_str(&transcript.get_code());
    out.push('\n');

    inputs_list.push("challenges".into());

    // Verify global constraints
    out.push_str("    // Verify global constraints\n");
    for (c, gc) in global_constraints.iter().enumerate() {
        let flat_code: Vec<Value> = gc["code"].as_array().cloned().unwrap_or_default();
        let chunks = &all_chunks[c];

        for (ci, chunk) in chunks.iter().enumerate() {
            // Declare output tmp signals
            for &id in &chunk.outputs {
                let dim = get_tmp_dim(&flat_code, id);
                if dim == 1 {
                    out.push_str(&format!("    signal output tmp_{id};\n"));
                } else {
                    out.push_str(&format!("    signal output tmp_{id}[3];\n"));
                }
            }

            // Instantiate chunk
            let all_args: Vec<String> =
                inputs_list.iter().cloned().chain(chunk.inputs.iter().map(|id| format!("tmp_{id}"))).collect();
            let out_list: Vec<String> = chunk.outputs.iter().map(|id| format!("tmp_{id}")).collect();
            out.push_str(&format!(
                "    ({}) <== GlobalConstraint{c}_chunk{ci}()({});\n",
                out_list.join(","),
                all_args.join(",")
            ));
        }

        // Constraint check on last instruction's dest
        if let Some(last_inst) = flat_code.last() {
            let dest_id = last_inst["dest"]["id"].as_u64().unwrap_or(0);
            let dest_dim = last_inst["dest"]["dim"].as_u64().unwrap_or(1);
            if dest_dim == 1 {
                out.push_str(&format!("    tmp_{dest_id} === 0;\n"));
            } else {
                out.push_str(&format!("    tmp_{dest_id}[0] === 0;\n"));
                out.push_str(&format!("    tmp_{dest_id}[1] === 0;\n"));
                out.push_str(&format!("    tmp_{dest_id}[2] === 0;\n"));
            }
        }
        out.push('\n');
    }

    out.push_str("}\n");
    out
}
