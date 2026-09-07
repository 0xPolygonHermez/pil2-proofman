//! Rust port of the VADCOP input sub-templates:
//! - `define_vadcop_inputs.circom.ejs`
//! - `assign_vadcop_inputs.circom.ejs`
//! - `init_vadcop_inputs.circom.ejs`
//! - `agg_vadcop_inputs.circom.ejs`

use serde_json::Value;

/// Parse `numProofValues` — which is emitted as a JSON array (e.g. `[2]`) in
/// globalInfo.json.  Sum the elements, or fall back to a plain scalar.
fn parse_num_proof_values(v: &Value) -> usize {
    if let Some(arr) = v.as_array() {
        arr.iter().map(|e| e.as_u64().unwrap_or(0)).sum::<u64>() as usize
    } else {
        v.as_u64().unwrap_or(0) as usize
    }
}

// ── define_vadcop_inputs ──────────────────────────────────────────────────────

/// Port of `main_templates/vadcop/define_vadcop_inputs.circom.ejs`.
///
/// Emits signal declarations for the vadcop inter-circuit signals.
/// If `publics_names` is `Some`, each declared signal name is appended to it
/// (mirrors `options.publicsNames` in the JS).
pub fn define_vadcop_inputs(
    vadcop_info: &Value,
    airgroup_id: usize,
    prefix: &str,
    is_input: bool,
    mut publics_names: Option<&mut Vec<String>>,
) -> String {
    let mut out = String::new();
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };
    let signal_type = if is_input { "input" } else { "output" };

    let agg_types_len = vadcop_info["aggTypes"]
        .as_array()
        .and_then(|a| a.get(airgroup_id))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;

    // circuitType
    out.push_str(&format!("    signal {signal_type} {prefix_}circuitType;\n"));
    if let Some(ref mut pn) = publics_names {
        pn.push(format!("{prefix_}circuitType"));
    }

    // aggregatedProofs
    out.push_str(&format!("    signal {signal_type} {prefix_}aggregatedProofs;\n"));
    if let Some(ref mut pn) = publics_names {
        pn.push(format!("{prefix_}aggregatedProofs"));
    }

    if agg_types_len > 0 {
        out.push_str(&format!("    signal {signal_type} {prefix_}aggregationTypes[{agg_types_len}];\n"));
        if let Some(ref mut pn) = publics_names {
            pn.push(format!("{prefix_}aggregationTypes"));
        }
        out.push_str(&format!("    signal {signal_type} {prefix_}airgroupvalues[{agg_types_len}][3];\n"));
        if let Some(ref mut pn) = publics_names {
            pn.push(format!("{prefix_}airgroupvalues"));
        }
    }

    if curve == "None" {
        out.push_str(&format!("    signal {signal_type} {prefix_}stage1Hash[{lattice_size}];\n"));
    } else {
        out.push_str(&format!("    signal {signal_type} {prefix_}stage1Hash[2][5];\n"));
    }
    if let Some(ref mut pn) = publics_names {
        pn.push(format!("{prefix_}stage1Hash"));
    }

    out
}

// ── assign_vadcop_inputs ──────────────────────────────────────────────────────

/// Options for `assign_vadcop_inputs`.
#[derive(Debug, Default, Clone)]
pub struct AssignVadcopOptions {
    /// Prefix aggregation types signal with `prefix_` (for recursive1 with compressor).
    pub add_prefix_agg_types: bool,
    /// Emit `IsZero()` enable wire (for recursive2 / vadcop final with multiple air groups).
    pub set_enable_input: bool,
    /// Emit `parallel` keyword (for recursive2).
    #[allow(dead_code)]
    pub parallel: bool,
}

/// Port of `main_templates/vadcop/assign_vadcop_inputs.circom.ejs`.
///
/// Wires all vadcop signals into the verifier `component.publics[...]`.
pub fn assign_vadcop_inputs(
    component_name: &str,
    vadcop_info: &Value,
    airgroup_id: usize,
    prefix: &str,
    _prefix_stark: &str,
    opts: &AssignVadcopOptions,
) -> String {
    let mut out = String::new();
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };

    let agg_types_len = vadcop_info["aggTypes"]
        .as_array()
        .and_then(|a| a.get(airgroup_id))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let n_publics = vadcop_info["nPublics"].as_u64().unwrap_or(0) as usize;
    let num_proof_values = parse_num_proof_values(&vadcop_info["numProofValues"]);
    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;

    let mut n = 0usize; // running publics counter

    // circuitType
    out.push_str(&format!("    {component_name}.publics[{n}] <== {prefix_}circuitType;\n"));
    n += 1;

    // aggregatedProofs
    out.push_str(&format!("    {component_name}.publics[{n}] <== {prefix_}aggregatedProofs;\n"));
    n += 1;

    if agg_types_len > 0 {
        // aggregationTypes
        let agg_prefix = if opts.add_prefix_agg_types { prefix_.as_str() } else { "" };
        out.push_str(&format!(
            "    for(var i = 0; i < {agg_types_len}; i++) {{\n        {component_name}.publics[{n} + i] <== {agg_prefix}aggregationTypes[i];\n    }}\n"
        ));
        n += agg_types_len;

        // airgroupvalues
        out.push_str(&format!(
            "    for(var i = 0; i < {agg_types_len}; i++) {{\n        {component_name}.publics[{n} + 3*i] <== {prefix_}airgroupvalues[i][0];\n        {component_name}.publics[{n} + 3*i + 1] <== {prefix_}airgroupvalues[i][1];\n        {component_name}.publics[{n} + 3*i + 2] <== {prefix_}airgroupvalues[i][2];\n    }}\n"
        ));
        n += 3 * agg_types_len;
    }

    // stage1Hash
    if curve != "None" {
        out.push_str(&format!(
            "    for (var i = 0; i < 2; i++) {{\n        for (var j = 0; j < 5; j++) {{\n            {component_name}.publics[{n} + 5*i + j] <== {prefix_}stage1Hash[i][j];\n        }}\n    }}\n"
        ));
        n += 10;
    } else {
        out.push_str(&format!(
            "    for (var i = 0; i < {lattice_size}; i++) {{\n        {component_name}.publics[{n} + i] <== {prefix_}stage1Hash[i];\n    }}\n"
        ));
        n += lattice_size;
    }

    // publics
    if n_publics > 0 {
        out.push_str(&format!(
            "    for(var i = 0; i < {n_publics}; i++) {{\n        {component_name}.publics[{n} + i] <== publics[i];\n    }}\n"
        ));
        n += n_publics;
    }

    // proofValues
    if num_proof_values > 0 {
        out.push_str(&format!(
            "    for(var i = 0; i < {num_proof_values}; i++) {{\n        {component_name}.publics[{n} + 3*i] <== proofValues[i][0];\n        {component_name}.publics[{n} + 3*i + 1] <== proofValues[i][1];\n        {component_name}.publics[{n} + 3*i + 2] <== proofValues[i][2];\n    }}\n"
        ));
        n += num_proof_values * 3;
    }

    // globalChallenge
    out.push_str(&format!(
        "    {component_name}.publics[{n}] <== globalChallenge[0];\n    {component_name}.publics[{n} +1] <== globalChallenge[1];\n    {component_name}.publics[{n} +2] <== globalChallenge[2];\n"
    ));

    // enable signal
    if opts.set_enable_input {
        out.push_str(&format!(
            "    signal {{binary}} {prefix_}isNull <== IsZero()({prefix_}circuitType);\n    {component_name}.enable <== 1 - {prefix_}isNull;\n"
        ));
    }

    out
}

// ── init_vadcop_inputs ────────────────────────────────────────────────────────

/// Port of `main_templates/vadcop/init_vadcop_inputs.circom.ejs`.
///
/// Initialises VADCOP outputs from a StarkVerifier component (compressor / recursive1 non-compressed).
pub fn init_vadcop_inputs(
    component_name: &str,
    prefix: &str,
    prefix_stark: &str,
    airgroup_id: usize,
    stark_info: &Value,
    vadcop_info: &Value,
) -> String {
    let mut out = String::new();
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };
    let prefix_stark_ = if prefix_stark.is_empty() { String::new() } else { format!("{prefix_stark}_") };

    let air_groups_len = vadcop_info["air_groups"].as_array().map(|a| a.len()).unwrap_or(0);
    let airs_0_len =
        vadcop_info["airs"].as_array().and_then(|a| a.first()).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);

    let air_id = stark_info["airId"].as_u64().unwrap_or(0);
    let circuit_type = if air_groups_len > 1 || airs_0_len > 1 { air_id + 2 } else { air_id + 1 };

    let agg_types_len = vadcop_info["aggTypes"]
        .as_array()
        .and_then(|a| a.get(airgroup_id))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let air_values_stage1_len = stark_info["airValuesMap"]
        .as_array()
        .map(|a| a.iter().filter(|v| v["stage"].as_u64() == Some(1)).count())
        .unwrap_or(0);

    // Wire globalChallenge
    out.push_str(&format!("    {component_name}.globalChallenge <== globalChallenge;\n\n"));
    out.push_str("    // --> Assign the VADCOP data\n");

    // circuitType
    out.push_str(&format!("    {prefix_}circuitType <== {circuit_type};\n"));
    // aggregatedProofs
    out.push_str(&format!("    {prefix_}aggregatedProofs <== 1;\n"));

    if agg_types_len > 0 {
        // aggregationTypes — constant values from vadcop_info
        let agg_type_vals: Vec<String> = vadcop_info["aggTypes"]
            .as_array()
            .and_then(|a| a.get(airgroup_id))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|v| v["aggType"].as_u64().unwrap_or(0).to_string()).collect())
            .unwrap_or_default();
        out.push_str(&format!("    {prefix_}aggregationTypes <== [{}];\n", agg_type_vals.join(",")));

        // airgroupvalues — pass through from stark component
        for i in 0..agg_types_len {
            out.push_str(&format!("    {prefix_}airgroupvalues[{i}] <== {prefix_stark_}airgroupvalues[{i}];\n"));
        }
    }

    // stage1Hash — computed via CalculateStage1Hash
    let air_values_map_len = stark_info["airValuesMap"].as_array().map(|a| a.len()).unwrap_or(0);
    if air_values_stage1_len > 0 && air_values_map_len > 0 {
        out.push_str(&format!(
            "    {prefix_}stage1Hash <== CalculateStage1Hash()({component_name}.rootC, {prefix_stark_}root1, {prefix_stark_}airvalues);\n"
        ));
    } else {
        out.push_str(&format!(
            "    {prefix_}stage1Hash <== CalculateStage1Hash()({component_name}.rootC, {prefix_stark_}root1);\n"
        ));
    }

    out
}

// ── agg_vadcop_inputs ─────────────────────────────────────────────────────────

/// Port of `main_templates/vadcop/agg_vadcop_inputs.circom.ejs`.
///
/// Aggregates VADCOP signals from `slot_prefixes.len()` verifier instances into one
/// output, folding them left-to-right.  Used by `recursive2`.
pub fn agg_vadcop_inputs(vadcop_info: &Value, airgroup_id: usize, slot_prefixes: &[String], prefix: &str) -> String {
    let mut out = String::new();
    let p: Vec<String> =
        slot_prefixes.iter().map(|s| if s.is_empty() { String::new() } else { format!("{s}_") }).collect();
    let n = p.len();
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };

    // Name of the accumulator after folding the first k slots: the slots' initial
    // letters, uppercased. k=2 gives "AB", which is what the arity-3 text uses.
    let acc = |k: usize| -> String {
        p[..k].iter().filter_map(|q| q.chars().next()).map(|c| c.to_ascii_uppercase()).collect()
    };

    let agg_types_len = vadcop_info["aggTypes"]
        .as_array()
        .and_then(|a| a.get(airgroup_id))
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let air_groups_len = vadcop_info["air_groups"].as_array().map(|a| a.len()).unwrap_or(0);
    let airs_0_len =
        vadcop_info["airs"].as_array().and_then(|a| a.first()).and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);

    let curve = vadcop_info["curve"].as_str().unwrap_or("None");
    let lattice_size = vadcop_info["latticeSize"].as_u64().unwrap_or(0) as usize;

    let multi_air = air_groups_len > 1 || airs_0_len > 1;

    // circuitType — constant based on topology
    let circuit_type_val = if multi_air { 1 } else { 0 };
    out.push_str(&format!("    {prefix_}circuitType <== {circuit_type_val};\n\n"));

    if agg_types_len > 0 {
        out.push_str(&format!(
            "    {prefix_}aggregationTypes <== aggregationTypes;\n    signal {{binary}} aggTypes[{agg_types_len}];\n    for (var i = 0; i < {agg_types_len}; i++) {{\n        {prefix_}aggregationTypes[i] * ({prefix_}aggregationTypes[i] - 1) === 0;\n        aggTypes[i] <== {prefix_}aggregationTypes[i];\n    }}\n\n"
        ));
    }

    if multi_air {
        // Running isNull for each intermediate accumulator.
        for k in 2..n {
            let terms: String = p[..k].iter().map(|q| format!(" - {q}isNull")).collect();
            out.push_str(&format!("    signal {{binary}} {}_isNull <== IsZero()({k}{terms});\n", acc(k)));
        }

        if agg_types_len > 0 {
            let mut decls = String::new();
            for k in 2..n {
                decls.push_str(&format!("    signal airgroupValues_{}[{agg_types_len}][3];\n", acc(k)));
            }
            let mut body = String::new();
            for k in 1..n {
                let (lhs, lhs_isnull) = if k == 1 {
                    (format!("{}airgroupvalues[i]", p[0]), format!("{}isNull", p[0]))
                } else {
                    (format!("airgroupValues_{}[i]", acc(k)), format!("{}_isNull", acc(k)))
                };
                let target = if k == n - 1 {
                    format!("{prefix_}airgroupvalues[i]")
                } else {
                    format!("airgroupValues_{}[i]", acc(k + 1))
                };
                body.push_str(&format!(
                    "        {target} <== AggregateAirgroupValuesNull()({lhs}, {}airgroupvalues[i], aggTypes[i], {lhs_isnull}, {}isNull);\n",
                    p[k], p[k]
                ));
            }
            out.push_str(&format!("{decls}    for (var i = 0; i < {agg_types_len}; i++) {{\n{body}    }}\n\n"));
        }

        let null_list = p.iter().map(|q| format!("{q}isNull")).collect::<Vec<_>>().join(", ");
        let proof_list = p.iter().map(|q| format!("{q}aggregatedProofs")).collect::<Vec<_>>().join(", ");
        out.push_str(&format!(
            "    signal {{binary}} isNull[{n}] <== [{null_list}];\n    {prefix_}aggregatedProofs <== AggregateProofsNull({n})([{proof_list}], isNull);\n\n"
        ));

        for k in 1..n {
            let (lhs, lhs_isnull) = if k == 1 {
                (format!("{}stage1Hash", p[0]), format!("{}isNull", p[0]))
            } else {
                (format!("{}_stage1Hash", acc(k)), format!("{}_isNull", acc(k)))
            };
            let call = if curve != "None" {
                format!("AccumulatePointsNull()({lhs}, {}stage1Hash, {lhs_isnull}, {}isNull)", p[k], p[k])
            } else {
                format!("AggregateValuesNull({lattice_size})({lhs}, {}stage1Hash, {lhs_isnull}, {}isNull)", p[k], p[k])
            };
            if k == n - 1 {
                out.push_str(&format!("    {prefix_}stage1Hash <== {call};\n"));
            } else {
                let name = acc(k + 1);
                let dim = if curve != "None" { "[2][5]".to_string() } else { format!("[{lattice_size}]") };
                out.push_str(&format!("    signal {name}_stage1Hash{dim} <== {call};\n"));
            }
        }
    } else {
        if agg_types_len > 0 {
            let mut decls = String::new();
            for k in 2..n {
                decls.push_str(&format!("    signal airgroupValues{}[{agg_types_len}][3];\n", acc(k)));
            }
            let mut body = String::new();
            for k in 1..n {
                let lhs =
                    if k == 1 { format!("{}airgroupvalues[i]", p[0]) } else { format!("airgroupValues{}[i]", acc(k)) };
                let target = if k == n - 1 {
                    format!("{prefix_}airgroupvalues[i]")
                } else {
                    format!("airgroupValues{}[i]", acc(k + 1))
                };
                body.push_str(&format!(
                    "        {target} <== AggregateAirgroupValues()({lhs}, {}airgroupvalues[i], aggTypes[i]);\n",
                    p[k]
                ));
            }
            out.push_str(&format!("{decls}    for (var i = 0; i < {agg_types_len}; i++) {{\n{body}    }}\n\n"));
        }

        let proof_list = p.iter().map(|q| format!("{q}aggregatedProofs")).collect::<Vec<_>>().join(", ");
        out.push_str(&format!("    {prefix_}aggregatedProofs <== AggregateProofs({n})([{proof_list}]);\n\n"));

        for k in 1..n {
            let lhs = if k == 1 { format!("{}stage1Hash", p[0]) } else { format!("{}_stage1Hash", acc(k)) };
            let call = if curve != "None" {
                format!("AccumulatePoints()({lhs}, {}stage1Hash)", p[k])
            } else {
                format!("AggregateValues({lattice_size})({lhs}, {}stage1Hash)", p[k])
            };
            if k == n - 1 {
                out.push_str(&format!("    {prefix_}stage1Hash <== {call};\n"));
            } else {
                let name = acc(k + 1);
                let dim = if curve != "None" { "[2][5]".to_string() } else { format!("[{lattice_size}]") };
                out.push_str(&format!("    signal {name}_stage1Hash{dim} <== {call};\n"));
            }
        }
    }

    out
}

#[cfg(test)]
mod fold_tests {
    use super::*;

    fn single_air() -> serde_json::Value {
        serde_json::json!({
            "aggTypes": [[{"aggType": 0, "stage": 2}]], "air_groups": ["g"],
            "airs": [[{"name": "a"}]], "curve": "None", "latticeSize": 368
        })
    }
    fn multi_air() -> serde_json::Value {
        serde_json::json!({
            "aggTypes": [[{"aggType": 0, "stage": 2}]], "air_groups": ["g"],
            "airs": [[{"name": "a"}, {"name": "b"}]], "curve": "None", "latticeSize": 368
        })
    }
    fn slots(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{}_sv", (b'a' + i as u8) as char)).collect()
    }

    // The exact text `agg_vadcop_inputs` emitted for three slots before it became a fold.
    // Inlined rather than kept as fixture files so the expected output is readable beside
    // the assertion. Byte-exact: this is the whole point of the test.
    const EXPECTED_N3_SINGLE: &str = r#"    sv_circuitType <== 0;

    sv_aggregationTypes <== aggregationTypes;
    signal {binary} aggTypes[1];
    for (var i = 0; i < 1; i++) {
        sv_aggregationTypes[i] * (sv_aggregationTypes[i] - 1) === 0;
        aggTypes[i] <== sv_aggregationTypes[i];
    }

    signal airgroupValuesAB[1][3];
    for (var i = 0; i < 1; i++) {
        airgroupValuesAB[i] <== AggregateAirgroupValues()(a_sv_airgroupvalues[i], b_sv_airgroupvalues[i], aggTypes[i]);
        sv_airgroupvalues[i] <== AggregateAirgroupValues()(airgroupValuesAB[i], c_sv_airgroupvalues[i], aggTypes[i]);
    }

    sv_aggregatedProofs <== AggregateProofs(3)([a_sv_aggregatedProofs, b_sv_aggregatedProofs, c_sv_aggregatedProofs]);

    signal AB_stage1Hash[368] <== AggregateValues(368)(a_sv_stage1Hash, b_sv_stage1Hash);
    sv_stage1Hash <== AggregateValues(368)(AB_stage1Hash, c_sv_stage1Hash);
"#;

    const EXPECTED_N3_MULTI: &str = r#"    sv_circuitType <== 1;

    sv_aggregationTypes <== aggregationTypes;
    signal {binary} aggTypes[1];
    for (var i = 0; i < 1; i++) {
        sv_aggregationTypes[i] * (sv_aggregationTypes[i] - 1) === 0;
        aggTypes[i] <== sv_aggregationTypes[i];
    }

    signal {binary} AB_isNull <== IsZero()(2 - a_sv_isNull - b_sv_isNull);
    signal airgroupValues_AB[1][3];
    for (var i = 0; i < 1; i++) {
        airgroupValues_AB[i] <== AggregateAirgroupValuesNull()(a_sv_airgroupvalues[i], b_sv_airgroupvalues[i], aggTypes[i], a_sv_isNull, b_sv_isNull);
        sv_airgroupvalues[i] <== AggregateAirgroupValuesNull()(airgroupValues_AB[i], c_sv_airgroupvalues[i], aggTypes[i], AB_isNull, c_sv_isNull);
    }

    signal {binary} isNull[3] <== [a_sv_isNull, b_sv_isNull, c_sv_isNull];
    sv_aggregatedProofs <== AggregateProofsNull(3)([a_sv_aggregatedProofs, b_sv_aggregatedProofs, c_sv_aggregatedProofs], isNull);

    signal AB_stage1Hash[368] <== AggregateValuesNull(368)(a_sv_stage1Hash, b_sv_stage1Hash, a_sv_isNull, b_sv_isNull);
    sv_stage1Hash <== AggregateValuesNull(368)(AB_stage1Hash, c_sv_stage1Hash, AB_isNull, c_sv_isNull);
"#;

    #[test]
    fn arity_three_output_is_unchanged() {
        assert_eq!(agg_vadcop_inputs(&single_air(), 0, &slots(3), "sv"), EXPECTED_N3_SINGLE);
        assert_eq!(agg_vadcop_inputs(&multi_air(), 0, &slots(3), "sv"), EXPECTED_N3_MULTI);
    }

    #[test]
    fn arity_two_folds_once_and_names_no_intermediate_twice() {
        let out = agg_vadcop_inputs(&single_air(), 0, &slots(2), "sv");
        // One combine, so exactly one AB intermediate and no reference to slot c.
        assert_eq!(out.matches("AggregateProofs(2)").count(), 1);
        assert!(!out.contains("c_sv"), "N=2 must not mention a third slot:\n{out}");
        // One combine means the result goes straight to the output signal, so the
        // fold declares no intermediate accumulator at all.
        assert!(!out.contains("AB_stage1Hash"), "N=2 needs no intermediate:\n{out}");
        assert!(out.contains("sv_stage1Hash <== "), "N=2 must write the output directly:\n{out}");
    }
}
