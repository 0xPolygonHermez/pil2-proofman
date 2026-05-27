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
/// Aggregates VADCOP signals from three verifier instances (a/b/c) into one output.
/// Used by `recursive2`.
pub fn agg_vadcop_inputs(
    vadcop_info: &Value,
    airgroup_id: usize,
    prefix1: &str,
    prefix2: &str,
    prefix3: &str,
    prefix: &str,
) -> String {
    let mut out = String::new();
    let prefix1_ = if prefix1.is_empty() { String::new() } else { format!("{prefix1}_") };
    let prefix2_ = if prefix2.is_empty() { String::new() } else { format!("{prefix2}_") };
    let prefix3_ = if prefix3.is_empty() { String::new() } else { format!("{prefix3}_") };
    let prefix_ = if prefix.is_empty() { String::new() } else { format!("{prefix}_") };

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
        out.push_str(&format!(
            "    signal {{binary}} AB_isNull <== IsZero()(2 - {prefix1_}isNull - {prefix2_}isNull);\n"
        ));

        if agg_types_len > 0 {
            out.push_str(&format!(
                "    signal airgroupValues_AB[{agg_types_len}][3];\n    for (var i = 0; i < {agg_types_len}; i++) {{\n        airgroupValues_AB[i] <== AggregateAirgroupValuesNull()({prefix1_}airgroupvalues[i], {prefix2_}airgroupvalues[i], aggTypes[i], {prefix1_}isNull, {prefix2_}isNull);\n        {prefix_}airgroupvalues[i] <== AggregateAirgroupValuesNull()(airgroupValues_AB[i], {prefix3_}airgroupvalues[i], aggTypes[i], AB_isNull, {prefix3_}isNull);\n    }}\n\n"
            ));
        }

        out.push_str(&format!(
            "    signal {{binary}} isNull[3] <== [{prefix1_}isNull, {prefix2_}isNull, {prefix3_}isNull];\n    {prefix_}aggregatedProofs <== AggregateProofsNull(3)([{prefix1_}aggregatedProofs, {prefix2_}aggregatedProofs, {prefix3_}aggregatedProofs], isNull);\n\n"
        ));

        if curve != "None" {
            out.push_str(&format!(
                "    signal AB_stage1Hash[2][5] <== AccumulatePointsNull()({prefix1_}stage1Hash, {prefix2_}stage1Hash, {prefix1_}isNull, {prefix2_}isNull);\n    {prefix_}stage1Hash <== AccumulatePointsNull()(AB_stage1Hash, {prefix3_}stage1Hash, AB_isNull, {prefix3_}isNull);\n"
            ));
        } else {
            out.push_str(&format!(
                "    signal AB_stage1Hash[{lattice_size}] <== AggregateValuesNull({lattice_size})({prefix1_}stage1Hash, {prefix2_}stage1Hash, {prefix1_}isNull, {prefix2_}isNull);\n    {prefix_}stage1Hash <== AggregateValuesNull({lattice_size})(AB_stage1Hash, {prefix3_}stage1Hash, AB_isNull, {prefix3_}isNull);\n"
            ));
        }
    } else {
        if agg_types_len > 0 {
            out.push_str(&format!(
                "    signal airgroupValuesAB[{agg_types_len}][3];\n    for (var i = 0; i < {agg_types_len}; i++) {{\n        airgroupValuesAB[i] <== AggregateAirgroupValues()({prefix1_}airgroupvalues[i], {prefix2_}airgroupvalues[i], aggTypes[i]);\n        {prefix_}airgroupvalues[i] <== AggregateAirgroupValues()(airgroupValuesAB[i], {prefix3_}airgroupvalues[i], aggTypes[i]);\n    }}\n\n"
            ));
        }

        out.push_str(&format!(
            "    {prefix_}aggregatedProofs <== AggregateProofs(3)([{prefix1_}aggregatedProofs, {prefix2_}aggregatedProofs, {prefix3_}aggregatedProofs]);\n\n"
        ));

        if curve != "None" {
            out.push_str(&format!(
                "    signal AB_stage1Hash[2][5] <== AccumulatePoints()({prefix1_}stage1Hash, {prefix2_}stage1Hash);\n    {prefix_}stage1Hash <== AccumulatePoints()(AB_stage1Hash, {prefix3_}stage1Hash);\n"
            ));
        } else {
            out.push_str(&format!(
                "    signal AB_stage1Hash[{lattice_size}] <== AggregateValues({lattice_size})({prefix1_}stage1Hash, {prefix2_}stage1Hash);\n    {prefix_}stage1Hash <== AggregateValues({lattice_size})(AB_stage1Hash, {prefix3_}stage1Hash);\n"
            ));
        }
    }

    out
}
