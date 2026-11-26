//! The vadcop data a recursion main hands its static templates (`helper_circuits/vadcop/` and `helper_circuits/mains/`): `vadcop_<name>()` and, per airgroup, `vadcop_<name>(i)`.

use anyhow::{bail, Context, Result};
use serde_json::Value;

use super::parse_num_proof_values;

/// The air's (airgroupId, airId); a basic starkinfo's circuitType must be the one circom computes.
pub fn air_id_vars(stark_info: &Value) -> Result<(u64, u64)> {
    let airgroup_id = stark_info["airgroupId"].as_u64().context("starkInfo has no airgroupId")?;
    let air_id = stark_info["airId"].as_u64().context("starkInfo has no airId")?;
    let circuit_type = proofman_common::circuit_type_of(airgroup_id, air_id + 1);
    if let Some(recorded) = stark_info["circuitType"].as_u64().filter(|&r| r != circuit_type) {
        bail!("starkInfo.circuitType {recorded} disagrees with ({airgroup_id}, {air_id})");
    }
    Ok((airgroup_id, air_id))
}

/// `function vadcop_{name}(i)`: `values[i]`, else `default`.
fn indexed(name: &str, values: &[String], default: &str) -> String {
    let branches: String =
        values.iter().enumerate().map(|(i, v)| format!("if (i == {i}) {{ return {v}; }} ")).collect();
    format!("function vadcop_{name}(i) {{ {branches}return {default}; }}\n")
}

fn list<T: ToString>(v: &[T]) -> String {
    if v.is_empty() {
        "[0]".into()
    } else {
        format!("[{}]", v.iter().map(T::to_string).collect::<Vec<_>>().join(", "))
    }
}

/// The vadcop data of a recursion main: globalInfo `vi` (Null for a circuit with none), the per-airgroup
/// recursive1 keys `basic_vk` and recursive2 keys `agg_vk`, the verified air's ids.
pub fn vadcop_data(
    vi: &Value,
    basic_vk: &[Vec<Vec<String>>],
    agg_vk: &[Vec<String>],
    airgroup_id: u64,
    air_id: u64,
) -> Result<String> {
    let agg_types = vi["aggTypes"].as_array().map_or(&[][..], Vec::as_slice);
    let ec = vi["curve"].as_str().is_some_and(|c| c != "None");
    let w = if ec {
        10
    } else if vi.is_null() {
        0
    } else {
        vi["latticeSize"].as_u64().filter(|&n| n > 0).context("vadcopInfo has no latticeSize")?
    };
    let airgroups = vi["air_groups"].as_array().map_or(0, Vec::len);
    let airs_0 = vi["airs"].get(0).and_then(Value::as_array).map_or(0, Vec::len);
    let stage = |v: &Value, what: &str| v["stage"].as_u64().with_context(|| format!("{what} has no stage"));
    let pv: Vec<u64> = vi["proofValuesMap"]
        .as_array()
        .map_or(Ok(vec![]), |a| a.iter().map(|v| stage(v, "a proofValuesMap entry")).collect())?;
    let n_publics = if vi.is_null() { 0 } else { vi["nPublics"].as_u64().context("vadcopInfo has no nPublics")? };

    let mut out =
        format!("\nfunction vadcop_airgroupId() {{ return {airgroup_id}; }}\nfunction vadcop_airId() {{ return {air_id}; }}\n");
    out += &format!("function vadcop_nPublics() {{ return {n_publics}; }}\n");
    out += &format!("function vadcop_nProofValues() {{ return {}; }}\n", parse_num_proof_values(&vi["numProofValues"]));
    out += &format!("function vadcop_nPVMap() {{ return {}; }}\nfunction vadcop_PV() {{ return {}; }}\n", pv.len(), list(&pv));
    out += &format!("function vadcop_W() {{ return {w}; }}\nfunction vadcop_ec() {{ return {}; }}\n", u8::from(ec));
    for k in ["A", "B", "Z", "C1", "C2"] {
        let v = if ec {
            vi["curveConstants"][k]
                .as_str()
                .map(|v| format!("[{v}]"))
                .with_context(|| format!("vadcopInfo has no curveConstants.{k}"))?
        } else {
            "[0, 0, 0, 0, 0]".into()
        };
        out += &format!("function vadcop_ec{k}() {{ return {v}; }}\n");
    }
    out += &format!("function vadcop_multiAir() {{ return {}; }}\n", u8::from(airgroups > 1 || airs_0 > 1));
    let types: Vec<Vec<u64>> = agg_types
        .iter()
        .map(|ag| {
            let ag = ag.as_array().context("an aggTypes entry is not a list")?;
            ag.iter().map(|t| t["aggType"].as_u64().context("an aggTypes entry has no aggType")).collect()
        })
        .collect::<Result<_>>()?;
    out += &indexed("nAgg", &types.iter().map(|t| t.len().to_string()).collect::<Vec<_>>(), "0");
    out += &indexed("AGG", &types.iter().map(|t| list(t)).collect::<Vec<_>>(), "[0]");
    let root = |vk: &[String]| format!("[{}]", vk.join(", "));
    out += &indexed("nBasics", &basic_vk.iter().map(|b| b.len().to_string()).collect::<Vec<_>>(), "0");
    let basics: Vec<String> =
        basic_vk.iter().map(|b| format!("[{}]", b.iter().map(|vk| root(vk)).collect::<Vec<_>>().join(", "))).collect();
    out += &indexed("basics", &basics, "[[0, 0, 0, 0]]");
    out += &indexed("rootCAgg", &agg_vk.iter().map(|vk| root(vk)).collect::<Vec<_>>(), "[0, 0, 0, 0]");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compressor's starkinfo carries the basic air's ids but no circuitType.
    #[test]
    fn air_ids_without_a_recorded_circuit_type() {
        assert_eq!(air_id_vars(&serde_json::json!({ "airgroupId": 0, "airId": 1 })).unwrap(), (0, 1));
    }

    #[test]
    fn air_ids_reject_a_circuit_type_that_disagrees() {
        let err = air_id_vars(&serde_json::json!({ "airgroupId": 0, "airId": 1, "circuitType": 99 })).unwrap_err();
        assert!(err.to_string().contains("circuitType"), "{err}");
    }

    #[test]
    fn per_airgroup_data_is_indexed() {
        let vi = serde_json::json!({
            "aggTypes": [[{"aggType": 0}], [{"aggType": 1}, {"aggType": 0}]], "air_groups": ["a", "b"],
            "airs": [[{}], [{}]], "curve": "None", "latticeSize": 368, "nPublics": 2, "numProofValues": [1]
        });
        let vk = vec!["1".to_string(), "2".into(), "3".into(), "4".into()];
        let out = vadcop_data(&vi, &[vec![vk.clone()], vec![vk.clone(), vk.clone()]], &[vk.clone(), vk], 0, 0).unwrap();
        for f in [
            "function vadcop_nAgg(i) { if (i == 0) { return 1; } if (i == 1) { return 2; } return 0; }",
            "function vadcop_AGG(i) { if (i == 0) { return [0]; } if (i == 1) { return [1, 0]; } return [0]; }",
            "function vadcop_multiAir() { return 1; }",
            "function vadcop_W() { return 368; }",
            "if (i == 1) { return [[1, 2, 3, 4], [1, 2, 3, 4]]; }",
        ] {
            assert!(out.contains(f), "missing `{f}`:\n{out}");
        }
    }
}
