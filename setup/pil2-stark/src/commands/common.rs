//! Helpers shared by the setup commands.

use anyhow::{bail, Result};
use serde_json::Value;

/// Parse a vadcop_final.verkey.json array ([[u64;4]]) into [u64;4].
pub(crate) fn parse_const_root(json: &Value) -> Result<[u64; 4]> {
    let arr = json.as_array().ok_or_else(|| anyhow::anyhow!("verkey.json is not an array"))?;
    if arr.len() < 4 {
        bail!("verkey.json has {} elements, expected 4", arr.len());
    }
    let parse_one = |v: &Value, idx: usize| -> Result<u64> {
        v.as_u64()
            .or_else(|| v.as_str()?.parse::<u64>().ok())
            .ok_or_else(|| anyhow::anyhow!("verkey.json element {} is not a valid u64: {}", idx, v))
    };
    Ok([parse_one(&arr[0], 0)?, parse_one(&arr[1], 1)?, parse_one(&arr[2], 2)?, parse_one(&arr[3], 3)?])
}

/// Append the intermediate-polynomial section of a per-AIR stats entry, then the blank separator line.
pub(crate) fn push_im_pols_stats(lines: &mut Vec<String>, base_field: &[String], extended_field: &[String]) {
    if !base_field.is_empty() {
        lines.push("Intermediate polynomials baseField:".to_string());
        lines.extend(base_field.iter().map(|pol| format!("    {pol}")));
    }
    if !extended_field.is_empty() {
        lines.push("Intermediate polynomials extendedField:".to_string());
        lines.extend(extended_field.iter().map(|pol| format!("    {pol}")));
    }
    lines.push(String::new());
}
