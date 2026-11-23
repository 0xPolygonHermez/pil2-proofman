//! Helpers shared by the setup commands.

use anyhow::{bail, Result};
use serde_json::Value;

/// Parse a verkey.json value (exactly 4 numeric u64 limbs) into [u64;4].
pub(crate) fn parse_const_root(json: &Value) -> Result<[u64; 4]> {
    let arr = json.as_array().ok_or_else(|| anyhow::anyhow!("verkey.json is not an array"))?;
    if arr.len() != 4 {
        bail!("verkey.json has {} entries, expected exactly 4", arr.len());
    }
    let mut limbs = [0u64; 4];
    for (i, v) in arr.iter().enumerate() {
        limbs[i] = v.as_u64().ok_or_else(|| anyhow::anyhow!("verkey.json limb {i} is not a valid u64: {v}"))?;
    }
    Ok(limbs)
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
