//! The data of `GetSha256Inputs` (static, `helper_circuits/sha256_inputs.circom`): publics_info's
//! definitions as the template's `DEFS` var.

use anyhow::{bail, Result};
use serde_json::Value;

/// `(nPublics, nDefs, "var DEFS[..] = [..];")` of a publics_info.
pub fn sha256_defs(publics: &Value) -> Result<(u64, usize, String)> {
    let n_publics = publics["nPublics"].as_u64().unwrap_or(0);
    let defs = publics["definitions"].as_array().map_or(&[][..], Vec::as_slice);
    let mut v: Vec<u64> = Vec::with_capacity(6 * defs.len());
    for d in defs {
        let chunk = |k: usize, default: u64| d["chunks"].get(k).and_then(Value::as_u64).unwrap_or(default);
        let (n_chunks, avoid, vk) =
            (chunk(0, 1), d["avoidAlias"].as_bool() == Some(true), d["verificationKey"].as_bool() == Some(true));
        if avoid && !(vk && n_chunks == 1) {
            bail!("publics definition {}: avoidAlias needs a single-chunk verificationKey", d["name"]);
        }
        v.extend([d["initialPos"].as_u64().unwrap_or(0), d["nValues"].as_u64().unwrap_or(0), n_chunks, chunk(1, 64)]);
        v.extend([u64::from(avoid), u64::from(vk)]);
    }
    let list: Vec<String> = v.iter().map(u64::to_string).collect();
    let n = list.len().max(1);
    let body = if list.is_empty() { "0".to_string() } else { list.join(", ") };
    Ok((n_publics, defs.len(), format!("    var DEFS[{n}] = [{body}];\n")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defs_in_template_order() {
        let p = json!({ "nPublics": 5, "definitions": [
            { "name": "a", "initialPos": 0, "nValues": 1, "chunks": [2, 32] },
            { "name": "vk", "initialPos": 1, "nValues": 4, "chunks": [1, 64], "avoidAlias": true, "verificationKey": true }
        ]});
        let (n, d, defs) = sha256_defs(&p).unwrap();
        assert_eq!((n, d), (5, 2));
        assert_eq!(defs, "    var DEFS[12] = [0, 1, 2, 32, 0, 0, 1, 4, 1, 64, 1, 1];\n");
    }

    #[test]
    fn refuses_avoid_alias_outside_a_single_chunk_key() {
        let p = json!({ "nPublics": 1, "definitions": [{ "name": "x", "nValues": 1, "chunks": [1, 64], "avoidAlias": true }] });
        assert!(sha256_defs(&p).unwrap_err().to_string().contains("avoidAlias"));
    }
}
