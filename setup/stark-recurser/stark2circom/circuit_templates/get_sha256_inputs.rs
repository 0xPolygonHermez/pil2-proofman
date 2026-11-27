//! The data of `GetSha256Inputs` (static, `helper_circuits/mains/sha256_inputs.circom`): publics_info's
//! definitions as the final circuit's `vadcop_DEFS()`.

use anyhow::{bail, Context, Result};
use proofman_common::PublicsInfo;
use serde::Deserialize;
use serde_json::Value;

/// The program verification key in the publics: its `initialPos`, and whether it is the first
/// definition (else it is the last).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramVk {
    pub pos: usize,
    pub first: bool,
}

/// A validated publics_info as the final circuit and the Solidity verifier read it.
#[derive(Debug)]
pub struct Sha256Defs {
    pub n_publics: usize,
    pub n_defs: usize,
    /// `function vadcop_DEFS() { return [..]; }`.
    pub defs: String,
    pub vk: Option<ProgramVk>,
}

/// The [`Sha256Defs`] of a publics_info (`None` or null: no publics). FinalSnark binds and Solidity
/// packs only a single 4-word VK that is the first or the last definition, so any other is refused.
pub fn sha256_defs(publics: Option<&Value>) -> Result<Sha256Defs> {
    let info = match publics {
        None | Some(Value::Null) => None,
        Some(p) => Some(PublicsInfo::deserialize(p).context("publics info")?),
    };
    let defs = info.as_ref().map_or(&[][..], |i| i.definitions.as_slice());
    let (mut v, mut vk) = (Vec::with_capacity(6 * defs.len()), None);
    for (k, d) in defs.iter().enumerate() {
        // An aggregate's VK is bound to rootC words, which need all 64 bits.
        if d.avoid_alias {
            bail!("publics definition {}: avoidAlias is not supported", d.name);
        }
        if d.verification_key {
            if vk.is_some() {
                bail!("publics definition {}: a second verificationKey", d.name);
            }
            if k != 0 && k + 1 != defs.len() {
                bail!("publics definition {}: the verificationKey must be the first or the last definition", d.name);
            }
            if d.n_values != 4 || d.chunks != [1, 64] {
                bail!("publics definition {}: the program VK needs nValues 4 and chunks [1, 64]", d.name);
            }
            vk = Some(ProgramVk { pos: d.initial_pos, first: k == 0 });
        }
        v.extend([d.initial_pos, d.n_values, d.chunks[0], d.chunks[1], 0, usize::from(d.verification_key)]);
    }
    let list: Vec<String> = v.iter().map(usize::to_string).collect();
    let body = if list.is_empty() { "0".to_string() } else { list.join(", ") };
    Ok(Sha256Defs {
        n_publics: info.as_ref().map_or(0, |i| i.n_publics),
        n_defs: defs.len(),
        defs: format!("function vadcop_DEFS() {{ return [{body}]; }}\n"),
        vk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vk_def(pos: usize) -> Value {
        json!({ "name": "vk", "initialPos": pos, "nValues": 4, "chunks": [1, 64], "verificationKey": true })
    }

    /// A VK first or last; chunks default to one 64-bit word.
    #[test]
    fn defs_in_template_order() {
        let p = json!({ "nPublics": 6, "definitions": [{ "name": "a", "initialPos": 0, "nValues": 1, "chunks": [2, 32] }, vk_def(2)] });
        let s = sha256_defs(Some(&p)).unwrap();
        assert_eq!((s.n_publics, s.n_defs, s.vk), (6, 2, Some(ProgramVk { pos: 2, first: false })));
        assert_eq!(s.defs, "function vadcop_DEFS() { return [0, 1, 2, 32, 0, 0, 2, 4, 1, 64, 0, 1]; }\n");
        let p = json!({ "nPublics": 5, "definitions": [vk_def(0), { "name": "a", "initialPos": 4, "nValues": 1 }] });
        let s = sha256_defs(Some(&p)).unwrap();
        assert_eq!(s.vk, Some(ProgramVk { pos: 0, first: true }));
        assert_eq!(s.defs, "function vadcop_DEFS() { return [0, 4, 1, 64, 0, 1, 4, 1, 1, 64, 0, 0]; }\n");
        let s = sha256_defs(Some(&Value::Null)).unwrap();
        assert_eq!(
            (s.n_publics, s.n_defs, s.vk, s.defs.as_str()),
            (0, 0, None, "function vadcop_DEFS() { return [0]; }\n")
        );
    }

    /// Only a single [1, 64] x 4 VK, first or last, without avoidAlias, is accepted.
    #[test]
    fn refuses_a_vk_the_final_circuit_cannot_bind() {
        let refused = |defs: Value, msg: &str| {
            let err = sha256_defs(Some(&json!({ "nPublics": 9, "definitions": defs }))).unwrap_err().to_string();
            assert!(err.contains(msg), "{err}");
        };
        let mut aliased = vk_def(0);
        aliased["avoidAlias"] = json!(true);
        refused(json!([aliased]), "avoidAlias");
        refused(json!([{ "name": "x", "initialPos": 0, "nValues": 1, "avoidAlias": true }]), "avoidAlias");
        refused(json!([vk_def(0), vk_def(4)]), "second verificationKey");
        let x = json!({ "name": "x", "initialPos": 8, "nValues": 1 });
        refused(json!([{ "name": "a", "initialPos": 0, "nValues": 1 }, vk_def(1), x]), "first or the last");
        let mut halves = vk_def(0);
        halves["chunks"] = json!([2, 32]);
        refused(json!([halves]), "chunks [1, 64]");
        let mut short = vk_def(0);
        short["nValues"] = json!(1);
        refused(json!([short]), "nValues 4");
    }
}
