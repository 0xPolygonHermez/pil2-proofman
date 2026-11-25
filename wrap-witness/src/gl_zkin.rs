//! The zkin of the final circuit of a blake3 key, from its recursivef proof: a Goldilocks STARK whose
//! flat proof (`Proofs::proof2pointer` order) is cut into the inputs `proof_inputs` names.

use proofman_common::proof_inputs::{proof_inputs, Origin};
use proofman_common::StarkInfo;
use serde_json::{Map, Value};

use crate::error::{WrapWitnessError, WrapWitnessResult};

/// The zkin of the proof `proof` (the words after its publics) of publics `publics`, of the STARK
/// `stark_info`, or why the words are not those of a proof of it.
pub fn gl_proof_zkin(stark_info: &Value, proof: &[u64], publics: &[u64]) -> WrapWitnessResult<Value> {
    let bad = |e: String| WrapWitnessError::Mismatch(format!("the recursivef proof's zkin: {e}"));
    let si = StarkInfo::try_from_value(stark_info).map_err(|e| bad(e.to_string()))?;
    let mut words = proof.iter();
    let mut zkin = Map::new();
    for i in proof_inputs(&si).map_err(bad)? {
        let value = match i.origin {
            Origin::Publics if publics.len() as u64 == i.dims[0] => shape(&mut publics.iter(), &i.dims),
            Origin::Publics => return Err(bad(format!("{} publics, the starkinfo has {}", publics.len(), i.dims[0]))),
            Origin::ProofValues => return Err(bad("the final circuit takes no proof values".into())),
            Origin::Proof | Origin::Unread => shape(&mut words, &i.dims),
        };
        zkin.insert(i.name, value.ok_or_else(|| bad(format!("the proof has {} words, too few", proof.len())))?);
    }
    let rest = words.count();
    if rest != 0 {
        return Err(bad(format!("the proof has {rest} words more than its starkinfo's")));
    }
    Ok(Value::Object(zkin))
}

/// The next words in the nested arrays of `dims` (a scalar for none), as decimal strings.
fn shape<'a>(words: &mut impl Iterator<Item = &'a u64>, dims: &[u64]) -> Option<Value> {
    match dims.split_first() {
        None => words.next().map(|w| Value::String(w.to_string())),
        Some((&n, rest)) => (0..n).map(|_| shape(words, rest)).collect::<Option<Vec<_>>>().map(Value::Array),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A blake3-shaped recursivef: binary trees, 2 last levels, an empty stage 2, no grinding.
    fn stark_info() -> Value {
        json!({
            "keyVersion": proofman_common::KEY_VERSION,
            "starkStruct": {
                "verificationHashType": "GL", "numQueries": 2, "nBits": 3, "nBitsExt": 4, "grindingBitsQueries": 0,
                "merkleTreeArity": 2, "merkleTreeCustom": true, "lastLevelVerification": 2, "hashCommits": false,
                "transcriptArity": 4, "foldingFactors": [1], "logDegrees": [3, 2], "logDomainSizes": [4, 3]
            },
            "nStages": 2, "nPublics": 5, "nConstants": 1, "qDim": 3, "cExpId": 0, "security": {"regime": "test"},
            "evMap": [{"type": "cm", "id": 0, "prime": 0, "openingPos": 0}],
            "mapSectionsN": {"cm1": 1, "cm2": 0, "cm3": 3}, "customCommits": [], "customCommitsMap": [],
            "cmPolsMap": [{"name": "q", "stage": 3, "stagePos": 0, "stageId": 0, "dim": 3}],
            "challengesMap": [], "airgroupValuesMap": [], "airValuesMap": [], "proofValuesMap": [],
            "boundaries": [{"name": "everyRow"}], "openingPoints": [0], "qDeg": 1
        })
    }

    /// Every word lands in one input, the unread ones too; publics come from their own slice.
    #[test]
    fn cuts_the_flat_proof_into_the_named_inputs() {
        let si = stark_info();
        let n = proof_inputs(&StarkInfo::try_from_value(&si).unwrap()).unwrap().iter().map(|i| i.words()).sum::<u64>();
        let proof: Vec<u64> = (0..n).collect();
        let zkin = gl_proof_zkin(&si, &proof, &[7; 5]).unwrap();
        assert_eq!(zkin["publics"], json!(["7", "7", "7", "7", "7"]));
        assert_eq!(zkin["root1"], json!(["0", "1", "2", "3"]));
        // Stage 2 has no columns: its path is in the proof but it has no values.
        assert!(zkin.get("s0_siblings2").is_some() && zkin.get("s0_vals2").is_none());
        assert_eq!(zkin["s0_last_levels2"].as_array().unwrap().len(), 4);
        assert_eq!(zkin["nonce"], json!((n - 1).to_string()));
        assert!(gl_proof_zkin(&si, &proof[1..], &[7; 5]).is_err());
        assert!(gl_proof_zkin(&si, &[proof.clone(), vec![0]].concat(), &[7; 5]).is_err());
        assert!(gl_proof_zkin(&si, &proof, &[7; 4]).is_err());
    }

    /// The final circuit's main declares exactly the inputs this zkin fills, with their shapes.
    #[test]
    fn the_zkin_is_the_final_mains_inputs() {
        let shape = |v: &Value| {
            let (mut out, mut v) = (String::new(), v);
            while let Value::Array(a) = v {
                out += &format!("[{}]", a.len());
                v = &a[0];
            }
            out
        };
        let si = stark_info();
        let n = proof_inputs(&StarkInfo::try_from_value(&si).unwrap()).unwrap().iter().map(|i| i.words()).sum::<u64>();
        let zkin = gl_proof_zkin(&si, &vec![0; n as usize], &[0; 5]).unwrap();
        let mut made: Vec<String> =
            zkin.as_object().unwrap().iter().map(|(k, v)| format!("{k}{};", shape(v))).collect();
        let main = pil2_stark_recurser::stark2circom::gen_recursion_final(&si, &["v.circom".into()], None).unwrap();
        let mut declared: Vec<String> =
            main.lines().filter_map(|l| l.trim().strip_prefix("signal input ")).map(String::from).collect();
        made.sort();
        declared.sort();
        assert_eq!(made, declared);
    }
}
