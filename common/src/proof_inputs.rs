//! The flat proof's inputs (`Proofs::proof2pointer` order): name, shape and origin. The one source of
//! the circom verifiers' offsets, the final SNARK's named zkin and its Rust producer (`gl_proof_zkin`).

use crate::{LowDegreeTest, PolMap, StarkInfo};

/// Where a proof input's value comes from in a circuit that verifies the proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The proof's flat buffer.
    Proof,
    /// The proof's flat buffer, but no verifier reads it: an empty stage's path, the nonce without grinding.
    Unread,
    /// The circuit's own `publics` (or `{prefix}_publics`).
    Publics,
    /// The circuit's `proofValues`.
    ProofValues,
}

/// A verifier input: its name and shape. An input of no words has no entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofInput {
    pub name: String,
    pub dims: Vec<u64>,
    pub origin: Origin,
}

impl ProofInput {
    /// Its words in the flat proof (0 unless it is there).
    pub fn words(&self) -> u64 {
        match self.origin {
            Origin::Proof | Origin::Unread => self.dims.iter().product(),
            Origin::Publics | Origin::ProofValues => 0,
        }
    }
}

/// The verifier's inputs in `Proofs::proof2pointer` order (publics and proof values, which the proof
/// does not carry, where the circuits wire them). A BN128 STARK has single-element roots and Poseidon
/// paths. The one validation of the Merkle arity and the low-degree schedule.
pub fn proof_inputs(si: &StarkInfo) -> Result<Vec<ProofInput>, String> {
    let ss = &si.stark_struct;
    let bn128 = ss.verification_hash_type == "BN128";
    let arity = ss.merkle_tree_arity;
    if arity < 2 || !arity.is_power_of_two() {
        return Err(format!("merkleTreeArity {arity} is not a power of two"));
    }
    let ll = ss.last_level_verification;
    let nodes = u32::try_from(ll).ok().and_then(|l| arity.checked_pow(l)).ok_or("lastLevelVerification too large")?;
    let (root, sib_w) = if bn128 { (vec![], arity) } else { (vec![4], (arity - 1) * 4) };
    let last = if bn128 { vec![nodes] } else { vec![nodes, 4] };
    let mut v = Inputs { v: Vec::new(), arity, ll, sib_w, last };

    let len = |m: &Option<Vec<PolMap>>| m.as_ref().map_or(0, Vec::len) as u64;
    v.push("publics".into(), vec![si.n_publics], Origin::Publics);
    v.push("airgroupvalues".into(), vec![len(&si.airgroupvalues_map), 3], Origin::Proof);
    v.push("airvalues".into(), vec![len(&si.airvalues_map), 3], Origin::Proof);
    v.push("proofvalues".into(), vec![len(&si.proofvalues_map), 3], Origin::ProofValues);
    let q = u64::from(si.n_stages) + 1;
    for s in 1..=q {
        v.push(format!("root{s}"), root.clone(), Origin::Proof);
    }
    v.push("evals".into(), vec![si.ev_map.len() as u64, 3], Origin::Proof);

    // Goldilocks' two-adicity bounds every domain (and the shifts below).
    let logs: Vec<u64> = match &ss.low_degree_test {
        LowDegreeTest::Stir(st) => [&st.log_domain_sizes[..], &st.log_degrees[..]].concat(),
        LowDegreeTest::Fri(f) => f.log_domain_sizes.clone(),
    };
    if logs.iter().any(|&l| l > 32) {
        return Err(format!("a low-degree domain past 2^32: {logs:?}"));
    }
    let (l0, nq) = match &ss.low_degree_test {
        LowDegreeTest::Stir(_) if bn128 => return Err("STIR is not supported for BN128 proofs".into()),
        LowDegreeTest::Stir(st) => {
            let m = st.num_iterations();
            let lens = [st.log_domain_sizes.len(), st.log_degrees.len(), st.num_queries.len() + 1];
            let folds = st.folding_factors.iter().zip(&st.log_domain_sizes).all(|(&k, &l)| k > 0 && k <= l);
            if m == 0 || lens != [m + 1; 3] || st.grinding_bits_queries.len() != m || !folds {
                return Err(format!("malformed STIR schedule: {m} folds, lengths {lens:?}"));
            }
            (st.log_domain_sizes[0], st.num_queries[0])
        }
        LowDegreeTest::Fri(f) => {
            if f.log_domain_sizes.is_empty() || f.log_domain_sizes.windows(2).any(|w| w[1] > w[0]) {
                return Err(format!("malformed FRI schedule: logDomainSizes {:?}", f.log_domain_sizes));
            }
            (f.log_domain_sizes[0], f.num_queries)
        }
    };
    // The fixed tree, the custom commits, then every stage tree (an empty one is unread).
    let sec = |k: &str| si.map_sections_n.get(k).copied().unwrap_or(0);
    v.tree("s0_", "C", si.n_constants, nq, l0, Origin::Proof);
    for cc in &si.custom_commits {
        v.tree("s0_", &format!("_{}_0", cc.name), sec(&format!("{}0", cc.name)), nq, l0, Origin::Proof);
    }
    for s in 1..=q {
        let w = sec(&format!("cm{s}"));
        let origin = if w > 0 || s == q { Origin::Proof } else { Origin::Unread };
        v.tree("s0_", &s.to_string(), w, nq, l0, origin);
    }
    match &ss.low_degree_test {
        LowDegreeTest::Stir(st) => {
            let m = st.num_iterations();
            for f in 1..=m {
                v.push(format!("s{f}_root"), root.clone(), Origin::Proof);
            }
            for (r, &k) in st.folding_factors.iter().enumerate() {
                v.tree(
                    &format!("s{}_", r + 1),
                    "",
                    3 << k,
                    st.num_queries[r],
                    st.log_domain_sizes[r] - k,
                    Origin::Proof,
                );
            }
            v.push("betas".into(), vec![m as u64 - 1, 3], Origin::Proof);
            v.push("finalPol".into(), vec![1 << st.log_degrees[m], 3], Origin::Proof);
            v.push("nonces".into(), vec![m as u64], Origin::Proof);
            for kind in ["ansCoeffs", "shakeCoeffs"] {
                for (i, &t) in st.num_queries.iter().enumerate().take(m - 1) {
                    v.push(format!("{kind}{i}"), vec![t + 1, 3], Origin::Proof);
                }
            }
        }
        LowDegreeTest::Fri(fri) => {
            let steps = &fri.log_domain_sizes;
            for f in 1..steps.len() {
                v.push(format!("s{f}_root"), root.clone(), Origin::Proof);
            }
            for f in 1..steps.len() {
                v.tree(&format!("s{f}_"), "", 3 << (steps[f - 1] - steps[f]), nq, steps[f], Origin::Proof);
            }
            v.push("finalPol".into(), vec![1 << steps[steps.len() - 1], 3], Origin::Proof);
            let nonce = if fri.grinding_bits_queries > 0 { Origin::Proof } else { Origin::Unread };
            v.push("nonce".into(), vec![], nonce);
        }
    }
    Ok(v.v)
}

/// [`proof_inputs`]' accumulator, with the key's Merkle shapes.
struct Inputs {
    v: Vec<ProofInput>,
    arity: u64,
    ll: u64,
    sib_w: u64,
    last: Vec<u64>,
}

impl Inputs {
    /// An input of no words is not one.
    fn push(&mut self, name: String, dims: Vec<u64>, origin: Origin) {
        if dims.iter().all(|&d| d > 0) {
            self.v.push(ProofInput { name, dims, origin });
        }
    }

    /// An opened tree over 2^bits leaves (`{pre}vals{sfx}`): n rows of `width` values, their paths, the
    /// levels sent in clear.
    fn tree(&mut self, pre: &str, sfx: &str, width: u64, n: u64, bits: u64, origin: Origin) {
        let levels = bits.div_ceil(u64::from(self.arity.trailing_zeros())).saturating_sub(self.ll);
        self.push(format!("{pre}vals{sfx}"), vec![n, width], origin);
        self.push(format!("{pre}siblings{sfx}"), vec![n, levels, self.sib_w], origin);
        let last = if self.ll > 0 { self.last.clone() } else { vec![0] };
        self.push(format!("{pre}last_levels{sfx}"), last, origin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Arity 4, one last level, an empty stage 2, no grinding: every word the prover writes is an input.
    fn stark_info(ldt: serde_json::Value) -> StarkInfo {
        let mut ss = json!({
            "verificationHashType": "GL", "nBits": 8, "nBitsExt": 10, "merkleTreeArity": 4, "transcriptArity": 4,
            "merkleTreeCustom": false, "lastLevelVerification": 1, "hashCommits": false
        });
        ss.as_object_mut().unwrap().extend(ldt.as_object().unwrap().clone());
        StarkInfo::try_from_value(&json!({
            "keyVersion": crate::KEY_VERSION, "starkStruct": ss, "nStages": 2, "nPublics": 3, "nConstants": 2,
            "qDim": 3, "cExpId": 0, "security": {"regime": "test"}, "qDeg": 1, "openingPoints": [0],
            "evMap": [{"type": "cm", "id": 0, "prime": 0, "openingPos": 0}], "boundaries": [{"name": "everyRow"}],
            "mapSectionsN": {"cm1": 1, "cm2": 0, "cm3": 3}, "customCommits": [], "customCommitsMap": [],
            "cmPolsMap": [], "challengesMap": [], "airgroupValuesMap": [], "airValuesMap": [], "proofValuesMap": []
        }))
        .unwrap()
    }

    fn words(si: &StarkInfo) -> u64 {
        proof_inputs(si).unwrap().iter().map(ProofInput::words).sum()
    }

    /// The inputs cover exactly the flat proof the prover writes (`ProofLayout`, `proof2pointer`).
    #[test]
    fn inputs_cover_the_flat_proof() {
        let fri = stark_info(json!({
            "numQueries": 5, "grindingBitsQueries": 0, "foldingFactors": [], "logDegrees": [], "logDomainSizes": [10, 7, 4]
        }));
        assert_eq!(words(&fri), fri.proof_layout().flat_proof_words() as u64);
        let unread: Vec<_> =
            proof_inputs(&fri).unwrap().into_iter().filter(|i| i.origin == Origin::Unread).map(|i| i.name).collect();
        assert_eq!(unread, ["s0_siblings2", "s0_last_levels2", "nonce"]);
        let stir = stark_info(json!({
            "lowDegreeTest": "STIR", "foldingFactors": [2, 2], "logDegrees": [8, 6, 4], "logDomainSizes": [10, 9, 8],
            "numQueries": [5, 3], "grindingBitsQueries": [0, 0]
        }));
        assert_eq!(words(&stir), stir.proof_layout().flat_proof_words() as u64);
    }

    #[test]
    fn malformed_schedules_are_errors() {
        let bad = |ldt| proof_inputs(&stark_info(ldt)).unwrap_err();
        assert!(bad(json!({"numQueries": 5, "grindingBitsQueries": 0, "foldingFactors": [], "logDegrees": [], "logDomainSizes": []}))
            .contains("FRI"));
        assert!(bad(json!({"numQueries": 5, "grindingBitsQueries": 0, "foldingFactors": [], "logDegrees": [], "logDomainSizes": [10, 40]}))
            .contains("2^32"));
        assert!(bad(json!({"lowDegreeTest": "STIR", "foldingFactors": [2, 2], "logDegrees": [8, 6, 4],
            "logDomainSizes": [10, 9, 8], "numQueries": [5], "grindingBitsQueries": [0, 0]}))
        .contains("STIR"));
    }
}
