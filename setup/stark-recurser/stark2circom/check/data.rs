//! The main's data functions against the starkinfo and globalInfo they copy: `key_<name>(k)` per key and the
//! `vadcop_<name>()` of a recursion main. Each expected value is read off the artefact (a field, a list's length,
//! its stages); the derived tables (proof offsets, the DEEP grouping, `batchSize`) are not checked here.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use proofman_common::{EvMapEType, LowDegreeTest, PolMap, StarkInfo};
use serde_json::Value;

use super::syntax::{DataFn, Val};

/// How a circuit uses a key, which the key's data records.
pub struct KeyUse<'a> {
    /// The verification key the data carries; `None` when it is an input of the circuit.
    pub const_root: Option<&'a [String; 4]>,
    pub enable_input: bool,
    /// Its transcript starts from the global challenge and absorbs the air's circuitType.
    pub input_challenges: bool,
    /// (nRoots, firstRootDim) of a GL key's `ConstraintRoots` (a BN128 key folds them in `ConstraintEval`).
    pub roots: Option<(usize, usize)>,
}

fn int(x: impl TryInto<i128>) -> Val {
    Val::Int(x.try_into().unwrap_or(i128::MAX))
}

fn ints<T: Copy + TryInto<i128>>(v: &[T]) -> Val {
    Val::ints(&v.iter().map(|&x| x.try_into().unwrap_or(i128::MAX)).collect::<Vec<_>>())
}

fn stages(m: &Option<Vec<PolMap>>) -> Vec<u64> {
    m.iter().flatten().map(|p| p.stage).collect()
}

/// What `key_<name>(k)` must return for key `si` used as `u`.
pub fn expected_key_data(si: &StarkInfo, u: &KeyUse) -> Result<Vec<(&'static str, Val)>> {
    let ss = &si.stark_struct;
    let n_stages = u64::from(si.n_stages);
    let q = n_stages + 1;
    let challenges = si.challenges_map.as_deref().unwrap_or(&[]);
    let mut per_stage = vec![0u64; q as usize];
    for c in challenges.iter().filter(|c| (2..=n_stages).contains(&c.stage)) {
        per_stage[c.stage as usize] += 1;
    }
    let circuit_type = match u.input_challenges {
        true => si.circuit_type.context("starkinfo has no circuitType")?,
        false => 0,
    };
    let cm = si.cm_pols_map.as_deref().unwrap_or(&[]);
    let q_pol = cm.iter().position(|p| p.stage == q && p.stage_id == 0).context("cmPolsMap has no quotient")?;
    let q_pol_ev = (si.ev_map.iter())
        .position(|e| e.type_ == EvMapEType::Cm && e.id == q_pol as u64)
        .context("evMap has no quotient evaluation")?;
    // L_0 trees: stage trees with columns (and Q's), the fixed one, each custom commit's
    let width = |s: String| si.map_sections_n.get(&s).copied().unwrap_or(0);
    let mut widths: Vec<u64> =
        (1..=q).filter(|&s| s == q || width(format!("cm{s}")) > 0).map(|s| width(format!("cm{s}"))).collect();
    widths.push(si.n_constants);
    widths.extend(si.custom_commits.iter().map(|c| width(format!("{}0", c.name))));
    let mut tree_start = vec![0u64];
    for w in &widths {
        tree_start.push(tree_start.last().unwrap() + w);
    }
    let root_c = match u.const_root {
        Some(r) => r.iter().map(|x| x.parse::<i128>().context("rootC")).collect::<Result<Vec<_>>>()?,
        None => vec![0; 4],
    };
    let (steps, n_queries, pow_bits) = match &ss.low_degree_test {
        LowDegreeTest::Fri(f) => (f.log_domain_sizes.clone(), f.num_queries, f.grinding_bits_queries),
        // the query templates see STIR as FRI with one fold, of L_0 by k_0
        LowDegreeTest::Stir(s) => {
            let l0 = *s.log_domain_sizes.first().context("STIR without domains")?;
            let k0 = *s.folding_factors.first().context("STIR without folds")?;
            let first = |v: &[u64]| v.first().copied().context("STIR without queries");
            (vec![l0, l0 - k0], first(&s.num_queries)?, first(&s.grinding_bits_queries)?)
        }
    };
    let stir = ss.low_degree_test.stir();
    let stir_list = |f: fn(&proofman_common::StirStruct) -> &Vec<u64>| ints(stir.map_or(&[][..], |s| f(s)));
    let len = |m: &Option<Vec<PolMap>>| m.as_ref().map_or(0, Vec::len);
    Ok(vec![
        ("nBits", int(ss.n_bits)),
        ("nBitsExt", int(ss.n_bits_ext)),
        ("arity", int(ss.merkle_tree_arity)),
        ("lastLevels", int(ss.last_level_verification)),
        ("hashCommits", int(u64::from(ss.hash_commits))),
        ("nQueries", int(n_queries)),
        ("powBits", int(pow_bits)),
        ("qDeg", int(si.q_deg)),
        ("qPolEv", int(q_pol_ev)),
        ("nEvals", int(si.ev_map.len())),
        ("nPublics", int(si.n_publics)),
        ("nAirValues", int(len(&si.airvalues_map))),
        ("nAirgroupValues", int(len(&si.airgroupvalues_map))),
        ("nProofValues", int(len(&si.proofvalues_map))),
        ("nStages", int(n_stages)),
        ("nChallenges", int(challenges.len())),
        ("circuitType", int(circuit_type)),
        ("nRoots", int(u.roots.map_or(0, |r| r.0))),
        ("firstRootDim", int(u.roots.map_or(0, |r| r.1))),
        ("verkeyInput", int(u64::from(u.const_root.is_none()))),
        ("enableInput", int(u64::from(u.enable_input))),
        ("nTrees", int(widths.len())),
        ("CHALLENGES_PER_STAGE", ints(&per_stage)),
        ("AIR_VALUE_STAGE", ints(&stages(&si.airvalues_map))),
        ("AIRGROUP_VALUE_STAGE", ints(&stages(&si.airgroupvalues_map))),
        ("L0_TREE_START", ints(&tree_start)),
        ("ROOT_C", ints(&root_c)),
        ("nSteps", int(steps.len())),
        ("STEPS", ints(&steps)),
        ("stir", int(u64::from(stir.is_some()))),
        ("stirM", int(stir.map_or(0, |s| s.num_iterations()))),
        ("stirFinalBits", int(stir.and_then(|s| s.log_degrees.last().copied()).unwrap_or(0))),
        ("STIR_LOG_DOMAIN_SIZES", stir_list(|s| &s.log_domain_sizes)),
        ("STIR_FOLDING_FACTORS", stir_list(|s| &s.folding_factors)),
        ("STIR_NUM_QUERIES", stir_list(|s| &s.num_queries)),
        ("STIR_GRINDING_BITS", stir_list(|s| &s.grinding_bits_queries)),
    ])
}

/// Key `k`'s data in `fns` (the main's `key_` functions) is `want`.
pub fn check_key_data(fns: &BTreeMap<String, DataFn>, k: usize, want: &[(&str, Val)]) -> Result<()> {
    for (name, v) in want {
        let f = fns.get(*name).with_context(|| format!("the main has no key_{name}"))?;
        let got = f.value(k as u64);
        if got != v {
            bail!("key_{name}({k}) is {got:?}, its starkinfo gives {v:?}");
        }
    }
    Ok(())
}

/// What the `vadcop_` functions of a main over globalInfo `gi` must return (`global`: it checks the global
/// constraints, so it also carries their challenge count).
pub fn expected_vadcop_data(gi: &Value, global: bool) -> Result<Vec<(&'static str, Option<u64>, Val)>> {
    let num = |v: &Value, what: &str| v.as_u64().with_context(|| format!("globalInfo {what}"));
    let agg_types = gi["aggTypes"].as_array().context("globalInfo has no aggTypes")?;
    let ec = gi["curve"].as_str().is_some_and(|c| c != "None");
    let pv: Vec<u64> = (gi["proofValuesMap"].as_array().into_iter().flatten())
        .map(|v| num(&v["stage"], "proofValuesMap stage"))
        .collect::<Result<_>>()?;
    let n_proof_values = match &gi["numProofValues"] {
        Value::Array(a) => a.iter().map(|e| num(e, "numProofValues")).sum::<Result<u64>>()?,
        v => v.as_u64().unwrap_or(0),
    };
    let w = if ec { 10 } else { num(&gi["latticeSize"], "latticeSize")? };
    let airgroups = gi["air_groups"].as_array().map_or(0, Vec::len);
    let airs_0 = gi["airs"].get(0).and_then(Value::as_array).map_or(0, Vec::len);
    let mut want = vec![
        ("nPublics", None, int(num(&gi["nPublics"], "nPublics")?)),
        ("nProofValues", None, int(n_proof_values)),
        ("nPVMap", None, int(pv.len())),
        ("PV", None, ints(&pv)),
        ("W", None, int(w)),
        ("ec", None, int(u64::from(ec))),
        ("multiAir", None, int(u64::from(airgroups > 1 || airs_0 > 1))),
    ];
    for (name, k) in [("ecA", "A"), ("ecB", "B"), ("ecZ", "Z"), ("ecC1", "C1"), ("ecC2", "C2")] {
        let v = match ec {
            true => {
                let c = gi["curveConstants"][k].as_str().with_context(|| format!("globalInfo curveConstants.{k}"))?;
                let v = c
                    .split(',')
                    .map(|x| x.trim().parse::<i128>().context("a curve constant"))
                    .collect::<Result<Vec<_>>>()?;
                Val::ints(&v)
            }
            false => Val::ints(&[0; 5]),
        };
        want.push((name, None, v));
    }
    for (i, ag) in agg_types.iter().enumerate() {
        let types = (ag.as_array().context("an aggTypes entry")?.iter())
            .map(|t| num(&t["aggType"], "aggType"))
            .collect::<Result<Vec<_>>>()?;
        want.push(("nAgg", Some(i as u64), int(types.len())));
        want.push(("AGG", Some(i as u64), ints(&types)));
    }
    if global {
        let n = gi["numChallenges"].get(1).and_then(Value::as_u64).unwrap_or(0);
        want.push(("nGlobalChallenges", None, int(n)));
    }
    Ok(want)
}

pub fn check_vadcop_data(fns: &BTreeMap<String, DataFn>, want: &[(&str, Option<u64>, Val)]) -> Result<()> {
    for (name, i, v) in want {
        let f = fns.get(*name).with_context(|| format!("the main has no vadcop_{name}"))?;
        let got = f.value(i.unwrap_or(0));
        if got != v {
            let at = i.map(|i| i.to_string()).unwrap_or_default();
            bail!("vadcop_{name}({at}) is {got:?}, its globalInfo gives {v:?}");
        }
    }
    Ok(())
}
