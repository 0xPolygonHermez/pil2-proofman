//! The main's data functions against the starkinfo and globalInfo they copy: `key_<name>(k)` per key and the
//! `vadcop_<name>()` of a recursion main. Each expected value is read off the artefact (a field, a list's length,
//! its stages); the proof offsets and the DEEP tables have their own modules ([`super::proof`], [`super::deep`]).

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use proofman_common::{EvMapEType, LowDegreeTest, PolMap, StarkInfo};
use serde_json::Value;

use super::syntax::{DataFn, Val};
use super::Vadcop;

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

/// An L_0 tree the verifier opens.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum L0Tree {
    Stage(u64),
    Fixed,
    Custom(usize),
}

/// The L_0 trees and their widths in the main's order: stage trees with columns (and Q's), the fixed one, each
/// custom commit's.
pub fn l0_trees(si: &StarkInfo) -> Vec<(L0Tree, u64)> {
    let q = u64::from(si.n_stages) + 1;
    let width = |s: String| si.map_sections_n.get(&s).copied().unwrap_or(0);
    let mut trees: Vec<(L0Tree, u64)> = (1..=q)
        .filter(|&s| s == q || width(format!("cm{s}")) > 0)
        .map(|s| (L0Tree::Stage(s), width(format!("cm{s}"))))
        .collect();
    trees.push((L0Tree::Fixed, si.n_constants));
    trees.extend(
        si.custom_commits.iter().enumerate().map(|(c, cc)| (L0Tree::Custom(c), width(format!("{}0", cc.name)))),
    );
    trees
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
    let widths: Vec<u64> = l0_trees(si).into_iter().map(|t| t.1).collect();
    let mut tree_start = vec![0u64];
    for w in &widths {
        tree_start.push(tree_start.last().unwrap() + w);
    }
    // a BN128 key's root is one field element of ~254 bits
    let root_c = match u.const_root {
        Some(r) => Val::List(r.iter().map(|x| Val::decimal(x).context("rootC")).collect::<Result<Vec<_>>>()?),
        None => Val::ints(&[0; 4]),
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
        ("ROOT_C", root_c),
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

/// The queries go in batches of `key_batchSize(k)` and one tail batch, so any size in 1..=nQueries covers them all.
pub fn check_batch_size(fns: &BTreeMap<String, DataFn>, k: usize) -> Result<()> {
    let get = |name: &str| fns.get(name).map(|f| f.value(k as u64)).with_context(|| format!("no key_{name}"));
    let (Val::Int(batch), Val::Int(n_queries)) = (get("batchSize")?, get("nQueries")?) else {
        bail!("key_batchSize({k}) or key_nQueries({k}) is not a number");
    };
    ensure!((1..=*n_queries).contains(batch), "key_batchSize({k}) is {batch}, outside 1..={n_queries}");
    Ok(())
}

/// What the `vadcop_` functions of a main over globalInfo `gi` that verifies `role` must return; `first_key` is
/// the starkinfo of its first key.
pub fn expected_vadcop_data(
    gi: &Value,
    role: &Vadcop,
    first_key: Option<&Value>,
) -> Result<Vec<(&'static str, Option<u64>, Val)>> {
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
    let airs: Vec<u64> =
        (gi["airs"].as_array().into_iter().flatten()).map(|a| a.as_array().map_or(0, Vec::len) as u64).collect();
    let (airgroup, air) = match role {
        Vadcop::Air => {
            let si = first_key.context("a recursion main without its key")?;
            let (ag, air) = (num(&si["airgroupId"], "key airgroupId")?, num(&si["airId"], "key airId")?);
            ensure!(airs.get(ag as usize).is_some_and(|&n| air < n), "air ({ag}, {air}) is not in globalInfo");
            (ag, air)
        }
        Vadcop::Airgroup { airgroup, .. } => (*airgroup, 0),
        _ => (0, 0),
    };
    want.push(("airgroupId", None, int(airgroup)));
    want.push(("airId", None, int(air)));
    if let Vadcop::Final { recursive1, recursive2 } = role {
        ensure!(
            recursive1.len() == airs.len() && recursive2.len() == airs.len(),
            "{} airgroups, {} recursive1 key sets and {} recursive2 keys",
            airs.len(),
            recursive1.len(),
            recursive2.len()
        );
    }
    let key = |k: &[u64; 4]| ints(&k[..]);
    for (i, &n_airs) in airs.iter().enumerate() {
        let basics: &[[u64; 4]] = match role {
            Vadcop::Airgroup { airgroup, recursive1 } if *airgroup == i as u64 => recursive1,
            Vadcop::Final { recursive1, .. } => &recursive1[i],
            _ => &[],
        };
        ensure!(
            basics.is_empty() || basics.len() as u64 == n_airs,
            "airgroup {i} has {n_airs} airs and {} recursive1 keys",
            basics.len()
        );
        let listed = match basics {
            [] => Val::List(vec![Val::ints(&[0; 4])]),
            b => Val::List(b.iter().map(key).collect()),
        };
        let agg = match role {
            Vadcop::Final { recursive2, .. } => key(&recursive2[i]),
            _ => Val::ints(&[0; 4]),
        };
        want.push(("nBasics", Some(i as u64), int(basics.len())));
        want.push(("basics", Some(i as u64), listed));
        want.push(("rootCAgg", Some(i as u64), agg));
    }
    if matches!(role, Vadcop::Final { .. }) {
        let n = gi["numChallenges"].get(1).and_then(Value::as_u64).unwrap_or(0);
        want.push(("nGlobalChallenges", None, int(n)));
    }
    Ok(want)
}

/// The final main's `vadcop_DEFS()` and its `FinalSnark(nPublics, nDefs, vkPos)` against `publics_info` (Null:
/// no publics): one `[initialPos, nValues, nChunks, bitsPerChunk, avoidAlias, verificationKey]` per definition,
/// each within the publics.
pub fn check_publics_defs(fns: &BTreeMap<String, DataFn>, main: &str, publics_info: &Value) -> Result<()> {
    let n_publics = publics_info["nPublics"].as_u64().unwrap_or(0);
    let (mut flat, mut vk_pos) = (Vec::new(), -1i128);
    let defs = publics_info["definitions"].as_array().map_or(&[][..], Vec::as_slice);
    for d in defs {
        let num = |k: &str| d[k].as_u64().with_context(|| format!("publics definition {d}: {k}"));
        let (pos, n) = (num("initialPos")?, num("nValues")?);
        let chunks = match d["chunks"].as_array() {
            Some(c) => c.iter().map(|x| x.as_u64().context("publics definition chunks")).collect::<Result<Vec<_>>>()?,
            None => vec![1, 64],
        };
        ensure!(chunks.len() == 2, "publics definition {d}: chunks");
        ensure!(pos + n * chunks[0] <= n_publics, "publics definition {d} past the {n_publics} publics");
        let flag = |k: &str| u64::from(d[k].as_bool().unwrap_or(false));
        if flag("verificationKey") == 1 {
            vk_pos = pos as i128;
        }
        flat.extend([pos, n, chunks[0], chunks[1], flag("avoidAlias"), flag("verificationKey")]);
    }
    let got = fns.get("DEFS").context("the main has no vadcop_DEFS")?.value(0);
    let want = ints(&flat);
    ensure!(got == &want, "vadcop_DEFS() is {got:?}, publics_info gives {want:?}");
    let call = format!("FinalSnark({n_publics}, {}, {vk_pos})", defs.len());
    ensure!(main.contains(&call), "the final main does not instantiate {call}");
    Ok(())
}

/// The publics in no definition of `publics_info`: the final SNARK's publics hash does not bind them.
pub fn unhashed_publics(publics_info: &Value) -> Vec<u64> {
    let n_publics = publics_info["nPublics"].as_u64().unwrap_or(0);
    let mut hashed = vec![false; n_publics as usize];
    for d in publics_info["definitions"].as_array().into_iter().flatten() {
        let pos = d["initialPos"].as_u64().unwrap_or(0);
        let chunks = d["chunks"].get(0).and_then(Value::as_u64).unwrap_or(1);
        let end = pos + d["nValues"].as_u64().unwrap_or(0) * chunks;
        for i in pos..end.min(n_publics) {
            hashed[i as usize] = true;
        }
    }
    (0..n_publics).filter(|&i| !hashed[i as usize]).collect()
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
