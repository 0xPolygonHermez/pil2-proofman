//! The STARK setup entry: the thin core's pipeline behind the signature `pil_info` had.

use anyhow::{anyhow, Context, Result};
use pil2_pilout::pilout as pb;
use pil_info::core::dag::build::build_dag;
use pil_info::core::from_pilout::from_pilout;
use pil_info::core::im::{check_im, im_order, ImChoice};
use pil_info::core::layout::{layout, quotient_dim, LayoutParams};
use pil_info::core::openings::openings;
use pil_info::core::project::{stark_info, StarkParams};
use pil_info::emit::prover::{emit_prover, to_arena, EmitParams};
use pil_info::search::im::search_im;
use pil_info::PilInfoCfg;

use crate::output::stark_info::{build_starkinfo_output_from_core, solve_low_degree_test};
use crate::pil::info::{report, validate_stark_struct, PilInfoResult};
use crate::pil::prepare::PrepareOptions;
use crate::types::stark_info::{StarkInfo, VerifierInfo};
use crate::types::stark_struct::StarkStruct;

/// Assemble the pil info of one air through the thin core.
///
/// Fails on a pilout the core rejects, a starkStruct that does not fit the air, and a constraint
/// DAG that disagrees with the qVerifier code; panics (the 128-bit gate) on a low-degree test that
/// fails the exact security check.
pub fn stark_setup(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    stark_struct: &StarkStruct,
    options: &PrepareOptions,
) -> Result<PilInfoResult> {
    let at = || format!("the STARK setup of air {air_id} of airgroup {airgroup_id}");
    // The degree bound comes from the blowup: (1 << (nBitsExt - nBits)) + 1.
    let cfg = PilInfoCfg::goldilocks(stark_struct.n_bits_ext - stark_struct.n_bits);
    let cs = from_pilout(pilout, airgroup_id, air_id, &cfg.field, cfg.opening).with_context(at)?;
    if !options.debug {
        validate_stark_struct(stark_struct, cs.n_rows_bits, airgroup_id, air_id)?;
    }
    let air_name = pilout.air_groups[airgroup_id].airs[air_id].name.clone().unwrap_or_default();

    let none = ImChoice { exprs: Vec::new() };
    let set = search_im(&cs, &cfg.degree_policy, quotient_dim(&cs, &none) as usize).with_context(at)?;
    let im = im_order(&set).map_err(|e| anyhow!("{}: im order: {e:?}", at()))?;
    let max_deg = cfg.degree_policy.max_constraint_degree() as u32;
    let chk = check_im(&cs, &im, max_deg, true).map_err(|e| anyhow!("{}: im check: {e:?}", at()))?;
    let dag = build_dag(&cs, &im).map_err(|e| anyhow!("{}: constraint DAG: {e:?}", at()))?;
    let lay_params = LayoutParams {
        im_pols_stages: options.im_pols_stages,
        deep_challenges: true,
        q_deg: chk.q_deg,
        air_name: air_name.clone(),
    };
    let lay = layout(&cs, &im, &lay_params).map_err(|e| anyhow!("{}: layout: {e:?}", at()))?;
    let exprs = to_arena(&cs, &im, &lay).with_context(at)?;
    let ops = openings(&cs, &im, &lay, &exprs.roles.hint_roots()).map_err(|e| anyhow!("{}: openings: {e:?}", at()))?;
    let c_exp_id = exprs.roles.c_exp;
    let emit = EmitParams { air_id, airgroup_id, air_name: air_name.clone(), debug: options.debug };
    let (mut pil_code, setup) = emit_prover(&cs, &im, &lay, &ops, &chk, exprs, &emit).with_context(at)?;

    // the 128-bit gate sits inside the solve
    let ldt = solve_low_degree_test(stark_struct, pil_code.fri_batch_size.max(1) as u64);
    let p = StarkParams {
        n_bits: stark_struct.n_bits as u32,
        n_bits_ext: stark_struct.n_bits_ext as u32,
        c_exp_id: c_exp_id as u32,
    };
    let core = stark_info(&cs, &im, &chk, &lay, &ops, &p).map_err(|e| anyhow!("{}: starkinfo: {e:?}", at()))?;
    if !options.debug {
        let out = build_starkinfo_output_from_core(&core, stark_struct, &ldt, airgroup_id, air_id, None, &air_name)?;
        let si = StarkInfo::from_json(&serde_json::to_value(&out)?)?;
        let vi = VerifierInfo::try_from(&pil_code.verifier_info)?;
        crate::output::dag_eval::check_dag_parity(&dag, &si, &vi).with_context(at)?;
        pil_code.verifier_info.constraint_dag = Some(dag);
    }

    let q_deg = i64::from(chk.q_deg);
    let (summary, prover_memory) =
        report(&setup, &pil_code, stark_struct, q_deg, core.verifier.boundaries.len(), options);
    let im_pols_info = setup.im_pols_info.clone();
    Ok(PilInfoResult { setup, pil_code, summary, prover_memory, im_pols_info, c_exp_id, q_deg })
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::output::json::to_json_string;
    use crate::output::stark_info::{build_starkinfo_output, collect_opening_points, solve_low_degree_test};
    use crate::output::test_support::common::{self, real_pilouts};
    use crate::types::security::pcs::LowDegreeTest;
    use crate::types::stark_info::{StarkInfo, VerifierInfo};
    use crate::types::stark_struct::{generate_stark_struct, StarkSettings};

    const N_BITS: usize = 18;

    /// The old body.
    fn oracle(p: &pb::PilOut, g: usize, a: usize, ss: &StarkStruct, o: &PrepareOptions) -> Result<PilInfoResult> {
        crate::pil::info::pil_info_old(p, g, a, ss, o)
    }

    /// The 2^18-row stark struct and solved test of a blowup of 1..=3 bits, solved once.
    fn solved(bits: usize) -> &'static (StarkStruct, LowDegreeTest) {
        static CACHE: [OnceLock<(StarkStruct, LowDegreeTest)>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
        CACHE[bits - 1].get_or_init(|| {
            let settings = StarkSettings { initial_blowup_factor: Some(bits), ..Default::default() };
            let ss = generate_stark_struct(&settings, N_BITS, "blake3");
            let ldt = solve_low_degree_test(&ss, 200);
            (ss, ldt)
        })
    }

    /// starkinfo, verifierinfo (as JSON) and expressionsinfo of a result, the way `commands/setup.rs` writes them.
    fn outputs(r: &PilInfoResult, ss: &StarkStruct, ldt: &LowDegreeTest, g: usize, a: usize) -> [String; 3] {
        let out = build_starkinfo_output(
            &r.setup,
            ss,
            &r.pil_code,
            &collect_opening_points(&r.setup),
            ldt,
            g,
            a,
            Some(proofman_common::circuit_type_of(0, 1)),
            &r.setup.name,
            r.c_exp_id,
            r.q_deg,
        );
        [
            to_json_string(&out).unwrap(),
            to_json_string(&r.pil_code.verifier_info).unwrap(),
            to_json_string(&r.pil_code.expressions_info).unwrap(),
        ]
    }

    /// A verifierinfo JSON without its `constraintDag`, re-serialised.
    fn without_dag(json: &str) -> String {
        let mut v: serde_json::Value = serde_json::from_str(json).unwrap();
        v.as_object_mut().unwrap().remove("constraintDag");
        v.to_string()
    }

    /// Both entries on one air at the first blowup the old one accepts; false when none does (then the new must refuse too).
    fn equals_old(p: &pb::PilOut, g: usize, a: usize, what: &str) -> bool {
        let opts = PrepareOptions::default();
        for bits in 1..=3 {
            let (ss, ldt) = solved(bits);
            let Ok(old) = oracle(p, g, a, ss, &opts) else { continue };
            let new = stark_setup(p, g, a, ss, &opts).unwrap_or_else(|e| panic!("{what}: new refuses: {e:#}"));
            assert_eq!(new.pil_code.fri_batch_size, old.pil_code.fri_batch_size, "{what}");
            let (n, o) = (outputs(&new, ss, ldt, g, a), outputs(&old, ss, ldt, g, a));
            assert_eq!(n[0], o[0], "{what}: starkinfo");
            assert_eq!(without_dag(&n[1]), without_dag(&o[1]), "{what}: verifierinfo minus constraintDag");
            assert_eq!(n[2], o[2], "{what}: expressionsinfo");
            assert_eq!(new.summary, old.summary, "{what}: summary");
            return true;
        }
        let ss = &solved(1).0;
        assert!(stark_setup(p, g, a, ss, &opts).is_err(), "{what}: old refuses at every blowup, new accepts");
        false
    }

    fn sized(mut p: pb::PilOut) -> pb::PilOut {
        p.air_groups[0].airs[0].num_rows = Some(1 << N_BITS);
        p
    }

    /// The first `n` seeded random pilouts one blowup of 1..=3 bits accepts.
    fn feasible(n: usize) -> Vec<(u64, pb::PilOut)> {
        let opts = PrepareOptions::default();
        (0..1000u64)
            .map(|seed| (seed, sized(common::random_pilout(seed).0)))
            .filter(|(_, p)| (1..=3).any(|b| oracle(p, 0, 0, &solved(b).0, &opts).is_ok()))
            .take(n)
            .collect()
    }

    #[test]
    fn runs_on_every_code_built_pilout() {
        let ps = feasible(10);
        assert_eq!(ps.len(), 10, "not enough feasible random pilouts");
        for (seed, p) in &ps {
            let ok = (1..=3).any(|b| stark_setup(p, 0, 0, &solved(b).0, &PrepareOptions::default()).is_ok());
            assert!(ok, "seed {seed}: stark_setup fails at every blowup");
        }
    }

    #[test]
    fn verifier_info_has_dag_and_qverifier() {
        use pil_info::core::dag::format::ConstraintDag;
        let (seed, p) = feasible(1).remove(0);
        let (ss, ldt) = (1..=3)
            .map(solved)
            .find(|(ss, _)| stark_setup(&p, 0, 0, ss, &PrepareOptions::default()).is_ok())
            .unwrap_or_else(|| panic!("seed {seed}: no blowup works"));
        let r = stark_setup(&p, 0, 0, ss, &PrepareOptions::default()).unwrap();
        let [si_json, vi_json, _] = outputs(&r, ss, ldt, 0, 0);
        let (q, d) = (vi_json.find("\"qVerifier\""), vi_json.find("\"constraintDag\""));
        assert!(matches!((q, d), (Some(q), Some(d)) if q < d), "qVerifier then constraintDag: {q:?} {d:?}");
        let v: serde_json::Value = serde_json::from_str(&vi_json).unwrap();
        let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["constraintDag", "qVerifier"]);
        let dag: ConstraintDag = serde_json::from_value(v["constraintDag"].clone()).unwrap();
        let si = StarkInfo::from_json(&serde_json::from_str(&si_json).unwrap()).unwrap();
        let vi = VerifierInfo::try_from(&r.pil_code.verifier_info).unwrap();
        crate::output::dag_eval::check_dag_parity(&dag, &si, &vi).unwrap();
    }

    #[test]
    fn result_equals_old_oracle() {
        let ps = feasible(10);
        assert_eq!(ps.len(), 10, "not enough feasible random pilouts");
        for (seed, p) in &ps {
            assert!(equals_old(p, 0, 0, &format!("seed {seed}")));
        }
        let mut compared = 0;
        for (f, mut p) in real_pilouts() {
            for g in 0..p.air_groups.len() {
                for a in 0..p.air_groups[g].airs.len() {
                    p.air_groups[g].airs[a].num_rows = Some(1 << N_BITS);
                    compared += equals_old(&p, g, a, &format!("{f:?} {g}/{a}")) as usize;
                }
            }
        }
        assert!(compared >= 40, "only {compared} real airs compared");
    }

    /// TEMPORARY twin of the old generator's `ev_map`, until Task 8 deletes it.
    #[test]
    fn evmap_equals_old_generator_evmap() {
        let opts = PrepareOptions::default();
        let mut compared = 0;
        for (seed, p) in feasible(10) {
            for bits in 1..=3 {
                let ss = &solved(bits).0;
                let Ok(old) = oracle(&p, 0, 0, ss, &opts) else { continue };
                let new = stark_setup(&p, 0, 0, ss, &opts).unwrap();
                assert_eq!(format!("{:?}", new.pil_code.ev_map), format!("{:?}", old.pil_code.ev_map), "seed {seed}");
                assert_eq!(new.setup.opening_points, old.setup.opening_points, "seed {seed}");
                compared += 1;
                break;
            }
        }
        assert_eq!(compared, 10);
    }

    #[test]
    fn rejects_first_row_constraint_with_message() {
        let p = sized(common::pilout(
            &["a"],
            &[],
            vec![common::sub(common::witness(0, 0), common::constant("1"))],
            vec![common::first_row(0)],
        ));
        let ss = &solved(1).0;
        let err = match stark_setup(&p, 0, 0, ss, &PrepareOptions::default()) {
            Ok(_) => panic!("a firstRow constraint was accepted"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("is firstRow, and the STARK takes only everyRow"), "{err}");
        assert!(err.contains("firstRow 0"), "{err}");
    }

    /// The 128-bit gate sits in `solve_low_degree_test`; a schedule it refuses must stop the setup.
    #[test]
    fn below_128_bits_still_fails_setup() {
        let p = feasible(1).remove(0).1;
        let mut ss = solved(1).0.clone();
        match &mut ss.low_degree_test {
            crate::types::stark_struct::LowDegreeTest::Fri(f) => f.log_degrees[1] += 1,
            crate::types::stark_struct::LowDegreeTest::Stir(f) => f.log_degrees[1] += 1,
        }
        let run = std::panic::catch_unwind(|| stark_setup(&p, 0, 0, &ss, &PrepareOptions::default()));
        let msg = match run {
            Ok(r) => panic!("a refused schedule set up: ok={}", r.is_ok()),
            Err(e) => e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())),
        };
        let msg = msg.unwrap_or_default();
        assert!(msg.contains("exact security check"), "{msg}");
    }
}
