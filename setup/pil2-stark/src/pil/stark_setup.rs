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

/// What names a key beside its pilout air; each `None` takes the pilout's.
#[derive(Clone, Debug, Default)]
pub struct KeyIdentity {
    pub name: Option<String>,
    /// The key's (airgroup id, air id).
    pub ids: Option<(usize, usize)>,
    /// Basic airs only: what their transcript and stage-1 contribution absorb.
    pub circuit_type: Option<u64>,
    /// The proof travels as a Merkle multiproof.
    pub multiproof: bool,
}

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
    key: &KeyIdentity,
) -> Result<PilInfoResult> {
    let at = || format!("the STARK setup of air {air_id} of airgroup {airgroup_id}");
    let failed = |step: &str, e: &dyn std::fmt::Debug| anyhow!("{}: {step}: {e:?}", at());
    // The degree bound comes from the blowup: (1 << (nBitsExt - nBits)) + 1.
    let cfg = PilInfoCfg::goldilocks(stark_struct.n_bits_ext - stark_struct.n_bits);
    let cs = from_pilout(pilout, airgroup_id, air_id, &cfg.field, cfg.opening).with_context(at)?;
    validate_stark_struct(stark_struct, cs.n_rows_bits, airgroup_id, air_id)?;
    let air_name = pilout.air_groups[airgroup_id].airs[air_id].name.clone().unwrap_or_default();

    let none = ImChoice { exprs: Vec::new() };
    let q_dim = quotient_dim(&cs, &none).map_err(|e| failed("quotient dim", &e))?;
    let set = search_im(&cs, &cfg.degree_policy, q_dim as usize).with_context(at)?;
    let im = im_order(&cs, &set).map_err(|e| failed("im order", &e))?;
    let max_deg = cfg.degree_policy.max_constraint_degree() as u32;
    let chk = check_im(&cs, &im, max_deg, true).map_err(|e| failed("im check", &e))?;
    let dag = build_dag(&cs, &im).map_err(|e| failed("constraint DAG", &e))?;
    let lay_params = LayoutParams {
        im_pols_stages: options.im_pols_stages,
        deep_challenges: true,
        q_deg: chk.q_deg,
        air_name: air_name.clone(),
    };
    let lay = layout(&cs, &im, &lay_params).map_err(|e| failed("layout", &e))?;
    let exprs = to_arena(&cs, &im, &lay).with_context(at)?;
    let ops = openings(&cs, &im, &lay).map_err(|e| failed("openings", &e))?;
    let p = StarkParams {
        n_bits: stark_struct.n_bits as u32,
        n_bits_ext: stark_struct.n_bits_ext as u32,
        c_exp_id: exprs.roles.c_exp as u32,
        circuit_type: key.circuit_type,
        multiproof: key.multiproof,
    };
    let emitted =
        emit_prover(&cs, &im, &lay, &ops, &chk, exprs, &EmitParams { air_id, airgroup_id }).with_context(at)?;
    let core = stark_info(&cs, &im, &chk, &lay, &ops, &p).map_err(|e| failed("starkinfo", &e))?;

    // the 128-bit gate sits inside the solve
    let ldt = solve_low_degree_test(stark_struct, core.verifier.fri_batch_size.max(1) as u64);
    let name = key.name.as_deref().unwrap_or(&air_name);
    let (key_airgroup, key_air) = key.ids.unwrap_or((airgroup_id, air_id));
    let starkinfo = build_starkinfo_output_from_core(&core, stark_struct, &ldt, key_airgroup, key_air, name)?;
    let mut pil_code = emitted.code;
    let si = StarkInfo::from_json(&serde_json::to_value(&starkinfo)?)?;
    let vi = VerifierInfo::try_from(&pil_code.verifier_info)?;
    crate::output::dag_eval::check_dag_parity(&dag, &si, &vi).with_context(at)?;
    pil_code.verifier_info.constraint_dag = Some(dag);

    let stages = &emitted.constraint_stages;
    let (summary, prover_memory) =
        report(&core.verifier, &air_name, stages, emitted.n_expressions, stark_struct, options);
    Ok(PilInfoResult { starkinfo, pil_code, summary, prover_memory, im_pols_info: emitted.im_pols_info })
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::output::json::to_json_string;
    use crate::output::test_support::common;
    use crate::types::stark_struct::{generate_stark_struct, StarkSettings};

    const N_BITS: usize = 18;

    /// The 2^18-row stark struct of a blowup of 1..=3 bits.
    fn stark_struct(bits: usize) -> &'static StarkStruct {
        static CACHE: [OnceLock<StarkStruct>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
        CACHE[bits - 1].get_or_init(|| {
            let settings = StarkSettings { initial_blowup_factor: Some(bits), ..Default::default() };
            generate_stark_struct(&settings, N_BITS, "blake3")
        })
    }

    fn sized(mut p: pb::PilOut) -> pb::PilOut {
        p.air_groups[0].airs[0].num_rows = Some(1 << N_BITS);
        p
    }

    /// The first `n` seeded random pilouts with their setup at the first blowup of 1..=3 bits that takes them.
    fn feasible(n: usize) -> Vec<(u64, PilInfoResult)> {
        let opts = PrepareOptions::default();
        (0..1000u64)
            .filter_map(|seed| {
                let p = sized(common::random_pilout(seed).0);
                (1..=3)
                    .find_map(|b| stark_setup(&p, 0, 0, stark_struct(b), &opts, &KeyIdentity::default()).ok())
                    .map(|r| (seed, r))
            })
            .take(n)
            .collect()
    }

    #[test]
    fn runs_on_every_code_built_pilout() {
        assert_eq!(feasible(10).len(), 10, "not enough feasible random pilouts");
    }

    #[test]
    fn verifier_info_has_dag_and_qverifier() {
        use pil_info::core::dag::format::ConstraintDag;
        let (_, r) = feasible(1).remove(0);
        let vi_json = to_json_string(&r.pil_code.verifier_info).unwrap();
        let (q, d) = (vi_json.find("\"qVerifier\""), vi_json.find("\"constraintDag\""));
        assert!(matches!((q, d), (Some(q), Some(d)) if q < d), "qVerifier then constraintDag: {q:?} {d:?}");
        let v: serde_json::Value = serde_json::from_str(&vi_json).unwrap();
        let mut keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["constraintDag", "qVerifier"]);
        let dag: ConstraintDag = serde_json::from_value(v["constraintDag"].clone()).unwrap();
        let si = StarkInfo::from_json(&serde_json::to_value(&r.starkinfo).unwrap()).unwrap();
        let vi = VerifierInfo::try_from(&r.pil_code.verifier_info).unwrap();
        crate::output::dag_eval::check_dag_parity(&dag, &si, &vi).unwrap();
    }

    #[test]
    fn rejects_first_row_constraint_with_message() {
        let p = sized(common::pilout(
            &["a"],
            &[],
            vec![common::sub(common::witness(0, 0), common::constant("1"))],
            vec![common::first_row(0)],
        ));
        let err = match stark_setup(&p, 0, 0, stark_struct(1), &PrepareOptions::default(), &KeyIdentity::default()) {
            Ok(_) => panic!("a firstRow constraint was accepted"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("is firstRow, and the STARK takes only everyRow"), "{err}");
        assert!(err.contains("firstRow 0"), "{err}");
    }

    /// The 128-bit gate sits in `solve_low_degree_test`; a schedule it refuses must stop the setup.
    #[test]
    fn below_128_bits_still_fails_setup() {
        let (seed, r) = feasible(1).remove(0);
        let p = sized(common::random_pilout(seed).0);
        let mut ss =
            (1..=3).map(stark_struct).find(|ss| ss.n_bits_ext == r.starkinfo.stark_struct.n_bits_ext).unwrap().clone();
        match &mut ss.low_degree_test {
            crate::types::stark_struct::LowDegreeTest::Fri(f) => f.log_degrees[1] += 1,
            crate::types::stark_struct::LowDegreeTest::Stir(f) => f.log_degrees[1] += 1,
        }
        let run = std::panic::catch_unwind(|| {
            stark_setup(&p, 0, 0, &ss, &PrepareOptions::default(), &KeyIdentity::default())
        });
        let msg = match run {
            Ok(r) => panic!("a refused schedule set up: ok={}", r.is_ok()),
            Err(e) => e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())),
        };
        let msg = msg.unwrap_or_default();
        assert!(msg.contains("exact security check"), "{msg}");
    }
}
