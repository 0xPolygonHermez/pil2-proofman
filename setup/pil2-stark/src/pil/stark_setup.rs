//! The STARK setup entry: the thin core's pipeline behind the signature `pil_info` had.

use anyhow::{anyhow, Context, Result};
use pil2_pilout::pilout as pb;
use pil_info::core::dag::build::build_dag;
use pil_info::core::project::{stark_info, StarkParams};
use pil_info::emit::prover::{emit_prover, EmitParams};
use pil_info::pipeline::{self, CorePass};
use pil_info::PilInfoCfg;

use crate::output::stark_info::{build_starkinfo_output, solve_low_degree_test};
use crate::pil::info::{report, validate_stark_struct, PilInfoResult};
use pil_info::PrepareOptions;
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
/// Fails on a pilout the core rejects, a starkStruct that does not fit the air and a low-degree test that fails
/// the exact security check (the 128-bit gate).
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
    let blowup = (stark_struct.n_bits_ext.checked_sub(stark_struct.n_bits)).ok_or_else(|| {
        anyhow!("{}: nBitsExt {} is below nBits {}", at(), stark_struct.n_bits_ext, stark_struct.n_bits)
    })?;
    // the degree bound comes from the blowup: (1 << (nBitsExt - nBits)) + 1
    let cfg = PilInfoCfg::goldilocks(blowup);
    let CorePass { cs, im, chk, lay, exprs, ops } =
        pipeline::run(pilout, airgroup_id, air_id, &cfg, options.im_pols_stages).with_context(at)?;
    validate_stark_struct(stark_struct, cs.n_rows_bits, airgroup_id, air_id)?;
    let dag = build_dag(&cs, &im).map_err(|e| failed("constraint DAG", &e))?;
    let p = StarkParams {
        n_bits: stark_struct.n_bits as u32,
        n_bits_ext: stark_struct.n_bits_ext as u32,
        c_exp_id: exprs.roles.c_exp as u32,
        circuit_type: key.circuit_type,
        multiproof: key.multiproof,
    };
    let emitted = emit_prover(&cs, &im, &lay, &ops, exprs, &EmitParams { air_id, airgroup_id }).with_context(at)?;
    let view = stark_info(&cs, &im, &chk, &lay, &ops, &p);

    // the 128-bit gate sits inside the solve
    let ldt = solve_low_degree_test(stark_struct, view.fri_batch_size.max(1) as u64).with_context(at)?;
    let name = key.name.as_deref().unwrap_or(&cs.name);
    let starkinfo = build_starkinfo_output(&view, stark_struct, &ldt, key.ids.unwrap_or((airgroup_id, air_id)), name)?;
    let mut pil_code = emitted.code;
    pil_code.verifier_info.constraint_dag = Some(dag);

    let stages: Vec<usize> = emitted.constraints.iter().map(|c| c.stage.unwrap_or(0)).collect();
    let (summary, prover_memory) = report(&view, &cs.name, &stages, emitted.arena.len(), stark_struct, options);
    Ok(PilInfoResult { starkinfo, pil_code, summary, prover_memory, im_pols_info: emitted.im_pols_info, im })
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
            generate_stark_struct(&settings, N_BITS, proofman_common::hash_family::HashFamily::Blake3)
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
    fn deep_verifierinfo_has_only_the_dag() {
        use crate::types::stark_info::VerifierInfo;
        let (_, r) = feasible(1).remove(0);
        let v: serde_json::Value = serde_json::from_str(&to_json_string(&r.pil_code.verifier_info).unwrap()).unwrap();
        let keys: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys, ["constraintDag"]);
        assert!(VerifierInfo::from_json(&v).is_ok());
        // a stale key's qVerifier is ignored; a key without the DAG is refused
        let mut stale = v.clone();
        stale["qVerifier"] = serde_json::json!({"tmpUsed": 0, "code": []});
        assert!(VerifierInfo::from_json(&stale).is_ok());
        let err = VerifierInfo::from_json(&serde_json::json!({"qVerifier": stale["qVerifier"]})).unwrap_err();
        assert!(err.to_string().contains("regenerate the setup"), "{err}");
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
    fn below_128_bits_fails_setup() {
        let (seed, r) = feasible(1).remove(0);
        let p = sized(common::random_pilout(seed).0);
        let mut ss =
            (1..=3).map(stark_struct).find(|ss| ss.n_bits_ext == r.starkinfo.stark_struct.n_bits_ext).unwrap().clone();
        match &mut ss.low_degree_test {
            crate::types::stark_struct::LowDegreeTest::Fri(f) => f.log_degrees[1] += 1,
            crate::types::stark_struct::LowDegreeTest::Stir(f) => f.log_degrees[1] += 1,
        }
        let err = match stark_setup(&p, 0, 0, &ss, &PrepareOptions::default(), &KeyIdentity::default()) {
            Ok(_) => panic!("a refused schedule set up"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("exact security check"), "{err}");
    }
}
