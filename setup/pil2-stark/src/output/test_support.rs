//! Test fixtures: the old pipeline and the core run on the same random pilout (cfg(test) only).

use std::sync::OnceLock;

use pil2_pilout::pilout::PilOut;

use crate::output::json::to_json_string;
use crate::output::stark_info::{
    build_starkinfo_output, build_starkinfo_output_from_core, collect_opening_points, solve_low_degree_test,
};
use crate::types::output::StarkInfoOutput;
use crate::types::security::pcs::LowDegreeTest;
use crate::types::stark_info::{StarkInfo, VerifierInfo};
use crate::types::stark_struct::{generate_stark_struct, StarkSettings, StarkStruct};

// The pil-info test helpers, included (not copied) so both crates share one source.
#[path = "../../../pil-info/tests/common/mod.rs"]
pub mod common;

/// `OldRun` plus what the STARK output needs: struct, solved test, old output, loaded starkinfo/verifierinfo.
pub struct Full {
    pub run: common::OldRun,
    pub stark_struct: StarkStruct,
    pub ldt: LowDegreeTest,
    pub old_out: StarkInfoOutput,
    pub si: StarkInfo,
    pub vi: VerifierInfo,
}

pub const N_BITS: usize = 18;

/// The stark struct and solved test of one blowup, solved once (the security solve costs ~1 s in debug).
pub fn solved(bits: usize) -> &'static (StarkStruct, LowDegreeTest) {
    static CACHE: [OnceLock<(StarkStruct, LowDegreeTest)>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    CACHE[bits - 1].get_or_init(|| {
        let settings = StarkSettings { initial_blowup_factor: Some(bits), ..Default::default() };
        let stark_struct = generate_stark_struct(&settings, N_BITS, "blake3");
        let ldt = solve_low_degree_test(&stark_struct, 200);
        (stark_struct, ldt)
    })
}

/// `p` with 2^18 rows (a real stark struct needs them); None if no blowup of 1..=3 bits is feasible.
pub fn full(p: PilOut) -> Option<Full> {
    full_in(p, false)
}

/// `full`, both pipelines given `im_pols_stages`.
pub fn full_in(mut p: PilOut, im_pols_stages: bool) -> Option<Full> {
    p.air_groups[0].airs[0].num_rows = Some(1 << N_BITS);
    for bits in 1..=3 {
        let (stark_struct, ldt) = solved(bits).clone();
        let blowup = stark_struct.n_bits_ext - stark_struct.n_bits;
        let Some(run) = common::old_run_with(&p, 0, 0, blowup, im_pols_stages) else { continue };
        let old = &run.old;
        let old_out = build_starkinfo_output(
            &old.setup,
            &stark_struct,
            &old.pil_code,
            &collect_opening_points(&old.setup),
            &ldt,
            0,
            0,
            Some(proofman_common::circuit_type_of(0, 1)),
            "Synthetic",
            old.c_exp_id,
            old.q_deg,
        );
        let json: serde_json::Value = serde_json::from_str(&to_json_string(&old_out).ok()?).ok()?;
        let si = StarkInfo::from_json(&json).ok()?;
        let vi = VerifierInfo::try_from(&old.pil_code.verifier_info).ok()?;
        return Some(Full { run, stark_struct, ldt, old_out, si, vi });
    }
    None
}

/// The first `n` feasible seeded random pilouts.
pub fn full_random(n: usize) -> Vec<Full> {
    full_random_in(n, false)
}

pub fn full_random_in(n: usize, im_pols_stages: bool) -> Vec<Full> {
    (0..1000u64).filter_map(|seed| full_in(common::random_pilout(seed).0, im_pols_stages)).take(n).collect()
}

/// The core's starkinfo of `run` through `common::core_out`, Q's id from the roles.
pub fn core_starkinfo(
    run: &common::OldRun,
    ss: &StarkStruct,
    ldt: &LowDegreeTest,
    airgroup_id: usize,
    air_id: usize,
    air_name: &str,
) -> StarkInfoOutput {
    use pil_info::core::project::{stark_info, StarkParams};
    let c = common::core_out(run);
    let p = StarkParams {
        n_bits: ss.n_bits as u32,
        n_bits_ext: ss.n_bits_ext as u32,
        c_exp_id: c.exprs.roles.c_exp as u32,
    };
    let core = stark_info(&run.cs, &run.im, &run.chk, &c.lay, &c.ops, &p).unwrap();
    let ct = Some(proofman_common::circuit_type_of(0, 1));
    build_starkinfo_output_from_core(&core, ss, ldt, airgroup_id, air_id, ct, air_name).unwrap()
}
