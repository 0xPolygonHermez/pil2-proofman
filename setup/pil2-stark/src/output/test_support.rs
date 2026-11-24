//! Test fixtures: the core run and `stark_setup`'s starkinfo/verifierinfo on the same pilout (cfg(test) only).

use pil2_pilout::pilout::PilOut;

use crate::pil::prepare::PrepareOptions;
use crate::pil::stark_setup::stark_setup;
use crate::types::stark_info::{StarkInfo, VerifierInfo};
use crate::types::stark_struct::{generate_stark_struct, StarkSettings};

// The pil-info test helpers, included (not copied) so both crates share one source.
#[path = "../../../pil-info/tests/common/mod.rs"]
pub mod common;

/// `common::Run` plus the loaded starkinfo/verifierinfo `stark_setup` writes for it.
pub struct Full {
    pub run: common::Run,
    pub si: StarkInfo,
    pub vi: VerifierInfo,
}

pub const N_BITS: usize = 18;

/// `p` with 2^18 rows (a real stark struct needs them); None if no blowup of 1..=3 bits is feasible.
pub fn full(mut p: PilOut) -> Option<Full> {
    p.air_groups[0].airs[0].num_rows = Some(1 << N_BITS);
    for bits in 1..=3 {
        let settings = StarkSettings { initial_blowup_factor: Some(bits), ..Default::default() };
        let ss = generate_stark_struct(&settings, N_BITS, "blake3");
        let Some(run) = common::run(&p, 0, 0, ss.n_bits_ext - ss.n_bits) else { continue };
        let r = stark_setup(&p, 0, 0, &ss, &PrepareOptions::default()).ok()?;
        let si = StarkInfo::from_json(&serde_json::to_value(&r.starkinfo).ok()?).ok()?;
        let vi = VerifierInfo::try_from(&r.pil_code.verifier_info).ok()?;
        return Some(Full { run, si, vi });
    }
    None
}

/// The first `n` feasible seeded random pilouts.
pub fn full_random(n: usize) -> Vec<Full> {
    (0..1000u64).filter_map(|seed| full(common::random_pilout(seed).0)).take(n).collect()
}
