//! PIL2-FFLONK: a SHPLONK/KZG prover over BN254 for PIL2 AIRs.
//!
//! Layout mirrors pil2-stark: the prover itself is C++ under `cpp/`, built by
//! its own Makefile, while Rust owns setup, parameterization and orchestration.
//!
//! This crate deliberately depends on nothing from `pil2-stark-setup`. The
//! `Pcs` trait there has no polymorphic consumer -- every call site constructs
//! a concrete `Fri::new(...)` -- so implementing it would buy nothing, and it
//! would create a cycle once `pil2-stark-setup` grows a `setup_pilfflonk`
//! command that depends on this crate.

mod proof;
mod setup;
mod shkey;
mod stark_info;
mod solidity;
mod shplonk_pcs;
mod zkey;

pub use proof::{
    INV_KEY, INV_ZH_KEY, PROOF_PROTOCOL, ShPlonkProof, UNEVALUATED_POL, W_KEY, WP_KEY, commitment_key, evaluation_key,
};
pub use stark_info::{EvMap, PolMap, StarkInfo, StarkStruct};
pub use shkey::{Candidate, bucket_size_for, normalize_opening_point, committed_degree, quotient_degree, omega_name, required_omegas, available_bucket_sizes, combined_degree, derive_f, plan_buckets};
pub use setup::{CURVE, PROTOCOL, ShPlonkPol, ShPlonkSetup, ShPlonkStage, ShPlonkStagePol};
pub use shplonk_pcs::{BN254_SECURITY_BITS, ShPlonk, ShPlonkConfig};
pub use solidity::{PROOF_LAYOUT_NOTE, ProofLayout, gen_iverifier, gen_solidity};
pub use zkey::{FCommitment, ZKey};
