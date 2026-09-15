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

mod air;
mod curve;
mod fr;
mod linearisation;
mod pairing;
mod proof;
#[cfg(test)]
mod reference;
mod roots;
mod setup;
mod shkey;
mod stark_info;
mod transcript;
mod verifier;
mod verifier_code;
mod verify;
mod solidity;
mod shplonk_pcs;
mod zkey;

pub use proof::{
    INV_KEY, INV_ZH_KEY, PROOF_PROTOCOL, ShPlonkProof, UNEVALUATED_POL, W_KEY, WP_KEY, commitment_key, evaluation_key,
};
pub use verifier::{
    Challenges, challenge_alpha, challenge_xi, challenge_xi_seed, challenge_y, enters_xi_seed_transcript,
    non_committed_pols, quotient_stage, recompute_challenges,
};
pub use transcript::{FR_BYTES, FR_MODULUS, Transcript, fr_modulus};
pub use curve::{CURVE_B, FQ_MODULUS, G1Affine, G2Affine, fq_modulus, g1_generator, g2_generator};
pub use verify::{AirInputs, Prepared, prepare};
pub use air::{check_inv_zh, inv_zh, quotient_at, zh_at};
pub use verifier_code::{EvalRef, Inputs, Instruction, Op, Operand, VerifierCode};
pub use pairing::{PairingCheck, Term, assemble};
pub use roots::{OpeningSet, all_roots, coset_key, flattened, omega_key, roots_for};
pub use linearisation::{Evaluations, Linearisation, f_at_root, linearise, r_at, resolve_evaluations};
pub use stark_info::{EvMap, PolMap, StarkInfo, StarkStruct};
pub use shkey::{
    Candidate, bucket_size_for, normalize_opening_point, committed_degree, quotient_degree, omega_name,
    required_omegas, available_bucket_sizes, combined_degree, derive_f, plan_buckets,
};
pub use setup::{CURVE, PROTOCOL, ShPlonkPol, ShPlonkSetup, ShPlonkStage, ShPlonkStagePol};
pub use shplonk_pcs::{BN254_SECURITY_BITS, ShPlonk, ShPlonkConfig};
pub use solidity::{PROOF_LAYOUT_NOTE, ProofLayout, gen_iverifier, gen_solidity};
pub use zkey::{FCommitment, SECTION_CONST_POLS_COEFS, SECTION_PTAU, ZKey};
