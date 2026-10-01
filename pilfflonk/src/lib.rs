//! Prover orchestration for the pilfflonk backend: the types of its own files, instance loading,
//! the stage loop over the C++ core and proof output.
//!
//! This crate owns the types of every pilfflonk file (spec §5.2): the setup (`pilfflonk-setup`)
//! writes them and the prover reads them, both through these types. They are those of spec
//! Annex A.6, version 1:
//!
//! | File | Type |
//! |---|---|
//! | `pilout.globalInfo.json` | [`PilfflonkGlobalInfo`] |
//! | `<air>.pilfflonkinfo.json` | [`PilfflonkInfo`] (read by C++ too: `pil2-stark/src/pilfflonk/pilfflonk_info.hpp`) |
//! | `<air>.verkey.json` | [`AirVerkey`] |
//! | `pilfflonk.vkey.json` | [`Vkey`] |
//! | the proof: bytes and `proof.json` | [`Proof`], [`ProofJson`], named by [`ProofNames`] |
//! | `publics.json` | [`Publics`] |
//! | the witness directory: `instances.json`, `instance_<ag>_<a>_<t>.bin`, `proof_values.json` | [`Witness`], read by [`FileWitnessSource`] |
//!
//! Each JSON file type is a [`JsonFile`]: it is validated when it is read and before it is
//! written, and written deterministically. [`canonical_json`] is the canonical form the digest of
//! the vkey is computed over.
//!
//! The prover takes its witness from a [`WitnessSource`] (spec §4.3, §5.3): a witness directory,
//! or the [`Witness`] a witness library computes over `Fr` (D4, [`witness_library`]), loaded with
//! [`load_witness_library`] and exported with [`pilfflonk_witness_library!`]. [`prover`] is the
//! orchestration of a proof over the C++ core (spec §4.4): [`ProvingKey::load`] and [`prove`], and,
//! for tests and diagnostics, [`stage_columns`], the columns its stages commit. On the GPU (spec
//! Fase 5): [`ProvingKey::load_on`] with [`Device::Gpu`], where [`gpu_available`].
//! [`check`](mod@check) checks a witness row by row without proving (§4.4, "Depuració"): [`check()`].
//!
//! The verifier is JS (`js/`, spec §4.5, D8): [`js_verifier::verify`] runs it with Node. The
//! Solidity verifier (§4.5, Fase 4), which `pilfflonk-setup` generates, takes the calldata
//! [`calldata`](mod@calldata) encodes for a proof: [`Calldata::read`] and [`Calldata::encode`].
//!
//! With the feature `oracle`, the module `oracle` is the Rust test oracle (plan M14, R8): an
//! evaluation of a pilout's constraints and of `Q` (A.1) with `num-bigint`, independent of
//! `pil-info`. It is for tests only; the crate's own tests turn it on.

pub mod calldata;
pub mod check;
pub mod degrees;
pub mod error;
pub mod field;
pub mod global_info;
pub mod js_verifier;
pub mod json;
pub mod layout;
pub mod names;
pub mod pilfflonk_info;
pub mod proof;
pub mod prover;
mod q_verifier;
pub mod tag;
pub mod verkey;
pub mod vkey;
pub mod witness;
pub mod witness_library;

#[cfg(feature = "oracle")]
pub mod oracle;

pub use calldata::{auxiliary_inverses, verifier_challenges, Calldata, CalldataLayout, VerifierChallenges};
pub use check::{
    check, check_challenges, check_columns, CheckOptions, CheckReport, ConstraintCheck, FailedRow, DEFAULT_MAX_ROWS,
};
pub use degrees::Degrees;
pub use error::{PilfflonkError, PilfflonkResult};
pub use field::{Digest, FqBytes, FrBytes, G1Affine, G2Affine, BN254_Q, BN254_R, G2_GENERATOR};
pub use global_info::{AggType, AirFile, GlobalInfoAir, PilfflonkGlobalInfo, SetupParams, FORMAT_VERSION};
pub use json::{canonical_json, JsonFile};
pub use layout::{Layout, LayoutEntry, LayoutPol};
pub use pilfflonk_info::{Boundary, ChallengeMapEntry, EvMapEntry, NameStageEntry, PilfflonkInfo, PolMapEntry, PolType};
pub use proof::{Proof, ProofJson, ProofNames, ProofShape, Publics, SnarkjsG1};
pub use prover::{
    gpu_available, prove, stage_columns, Device, ProofChallenges, ProofOutput, ProveOptions, ProvingKey, StageColumns,
};
pub use verkey::AirVerkey;
pub use vkey::{FixedCommitments, Vkey, DIGEST_DOMAIN};
pub use witness::{
    AirInstanceRef, AirShape, FileWitnessSource, InstanceWitness, ProofValues, Stage1Witness, Witness, WitnessShape,
    WitnessSource,
};
pub use witness_library::{
    compute_witness, load_witness_library, read_public_inputs, PilfflonkWitnessLibInitFn, PilfflonkWitnessLibrary,
};
