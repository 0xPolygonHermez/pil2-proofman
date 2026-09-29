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
//!
//! Each JSON file type is a [`JsonFile`]: it is validated when it is read and before it is
//! written, and written deterministically. [`canonical_json`] is the canonical form the digest of
//! the vkey is computed over.

pub mod error;
pub mod field;
pub mod global_info;
pub mod json;
pub mod layout;
pub mod names;
pub mod pilfflonk_info;
pub mod proof;
pub mod tag;
pub mod verkey;
pub mod vkey;

pub use error::{PilfflonkError, PilfflonkResult};
pub use field::{Digest, FqBytes, FrBytes, G1Affine, G2Affine, BN254_Q, BN254_R};
pub use global_info::{AggType, AirFile, GlobalInfoAir, PilfflonkGlobalInfo, SetupParams, FORMAT_VERSION};
pub use json::{canonical_json, JsonFile};
pub use layout::{Layout, LayoutEntry, LayoutPol};
pub use pilfflonk_info::{Boundary, ChallengeMapEntry, EvMapEntry, NameStageEntry, PilfflonkInfo, PolMapEntry, PolType};
pub use proof::{Proof, ProofJson, ProofNames, ProofShape, Publics, SnarkjsG1};
pub use verkey::AirVerkey;
pub use vkey::{FixedCommitments, Vkey, DIGEST_DOMAIN};
