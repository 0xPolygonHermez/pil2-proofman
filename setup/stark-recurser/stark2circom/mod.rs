//! In-process Rust port of the JS `stark2circom` code generators.
//!
//! Replaces subprocess calls to `node src/main_gencircom.js` and
//! `node src/gensolidity.js` with direct Rust string generation.
//!
//! Sub-modules:
//! - [`gencircom`] — circuit-level template generators (recursivef, recursive1/2, …)
//! - [`pil2circom`] — per-air stark verifier circom generator (GL + BN128)
//!
//! Shared:
//! - [`transcript`] — GL Poseidon2 transcript helper (used by both groups)

/// GL Poseidon2 transcript builder — shared by `pil2circom` and `gencircom`.
pub(crate) mod transcript;

/// Circuit-level template generators (port of `gencircom.js`).
pub mod circuit_templates;

/// Per-air stark verifier generators (port of `pil2circom.js`).
pub(crate) mod circom_verifier;

// ── Re-exports at the original paths (backward-compatible public API) ─────────

pub use circuit_templates::stark_inputs;
pub use circuit_templates::templates;
pub use circuit_templates::{gen_circom_circuit, CircomGenOptions, GenCircomCircuitInput};
pub use circuit_templates::templates::{
    gen_recursivef, gen_recursion_final, gen_solidity, gen_iverifier, gen_final_compressed, gen_compressor,
    gen_recursive1, gen_recursive2, gen_vadcop_final,
};
pub use circom_verifier::{gen_stark_verifier, Pil2CircomOptions as StarkVerifierOptions};
