//! The circom (and Solidity) generators of the recursion: [`circuit_templates`], the circuits
//! (recursivef, recursive1/2, compressor, vadcop_final, final), and `circom_verifier`, the
//! per-air STARK verifier (GL and BN128).

/// Compiles and runs the emitted BLAKE3 circom against the reference `blake3`
/// crate. Skips when circom or a C++ toolchain is unavailable.
#[cfg(test)]
mod blake3_equiv_tests;

/// Circuit-level generators.
pub mod circuit_templates;

/// Per-air stark verifier generators.
pub(crate) mod circom_verifier;

// ── Re-exports at the original paths (backward-compatible public API) ─────────

pub use circuit_templates::stark_inputs;
pub use circuit_templates::templates;
pub use circuit_templates::{gen_circom_circuit, CircomGenOptions, CircuitTemplate, GenCircomCircuitInput};
pub use circuit_templates::templates::{
    gen_recursivef, gen_recursion_final, gen_solidity, gen_iverifier, gen_final_compressed, gen_compressor,
    gen_recursive1, gen_recursive2, gen_vadcop_final, SnarkVerifier,
};
pub use circom_verifier::{gen_stark_verifier, gen_stark_verifier_bn128_blake3, Pil2CircomOptions as StarkVerifierOptions};
