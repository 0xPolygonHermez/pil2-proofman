//! Security-parameter calculator: query counts and grinding for FRI and STIR
//! low-degree testing at a target security level.
//!
//! Layout:
//!   - `hpf`        — high-precision float primitives (shared math plumbing)
//!   - `query_cost` — Merkle query hash-cost model (shared by fri/stir)
//!   - `regime`     — decoding regimes (JBR / UDR) and the `DecodingRegime` trait
//!   - `fri`        — FRI query-parameter calculator
//!   - `stir`       — STIR query-parameter calculator
//!
//! The verifier-hash FRI-vs-STIR comparison is an analysis tool, not part of this
//! calculator; see `examples/verifier_hash_comparison.rs`.

mod hpf;
mod query_cost;
mod regime;

mod fri;
mod stir;

// Public API (consumed by the setup pipeline).
pub use fri::{get_optimal_fri_query_params, FRIQueryResult, FRISecurityParams};
pub use hpf::goldilocks_cube_field_size;
pub use stir::{
    get_optimal_stir_query_params, try_get_optimal_stir_query_params, Protocol, StirQueryResult, StirRound,
    StirSecurityParams,
};

// Shared internals re-exported so sibling modules can reach them via `super::`.
pub(crate) use hpf::{hpf, hpf_from_f64, security_bits_from_error, truncate_decimal_places, PREC};
pub(crate) use query_cost::calculate_query_num_hashes;
