//! Proc-macro crate entrypoint. Keep only thin #[proc_macro] wrappers here.
//! Implementation details live in `trace.rs`, `trace_row.rs`, `packed_row.rs`, and `unpacked_row.rs`.

use proc_macro::TokenStream;

mod trace;
mod trace_row;
mod packed_row;
mod unpacked_row;
mod trait_row;
mod indexed_trace_row;

#[proc_macro]
pub fn trace(input: TokenStream) -> TokenStream {
    trace::trace_entrypoint(input)
}

#[proc_macro]
pub fn values(input: TokenStream) -> TokenStream {
    trace::values_entrypoint(input)
}

#[proc_macro]
pub fn trace_row(input: TokenStream) -> TokenStream {
    trace_row::trace_row_entrypoint(input)
}

/// Companion to `trace_row!` for the indexed (compact) variant of an already-defined row.
/// The base row + its `RowNameOps` trait must already exist (e.g. from `trace_row!`); this
/// only adds the indexed variant beside them. See `indexed_trace_row.rs` for what it emits.
#[proc_macro]
pub fn indexed_trace_row(input: TokenStream) -> TokenStream {
    indexed_trace_row::indexed_trace_row_entrypoint(input)
}

// Keep the old packed_row macro for backward compatibility
#[proc_macro]
pub fn packed_row(input: TokenStream) -> TokenStream {
    // For now, redirect to the old implementation if it exists
    // You can remove this once you've migrated all usage to trace_row
    trace_row::trace_row_entrypoint(input)
}
