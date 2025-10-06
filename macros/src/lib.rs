//! Proc-macro crate entrypoint. Keep only thin #[proc_macro] wrappers here.
//! Implementation details live in `trace.rs` and `packed_trace.rs`.

use proc_macro::TokenStream;

mod trace;
mod trace2;
mod packed_trace;

#[proc_macro]
pub fn trace(input: TokenStream) -> TokenStream {
    trace::trace_entrypoint(input)
}

#[proc_macro]
pub fn values(input: TokenStream) -> TokenStream {
    trace::values_entrypoint(input)
}

#[proc_macro]
pub fn packed_row(input: TokenStream) -> TokenStream {
    packed_trace::packed_row_entrypoint(input)
}
