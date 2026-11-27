// trait_row.rs - Shared trait generation for trace row types
//
// For each trace_row!(FooTraceRow<F> { ... }) invocation, this generates:
//
//   pub trait FooTraceRowOps<F: ...>: Default + Copy + Send { ... }
//   impl<F: ...> FooTraceRowOps<F> for FooTraceRow<F>       { ... }  // forwards to inherent methods
//   impl<F: ...> FooTraceRowOps<F> for FooTraceRowPacked<F> { ... }  // forwards to inherent methods
//
// Both forwarding impls use fully-qualified UFCS (e.g. `FooTraceRow::set_foo(self, v)`)
// to unambiguously call the inherent method rather than the trait method.
//
// The trait and the unpacked impl take any `F` an unpacked row can hold, a field of any size
// (BN128's `Fr` too); the accessors of the typed columns (`bit`, `ubit(N)`, `u8`…) convert
// through 64 bits, so they alone ask `where F: PrimeField64`. The packed row is 64-bit only.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::trace_row::{TraceField, collect_dimensions, compute_total_bits, contains_generic, is_array};

pub fn trait_impl(
    trait_name: &Ident,
    unpacked_name: &Ident,
    packed_name: Option<&Ident>,
    generic: &Option<Ident>,
    fields: &[TraceField],
    provided: TokenStream,
) -> TokenStream {
    let (generics, packed_bounds) = generic_tokens(generic);
    let (trait_bounds, unpacked_bounds) = match generic {
        // The unpacked row is `Sync` and `Debug`, as the trait asks, when its `F` is.
        Some(g) => (
            quote! { <#g: Copy + Default + Send + 'static> },
            quote! { <#g: Copy + Default + Send + Sync + std::fmt::Debug + 'static> },
        ),
        None => (quote! {}, quote! {}),
    };

    let trait_methods = trait_method_signatures(fields);
    let unpacked_fwd = forwarding_methods(unpacked_name, fields);
    let packed_impl = packed_name.map(|packed_name| {
        let packed_fwd = forwarding_methods(packed_name, fields);
        quote! {
            impl #packed_bounds #trait_name #generics for #packed_name #generics {
                #(#packed_fwd)*
            }
        }
    });

    quote! {
        pub trait #trait_name #trait_bounds: proofman_common::trace::TraceRow + Default + Copy + Send + Sync + std::fmt::Debug + 'static {
            #(#trait_methods)*
            #provided
        }

        impl #unpacked_bounds #trait_name #generics for #unpacked_name #generics {
            #(#unpacked_fwd)*
        }

        #packed_impl
    }
}

/// Each method the trait has for field `f`: its signature, name and the arguments it passes on.
fn field_methods(f: &TraceField) -> Vec<(TokenStream, Ident, TokenStream)> {
    let setter = format_ident!("set_{}", f.name);
    let getter = format_ident!("get_{}", f.name);
    let hidden = f.hidden.then(|| quote! { #[doc(hidden)] });

    if contains_generic(&f.ty) {
        if is_array(&f.ty) {
            return vec![];
        }
        return vec![
            (quote! { #hidden fn #setter(&mut self, value: F) }, setter, quote! { value }),
            (quote! { #hidden fn #getter(&self) -> F }, getter, quote! {}),
        ];
    }
    if !is_array(&f.ty) {
        let rust_ty = rust_type_for_bits(compute_total_bits(&f.ty));
        return vec![
            (quote! { #hidden fn #setter(&mut self, value: #rust_ty) where F: PrimeField64 }, setter, quote! { value }),
            (quote! { #hidden fn #getter(&self) -> #rust_ty where F: PrimeField64 }, getter, quote! {}),
        ];
    }

    let (bits, dims, _) = collect_dimensions(&f.ty);
    let rust_ty = rust_type_for_bits(bits);
    let idx: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("i{}", i)).collect();
    let nested = dims.iter().rev().fold(rust_ty.clone(), |acc, len| quote! { [#acc; #len] });
    let mut methods = vec![
        (
            quote! { #hidden fn #setter(&mut self, #(#idx: usize,)* value: #rust_ty) where F: PrimeField64 },
            setter,
            quote! { #(#idx,)* value },
        ),
        (
            quote! { #hidden fn #getter(&self, #(#idx: usize),*) -> #rust_ty where F: PrimeField64 },
            getter,
            quote! { #(#idx),* },
        ),
    ];
    // A lane's flattened column has no whole-array accessors.
    if !f.hidden {
        let (setter_all, getter_all) = (format_ident!("set_all_{}", f.name), format_ident!("get_all_{}", f.name));
        methods.push((
            quote! { fn #setter_all(&mut self, values: &#nested) where F: PrimeField64 },
            setter_all,
            quote! { values },
        ));
        methods.push((quote! { fn #getter_all(&self) -> #nested where F: PrimeField64 }, getter_all, quote! {}));
    }
    methods
}

fn trait_method_signatures(fields: &[TraceField]) -> Vec<TokenStream> {
    fields.iter().flat_map(field_methods).map(|(sig, _, _)| quote! { #sig; }).collect()
}

fn forwarding_methods(type_name: &Ident, fields: &[TraceField]) -> Vec<TokenStream> {
    fields
        .iter()
        .flat_map(field_methods)
        .map(|(sig, name, args)| {
            quote! {
                #[inline(always)]
                #sig {
                    #type_name::#name(self, #args)
                }
            }
        })
        .collect()
}

/// The generics of a row, bare and with the bounds of a packed row, whose words are 64-bit.
pub(crate) fn generic_tokens(generic: &Option<Ident>) -> (TokenStream, TokenStream) {
    match generic {
        Some(g) => (quote! { <#g> }, quote! { <#g: PrimeField64 + Copy + Default + Send + 'static> }),
        None => (quote! {}, quote! {}),
    }
}

pub(crate) fn rust_type_for_bits(bits: usize) -> TokenStream {
    match bits {
        1 => quote! { bool },
        2..=8 => quote! { u8 },
        9..=16 => quote! { u16 },
        17..=32 => quote! { u32 },
        33..=64 => quote! { u64 },
        _ => quote! { u128 },
    }
}
