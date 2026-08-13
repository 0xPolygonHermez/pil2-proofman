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

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::trace_row::{TraceField, collect_dimensions, compute_total_bits, contains_generic, is_array};

pub fn trait_impl(
    trait_name: &Ident,
    unpacked_name: &Ident,
    packed_name: &Ident,
    generic: &Option<Ident>,
    fields: &[TraceField],
) -> TokenStream {
    let (generics, generics_with_bounds) = generic_tokens(generic);

    let trait_methods = trait_method_signatures(fields);
    let unpacked_fwd = forwarding_methods(unpacked_name, fields);
    let packed_fwd = forwarding_methods(packed_name, fields);

    quote! {
        pub trait #trait_name #generics_with_bounds: proofman_common::trace::TraceRow + Default + Copy + Send + Sync + std::fmt::Debug + 'static {
            #(#trait_methods)*
        }

        impl #generics_with_bounds #trait_name #generics for #unpacked_name #generics {
            #(#unpacked_fwd)*
        }

        impl #generics_with_bounds #trait_name #generics for #packed_name #generics {
            #(#packed_fwd)*
        }
    }
}

fn trait_method_signatures(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut out = vec![];

    for f in fields.iter() {
        let setter = format_ident!("set_{}", f.name);
        let getter = format_ident!("get_{}", f.name);

        if contains_generic(&f.ty) {
            if !is_array(&f.ty) {
                out.push(quote! {
                    fn #setter(&mut self, value: F);
                    fn #getter(&self) -> F;
                });
            }
        } else if is_array(&f.ty) {
            let (bits, dims, _) = collect_dimensions(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            let idx_args: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("i{}", i)).collect();
            let setter_all = format_ident!("set_all_{}", f.name);
            let getter_all = format_ident!("get_all_{}", f.name);
            let mut nested_type = rust_ty.clone();
            for &len in dims.iter().rev() {
                nested_type = quote! { [#nested_type; #len] };
            }

            out.push(quote! {
                fn #setter(&mut self, #(#idx_args: usize,)* value: #rust_ty);
                fn #getter(&self, #(#idx_args: usize),*) -> #rust_ty;
                fn #setter_all(&mut self, values: &#nested_type);
                fn #getter_all(&self) -> #nested_type;
            });
        } else {
            let bits = compute_total_bits(&f.ty);
            let rust_ty = rust_type_for_bits(bits);

            out.push(quote! {
                fn #setter(&mut self, value: #rust_ty);
                fn #getter(&self) -> #rust_ty;
            });
        }
    }

    out
}

fn forwarding_methods(type_name: &Ident, fields: &[TraceField]) -> Vec<TokenStream> {
    let mut out = vec![];

    for f in fields.iter() {
        let setter = format_ident!("set_{}", f.name);
        let getter = format_ident!("get_{}", f.name);

        if contains_generic(&f.ty) {
            if !is_array(&f.ty) {
                out.push(quote! {
                    #[inline(always)]
                    fn #setter(&mut self, value: F) {
                        #type_name::#setter(self, value);
                    }
                    #[inline(always)]
                    fn #getter(&self) -> F {
                        #type_name::#getter(self)
                    }
                });
            }
        } else if is_array(&f.ty) {
            let (bits, dims, _) = collect_dimensions(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            let idx_args: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("i{}", i)).collect();
            let setter_all = format_ident!("set_all_{}", f.name);
            let getter_all = format_ident!("get_all_{}", f.name);
            let mut nested_type = rust_ty.clone();
            for &len in dims.iter().rev() {
                nested_type = quote! { [#nested_type; #len] };
            }

            out.push(quote! {
                #[inline(always)]
                fn #setter(&mut self, #(#idx_args: usize,)* value: #rust_ty) {
                    #type_name::#setter(self, #(#idx_args,)* value);
                }
                #[inline(always)]
                fn #getter(&self, #(#idx_args: usize),*) -> #rust_ty {
                    #type_name::#getter(self, #(#idx_args),*)
                }
                #[inline(always)]
                fn #setter_all(&mut self, values: &#nested_type) {
                    #type_name::#setter_all(self, values);
                }
                #[inline(always)]
                fn #getter_all(&self) -> #nested_type {
                    #type_name::#getter_all(self)
                }
            });
        } else {
            let bits = compute_total_bits(&f.ty);
            let rust_ty = rust_type_for_bits(bits);

            out.push(quote! {
                #[inline(always)]
                fn #setter(&mut self, value: #rust_ty) {
                    #type_name::#setter(self, value);
                }
                #[inline(always)]
                fn #getter(&self) -> #rust_ty {
                    #type_name::#getter(self)
                }
            });
        }
    }

    out
}

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
