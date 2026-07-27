// The compact/indexed companion for an already-defined trace row. From
// `indexed_trace_row!(RowName<F> { <fields, some tagged `@instr`> })` it emits, beside the
// existing `RowName` / `RowNameOps` (from `trace_row!`):
//   - `RowNamePackedIndexed` : packed row = leading `index: u32` + the untagged (runtime) cols
//   - `RowNameInstrTable`    : packed row of just the `@instr` (instruction-derived) cols
//   - impl `RowNameOps` for `RowNamePackedIndexed` : runtime setters forward, `@instr` setters no-op
//   - impl `IndexedFill` for the row family
//   - `RowNamePackedIndexed::{COL_SOURCE, INDEX_BITS}`
// It does NOT redefine `RowName` or `RowNameOps`; those come from the pristine pil-helpers.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{braced, parse_macro_input, Ident, Result, Token};

use crate::packed_row::packed_row_impl;
use crate::trace_row::{
    collect_dimensions, compute_total_bits, contains_generic, is_array, parse_bit_type, BitType, TraceField,
};
use crate::trait_row::{generic_tokens, rust_type_for_bits};

/// Fixed width of the compact row's leading instruction-index header.
const INDEX_BITS: u64 = 32;

struct IndexedInput {
    name: Ident,
    generic: Option<Ident>,
    /// Each field with a flag: `true` = `@instr` (instruction-derived / table column).
    fields: Vec<(TraceField, bool)>,
}

impl Parse for IndexedInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let generic = if input.peek(Token![<]) {
            let _lt: Token![<] = input.parse()?;
            let ident: Ident = input.parse()?;
            let _gt: Token![>] = input.parse()?;
            Some(ident)
        } else {
            None
        };

        let content;
        let _brace = braced!(content in input);

        let mut fields = vec![];
        while !content.is_empty() {
            let name: Ident = content.parse()?;
            let _colon: Token![:] = content.parse()?;
            let ty = parse_bit_type(&content, generic.as_ref())?;
            let mut instr = false;
            if content.peek(Token![@]) {
                let _at: Token![@] = content.parse()?;
                let tag: Ident = content.parse()?;
                if tag == "instr" {
                    instr = true;
                } else {
                    return Err(syn::Error::new_spanned(tag, "unknown tag; expected `@instr`"));
                }
            }
            fields.push((TraceField { name, ty }, instr));
            if content.peek(Token![,]) {
                let _comma: Token![,] = content.parse()?;
            }
        }
        Ok(IndexedInput { name, generic, fields })
    }
}

pub fn indexed_trace_row_entrypoint(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let inp = parse_macro_input!(input as IndexedInput);

    let name = &inp.name;
    let generic = &inp.generic;
    let trait_name = format_ident!("{}Ops", name);
    let packed_name = format_ident!("{}Packed", name);
    let indexed_name = format_ident!("{}PackedIndexed", name);
    let table_name = format_ident!("{}InstrTable", name);

    // Compact runtime subset: leading index header, then the untagged columns in order.
    let mut runtime_fields = vec![TraceField { name: format_ident!("index"), ty: BitType::Bit(INDEX_BITS as usize) }];
    let mut table_fields = vec![];
    for (f, instr) in &inp.fields {
        if *instr {
            table_fields.push(f.clone());
        } else {
            runtime_fields.push(f.clone());
        }
    }

    let indexed_struct = packed_row_impl(&indexed_name, generic, &runtime_fields);
    let table_struct = packed_row_impl(&table_name, generic, &table_fields);
    let ops_impl = indexed_ops_impl(&trait_name, &indexed_name, generic, &inp.fields);

    // Per output column (arrays expanded), in declaration order: 1 = instruction-derived.
    let mut col_source: Vec<u8> = vec![];
    for (f, instr) in &inp.fields {
        // collect_dimensions assumes an array; only call it for arrays (scalars are 1 column).
        let cols = if is_array(&f.ty) {
            let (_, dims, _) = collect_dimensions(&f.ty);
            dims.iter().product::<usize>()
        } else {
            1
        };
        col_source.extend(std::iter::repeat(if *instr { 1u8 } else { 0u8 }).take(cols));
    }
    let n_cols = col_source.len();
    let col_lits = col_source.iter().map(|&s| quote! { #s });

    let (generics, generics_with_bounds) = generic_tokens(generic);

    let expanded = quote! {
        #indexed_struct
        #table_struct
        #ops_impl

        impl #generics_with_bounds #indexed_name #generics {
            /// Per output column: 1 = instruction-derived (from the table), 0 = runtime row.
            pub const COL_SOURCE: [u8; #n_cols] = [ #(#col_lits),* ];
            /// Width of the leading instruction-index header (bits).
            pub const INDEX_BITS: u64 = #INDEX_BITS;
        }

        impl #generics_with_bounds proofman_common::trace::IndexedFill for #indexed_name #generics {
            const IS_INDEXED: bool = true;
            #[inline(always)]
            fn set_row_index(&mut self, index: u32) { #indexed_name::set_index(self, index); }
        }
        impl #generics_with_bounds proofman_common::trace::IndexedFill for #name #generics {}
        impl #generics_with_bounds proofman_common::trace::IndexedFill for #packed_name #generics {}
    };

    proc_macro::TokenStream::from(expanded)
}

/// `impl {trait} for {indexed}`: untagged columns forward to the compact buffer's inherent
/// accessors; `@instr` columns are no-ops on set and return the type default on get.
fn indexed_ops_impl(
    trait_name: &Ident,
    indexed_name: &Ident,
    generic: &Option<Ident>,
    fields: &[(TraceField, bool)],
) -> TokenStream {
    let (generics, generics_with_bounds) = generic_tokens(generic);
    let mut methods = vec![];

    for (f, instr) in fields.iter() {
        let setter = format_ident!("set_{}", f.name);
        let getter = format_ident!("get_{}", f.name);

        if contains_generic(&f.ty) {
            if !is_array(&f.ty) {
                if *instr {
                    methods.push(quote! {
                        #[inline(always)] fn #setter(&mut self, _value: F) {}
                        #[inline(always)] fn #getter(&self) -> F { F::default() }
                    });
                } else {
                    methods.push(quote! {
                        #[inline(always)] fn #setter(&mut self, value: F) { #indexed_name::#setter(self, value); }
                        #[inline(always)] fn #getter(&self) -> F { #indexed_name::#getter(self) }
                    });
                }
            }
        } else if is_array(&f.ty) {
            assert!(!*instr, "indexed_trace_row!: `@instr` unsupported on array column `{}`", f.name);
            let (bits, dims, _) = collect_dimensions(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            let idx_args: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("i{}", i)).collect();
            let setter_all = format_ident!("set_all_{}", f.name);
            let getter_all = format_ident!("get_all_{}", f.name);
            let mut nested = rust_ty.clone();
            for &len in dims.iter().rev() {
                nested = quote! { [#nested; #len] };
            }
            methods.push(quote! {
                #[inline(always)]
                fn #setter(&mut self, #(#idx_args: usize,)* value: #rust_ty) { #indexed_name::#setter(self, #(#idx_args,)* value); }
                #[inline(always)]
                fn #getter(&self, #(#idx_args: usize),*) -> #rust_ty { #indexed_name::#getter(self, #(#idx_args),*) }
                #[inline(always)]
                fn #setter_all(&mut self, values: &#nested) { #indexed_name::#setter_all(self, values); }
                #[inline(always)]
                fn #getter_all(&self) -> #nested { #indexed_name::#getter_all(self) }
            });
        } else {
            let bits = compute_total_bits(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            if *instr {
                methods.push(quote! {
                    #[inline(always)] fn #setter(&mut self, _value: #rust_ty) {}
                    #[inline(always)] fn #getter(&self) -> #rust_ty { <#rust_ty>::default() }
                });
            } else {
                methods.push(quote! {
                    #[inline(always)] fn #setter(&mut self, value: #rust_ty) { #indexed_name::#setter(self, value); }
                    #[inline(always)] fn #getter(&self) -> #rust_ty { #indexed_name::#getter(self) }
                });
            }
        }
    }

    quote! {
        impl #generics_with_bounds #trait_name #generics for #indexed_name #generics {
            #(#methods)*
        }
    }
}
