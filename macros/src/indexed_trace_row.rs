// The compact/indexed companion for an already-defined trace row. From
// `indexed_trace_row!(RowName<F> { <fields, some tagged `@instr`> })` it emits, beside the
// existing `RowName` / `RowNameOps` (from `trace_row!`):
//   - `RowNamePackedIndexed` : packed row = leading `index: [u32; LANES]` + the untagged cols
//   - `RowNameInstrTable`    : packed row of the `@instr` cols, lane dimension stripped
//   - impl `RowNameOps` for `RowNamePackedIndexed` : runtime setters forward, `@instr` no-op
//   - impl `RowNameOps` for `RowNameInstrTable`    : `@instr` setters land, runtime no-op
//   - impl `IndexedFill` for the row family
//   - `RowNamePackedIndexed::{COL_SOURCE, COL_LANE, INDEX_BITS, LANES}`
// It does NOT redefine `RowName` or `RowNameOps`; those come from the pristine pil-helpers.
//
// A row may pack several execution steps (lanes). The lane is the OUTER dimension of every
// `@instr` column, one table entry holds one lane's instruction, and the compact row carries
// one index per lane. `COL_LANE` tells the unpacker which lane's index selects the entry an
// instruction-derived output column is read from; that is what lets a single row mix columns
// coming from several different table entries.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{braced, parse_macro_input, Ident, Result, Token};

use crate::packed_row::packed_row_impl;
use crate::trace_row::{
    collect_dimensions, compute_total_bits, contains_generic, is_array, parse_bit_type, BitType, TraceField,
};
use crate::trait_row::{generic_tokens, rust_type_for_bits};

/// Fixed width of one instruction index in the compact row's header.
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

    let lanes = match infer_lanes(&inp.fields) {
        Ok(lanes) => lanes,
        Err(e) => return e.to_compile_error().into(),
    };

    let name = &inp.name;
    let generic = &inp.generic;
    let trait_name = format_ident!("{}Ops", name);
    let packed_name = format_ident!("{}Packed", name);
    let indexed_name = format_ident!("{}PackedIndexed", name);
    let table_name = format_ident!("{}InstrTable", name);

    // One index per lane, then the untagged columns in order. An array even for a single
    // lane, so the accessors have one shape.
    let mut runtime_fields = vec![TraceField {
        name: format_ident!("index"),
        ty: BitType::Array(Box::new(BitType::Bit(INDEX_BITS as usize)), lanes),
    }];
    let mut table_fields = vec![];
    for (f, instr) in &inp.fields {
        if *instr {
            table_fields.push(TraceField { name: f.name.clone(), ty: strip_lane(&f.ty) });
        } else {
            runtime_fields.push(f.clone());
        }
    }

    let indexed_struct = packed_row_impl(&indexed_name, generic, &runtime_fields);
    let table_struct = packed_row_impl(&table_name, generic, &table_fields);
    let ops_impl = indexed_ops_impl(&trait_name, &indexed_name, generic, &inp.fields);
    let table_ops = table_ops_impl(&trait_name, &table_name, generic, &inp.fields, lanes);

    // Per output column (arrays expanded), in declaration order. A field's columns are
    // lane-major, so within its block of `cols` columns each lane owns a run of
    // `cols / lanes`. Runtime columns select no entry, so their lane is 0.
    let mut col_source: Vec<u8> = vec![];
    let mut col_lane: Vec<u8> = vec![];
    for (f, instr) in &inp.fields {
        let cols = flat_cols(&f.ty);
        col_source.extend(std::iter::repeat_n(if *instr { 1u8 } else { 0u8 }, cols));
        if *instr {
            let per_lane = cols / lanes;
            col_lane.extend((0..cols).map(|k| (k / per_lane) as u8));
        } else {
            col_lane.extend(std::iter::repeat_n(0u8, cols));
        }
    }
    let n_cols = col_source.len();
    let source_lits = col_source.iter().map(|&s| quote! { #s });
    let lane_lits = col_lane.iter().map(|&l| quote! { #l });

    let (generics, generics_with_bounds) = generic_tokens(generic);

    let expanded = quote! {
        #indexed_struct
        #table_struct
        #ops_impl
        #table_ops

        impl #generics_with_bounds #indexed_name #generics {
            /// Per output column: 1 = instruction-derived (from the table), 0 = runtime row.
            pub const COL_SOURCE: [u8; #n_cols] = [ #(#source_lits),* ];
            /// Per output column: which lane's index selects its table entry (0 for runtime).
            pub const COL_LANE: [u8; #n_cols] = [ #(#lane_lits),* ];
            /// Width of one instruction index in the compact row's header (bits).
            pub const INDEX_BITS: u64 = #INDEX_BITS;
            /// Execution steps packed into one row; the header carries one index each.
            pub const LANES: usize = #lanes;
        }

        impl #generics_with_bounds proofman_common::trace::IndexedFill for #indexed_name #generics {
            const IS_INDEXED: bool = true;
            #[inline(always)]
            fn set_row_index(&mut self, lane: usize, index: u32) { #indexed_name::set_index(self, lane, index); }
        }
        impl #generics_with_bounds proofman_common::trace::IndexedFill for #name #generics {}
        impl #generics_with_bounds proofman_common::trace::IndexedFill for #packed_name #generics {}
    };

    proc_macro::TokenStream::from(expanded)
}

/// Lane count: the outer dimension of the `@instr` columns, which all of them must share.
/// Scalar instruction columns mean a single-lane row.
fn infer_lanes(fields: &[(TraceField, bool)]) -> Result<usize> {
    let mut lanes: Option<(usize, Ident)> = None;
    for (f, _) in fields.iter().filter(|(_, instr)| *instr) {
        let len = match &f.ty {
            BitType::Array(_, len) => *len,
            _ => 1,
        };
        match &lanes {
            None => {
                if len == 0 {
                    return Err(syn::Error::new_spanned(
                        &f.name,
                        "indexed_trace_row!: an `@instr` column cannot be zero lanes wide",
                    ));
                }
                if len > u8::MAX as usize {
                    return Err(syn::Error::new_spanned(
                        &f.name,
                        format!("indexed_trace_row!: at most {} lanes (COL_LANE is a u8)", u8::MAX),
                    ));
                }
                lanes = Some((len, f.name.clone()));
            }
            Some((first_len, first)) if *first_len != len => {
                return Err(syn::Error::new_spanned(
                    &f.name,
                    format!(
                        "indexed_trace_row!: every `@instr` column must carry the same number of \
                         lanes in its outer dimension; `{first}` has {first_len} but `{}` has {len}",
                        f.name
                    ),
                ));
            }
            _ => {}
        }
    }
    Ok(lanes.map_or(1, |(len, _)| len))
}

/// The declared column type with its outer (lane) dimension removed: what one entry holds.
fn strip_lane(ty: &BitType) -> BitType {
    match ty {
        BitType::Array(inner, _) => (**inner).clone(),
        other => other.clone(),
    }
}

/// Output columns a field expands to.
fn flat_cols(ty: &BitType) -> usize {
    if is_array(ty) {
        let (_, dims, _) = collect_dimensions(ty);
        dims.iter().product()
    } else {
        1
    }
}

/// `[[T; d1]; d0]` for dims `[d0, d1]`, outermost first.
fn nested_type(rust_ty: &TokenStream, dims: &[usize]) -> TokenStream {
    let mut ty = rust_ty.clone();
    for &len in dims.iter().rev() {
        ty = quote! { [#ty; #len] };
    }
    ty
}

/// A value of [`nested_type`] filled with `fill`.
fn nested_value(fill: &TokenStream, dims: &[usize]) -> TokenStream {
    let mut value = fill.clone();
    for &len in dims.iter().rev() {
        value = quote! { [#value; #len] };
    }
    value
}

/// `impl {trait} for {indexed}`: untagged columns forward to the compact buffer's inherent
/// accessors; `@instr` columns are no-ops on set and default on get.
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
            // The trait declares no accessors for generic array columns.
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
            let (bits, dims, _) = collect_dimensions(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            let idx_args: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("i{}", i)).collect();
            let ignored: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("_i{}", i)).collect();
            let setter_all = format_ident!("set_all_{}", f.name);
            let getter_all = format_ident!("get_all_{}", f.name);
            let nested = nested_type(&rust_ty, &dims);
            if *instr {
                let default_nested = nested_value(&quote! { <#rust_ty>::default() }, &dims);
                methods.push(quote! {
                    #[inline(always)]
                    fn #setter(&mut self, #(#ignored: usize,)* _value: #rust_ty) {}
                    #[inline(always)]
                    fn #getter(&self, #(#ignored: usize),*) -> #rust_ty { <#rust_ty>::default() }
                    #[inline(always)]
                    fn #setter_all(&mut self, _values: &#nested) {}
                    #[inline(always)]
                    fn #getter_all(&self) -> #nested { #default_nested }
                });
            } else {
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
            }
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

/// `impl {trait} for {table}`: the mirror of [`indexed_ops_impl`]. `@instr` columns land in
/// the entry with the lane index dropped -- an entry IS one lane -- so the writer that
/// fills a trace row lane by lane also fills an entry. Runtime columns are no-ops.
fn table_ops_impl(
    trait_name: &Ident,
    table_name: &Ident,
    generic: &Option<Ident>,
    fields: &[(TraceField, bool)],
    lanes: usize,
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
                        #[inline(always)] fn #setter(&mut self, value: F) { #table_name::#setter(self, value); }
                        #[inline(always)] fn #getter(&self) -> F { #table_name::#getter(self) }
                    });
                } else {
                    methods.push(quote! {
                        #[inline(always)] fn #setter(&mut self, _value: F) {}
                        #[inline(always)] fn #getter(&self) -> F { F::default() }
                    });
                }
            }
        } else if is_array(&f.ty) {
            let (bits, dims, _) = collect_dimensions(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            let setter_all = format_ident!("set_all_{}", f.name);
            let getter_all = format_ident!("get_all_{}", f.name);
            let nested = nested_type(&rust_ty, &dims);
            // Indices past the first address the entry's own column; the first is the lane.
            let inner_args: Vec<Ident> = (1..dims.len()).map(|i| format_ident!("i{}", i)).collect();

            if *instr {
                // Whole-array access spans lanes, which one entry cannot hold: lane 0 stands
                // for all of them.
                let (set_one, get_one, set_all, get_all) = if inner_args.is_empty() {
                    (
                        quote! { #table_name::#setter(self, value) },
                        quote! { #table_name::#getter(self) },
                        quote! { #table_name::#setter(self, values[0]) },
                        quote! { [#table_name::#getter(self); #lanes] },
                    )
                } else {
                    (
                        quote! { #table_name::#setter(self, #(#inner_args,)* value) },
                        quote! { #table_name::#getter(self, #(#inner_args),*) },
                        quote! { #table_name::#setter_all(self, &values[0]) },
                        quote! { [#table_name::#getter_all(self); #lanes] },
                    )
                };
                methods.push(quote! {
                    #[inline(always)]
                    fn #setter(&mut self, _lane: usize, #(#inner_args: usize,)* value: #rust_ty) { #set_one; }
                    #[inline(always)]
                    fn #getter(&self, _lane: usize, #(#inner_args: usize),*) -> #rust_ty { #get_one }
                    #[inline(always)]
                    fn #setter_all(&mut self, values: &#nested) { #set_all; }
                    #[inline(always)]
                    fn #getter_all(&self) -> #nested { #get_all }
                });
            } else {
                let ignored: Vec<Ident> = (0..dims.len()).map(|i| format_ident!("_i{}", i)).collect();
                let default_nested = nested_value(&quote! { <#rust_ty>::default() }, &dims);
                methods.push(quote! {
                    #[inline(always)]
                    fn #setter(&mut self, #(#ignored: usize,)* _value: #rust_ty) {}
                    #[inline(always)]
                    fn #getter(&self, #(#ignored: usize),*) -> #rust_ty { <#rust_ty>::default() }
                    #[inline(always)]
                    fn #setter_all(&mut self, _values: &#nested) {}
                    #[inline(always)]
                    fn #getter_all(&self) -> #nested { #default_nested }
                });
            }
        } else {
            let bits = compute_total_bits(&f.ty);
            let rust_ty = rust_type_for_bits(bits);
            if *instr {
                methods.push(quote! {
                    #[inline(always)] fn #setter(&mut self, value: #rust_ty) { #table_name::#setter(self, value); }
                    #[inline(always)] fn #getter(&self) -> #rust_ty { #table_name::#getter(self) }
                });
            } else {
                methods.push(quote! {
                    #[inline(always)] fn #setter(&mut self, _value: #rust_ty) {}
                    #[inline(always)] fn #getter(&self) -> #rust_ty { <#rust_ty>::default() }
                });
            }
        }
    }

    quote! {
        impl #generics_with_bounds #trait_name #generics for #table_name #generics {
            #(#methods)*
        }
    }
}
