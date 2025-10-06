// packed_row.rs - Packed row implementation

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::trace_row::{TraceField, BitType, contains_generic, compute_total_bits, is_array, collect_dimensions};

pub fn packed_row_impl(
    name: &Ident,
    generic: &Option<Ident>,
    fields: &[TraceField],
) -> TokenStream {
    let packed_bits: usize = fields.iter().map(|f| compute_total_bits(&f.ty)).sum();
    let packed_words = packed_bits.div_ceil(64);

    let generics = if let Some(g) = generic {
        quote! { <#g> }
    } else {
        quote! {}
    };
    let generics_with_bounds = if let Some(g) = generic {
        quote! { <#g: PrimeField64 + Copy + Default + Send> }
    } else {
        quote! {}
    };

    let generic_fields = get_packed_fields(fields);
    let setter_getters = get_packed_setters_getters(fields);

    quote! {
        #[derive(Debug, Copy, Clone, Default)]
        pub struct #name #generics_with_bounds {
            #(#generic_fields,)*
            pub packed: [u64; #packed_words],
        }

        impl #generics_with_bounds #name #generics {
            pub const PACKED_BITS: usize = #packed_bits;
            pub const PACKED_WORDS: usize = #packed_words;
            #(#setter_getters)*
        }

        impl #generics_with_bounds proofman_common::trace2::TraceRow for #name #generics {
            const ROW_SIZE: usize = #name::#generics::PACKED_WORDS;
        }
    }
}

fn get_packed_fields(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut packed_fields = vec![];
    let mut has_true_generic = false;
    
    for f in fields.iter() {
        let name = &f.name;
        if contains_generic(&f.ty) {
            // Always include truly generic fields
            packed_fields.push(quote! { pub #name: F });
            has_true_generic = true;
        }
        // Non-generic fields are stored in the packed array, not as separate fields
    }
    
    // If we have a generic parameter F but no truly generic fields, add PhantomData
    if !has_true_generic {
        packed_fields.push(quote! {
            _phantom: std::marker::PhantomData<F>
        });
    }
    
    packed_fields
}

fn get_packed_setters_getters(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut offset = 0usize;
    let mut setter_getters = vec![];

    for f in fields.iter() {
        if contains_generic(&f.ty) {
            // Generate direct F field accessors for truly generic fields
            add_generic_setter_getter(&f.name, &mut setter_getters);
        } else {
            // For non-generic fields, generate packed accessors
            if is_array(&f.ty) {
                add_packed_array_setter_getter(&f.name, &f.ty, &mut offset, &mut setter_getters);
            } else {
                add_packed_setter_getter(&f.name, &f.ty, &mut offset, &mut setter_getters);
            }
        }
    }

    setter_getters
}

fn add_generic_setter_getter(
    field_name: &Ident,
    setter_getters: &mut Vec<TokenStream>,
) {
    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);

    setter_getters.push(quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: F) {
            self.#field_name = value;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> F {
            self.#field_name
        }
    });
}

fn add_packed_setter_getter(
    field_name: &Ident,
    field_type: &BitType,
    offset: &mut usize,
    setter_getters: &mut Vec<TokenStream>,
) {
    let bit_width = compute_total_bits(field_type);
    let rust_type = type_for_bitwidth(bit_width);

    // Compute where the field starts and ends in the packed array
    let start = *offset;
    let end = *offset + bit_width;
    *offset = end;

    let word_start = start / 64;
    let word_end = (end - 1) / 64;
    let bit_start = start % 64;

    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);

    let tokens = if word_start == word_end {
        emit_contained_packed_accessor(&setter_name, &getter_name, word_start, bit_width, bit_start, rust_type)
    } else {
        emit_split_packed_accessor(&setter_name, &getter_name, word_start, word_end, bit_width, bit_start, rust_type)
    };

    setter_getters.push(tokens);
}

fn add_packed_array_setter_getter(
    field_name: &Ident,
    field_type: &BitType,
    offset: &mut usize,
    setter_getters: &mut Vec<TokenStream>,
) {
    let (bit_width, dims) = collect_dimensions(field_type);
    let total_len: usize = dims.iter().product();
    let base_offset = *offset;
    *offset += bit_width * total_len;

    let rust_type = type_for_bitwidth(bit_width);
    let args = dimension_args(&dims);
    let flat = flatten_index_expr(&args, &dims);
    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);

    setter_getters.push(quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, #(#args: usize,)* value: #rust_type) {
            debug_assert!((value as u128) < (1u128 << #bit_width), "Value out of range for {}", stringify!(#field_name));
            let index = #flat;
            let bit_offset = #base_offset + index * #bit_width;
            let word_start = bit_offset / 64;
            let bit_start = bit_offset % 64;

            if bit_start + #bit_width <= 64 {
                const mask: u64 = ((1u128 << #bit_width) - 1) as u64;
                self.packed[word_start] &= !(mask << bit_start);
                self.packed[word_start] |= ((value as u64) & mask) << bit_start;
            } else {
                let low_bits = 64 - bit_start;
                let high_bits = #bit_width - low_bits;
                let low_mask: u64 = ((1u128 << low_bits) - 1) as u64;
                let high_mask: u64 = ((1u128 << high_bits) - 1) as u64;

                self.packed[word_start] &= !(low_mask << bit_start);
                self.packed[word_start] |= ((value as u64) & low_mask) << bit_start;

                self.packed[word_start + 1] &= !high_mask;
                self.packed[word_start + 1] |= ((value as u64) >> low_bits) & high_mask;
            }
        }

        #[inline(always)]
        pub fn #getter_name(&self, #(#args: usize),*) -> F {
            let index = #flat;
            let bit_offset = #base_offset + index * #bit_width;
            let word_start = bit_offset / 64;
            let bit_start = bit_offset % 64;

            let raw_value = if bit_start + #bit_width <= 64 {
                const mask: u64 = ((1u128 << #bit_width) - 1) as u64;
                (self.packed[word_start] >> bit_start) & mask
            } else {
                let low_bits = 64 - bit_start;
                let high_bits = #bit_width - low_bits;
                let low_mask: u64 = ((1u128 << low_bits) - 1) as u64;
                let high_mask: u64 = ((1u128 << high_bits) - 1) as u64;

                let low = (self.packed[word_start] >> bit_start) & low_mask;
                let high = self.packed[word_start + 1] & high_mask;
                (high << low_bits) | low
            };
            F::from_u64(raw_value)
        }
    });
}

fn type_for_bitwidth(width: usize) -> TokenStream {
    match width {
        1 => quote! { bool },
        2..=8 => quote! { u8 },
        9..=16 => quote! { u16 },
        17..=32 => quote! { u32 },
        33..=64 => quote! { u64 },
        _ => quote! { u128 },
    }
}

fn dimension_args(dims: &[usize]) -> Vec<Ident> {
    dims.iter().enumerate().map(|(i, _)| format_ident!("i{}", i)).collect()
}

fn flatten_index_expr(idents: &[Ident], dims: &[usize]) -> TokenStream {
    let mut expr = quote! { #(#idents)* };
    let mut iter = idents.iter().zip(dims.iter()).rev();
    if let Some((last_id, _)) = iter.next() {
        expr = quote! { #last_id };
    }
    for (id, dim) in iter {
        expr = quote! { (#id * #dim) + #expr };
    }
    expr
}

fn emit_contained_packed_accessor(
    setter_name: &Ident,
    getter_name: &Ident,
    word_start: usize,
    bit_width: usize,
    bit_start: usize,
    rust_type: TokenStream,
) -> TokenStream {
    let mask_bits = ((1u128 << bit_width) - 1) as u64;
    let mask = mask_bits << bit_start;

    quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_type) {
            debug_assert!((value as u128) < (1u128 << #bit_width), "Value out of range");
            const MASK_BITS: u64 = #mask_bits;
            const MASK: u64 = #mask;
            self.packed[#word_start] &= !MASK;
            self.packed[#word_start] |= ((value as u64) & MASK_BITS) << #bit_start;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> F {
            const MASK_BITS: u64 = #mask_bits;
            let raw_value = (self.packed[#word_start] >> #bit_start) & MASK_BITS;
            F::from_u64(raw_value)
        }
    }
}

fn emit_split_packed_accessor(
    setter_name: &Ident,
    getter_name: &Ident,
    word_start: usize,
    word_end: usize,
    bit_width: usize,
    bit_start: usize,
    rust_type: TokenStream,
) -> TokenStream {
    let low_bits = 64 - bit_start;
    let high_bits = bit_width - low_bits;
    let low_mask = ((1u128 << low_bits) - 1) as u64;
    let high_mask = ((1u128 << high_bits) - 1) as u64;

    quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_type) {
            debug_assert!((value as u128) < (1u128 << #bit_width), "Value out of range");
            const LOW_MASK: u64 = #low_mask;
            const HIGH_MASK: u64 = #high_mask;
            self.packed[#word_start] &= !(LOW_MASK << #bit_start);
            self.packed[#word_start] |= ((value as u64) & LOW_MASK) << #bit_start;
            self.packed[#word_end] &= !HIGH_MASK;
            self.packed[#word_end] |= ((value as u64) >> #low_bits) & HIGH_MASK;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> F {
            const LOW_MASK: u64 = #low_mask;
            const HIGH_MASK: u64 = #high_mask;
            let low = (self.packed[#word_start] >> #bit_start) & LOW_MASK;
            let high = self.packed[#word_end] & HIGH_MASK;
            let raw_value = (high << #low_bits) | low;
            F::from_u64(raw_value)
        }
    }
}