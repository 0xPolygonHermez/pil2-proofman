// packed_row.rs - Packed row implementation

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::trace_row::{TraceField, BitType, contains_generic, compute_total_bits, is_array, collect_dimensions};

pub fn packed_row_impl(name: &Ident, generic: &Option<Ident>, fields: &[TraceField]) -> TokenStream {
    // Calculate bits needed for non-generic fields (these go in the packed array)
    let packed_bits: usize =
        fields.iter().filter(|f| !contains_generic(&f.ty)).map(|f| compute_total_bits(&f.ty)).sum();
    let packed_words = packed_bits.div_ceil(64);

    // Count generic F fields (these are stored separately, each takes one u64)
    let generic_field_count: usize =
        fields.iter().filter(|f| contains_generic(&f.ty)).map(|f| calculate_generic_field_size(&f.ty)).sum();

    // Total ROW_SIZE is packed words + generic fields
    let total_row_size = packed_words + generic_field_count;

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

    let default_field_exprs = get_default_field_exprs(fields);

    quote! {
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct #name #generics_with_bounds {
            #(#generic_fields,)*
            pub packed: [u64; #packed_words],
        }

        impl #generics_with_bounds Default for #name #generics {
            fn default() -> Self {
                Self {
                    #(#default_field_exprs,)*
                    packed: [0u64; #packed_words],
                }
            }
        }

        impl #generics_with_bounds #name #generics {
            pub const PACKED_BITS: usize = #packed_bits;
            pub const PACKED_WORDS: usize = #packed_words;
            #(#setter_getters)*
        }

        impl #generics_with_bounds proofman_common::trace::TraceRow for #name #generics {
            const ROW_SIZE: usize = #total_row_size; // Packed words + generic fields
            const IS_PACKED: bool = true;
        }
    }
}

fn get_packed_fields(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut packed_fields = vec![];
    let mut has_true_generic = false;

    for f in fields.iter() {
        let name = &f.name;
        if contains_generic(&f.ty) {
            // Always include truly generic fields with proper array expansion
            let field_type = generate_f_field_type(&f.ty);
            packed_fields.push(quote! { pub #name: #field_type });
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
            // For generic fields, only generate setters/getters for non-array fields
            // Array fields can be accessed directly
            if !is_array(&f.ty) {
                add_generic_setter_getter(&f.name, &mut setter_getters);
            }
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

fn add_generic_setter_getter(field_name: &Ident, setter_getters: &mut Vec<TokenStream>) {
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
    let (bit_width, dims, acc_dims) = collect_dimensions(field_type);
    let total_len: usize = dims.iter().product();
    let base_offset = *offset;
    *offset += bit_width * total_len;
    let rust_type = type_for_bitwidth(bit_width);

    let max_bit_start = 64 - bit_width;

    // Runtime params: i0: usize, ...
    let runtime_idents: Vec<Ident> = dims.iter().enumerate().map(|(i, _)| format_ident!("i{}", i)).collect();

    let runtime_flat = flatten_index_expr(&runtime_idents, &acc_dims);

    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);
    let setter_name_all = format_ident!("set_all_{}", field_name);
    let getter_name_all = format_ident!("get_all_{}", field_name);

    // Build the nested array type matching the declared shape, e.g. [[u8; 4]; 3]
    // dims is outermost-first, so wrap innermost-first.
    let nested_type = {
        let mut ty = rust_type.clone();
        for &len in dims.iter().rev() {
            ty = quote! { [#ty; #len] };
        }
        ty
    };

    let conversion = if bit_width == 1 {
        quote! { raw_value != 0 }
    } else {
        quote! { raw_value as #rust_type }
    };

    // Per-dimension bounds checks for the runtime accessor. These fire in debug builds
    // whenever an individual index is out of range for its dimension, even if the
    // resulting flat index would (by chance) still map to a valid element.
    let bounds_checks: Vec<TokenStream> = runtime_idents
        .iter()
        .zip(dims.iter())
        .enumerate()
        .map(|(dim_idx, (id, &dim))| {
            quote! {
                debug_assert!(
                    #id < #dim,
                    "Index out of bounds for field {} (dimension {}: {} >= {})",
                    stringify!(#field_name), #dim_idx, #id, #dim
                );
            }
        })
        .collect();

    // Whole-array setter/getter: one statement per element, fully unrolled at
    // compile time with all offsets/masks as immediate literals — no match, no branch.
    // LLVM can vectorise the resulting straight-line code across the whole array.
    let mut all_setter_stmts: Vec<TokenStream> = Vec::with_capacity(total_len);
    let mut all_getter_stmts: Vec<TokenStream> = Vec::with_capacity(total_len);
    // Combined clear mask per packed word, accumulated across all elements of this field.
    // Enables a single AND per word in set_all instead of one AND per element.
    let mut word_clear_masks: std::collections::BTreeMap<usize, u64> = std::collections::BTreeMap::new();

    for flat_idx in 0..total_len {
        let start = base_offset + flat_idx * bit_width;
        let end = start + bit_width;
        let word_start = start / 64;
        let word_end = (end - 1) / 64;
        let bit_start = start % 64;

        // Compute per-dimension indices from flat_idx (row-major, dims outermost-first).
        // e.g. for dims=[3,4]: idx0 = flat_idx / 4, idx1 = flat_idx % 4
        let idx_exprs: Vec<TokenStream> = {
            let mut exprs = Vec::with_capacity(dims.len());
            // strides[k] = product of dims[k+1..]
            let strides: Vec<usize> = (0..dims.len()).map(|k| dims[k + 1..].iter().product()).collect();
            let mut rem = flat_idx;
            for &stride in &strides {
                let idx = rem / stride;
                rem %= stride;
                exprs.push(quote! { #idx });
            }
            exprs
        };

        // Nested index into `values`, e.g. values[2][3]
        let values_access = quote! { values #( [#idx_exprs] )* };
        // Nested index into `result`, e.g. result[2][3]
        let result_access = quote! { result #( [#idx_exprs] )* };

        // Debug-only range validation for set_all (mirrors the per-element debug_assert
        // in the scalar setter so the two APIs behave consistently in debug builds).
        let set_all_debug_check = quote! {
            debug_assert!(
                (#values_access as u128) < (1u128 << #bit_width),
                "set_all_{}: element [{}] value {} exceeds {}-bit max {}",
                stringify!(#field_name), #flat_idx, #values_access as u128,
                #bit_width, (1u128 << #bit_width) - 1
            );
        };

        if word_start == word_end {
            let mask_bits = ((1u128 << bit_width) - 1) as u64;
            let mask = mask_bits << bit_start;

            *word_clear_masks.entry(word_start).or_insert(0) |= mask;
            all_setter_stmts.push(quote! {
                #set_all_debug_check
                self.packed[#word_start] |= ((#values_access as u64) & (#mask_bits as u64)) << #bit_start;
            });
            all_getter_stmts.push(quote! {
                #result_access = {
                    let raw_value = (self.packed[#word_start] >> #bit_start) & (#mask_bits as u64);
                    #conversion
                };
            });
        } else {
            let low_bits = 64 - bit_start;
            let high_bits = bit_width - low_bits;
            let low_mask = ((1u128 << low_bits) - 1) as u64;
            let high_mask = ((1u128 << high_bits) - 1) as u64;

            *word_clear_masks.entry(word_start).or_insert(0) |= low_mask << bit_start;
            *word_clear_masks.entry(word_end).or_insert(0) |= high_mask;
            all_setter_stmts.push(quote! {
                #set_all_debug_check
                self.packed[#word_start] |= ((#values_access as u64) & #low_mask) << #bit_start;
                self.packed[#word_end] |= ((#values_access as u64) >> #low_bits) & #high_mask;
            });
            all_getter_stmts.push(quote! {
                #result_access = {
                    let low = (self.packed[#word_start] >> #bit_start) & #low_mask;
                    let high = self.packed[#word_end] & #high_mask;
                    let raw_value = (high << #low_bits) | low;
                    #conversion
                };
            });
        }
    }

    // One AND per packed word to clear all bits occupied by this field — emitted before the ORs.
    let bulk_clear_stmts: Vec<TokenStream> = word_clear_masks
        .iter()
        .map(|(&word, &combined_mask)| {
            quote! { self.packed[#word] &= !(#combined_mask as u64); }
        })
        .collect();

    // Typed zero initializer for the nested array type (e.g. `[false; 64]` or `[[0; 8]; 4]`).
    // Generated instead of relying on `Default::default()` so that large array sizes like
    // `[bool; 64]` work on all stable Rust versions (the const-generic Default impl for
    // arrays was only stabilized in Rust 1.55).
    let zero_init = zero_init_expr(bit_width, field_type);

    setter_getters.push(quote! {
        // Runtime-indexed version — arithmetic-based, O(1) generated code regardless of
        // array size. Per-dimension bounds checks catch aliased out-of-range indices in
        // debug builds (a flat-index match cannot detect e.g. i1 == dim1 when i0 is
        // small enough that the flat index still falls within bounds).
        #[inline(always)]
        pub fn #setter_name(&mut self, #(#runtime_idents: usize,)* value: #rust_type) {
            #(#bounds_checks)*
            debug_assert!(
                (value as u128) < (1u128 << #bit_width),
                "Value out of range for field {} (setter: {}): value={}, bit_width={}, max_value={}",
                stringify!(#field_name), stringify!(#setter_name), value as u128, #bit_width, (1u128 << #bit_width) - 1
            );
            let flat: usize = #runtime_flat;
            let bit_offset: usize = #base_offset + flat * #bit_width;
            let word: usize = bit_offset / 64;
            let bit_start: usize = bit_offset % 64;
            if bit_start <= #max_bit_start {
                let mask_bits: u64 = ((1u128 << #bit_width) - 1) as u64;
                self.packed[word] &= !(mask_bits << bit_start);
                self.packed[word] |= ((value as u64) & mask_bits) << bit_start;
            } else {
                let low_bits: usize = 64 - bit_start;
                let high_bits: usize = #bit_width - low_bits;
                let low_mask: u64 = ((1u128 << low_bits) - 1) as u64;
                let high_mask: u64 = ((1u128 << high_bits) - 1) as u64;
                self.packed[word] &= !(low_mask << bit_start);
                self.packed[word] |= ((value as u64) & low_mask) << bit_start;
                self.packed[word + 1] &= !high_mask;
                self.packed[word + 1] |= ((value as u64) >> low_bits) & high_mask;
            }
        }

        #[inline(always)]
        pub fn #getter_name(&self, #(#runtime_idents: usize),*) -> #rust_type {
            #(#bounds_checks)*
            let flat: usize = #runtime_flat;
            let bit_offset: usize = #base_offset + flat * #bit_width;
            let word: usize = bit_offset / 64;
            let bit_start: usize = bit_offset % 64;
            let raw_value: u64 = if bit_start <= #max_bit_start {
                let mask_bits: u64 = ((1u128 << #bit_width) - 1) as u64;
                (self.packed[word] >> bit_start) & mask_bits
            } else {
                let low_bits: usize = 64 - bit_start;
                let high_bits: usize = #bit_width - low_bits;
                let low_mask: u64 = ((1u128 << low_bits) - 1) as u64;
                let high_mask: u64 = ((1u128 << high_bits) - 1) as u64;
                let low = (self.packed[word] >> bit_start) & low_mask;
                let high = self.packed[word + 1] & high_mask;
                (high << low_bits) | low
            };
            #conversion
        }

        // Whole-array versions — fully unrolled, all bit-offsets and masks are immediate
        // literals.  No branches; LLVM can auto-vectorise the straight-line code.
        // `values` and the return type preserve the declared shape (e.g. [[u8; 4]; 3]).
        // #[inline] (not always) to let the compiler decide at each call site.
        #[inline]
        pub fn #setter_name_all(&mut self, values: &#nested_type) {
            #(#bulk_clear_stmts)*
            #(#all_setter_stmts)*
        }

        #[inline]
        pub fn #getter_name_all(&self) -> #nested_type {
            let mut result: #nested_type = #zero_init;
            #(#all_getter_stmts)*
            result
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

    let getter_conversion = if bit_width == 1 {
        quote! { raw_value != 0 }
    } else {
        quote! { raw_value as #rust_type }
    };

    quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_type) {
            debug_assert!(
                (value as u128) < (1u128 << #bit_width),
                "Value out of range (setter: {}): value={}, bit_width={}, max_value={}",
                stringify!(#setter_name), value as u128, #bit_width, (1u128 << #bit_width) - 1
            );
            const MASK_BITS: u64 = #mask_bits;
            const MASK: u64 = #mask;
            self.packed[#word_start] &= !MASK;
            self.packed[#word_start] |= ((value as u64) & MASK_BITS) << #bit_start;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> #rust_type {
            const MASK_BITS: u64 = #mask_bits;
            let raw_value = (self.packed[#word_start] >> #bit_start) & MASK_BITS;
            #getter_conversion
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

    let getter_conversion = if bit_width == 1 {
        quote! { raw_value != 0 }
    } else {
        quote! { raw_value as #rust_type }
    };

    quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_type) {
            debug_assert!(
                (value as u128) < (1u128 << #bit_width),
                "Value out of range (setter: {}): value={}, bit_width={}, max_value={}",
                stringify!(#setter_name), value as u128, #bit_width, (1u128 << #bit_width) - 1
            );
            const LOW_MASK: u64 = #low_mask;
            const HIGH_MASK: u64 = #high_mask;
            self.packed[#word_start] &= !(LOW_MASK << #bit_start);
            self.packed[#word_start] |= ((value as u64) & LOW_MASK) << #bit_start;
            self.packed[#word_end] &= !HIGH_MASK;
            self.packed[#word_end] |= ((value as u64) >> #low_bits) & HIGH_MASK;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> #rust_type {
            const LOW_MASK: u64 = #low_mask;
            const HIGH_MASK: u64 = #high_mask;
            let low = (self.packed[#word_start] >> #bit_start) & LOW_MASK;
            let high = self.packed[#word_end] & HIGH_MASK;
            let raw_value = (high << #low_bits) | low;
            #getter_conversion
        }
    }
}

fn zero_init_expr(bit_width: usize, ty: &BitType) -> TokenStream {
    let leaf = if bit_width == 1 {
        quote! { false }
    } else {
        quote! { 0 }
    };
    array_zero_init(ty, leaf)
}

fn array_zero_init(ty: &BitType, leaf: TokenStream) -> TokenStream {
    match ty {
        BitType::Bit(_) | BitType::Generic => leaf,
        BitType::Array(inner, len) => {
            let inner_init = array_zero_init(inner, leaf);
            quote! { [#inner_init; #len] }
        }
    }
}

fn calculate_generic_field_size(ty: &BitType) -> usize {
    match ty {
        BitType::Bit(_) => 1,  // This shouldn't happen for generic fields, but just in case
        BitType::Generic => 1, // Generic F field is one F element
        BitType::Array(inner, len) => {
            calculate_generic_field_size(inner) * len // Recursively calculate array size
        }
    }
}

fn generate_f_field_type(ty: &BitType) -> TokenStream {
    match ty {
        BitType::Bit(_) => quote! { F },
        BitType::Generic => quote! { F },
        BitType::Array(inner, len) => {
            let inner_type = generate_f_field_type(inner);
            quote! { [#inner_type; #len] }
        }
    }
}

fn get_default_field_exprs(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut default_exprs = vec![];
    let mut has_true_generic = false;

    for f in fields.iter() {
        if contains_generic(&f.ty) {
            let name = &f.name;
            let init = default_expr(&f.ty);
            default_exprs.push(quote! { #name: #init });
            has_true_generic = true;
        }
    }

    // If we have a generic parameter F but no truly generic fields, add PhantomData
    if !has_true_generic {
        default_exprs.push(quote! {
            _phantom: std::marker::PhantomData
        });
    }

    default_exprs
}

fn default_expr(ty: &BitType) -> TokenStream {
    match ty {
        BitType::Bit(_) => quote! { F::default() },
        BitType::Generic => quote! { F::default() },
        BitType::Array(inner, len) => {
            let inner_default = default_expr(inner);
            quote! { [#inner_default; #len] }
        }
    }
}
