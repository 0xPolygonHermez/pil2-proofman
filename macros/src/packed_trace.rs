// trace_macro/src/lib.rs

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Result, Token, braced, parse_macro_input, token};

/// This macro generates a packed row structure representing a packed trace row.
/// It takes a name and a list of fields with their types, and generates
/// a structure that packs these fields into a fixed-size array of u64.
///
/// # Example
/// ```rust
/// use proofman_macros::packed_row;
///
/// packed_row! {
///     PackedRowExample<F> {
///         field0: F,
///         field1: u8,
///         field2: [[ubit(40);2]; 3],
///         field3: bit,
///         field4: [F; 2],
///         field5: [i32; 5],
///     }
/// }
struct TraceRowInput {
    name: Ident,
    generic: Option<Ident>,
    fields: Vec<TraceField>,
}

/// This struct represents a field in the packed row.
/// It contains the field name and its type.
#[allow(dead_code)]
struct TraceField {
    name: Ident,
    ty: BitType,
}

/// This enum represents the signedness of a bit type.
/// It can be either signed or unsigned.
#[allow(dead_code)]
#[derive(Clone)]
enum Signedness {
    Signed,
    Unsigned,
}

/// This enum represents the type of a field in the packed row.
/// It can be a bit type, a generic type, or an array of BitType.
#[allow(dead_code)]
#[derive(Clone)]
enum BitType {
    Bit(usize, Signedness),
    Generic,
    Array(Box<BitType>, usize),
}

/// DSL parsing
impl Parse for TraceRowInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        let generic = if input.peek(Token![<]) {
            let _lt: Token![<] = input.parse()?;
            let ident: Ident = input.parse()?;
            let _gt: Token![>] = input.parse()?;
            Some(ident)
        } else {
            None
        };

        let content;
        let _brace_token = braced!(content in input);

        let mut fields = vec![];
        while !content.is_empty() {
            let name = content.parse()?;
            let _colon_token: Token![:] = content.parse()?;
            let ty = parse_bit_type(&content, generic.as_ref())?;
            fields.push(TraceField { name, ty });
            if content.peek(Token![,]) {
                let _comma: Token![,] = content.parse()?;
            }
        }
        Ok(TraceRowInput { name, generic, fields })
    }
}

fn parse_bit_type(input: ParseStream, generic: Option<&Ident>) -> Result<BitType> {
    if input.peek(token::Bracket) {
        let content;
        let _ = syn::bracketed!(content in input);
        let elem_ty = parse_bit_type(&content, generic)?;
        let _semi: Token![;] = content.parse()?;
        let len_expr: syn::Expr = content.parse()?;
        let len = if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit), .. }) = len_expr {
            lit.base10_parse::<usize>()?
        } else {
            return Err(syn::Error::new_spanned(len_expr, "Expected integer length"));
        };
        Ok(BitType::Array(Box::new(elem_ty), len))
    } else if input.peek(Ident) {
        let ident: Ident = input.parse()?;
        let ident_str = ident.to_string();

        if let Some(g) = generic {
            if ident == *g {
                return Ok(BitType::Generic);
            }
        }

        match ident_str.as_str() {
            "bit" => Ok(BitType::Bit(1, Signedness::Unsigned)),
            "ubit" => {
                if input.peek(token::Paren) {
                    let bit_count = get_bit_count(input, "ubit", 1, 64)?;
                    Ok(BitType::Bit(bit_count, Signedness::Signed))
                } else {
                    Err(input.error("Expected parentheses after `ibit`, like `ibit(5)`"))
                }
            }
            "ibit" => {
                if input.peek(token::Paren) {
                    let bit_count = get_bit_count(input, "ibit", 2, 64)?;
                    Ok(BitType::Bit(bit_count, Signedness::Signed))
                } else {
                    Err(input.error("Expected parentheses after `ibit`, like `ibit(5)`"))
                }
            }
            "u8" => Ok(BitType::Bit(8, Signedness::Unsigned)),
            "u16" => Ok(BitType::Bit(16, Signedness::Unsigned)),
            "u32" => Ok(BitType::Bit(32, Signedness::Unsigned)),
            "u64" => Ok(BitType::Bit(64, Signedness::Unsigned)),
            "i8" => Ok(BitType::Bit(8, Signedness::Signed)),
            "i16" => Ok(BitType::Bit(16, Signedness::Signed)),
            "i32" => Ok(BitType::Bit(32, Signedness::Signed)),
            "i64" => Ok(BitType::Bit(64, Signedness::Signed)),
            _ => Ok(BitType::Generic),
        }
    } else {
        Err(input.error("Expected `bit`, `ubit(N)`, `ibit(N)`, `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, generic or array"))
    }
}

fn get_bit_count(input: &syn::parse::ParseBuffer<'_>, field_type: &str, min: usize, max: usize) -> Result<usize> {
    let content;
    syn::parenthesized!(content in input);
    let bits: syn::LitInt = content.parse()?;
    let bit_count = bits.base10_parse::<usize>()?;
    if bit_count < min || bit_count > max {
        return Err(syn::Error::new_spanned(
            bits,
            format!("`{}` fields must be between {} and {} bits wide", field_type, min, max),
        ));
    }
    Ok(bit_count)
}

/// Main macro function
/// This function generates the packed row structure and its associated methods.
/// It takes the parsed input and generates the necessary code.
pub(crate) fn packed_row_entrypoint(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let TraceRowInput { name, generic, fields } = parse_macro_input!(input as TraceRowInput);

    let packed_bits: usize = fields.iter().map(|f| compute_total_bits(&f.ty)).sum();
    let packed_words = packed_bits.div_ceil(64);


    let generics = if let Some(g) = &generic {
        quote! { <#g> }
    } else {
        quote! {}
    };
    let generic_fields = get_generic_fields(&fields);
    let setter_getters = get_setters_getters(&fields);

    let packed_row = quote! {
        #[derive(Debug, Default)]
        pub struct #name #generics {
            #(#generic_fields,)*
            pub packed: [u64; #packed_words],
        }

        impl #generics #name #generics {
            pub const PACKED_BITS: usize = #packed_bits;
            pub const PACKED_WORDS: usize = #packed_words;
            #(#setter_getters)*
        }
    };

    proc_macro::TokenStream::from(packed_row)
}

fn get_generic_fields(fields: &[TraceField]) -> Vec<proc_macro2::TokenStream> {
    let generic_fields_vec: Vec<_> = fields
        .iter()
        .filter_map(|f| {
            if contains_generic(&f.ty) {
                let name = &f.name;
                let ty = quote! { F };
                Some(quote! { pub #name: #ty })
            } else {
                None
            }
        })
        .collect();

    generic_fields_vec
}

fn get_setters_getters(fields: &[TraceField]) -> Vec<proc_macro2::TokenStream> {
    let mut offset = 0usize;
    let mut setter_getters = vec![];

    for f in fields.iter().filter(|f| !contains_generic(&f.ty)) {
        if is_array(&f.ty) {
            add_array_setter_getter(&f.name, &f.ty, &mut offset, &mut setter_getters);
        } else {
            add_setter_getter(&f.name, &f.ty, &mut offset, &mut setter_getters);
        }
    }

    setter_getters
}

fn add_setter_getter(
    field_name: &Ident,
    field_type: &BitType,
    offset: &mut usize,
    setter_getters: &mut Vec<proc_macro2::TokenStream>,
) {
    let bit_width = compute_total_bits(field_type);
    let rust_field_type = type_for_bitwidth(bit_width);

    // Compute where the field starts and ends in the packed array
    let start = *offset;
    let end = *offset + bit_width;
    *offset = end;

    let word_start = start / 64;
    let word_end = (end - 1) / 64;
    let bit_start = start % 64;

    let tokens = if word_start == word_end {
        emit_contained_accessor(field_name, word_start, bit_width, bit_start, rust_field_type)
    } else {
        emit_split_accessor(field_name, word_start, word_end, bit_width, bit_start, rust_field_type)
    };

    setter_getters.push(tokens);
}

fn add_array_setter_getter(
    field_name: &Ident,
    field_type: &BitType,
    offset: &mut usize,
    setter_getters: &mut Vec<proc_macro2::TokenStream>,
) {
    let (bit_width, dims) = collect_dimensions(field_type);
    let total_len: usize = dims.iter().product();
    let base_offset = *offset;
    *offset += bit_width * total_len;

    let rust_field_type = type_for_bitwidth(bit_width);
    let args = dimension_args(&dims);
    let flat = flatten_index_expr(&args, &dims);
    let (setter_name, getter_name) = setter_getter_names(field_name);

    let low_bits = quote! { 64 - (bit_offset % 64) };
    let high_bits = quote! { #bit_width - (#low_bits) };

    setter_getters.push(quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, #(#args: usize,)* value: #rust_field_type) {
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
                let low_mask: u64 = ((1u128 << low_bits) - 1) as u64;
                let high_mask: u64 = ((1u128 << #high_bits) - 1) as u64;

                self.packed[word_start] &= !(low_mask << bit_start);
                self.packed[word_start] |= ((value as u64) & low_mask) << bit_start;

                self.packed[word_start + 1] &= !high_mask;
                self.packed[word_start + 1] |= ((value as u64) >> low_bits) & high_mask;
            }
        }

        #[inline(always)]
        pub fn #getter_name(&self, #(#args: usize),*) -> #rust_field_type {
            let index = #flat;
            let bit_offset = #base_offset + index * #bit_width;
            let word_start = bit_offset / 64;
            let bit_start = bit_offset % 64;

            if bit_start + #bit_width <= 64 {
                const mask: u64 = ((1u128 << #bit_width) - 1) as u64;
                ((self.packed[word_start] >> bit_start) & mask) as #rust_field_type
            } else {
                let low_bits = 64 - bit_start;
                let high_bits = #bit_width - low_bits;
                let low_mask: u64 = ((1u128 << low_bits) - 1) as u64;
                let high_mask: u64 = ((1u128 << high_bits) - 1) as u64;

                let low = (self.packed[word_start] >> bit_start) & low_mask;
                let high = self.packed[word_start + 1] & high_mask;
                ((high << low_bits) | low) as #rust_field_type
            }
        }
    });
}

/// This function checks if the given type contains a generic type.
fn contains_generic(ty: &BitType) -> bool {
    match ty {
        BitType::Generic => true,
        BitType::Array(inner, _) => contains_generic(inner),
        _ => false,
    }
}

/// This function generates a list of identifiers for the dimensions of an array.
fn dimension_args(dims: &[usize]) -> Vec<Ident> {
    dims.iter().enumerate().map(|(i, _)| format_ident!("i{}", i)).collect()
}

/// This function generates a flattened index expression for the given identifiers and dimensions.
fn flatten_index_expr(idents: &[Ident], dims: &[usize]) -> proc_macro2::TokenStream {
    let mut expr = quote! { #(#idents)* }; // Use in actual calculation below
    let mut iter = idents.iter().zip(dims.iter()).rev();
    if let Some((last_id, _)) = iter.next() {
        expr = quote! { #last_id };
    }
    for (id, dim) in iter {
        expr = quote! { (#id * #dim) + #expr };
    }
    expr
}

fn collect_dimensions(mut ty: &BitType) -> (usize, Vec<usize>) {
    let mut dims = vec![];
    while let BitType::Array(inner, len) = ty {
        dims.push(*len);
        ty = inner;
    }
    (compute_total_bits(ty), dims)
}

fn compute_total_bits(ty: &BitType) -> usize {
    match ty {
        BitType::Bit(n, _) => *n,
        BitType::Generic => 0,
        BitType::Array(inner, len) => compute_total_bits(inner) * len,
    }
}

fn type_for_bitwidth(width: usize) -> proc_macro2::TokenStream {
    match width {
        0..=8 => quote! { u8 },
        9..=16 => quote! { u16 },
        17..=32 => quote! { u32 },
        33..=64 => quote! { u64 },
        _ => quote! { u128 },
    }
}

fn is_array(ty: &BitType) -> bool {
    matches!(ty, BitType::Array(_, _))
}

fn setter_getter_names(field_name: &Ident) -> (Ident, Ident) {
    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);
    (setter_name, getter_name)
}

fn emit_contained_accessor(
    field_name: &Ident,
    word_start: usize,
    bit_width: usize,
    bit_start: usize,
    rust_field_type: TokenStream,
) -> TokenStream {
    let mask_bits = ((1u128 << bit_width) - 1) as u64;
    let mask = mask_bits << bit_start;

    let (setter_name, getter_name) = setter_getter_names(field_name);

    quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_field_type) {
            debug_assert!((value as u128) < (1u128 << #bit_width), "Value out of range for {}", stringify!(#field_name));
            const MASK_BITS: u64 = #mask_bits;
            const MASK: u64 = #mask;
            self.packed[#word_start] &= !MASK;
            self.packed[#word_start] |= ((value as u64) & MASK_BITS) << #bit_start;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> #rust_field_type {
            const MASK_BITS: u64 = #mask_bits;
            (((self.packed[#word_start] >> #bit_start) & MASK_BITS) as #rust_field_type)
        }
    }
}

fn emit_split_accessor(
    field_name: &Ident,
    word_start: usize,
    word_end: usize,
    bit_width: usize,
    bit_start: usize,
    rust_field_type: TokenStream,
) -> TokenStream {
    let low_bits = 64 - bit_start;
    let high_bits = bit_width - low_bits;
    let low_mask = ((1u128 << low_bits) - 1) as u64;
    let high_mask = ((1u128 << high_bits) - 1) as u64;

    let (setter_name, getter_name) = setter_getter_names(field_name);

    quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_field_type) {
            debug_assert!((value as u128) < (1u128 << #bit_width), "Value out of range for {}", stringify!(#field_name));
            const LOW_MASK: u64 = #low_mask;
            const HIGH_MASK: u64 = #high_mask;
            self.packed[#word_start] &= !(LOW_MASK << #bit_start);
            self.packed[#word_start] |= ((value as u64) & LOW_MASK) << #bit_start;
            self.packed[#word_end] &= !HIGH_MASK;
            self.packed[#word_end] |= ((value as u64) >> #low_bits) & HIGH_MASK;
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> #rust_field_type {
            const LOW_MASK: u64 = #low_mask;
            const HIGH_MASK: u64 = #high_mask;
            let low = (self.packed[#word_start] >> #bit_start) & LOW_MASK;
            let high = self.packed[#word_end] & HIGH_MASK;
            ((high << #low_bits) | low) as #rust_field_type
        }
    }
}
