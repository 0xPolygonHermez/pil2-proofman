// trace_macro/src/lib.rs
use proc_macro2::TokenStream as TokenStream2;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{braced, token, Token};
use syn::{parse2, Generics, Ident, Result, LitInt};

/// This macro generates a packed row structure representing a packed trace row.
/// It takes a name and a list of fields with their types, and generates
/// a structure that packs these fields into a fixed-size array of u64.
///
/// # Example
/// ```rust
/// use trace_macro::packed_row;
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
struct ParsedPackedTraceInput {
    pub row_struct_name: Ident,
    pub struct_name: Ident,
    pub generics: Generics,
    pub fields: Vec<TraceField>,
    pub airgroup_id: LitInt,
    pub air_id: LitInt,
    pub num_rows: LitInt,
    pub commit_id: TokenStream2,
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
#[derive(Clone, Debug)]
enum Signedness {
    Signed,
    Unsigned,
}

/// This enum represents the type of a field in the packed row.
/// It can be a bit type, a generic type, or an array of BitType.
#[allow(dead_code)]
#[derive(Clone, Debug)]
enum BitType {
    Bit(usize, Signedness),
    Generic,
    Array(Box<BitType>, usize),
}

fn calculate_field_size_bittype(field_type: &BitType) -> usize {
    match field_type {
        BitType::Bit(_, _) => 1, // Each primitive bit type counts as 1 unit
        BitType::Generic => 1,   // Generic type defaults to size 1
        BitType::Array(inner, len) => calculate_field_size_bittype(inner) * len,
    }
}

/// DSL parsing
impl Parse for ParsedPackedTraceInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let struct_name = input.parse::<Ident>()?;
        let row_struct_name = format_ident!("{}Row", struct_name);

        let generics: Generics = input.parse()?;

        let content;
        let _brace_token = braced!(content in input);

        let mut fields = Vec::new();
        while !content.is_empty() {
            let name: Ident = content.parse()?;
            let _: Token![:] = content.parse()?;
            let ty: BitType = parse_bit_type(&content, &generics)?;
            fields.push(TraceField { name, ty });

            // consume optional comma
            if content.peek(Token![,]) {
                let _: Token![,] = content.parse()?;
            } else {
                break;
            }
        }

        input.parse::<Token![,]>()?;
        let airgroup_id: LitInt = input.parse()?;

        input.parse::<Token![,]>()?;
        let air_id: LitInt = input.parse()?;

        input.parse::<Token![,]>()?;
        let num_rows: LitInt = input.parse()?;

        // commit_id is optional
        let commit_id: TokenStream2 = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let commit_id = input.parse::<LitInt>()?;
            quote!(Some(#commit_id))
        } else {
            quote!(None)
        };

        Ok(ParsedPackedTraceInput {
            row_struct_name,
            struct_name,
            generics,
            fields,
            airgroup_id,
            air_id,
            num_rows,
            commit_id,
        })
    }
}

fn parse_bit_type(input: ParseStream, generics: &Generics) -> Result<BitType> {
    if input.peek(token::Bracket) {
        let content;
        let _ = syn::bracketed!(content in input);
        let elem_ty = parse_bit_type(&content, generics)?;
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

        if generics.params.iter().any(|param| {
            if let syn::GenericParam::Type(type_param) = param {
                type_param.ident == ident
            } else {
                false
            }
        }) {
            return Ok(BitType::Generic);
        }

        let result = match ident_str.as_str() {
            "bit" => Ok(BitType::Bit(1, Signedness::Unsigned)),
            "ubit" => {
                if input.peek(token::Paren) {
                    let bit_count = get_bit_count(input, "ubit", 1, 64)?;
                    Ok(BitType::Bit(bit_count, Signedness::Unsigned))
                } else {
                    return Err(input.error("Expected parentheses after `ubit`, like `ubit(5)`"));
                }
            }
            "ibit" => {
                if input.peek(token::Paren) {
                    let bit_count = get_bit_count(input, "ibit", 2, 64)?;
                    Ok(BitType::Bit(bit_count, Signedness::Signed))
                } else {
                    return Err(input.error("Expected parentheses after `ibit`, like `ibit(5)`"));
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
        };
        result
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
pub(crate) fn packed_trace_entrypoint(input: TokenStream) -> TokenStream {
    match packed_trace_impl(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn packed_trace_impl(input: TokenStream2) -> Result<TokenStream2> {
    let parsed_input: ParsedPackedTraceInput = parse2(input)?;

    let packed_row_struct_name = parsed_input.row_struct_name;
    let trace_struct_name = parsed_input.struct_name;
    let generics = parsed_input.generics.params;
    let fields = parsed_input.fields;
    let airgroup_id = parsed_input.airgroup_id;
    let air_id = parsed_input.air_id;
    let num_rows = parsed_input.num_rows;
    let commit_id = parsed_input.commit_id;

    let total_row_size: usize = fields.iter().map(|field| calculate_field_size_bittype(&field.ty)).sum();

    let packed_bits_info: Vec<usize> =
        fields.iter().flat_map(|field| compute_total_bits_vec(&field.ty).into_iter()).collect();

    let packed_bits: usize = packed_bits_info.iter().sum();
    let packed_words = packed_bits.div_ceil(64);

    let generic_fields = get_generic_fields(&fields);
    let setter_getters = get_setters_getters(&fields);
    let packed_row_struct = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct #packed_row_struct_name<#generics> {
            #(#generic_fields,)*
            pub packed: [u64; #packed_words],
            _phantom: std::marker::PhantomData<#generics>,
        }

        impl<#generics: Copy> #packed_row_struct_name<#generics> {
            pub const TOTAL_ROW_SIZE: usize = #total_row_size;
            pub const PACKED_BITS_INFO: [usize; #total_row_size] = [#(#packed_bits_info),*];
            pub const PACKED_BITS: usize = #packed_bits;
            pub const PACKED_WORDS: usize = #packed_words;
            #(#setter_getters)*
        }

        impl<#generics: Default + Copy> Default for #packed_row_struct_name<#generics> {
            fn default() -> Self {
                Self {
                    packed: [0u64; #packed_words],
                    _phantom: std::marker::PhantomData,
                }
            }
        }
    };

    // Generate trace struct
    let trace_struct = quote! {
        use rayon::prelude::*;

        pub struct #trace_struct_name<#generics> {
            pub buffer: Vec<#generics>,
            pub num_rows: usize,
            pub total_row_size: usize,
            pub packed_bits_info: [usize; #total_row_size],
            pub airgroup_id: usize,
            pub air_id: usize,
            pub commit_id: Option<usize>,
            pub shared_buffer: bool,
            pub num_packed_words: usize,
            pub unpack_info: Vec<u64>,
        }

        impl<#generics: Default + Clone + Copy + Send> #trace_struct_name<#generics> {
            pub const NUM_ROWS: usize = #num_rows;
            pub const TOTAL_ROW_SIZE: usize = #total_row_size;
            pub const PACKED_BITS_INFO: [usize; #total_row_size] = [#(#packed_bits_info),*];
            pub const AIRGROUP_ID: usize = #airgroup_id;
            pub const AIR_ID: usize = #air_id;
            pub const PACKED_WORDS: usize = #packed_words;

            pub fn new() -> Self {
                #trace_struct_name::with_capacity(Self::NUM_ROWS)
            }

            pub fn new_zeroes() -> Self {
                let num_rows = Self::NUM_ROWS;
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let buffer: Vec<#generics> = vec![#generics::default(); num_rows * #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE];

                #trace_struct_name {
                    buffer,
                    num_rows,
                    total_row_size: #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE,
                    packed_bits_info: #packed_row_struct_name::<#generics>::PACKED_BITS_INFO,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: false,
                    num_packed_words: Self::PACKED_WORDS,
                    unpack_info: Self::PACKED_BITS_INFO.iter().map(|&s| s as u64).collect(),
                }
            }

            pub fn with_capacity(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                // Allocate uninitialized memory for performance
                let mut vec: Vec<std::mem::MaybeUninit<#generics>> = Vec::with_capacity(num_rows * #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE);
                let mut buffer: Vec<#generics> = unsafe {
                    vec.set_len(num_rows * #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE);
                    std::mem::transmute(vec)
                };

                #[cfg(feature = "diagnostic")]
                unsafe {
                    let ptr = buffer.as_mut_ptr() as *mut u64;
                    let expected_len = num_rows * #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE;
                    for i in 0..expected_len {
                        ptr.add(i).write(u64::MAX - 1);
                    }
                }

                #trace_struct_name {
                    buffer,
                    num_rows,
                    total_row_size: #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE,
                    packed_bits_info: #packed_row_struct_name::<#generics>::PACKED_BITS_INFO,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: false,
                    num_packed_words: Self::PACKED_WORDS,
                    unpack_info: Self::PACKED_BITS_INFO.iter().map(|&s| s as u64).collect(),
                }
            }

            pub fn new_from_vec_zeroes(mut buffer: Vec<#generics>) -> Self {
                let total_row_size = #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE;
                let num_rows = Self::NUM_ROWS;
                let used_len = num_rows * total_row_size;

                assert!(
                    buffer.len() >= used_len,
                    "Provided buffer too small: got {}, expected at least {}",
                    buffer.len(),
                    used_len
                );

                buffer[..used_len]
                .par_iter_mut()
                .for_each(|x| {
                    *x = <#generics>::default();
                });

                Self {
                    buffer,
                    num_rows,
                    total_row_size: #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE,
                    packed_bits_info: #packed_row_struct_name::<#generics>::PACKED_BITS_INFO,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: true,
                    num_packed_words: Self::PACKED_WORDS,
                    unpack_info: Self::PACKED_BITS_INFO.iter().map(|&s| s as u64).collect(),
                }
            }

            pub fn new_from_vec(mut buffer: Vec<#generics>) -> Self {
                let total_row_size = #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE;
                let num_rows = Self::NUM_ROWS;
                let expected_len = num_rows * total_row_size;

                assert!(buffer.len() >= expected_len, "Flat buffer too small");
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                if cfg!(feature = "diagnostic") {
                    unsafe {
                        let ptr = buffer.as_mut_ptr() as *mut u64;
                        for i in 0..expected_len {
                            ptr.add(i).write(u64::MAX - 1);
                        }
                    }
                }

                Self {
                    buffer,
                    num_rows,
                    total_row_size: #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE,
                    packed_bits_info: #packed_row_struct_name::<#generics>::PACKED_BITS_INFO,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: true,
                    num_packed_words: Self::PACKED_WORDS,
                    unpack_info: Self::PACKED_BITS_INFO.iter().map(|&s| s as u64).collect(),
                }
            }

            pub fn from_vec(buffer: Vec<#generics>) -> Self {
                let total_row_size = #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE;
                let num_rows = Self::NUM_ROWS;
                let expected_len = num_rows * total_row_size;

                assert!(buffer.len() >= expected_len, "Flat buffer too small");
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                Self {
                    buffer,
                    num_rows,
                    total_row_size: #packed_row_struct_name::<#generics>::TOTAL_ROW_SIZE,
                    packed_bits_info: #packed_row_struct_name::<#generics>::PACKED_BITS_INFO,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: true,
                    num_packed_words: Self::PACKED_WORDS,
                    unpack_info: Self::PACKED_BITS_INFO.iter().map(|&s| s as u64).collect(),
                }
            }

            pub fn par_iter_mut_chunks(&mut self, n: usize) -> impl rayon::iter::IndexedParallelIterator<Item = &mut [#packed_row_struct_name<#generics>]> {
                assert!(n > 0 && (n & (n - 1)) == 0, "n must be a power of two");
                assert!(n <= self.num_rows, "n must be less than or equal to NUM_ROWS");
                let chunk_size = self.num_rows / n;
                assert!(chunk_size > 0, "Chunk size must be greater than zero");
                self.row_slice_mut().par_chunks_mut(chunk_size)
            }

            pub fn num_rows(&self) -> usize {
                self.num_rows
            }

            pub fn airgroup_id(&self) -> usize {
                self.airgroup_id
            }

            pub fn air_id(&self) -> usize {
                self.air_id
            }

            pub fn get_buffer(&mut self) -> Vec<#generics> {
                std::mem::take(&mut self.buffer)
            }

            pub fn is_shared_buffer(&self) -> bool {
                self.shared_buffer
            }

            pub fn is_packed(&self) -> bool {
                true
            }

            pub fn num_packed_words(&self) -> u64 {
                self.num_packed_words as u64
            }

            pub fn unpack_info(&self) -> Vec<u64> {
                self.unpack_info.clone()
            }
        }

        impl<#generics> #trace_struct_name<#generics> {
            pub fn row_slice(&self) -> &[#packed_row_struct_name<#generics>] {
                unsafe {
                    std::slice::from_raw_parts(
                        self.buffer.as_ptr() as *const #packed_row_struct_name<#generics>,
                        self.num_rows,
                    )
                }
            }

            pub fn row_slice_mut(&mut self) -> &mut [#packed_row_struct_name<#generics>] {
                unsafe {
                    std::slice::from_raw_parts_mut(
                        self.buffer.as_mut_ptr() as *mut #packed_row_struct_name<#generics>,
                        self.num_rows,
                    )
                }
            }
        }

        impl<#generics> std::ops::Index<usize> for #trace_struct_name<#generics> {
            type Output = #packed_row_struct_name<#generics>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.row_slice()[index]
            }
        }

        impl<#generics> std::ops::IndexMut<usize> for #trace_struct_name<#generics> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.row_slice_mut()[index]
            }
        }

        impl<#generics: Send> common::trace::Trace<#generics> for #trace_struct_name<#generics> {
            fn num_rows(&self) -> usize {
                self.num_rows
            }

            fn n_cols(&self) -> usize {
                self.total_row_size
            }

            fn airgroup_id(&self) -> usize {
                self.airgroup_id
            }

            fn air_id(&self) -> usize {
                self.air_id
            }

            fn commit_id(&self) -> Option<usize> {
                self.commit_id
            }

            fn get_buffer(&mut self) -> Vec<#generics> {
                std::mem::take(&mut self.buffer)
            }

            fn is_shared_buffer(&self) -> bool {
                self.shared_buffer
            }

            fn is_packed(&self) -> bool {
                true
            }

            fn num_packed_words(&self) -> u64 {
                self.num_packed_words as u64
            }

            fn unpack_info(&self) -> Vec<u64> {
                self.unpack_info.clone()
            }
        }
    };

    let result = quote! {
        #packed_row_struct
        #trace_struct
    };

    Ok(result)
}

fn get_generic_fields(fields: &[TraceField]) -> Vec<TokenStream2> {
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

fn get_setters_getters(fields: &[TraceField]) -> Vec<TokenStream2> {
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
    setter_getters: &mut Vec<TokenStream2>,
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
    setter_getters: &mut Vec<TokenStream2>,
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
fn flatten_index_expr(idents: &[Ident], dims: &[usize]) -> TokenStream2 {
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

fn compute_total_bits_vec(ty: &BitType) -> Vec<usize> {
    match ty {
        BitType::Bit(n, _) => vec![*n], // single bit type → one element
        BitType::Generic => vec![0],    // generic → zero or placeholder
        BitType::Array(inner, len) => {
            let inner_bits = compute_total_bits_vec(inner);
            let mut result = Vec::new();
            for _ in 0..*len {
                result.extend(inner_bits.iter());
            }
            result
        }
    }
}

fn type_for_bitwidth(width: usize) -> TokenStream2 {
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
    rust_field_type: TokenStream2,
) -> TokenStream2 {
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
    rust_field_type: TokenStream2,
) -> TokenStream2 {
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
