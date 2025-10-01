use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, format_ident, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    parse2, Field, FieldsNamed, Generics, Ident, LitInt, Result, Token,
};

#[proc_macro]
pub fn trace(input: TokenStream) -> TokenStream {
    match trace_impl(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn trace_impl(input: TokenStream2) -> Result<TokenStream2> {
    let parsed_input: ParsedTraceInput = parse2(input)?;

    let row_struct_name = parsed_input.row_struct_name;
    let trace_struct_name = parsed_input.struct_name;
    let split_struct_name = parsed_input.split_struct_name;
    let generics = parsed_input.generics.params;
    let fields = parsed_input.fields;
    let airgroup_id = parsed_input.airgroup_id;
    let air_id = parsed_input.air_id;
    let num_rows = parsed_input.num_rows;
    let commit_id = parsed_input.commit_id;

    // Calculate ROW_SIZE based on the field types
    let row_size = fields
        .named
        .iter()
        .map(|field| calculate_field_size_literal(&field.ty))
        .collect::<Result<Vec<usize>>>()?
        .into_iter()
        .sum::<usize>();

    // Generate row struct
    let field_definitions = fields.named.iter().map(|field| {
        let Field { ident, ty, .. } = field;
        quote! { pub #ident: #ty, }
    });

    fn default_expr(ty: &syn::Type) -> proc_macro2::TokenStream {
        match ty {
            syn::Type::Array(array) => {
                let len = &array.len;
                let inner = default_expr(&array.elem);
                quote! { [#inner; #len] }
            }
            _ => {
                quote! { <#ty as Default>::default() }
            }
        }
    }

    let default_field_exprs = fields.named.iter().map(|field| {
        let ident = field.ident.as_ref().unwrap();
        let init = default_expr(&field.ty);
        quote! { #ident: #init }
    });

    let row_struct = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct #row_struct_name<#generics> {
            #(#field_definitions)*
        }

        impl<#generics: Copy> #row_struct_name<#generics> {
            pub const ROW_SIZE: usize = #row_size;
        }

        impl<#generics: Default + Copy> Default for #row_struct_name<#generics> {
            fn default() -> Self {
                Self {
                    #(
                        #default_field_exprs
                    ),*
                }
            }
        }
    };

    // Generate trace struct
    let trace_struct = quote! {
        use rayon::prelude::*;

        pub struct #split_struct_name<#generics> {
            original_buffer: Vec<#generics>,
            pub chunks: Vec<std::mem::ManuallyDrop<Vec<#row_struct_name<#generics>>>>,
            leftover: Option<std::mem::ManuallyDrop<Vec<#row_struct_name<#generics>>>>,
            shared_buffer: bool,
        }

        impl<#generics> #split_struct_name<#generics> {
            pub fn leftover_size(&self) -> usize {
                self.leftover.as_ref().map_or(0, |v| v.len())
            }
        }

        pub struct #trace_struct_name<#generics> {
            pub buffer: Vec<#generics>,
            pub row_slice_mut: std::mem::ManuallyDrop<Vec<#row_struct_name<#generics>>>,
            pub num_rows: usize,
            pub row_size: usize,
            pub airgroup_id: usize,
            pub air_id: usize,
            pub commit_id: Option<usize>,
            pub shared_buffer: bool,
        }

        impl<#generics: Default + Clone + Copy + Send> #trace_struct_name<#generics> {
            pub const NUM_ROWS: usize = #num_rows;
            pub const ROW_SIZE: usize = #row_size;
            pub const AIRGROUP_ID: usize = #airgroup_id;
            pub const AIR_ID: usize = #air_id;

            pub fn new() -> Self {
                #trace_struct_name::with_capacity(Self::NUM_ROWS)
            }

            pub fn new_zeroes() -> Self {
                let num_rows = Self::NUM_ROWS;
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let mut buffer: Vec<#generics> = vec![#generics::default(); num_rows * #row_struct_name::<#generics>::ROW_SIZE];

                let ptr = buffer.as_mut_ptr() as *mut #row_struct_name<#generics>;
                #trace_struct_name {
                    buffer,
                    row_slice_mut: unsafe { std::mem::ManuallyDrop::new(Vec::from_raw_parts(ptr, num_rows, num_rows)) },
                    num_rows,
                    row_size: #row_struct_name::<#generics>::ROW_SIZE,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: false,
                }
            }

            pub fn with_capacity(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                // Allocate uninitialized memory for performance
                let mut vec: Vec<std::mem::MaybeUninit<#generics>> = Vec::with_capacity(num_rows * #row_struct_name::<#generics>::ROW_SIZE);
                let mut buffer: Vec<#generics> = unsafe {
                    vec.set_len(num_rows * #row_struct_name::<#generics>::ROW_SIZE);
                    std::mem::transmute(vec)
                };

                #[cfg(feature = "diagnostic")]
                unsafe {
                    let ptr = buffer.as_mut_ptr() as *mut u64;
                    let expected_len = num_rows * #row_struct_name::<#generics>::ROW_SIZE;
                    for i in 0..expected_len {
                        ptr.add(i).write(u64::MAX - 1);
                    }
                }

                let ptr = buffer.as_mut_ptr() as *mut #row_struct_name<#generics>;
                #trace_struct_name {
                    buffer,
                    row_slice_mut: unsafe { std::mem::ManuallyDrop::new(Vec::from_raw_parts(ptr, num_rows, num_rows)) },
                    num_rows,
                    row_size: #row_struct_name::<#generics>::ROW_SIZE,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: false,
                }
            }

            pub fn new_from_vec_zeroes(mut buffer: Vec<#generics>) -> Self {
                let row_size = #row_struct_name::<#generics>::ROW_SIZE;
                let num_rows = Self::NUM_ROWS;
                let used_len = num_rows * row_size;

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
                    row_size,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: true,
                }
            }

            pub fn new_from_vec(mut buffer: Vec<#generics>) -> Self {
                let row_size = #row_struct_name::<#generics>::ROW_SIZE;
                let num_rows = Self::NUM_ROWS;
                let expected_len = num_rows * row_size;

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

                let ptr = buffer.as_mut_ptr() as *mut #row_struct_name<#generics>;
                Self {
                    buffer,
                    row_slice_mut: unsafe { std::mem::ManuallyDrop::new(Vec::from_raw_parts(ptr, num_rows, num_rows)) },
                    num_rows: Self::NUM_ROWS,
                    row_size,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: true,
                }
            }

            pub fn from_vec(buffer: Vec<#generics>) -> Self {
                let row_size = #row_struct_name::<#generics>::ROW_SIZE;
                let num_rows = Self::NUM_ROWS;
                let expected_len = num_rows * row_size;

                assert!(buffer.len() >= expected_len, "Flat buffer too small");
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                Self {
                    buffer,
                    num_rows,
                    row_size,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: true,
                }
            }

            pub fn par_iter_mut_chunks(&mut self, n: usize) -> impl rayon::iter::IndexedParallelIterator<Item = &mut [#row_struct_name<#generics>]> {
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

            /// Splits the internal buffer into multiple disjoint owned chunks according to the specified sizes,
            /// returning a `#split_struct_name` containing these chunks and any leftover elements.
            ///
            /// # Arguments
            /// * `sizes` - A slice of sizes, where each value specifies the number of elements for the corresponding chunk.
            ///
            /// # Returns
            /// A `#split_struct_name<#generics>` containing:
            /// - `original_buffer`: The original buffer that was split.
            /// - `chunks`: A vector of `Vec<#row_struct_name<#generics>>` representing the split chunks.
            /// - `leftover`: An optional `Vec<#row_struct_name<#generics>>` containing any remaining elements after the splits.
            ///
            /// # Panics
            /// This function will panic if:
            /// - `sizes` is empty.
            /// - Any size in `sizes` is zero.
            /// - The total of `sizes` exceeds `Self::NUM_ROWS`.
            /// - The internal buffer's length does not match its capacity (i.e., buffer must be fully initialized).
            ///
            /// # Safety
            /// This function performs unsafe operations to split the buffer. The chunks are expected to be contiguous in memory.
            /// The original buffer  is not dropped, is stored in `original_buffer`, and will not be deallocated until
            /// the `#split_struct_name` is dropped.
            /// The split chunks are manually dropped to prevent double deallocation.
            pub fn to_split_struct(self, sizes: &[usize]) -> #split_struct_name<#generics> {
                assert!(!sizes.is_empty(), "Sizes cannot be empty");
                assert!(sizes.iter().all(|&size| size > 0), "All sizes must be greater than zero");

                assert!(self.buffer.len() == self.buffer.capacity(), "Buffer length must match its capacity");

                // Calculate total size and ensure it matches the buffer length
                let total_size: usize = sizes.iter().sum();

                // Ensure the total size does not exceed the capacity of the vector
                assert!(total_size <= Self::NUM_ROWS, "Total size exceeds vector capacity");

                let mut splits = if total_size < Self::NUM_ROWS {
                    Vec::with_capacity(sizes.len() + 1)
                } else {
                    Vec::with_capacity(sizes.len())
                };

                let mut offset = 0;
                let ptr = self.buffer.as_ptr() as *mut #row_struct_name<#generics>;

                // Push chunks based on sizes
                for &size in sizes {
                    unsafe {
                        let chunk_ptr = ptr.add(offset);
                        let chunk = Vec::from_raw_parts(chunk_ptr, size, size);
                        splits.push(std::mem::ManuallyDrop::new(chunk));
                    }

                    offset += size;
                }

                // Push the remaining part of the buffer if any
                let remaining = Self::NUM_ROWS - total_size;

                let leftover = if remaining > 0 {
                    unsafe {
                        let chunk_ptr = ptr.add(offset);
                        let chunk = Vec::from_raw_parts(chunk_ptr, remaining, remaining);
                        Some(std::mem::ManuallyDrop::new(chunk))
                    }
                } else {
                    None
                };

                debug_assert_eq!(offset + leftover.as_ref().map_or(0, |v| v.len()), Self::NUM_ROWS);

                #split_struct_name {
                    original_buffer: self.buffer,
                    chunks: splits,
                    leftover,
                    shared_buffer: self.shared_buffer,
                }
            }

            /// Reconstructs a trace buffer from contiguous, previously split chunks.
            ///
            /// This function takes a `#split_struct_name<#generics>` and reconstructs the original
            /// `#trace_struct_name<#generics>` instance that owns the entire original buffer.
            ///
            /// # Parameters
            /// * `split_struct` - A `#split_struct_name<#generics>` containing the split chunks and the original buffer.
            ///
            /// # Returns
            /// The original `#trace_struct_name<#generics>` instance that owns the buffer.
            pub fn from_split_struct(
                mut split_struct: #split_struct_name<#generics>,
            ) -> Self {
                let ptr = split_struct.original_buffer.as_mut_ptr() as *mut #row_struct_name<#generics>;
                #trace_struct_name {
                    buffer: unsafe { split_struct.original_buffer },
                    row_slice_mut: unsafe { std::mem::ManuallyDrop::new(Vec::from_raw_parts(ptr, Self::NUM_ROWS, Self::NUM_ROWS)) },
                    num_rows: Self::NUM_ROWS,
                    row_size: #row_struct_name::<#generics>::ROW_SIZE,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                    shared_buffer: split_struct.shared_buffer,
                }
            }
        }

        impl<#generics> #trace_struct_name<#generics> {
            #[inline]
            pub fn row_slice(&self) -> &[#row_struct_name<#generics>] {
                self.row_slice_mut.as_ref()
            }

            #[inline]
            pub fn row_slice_mut(&mut self) -> &mut [#row_struct_name<#generics>] {
                self.row_slice_mut.as_mut()
            }
        }

        impl<#generics> std::ops::Index<usize> for #trace_struct_name<#generics> {
            type Output = #row_struct_name<#generics>;

            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                &self.row_slice_mut[index]
            }
        }

        impl<#generics> std::ops::IndexMut<usize> for #trace_struct_name<#generics> {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.row_slice_mut[index]
            }
        }

        impl<#generics: Send> common::trace::Trace<#generics> for #trace_struct_name<#generics> {
            fn num_rows(&self) -> usize {
                self.num_rows
            }

            fn n_cols(&self) -> usize {
                self.row_size
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
        }
    };

    Ok(quote! {
        #row_struct
        #trace_struct
    })
}

// A struct to handle parsing the input and all the syntactic variations
struct ParsedTraceInput {
    row_struct_name: Ident,
    struct_name: Ident,
    split_struct_name: Ident,
    generics: Generics,
    fields: FieldsNamed,
    airgroup_id: LitInt,
    air_id: LitInt,
    num_rows: LitInt,
    commit_id: TokenStream2,
}

impl Parse for ParsedTraceInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        let row_struct_name;

        // Handle explicit or implicit row struct names
        if lookahead.peek(Ident) && input.peek2(Token![,]) {
            row_struct_name = Some(input.parse::<Ident>()?);
            input.parse::<Token![,]>()?; // Skip comma after explicit row name
        } else {
            row_struct_name = None;
        }

        let struct_name = input.parse::<Ident>()?;
        let row_struct_name = row_struct_name.unwrap_or_else(|| format_ident!("{}Row", struct_name));

        let split_struct_name = format_ident!("{}Split", struct_name);

        let generics: Generics = input.parse()?;
        let fields: FieldsNamed = input.parse()?;

        input.parse::<Token![,]>()?;
        let airgroup_id = input.parse::<LitInt>()?;

        input.parse::<Token![,]>()?;
        let air_id = input.parse::<LitInt>()?;

        input.parse::<Token![,]>()?;
        let num_rows = input.parse::<LitInt>()?;

        let commit_id: TokenStream2 = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let commit_id = input.parse::<LitInt>()?;
            quote!(Some(#commit_id))
        } else {
            quote!(None)
        };

        Ok(ParsedTraceInput {
            row_struct_name,
            struct_name,
            split_struct_name,
            generics,
            fields,
            airgroup_id,
            air_id,
            num_rows,
            commit_id,
        })
    }
}

// Calculate the size of a field based on its type and return it as a Result<usize>
fn calculate_field_size_literal(field_type: &syn::Type) -> Result<usize> {
    match field_type {
        // Handle arrays with multiple dimensions
        syn::Type::Array(type_array) => {
            let len =
                type_array.len.to_token_stream().to_string().parse::<usize>().map_err(|e| {
                    syn::Error::new_spanned(&type_array.len, format!("Failed to parse array length: {e}"))
                })?;
            let elem_size = calculate_field_size_literal(&type_array.elem)?;
            Ok(len * elem_size)
        }
        // For simple types, the size is 1
        _ => Ok(1),
    }
}

#[proc_macro]
pub fn values(input: TokenStream) -> TokenStream {
    match values_impl(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn values_impl(input: TokenStream2) -> Result<TokenStream2> {
    let parsed_input: ParsedValuesInput = parse2(input)?;

    let row_struct_name = parsed_input.row_struct_name;
    let values_struct_name = parsed_input.struct_name;
    let generics = parsed_input.generics.params;
    let fields = parsed_input.fields;

    // Calculate TOTAL_SIZE based on the field types
    let total_size = fields
        .named
        .iter()
        .map(|field| calculate_field_slots(&field.ty))
        .collect::<Result<Vec<usize>>>()?
        .into_iter()
        .sum::<usize>();

    // Generate row struct
    let field_definitions = fields.named.iter().map(|field| {
        let Field { ident, ty, .. } = field;
        quote! { pub #ident: #ty, }
    });

    let row_struct = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct #row_struct_name<#generics> {
            #(#field_definitions)*
        }

        impl<#generics: Copy> #row_struct_name<#generics> {
            pub const TOTAL_SIZE: usize = #total_size;
        }
    };

    let values_struct = quote! {
        pub struct #values_struct_name<'a, #generics> {
            pub buffer: Vec<#generics>,
            pub slice_values: &'a mut #row_struct_name<#generics>,
        }

        impl<'a, #generics: Default + Clone + Copy> #values_struct_name<'a, #generics> {
            pub fn new() -> Self {
                let mut buffer = vec![#generics::default(); #row_struct_name::<#generics>::TOTAL_SIZE]; // Interpolate here as well

                let slice_values = unsafe {
                    let ptr = buffer.as_mut_ptr() as *mut #row_struct_name<#generics>;
                    &mut *ptr
                };

                #values_struct_name {
                    buffer: buffer,
                    slice_values,
                }
            }

            pub fn from_vec(
                mut external_buffer: Vec<#generics>,
            ) -> Self {
                let slice_values = unsafe {
                    // Create a mutable slice from the raw pointer of external_buffer
                    let ptr = external_buffer.as_mut_ptr() as *mut #row_struct_name<#generics>;
                    &mut *ptr
                };

                // Return the struct with the owned buffers and borrowed slices
                #values_struct_name {
                    buffer: external_buffer,
                    slice_values,
                }
            }

            pub fn from_vec_guard(
                mut external_buffer_rw: std::sync::RwLockWriteGuard<Vec<#generics>>,
            ) -> Self {
                let slice_values = unsafe {
                    let ptr = external_buffer_rw.as_mut_ptr() as *mut #row_struct_name<#generics>;
                    &mut *ptr
                };

                #values_struct_name {
                    buffer: Vec::new(),
                    slice_values,
                }
            }
        }

        impl<'a, #generics> std::ops::Deref for #values_struct_name<'a, #generics> {
            type Target = #row_struct_name<#generics>;

            fn deref(&self) -> &Self::Target {
                &self.slice_values
            }
        }

        impl<'a, #generics> std::fmt::Debug for #values_struct_name<'a, #generics>
        where #generics: std::fmt::Debug {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.slice_values, f)
            }
        }

        impl<'a, #generics> std::ops::DerefMut for #values_struct_name<'a, #generics> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.slice_values
            }
        }

        impl<'a, #generics: Send> common::trace::Values<#generics> for #values_struct_name<'a, #generics> {
            fn get_buffer(&mut self) -> Vec<#generics> {
                let buffer = std::mem::take(&mut self.buffer);
                buffer
            }
        }

    };

    Ok(quote! {
        #row_struct
        #values_struct
    })
}

struct ParsedValuesInput {
    row_struct_name: Ident,
    struct_name: Ident,
    generics: Generics,
    fields: FieldsNamed,
}

impl Parse for ParsedValuesInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        let row_struct_name;

        // Handle explicit or implicit row struct names
        if lookahead.peek(Ident) && input.peek2(Token![,]) {
            row_struct_name = Some(input.parse::<Ident>()?);
            input.parse::<Token![,]>()?; // Skip comma after explicit row name
        } else {
            row_struct_name = None;
        }

        let struct_name = input.parse::<Ident>()?;
        let row_struct_name = row_struct_name.unwrap_or_else(|| format_ident!("{}Row", struct_name));

        let generics: Generics = input.parse()?;
        let fields: FieldsNamed = input.parse()?;

        Ok(ParsedValuesInput { row_struct_name, struct_name, generics, fields })
    }
}

fn calculate_field_slots(ty: &syn::Type) -> Result<usize> {
    match ty {
        // Handle `F`
        syn::Type::Path(type_path) if is_ident(type_path, "F") => Ok(1),

        // Handle `FieldExtension<F>`
        syn::Type::Path(type_path) if is_ident(type_path, "FieldExtension") => {
            // Assuming FieldExtension size is always 3 slots for this example.
            Ok(3)
        }

        // Handle `[F; N]` and `[FieldExtension<F>; N]`
        syn::Type::Array(type_array) => {
            let len =
                type_array.len.to_token_stream().to_string().parse::<usize>().map_err(|e| {
                    syn::Error::new_spanned(&type_array.len, format!("Failed to parse array length: {e}"))
                })?;
            let elem_slots = calculate_field_slots(&type_array.elem)?;
            Ok(len * elem_slots)
        }

        _ => Err(syn::Error::new_spanned(ty, "Unsupported type for slot calculation")),
    }
}

// Helper to check if a type is a specific identifier
fn is_ident(type_path: &syn::TypePath, name: &str) -> bool {
    type_path.path.segments.last().is_some_and(|seg| seg.ident == name)
}

#[test]
fn test_trace_macro_generates_default_row_struct() {
    let input = quote! {
        Simple<F> { a: F, b: F }, 0,0,0,0
    };

    let _generated = trace_impl(input).unwrap();
}

#[test]
fn test_trace_macro_with_explicit_row_struct_name() {
    let input = quote! {
        SimpleRow, Simple<F> { a: F, b: F }, 0,0,0,0
    };

    let _generated = trace_impl(input).unwrap();
}

#[test]
fn test_parsing_01() {
    let input = quote! {
        TraceRow, MyTrace<F> { a: F, b: F }, 0, 0, 34, 38
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "TraceRow");
    assert_eq!(parsed.struct_name, "MyTrace");
    assert_eq!(parsed.airgroup_id.base10_parse::<usize>().unwrap(), 0);
    assert_eq!(parsed.air_id.base10_parse::<usize>().unwrap(), 0);
    assert_eq!(parsed.num_rows.base10_parse::<usize>().unwrap(), 34);
    // assert_eq!(parsed.commit_id.to_string().parse::<usize>().unwrap(), 38);
}

#[test]
fn test_parsing_02() {
    let input = quote! {
        SimpleRow, Simple<F> { a: F }, 0, 0, 127_456, 0
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "SimpleRow");
    assert_eq!(parsed.struct_name, "Simple");
    assert_eq!(parsed.airgroup_id.base10_parse::<usize>().unwrap(), 0);
    assert_eq!(parsed.air_id.base10_parse::<usize>().unwrap(), 0);
    assert_eq!(parsed.num_rows.base10_parse::<usize>().unwrap(), 127_456);
    // assert_eq!(parsed.commit_id.to_string().parse::<usize>().unwrap(), 0);
}

#[test]
fn test_parsing_03() {
    let input = quote! {
        Simple<F> { a: F }, 0, 0, 0, 0
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "SimpleRow");
    assert_eq!(parsed.struct_name, "Simple");
}

#[test]
fn test_simple_type_size() {
    // A simple type like `F` should return size 1
    let ty: syn::Type = syn::parse_quote! { F };
    let size = calculate_field_size_literal(&ty).unwrap();
    assert_eq!(size, 1);
}

#[test]
fn test_array_type_size_single_dimension() {
    // An array like `[F; 3]` should return size 3
    let ty: syn::Type = syn::parse_quote! { [F; 3] };
    let size = calculate_field_size_literal(&ty).unwrap();
    assert_eq!(size, 3);
}

#[test]
fn test_array_type_size_multi_dimension() {
    // A multi-dimensional array like `[[F; 3]; 2]` should return size 6 (2 * 3)
    let ty: syn::Type = syn::parse_quote! { [[F; 3]; 2] };
    let size = calculate_field_size_literal(&ty).unwrap();
    assert_eq!(size, 6);
}

#[test]
fn test_nested_array_type_size() {
    // A more deeply nested array like `[[[F; 2]; 3]; 4]` should return size 24 (4 * 3 * 2)
    let ty: syn::Type = syn::parse_quote! { [[[F; 2]; 3]; 4] };
    let size = calculate_field_size_literal(&ty).unwrap();
    assert_eq!(size, 24);
}

#[test]
fn test_empty_array() {
    // An empty array should return size 0
    let ty: syn::Type = syn::parse_quote! { [F; 0] };
    let size = calculate_field_size_literal(&ty).unwrap();
    assert_eq!(size, 0);
}
