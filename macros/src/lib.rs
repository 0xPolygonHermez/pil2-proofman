use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, format_ident, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    parse2, Error, Field, FieldsNamed, Generics, Ident, LitInt, Result, Token,
};
use regex::Regex;

struct ExtractedAsm {
    pub lines: Vec<String>,
    pub outputs: Vec<TokenStream2>,
    pub inputs: Vec<TokenStream2>,
    pub clobbers: Vec<TokenStream2>,
}

#[proc_macro]
pub fn import_asm(input: TokenStream) -> TokenStream {
    match import_asm_impl(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn import_asm_impl(input: TokenStream2) -> Result<TokenStream2> {
    let input = parse2::<Ident>(input)?;
    let extracted = if input == "goldilocks_base_field_scalar_add" {
        const IMPORT: &str = include_str!("../../pil2-stark/src/goldilocks/src/goldilocks_base_field_scalar.hpp");
        extract_asm_from_function(IMPORT, "Goldilocks::add").map_err(|e| Error::new(input.span(), e.to_string()))?
    } else {
        return Err(Error::new(
            input.span(),
            "Unsupported function, all asm imports must be manually hard-coded in the macro!",
        ));
    };

    let ExtractedAsm { lines, outputs, inputs, clobbers } = extracted;

    let quoted_lines: Vec<TokenStream2> = lines.iter().map(|line| quote! { #line }).collect();

    Ok(quote! {
        {
            use std::arch::asm;
            unsafe {
                asm!(
                    #(#quoted_lines),*,
                    #(#outputs,)*
                    #(#inputs,)*
                    #(#clobbers,)*
                );
            }
        }
    })
}

fn extract_asm_from_function(import: &str, function_path: &str) -> Result<ExtractedAsm> {
    let func_pattern = format!(r#"inline\s+void\s+{}\s*\([^)]*\)\s*\{{(?s)(.*?)\n\}}"#, regex::escape(function_path));
    let func_re = Regex::new(&func_pattern)
        .map_err(|e| Error::new(proc_macro2::Span::call_site(), format!("Invalid regex: {e}")))?;

    let func_body = func_re
        .captures(import)
        .and_then(|caps| caps.get(1))
        .ok_or_else(|| Error::new(proc_macro2::Span::call_site(), "Function body not found"))?
        .as_str();

    // Match multi-line __asm__ with multiple quoted strings before the colon
    let asm_block_re = Regex::new(
        r#"__asm__\s*\(\s*((?s)(?:"(?:\\.|[^"])*"\s*)+)\s*:\s*(.*?)\s*(?::\s*(.*?)\s*)?(?::\s*(.*?)\s*)?\);"#,
    )
    .map_err(|e| Error::new(proc_macro2::Span::call_site(), format!("Invalid regex: {e}")))?;

    let caps = asm_block_re
        .captures(func_body)
        .ok_or_else(|| Error::new(proc_macro2::Span::call_site(), "__asm__ block not found"))?;

    let raw_lines_block = caps.get(1).unwrap().as_str();
    let outputs = caps.get(2).map_or("", |m| m.as_str());
    let inputs = caps.get(3).map_or("", |m| m.as_str());
    let clobbers = caps.get(4).map_or("", |m| m.as_str());

    // Extract each quoted instruction line individually
    let line_re = Regex::new(r#""((?:\\.|[^"])*)""#).unwrap();
    let lines = line_re
        .captures_iter(raw_lines_block)
        .map(|cap| cap[1].trim().replace("\\t", "\t").replace("\\n", "").replace("\\\"", "\""))
        .collect::<Vec<_>>();

    // Convert C++-style operand lists into Rust token streams
    fn parse_operands(block: &str) -> Result<Vec<TokenStream2>> {
        let operand_re = Regex::new(r#"^"([^"]+)"\s*\(([^)]+)\)$"#).unwrap();

        block
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Some(caps) = operand_re.captures(s) {
                    let constraint = caps.get(1).unwrap().as_str();
                    let expr = caps.get(2).unwrap().as_str().trim().replace('.', "_"); // convert result.fe → result_fe

                    let operand = match constraint {
                        "=&a" => quote! { out("eax") #expr },
                        "r" => quote! { in(reg) #expr },
                        "m" => quote! { in("m") #expr },
                        _ => {
                            return Err(Error::new(
                                proc_macro2::Span::call_site(),
                                format!("Unsupported constraint: `{constraint}`"),
                            ));
                        }
                    };

                    Ok(operand)
                } else if s.starts_with('"') && s.ends_with('"') {
                    // Handle clobber like "%r10"
                    let clobber = s.trim_matches('"').trim_start_matches('%');
                    Ok(quote! { clobber(#clobber) })
                } else {
                    Err(Error::new(proc_macro2::Span::call_site(), format!("Malformed operand: `{s}`")))
                }
            })
            .collect()
    }

    Ok(ExtractedAsm {
        lines,
        outputs: parse_operands(outputs)?,
        inputs: parse_operands(inputs)?,
        clobbers: parse_operands(clobbers)?,
    })
}

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

    let row_struct = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct #row_struct_name<#generics> {
            #(#field_definitions)*
        }

        impl<#generics: Copy> #row_struct_name<#generics> {
            pub const ROW_SIZE: usize = #row_size;
        }
    };

    // Generate trace struct
    let trace_struct = quote! {
        use rayon::prelude::*;

        pub struct #trace_struct_name<#generics> {
            pub buffer: Vec<#row_struct_name<#generics>>,
            pub num_rows: usize,
            pub row_size: usize,
            pub airgroup_id: usize,
            pub air_id: usize,
            pub commit_id: Option<usize>,
        }

        impl<#generics: Default + Clone + Copy + Send> #trace_struct_name<#generics> {
            pub const NUM_ROWS: usize = #num_rows;
            pub const ROW_SIZE: usize = #row_size;
            pub const AIRGROUP_ID: usize = #airgroup_id;
            pub const AIR_ID: usize = #air_id;

            pub fn new() -> Self {
                #trace_struct_name::with_capacity(Self::NUM_ROWS)
            }

            pub fn with_capacity(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let buffer: Vec<#row_struct_name::<#generics>> = if cfg!(feature = "diagnostic") {
                    let mut buffer_u64 = vec![u64::MAX - 1; num_rows * #row_struct_name::<#generics>::ROW_SIZE];

                    // Convert safely by properly managing size & alignment
                    let ptr = buffer_u64.as_mut_ptr();
                    let len = buffer_u64.len() / #row_struct_name::<#generics>::ROW_SIZE;
                    let cap = buffer_u64.capacity() / #row_struct_name::<#generics>::ROW_SIZE;
                    std::mem::forget(buffer_u64);

                    unsafe { Vec::from_raw_parts(ptr as *mut #row_struct_name<#generics>, len, cap) }
                } else {
                    let mut buff_uninit: Vec<std::mem::MaybeUninit<#row_struct_name<#generics>>> = Vec::with_capacity(num_rows);
                    unsafe {
                        buff_uninit.set_len(num_rows);
                    }
                    unsafe { std::mem::transmute(buff_uninit) }
                };

                #trace_struct_name {
                    buffer,
                    num_rows,
                    row_size: #row_struct_name::<#generics>::ROW_SIZE,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                }
            }

            pub fn new_zeroes() -> Self {
                let num_rows = Self::NUM_ROWS;
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let buffer: Vec<#row_struct_name<#generics>> = vec![#row_struct_name::<#generics>::default(); num_rows];


                #trace_struct_name {
                    buffer,
                    num_rows,
                    row_size: #row_struct_name::<#generics>::ROW_SIZE,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                }
            }

            pub fn from_vec(
                mut external_buffer: Vec<#generics>,
            ) -> Self {
                let num_rows = Self::NUM_ROWS;
                let buffer: Vec<#row_struct_name::<#generics>> = unsafe { std::mem::transmute(external_buffer) };
                #trace_struct_name {
                    buffer,
                    num_rows,
                    row_size: #row_struct_name::<#generics>::ROW_SIZE,
                    airgroup_id: Self::AIRGROUP_ID,
                    air_id: Self::AIR_ID,
                    commit_id: #commit_id,
                }
            }

            pub fn from_slice(
                slice: &[#generics],
            ) -> Self {
                let num_rows = Self::NUM_ROWS;

                unsafe {
                    // Create a mutable slice from the raw pointer
                    let buffer: &mut [#row_struct_name<#generics>] = std::slice::from_raw_parts_mut(
                        slice.as_ptr() as *mut #row_struct_name<#generics>,
                        num_rows
                    );

                    // Convert the slice into a Vec without taking ownership (caller still owns the memory)
                    let buffer_vec = buffer.to_vec(); // This creates a new Vec, without modifying the original memory

                    Self {
                        buffer: buffer_vec,
                        num_rows,
                        row_size: #row_struct_name::<#generics>::ROW_SIZE,
                        airgroup_id: Self::AIRGROUP_ID,
                        air_id: Self::AIR_ID,
                        commit_id: #commit_id,
                    }
                }
            }

            /// Returns parallel mutable iterators to access the buffer.
            ///
            /// # Arguments
            /// * `n` - The number of segments to divide the buffer into. Must be a power of two and <= `NUM_ROWS`.
            ///
            /// # Panics
            /// Panics if `n` is not a power of two or if `n > NUM_ROWS`.
            pub fn par_iter_mut_chunks(&mut self, n: usize) -> impl rayon::iter::IndexedParallelIterator<Item = &mut [#row_struct_name<#generics>]> {
                assert!(n > 0 && (n & (n - 1)) == 0, "n must be a power of two");
                assert!(n <= self.num_rows, "n must be less than or equal to NUM_ROWS");
                let chunk_size = self.num_rows / n;
                assert!(chunk_size > 0, "Chunk size must be greater than zero");
                self.buffer.par_chunks_mut(chunk_size)
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
                let mut buffer = std::mem::take(&mut self.buffer);
                unsafe {
                    let ptr = buffer.as_mut_ptr();
                    let capacity = buffer.capacity() * self.row_size;
                    let len = buffer.len() * self.row_size;

                    std::mem::forget(buffer);

                    Vec::from_raw_parts(ptr.cast(), len, capacity)
                }
            }
        }

        impl<#generics> std::ops::Index<usize> for #trace_struct_name<#generics> {
            type Output = #row_struct_name<#generics>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.buffer[index]
            }
        }

        impl<#generics> std::ops::IndexMut<usize> for #trace_struct_name<#generics> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.buffer[index]
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
                let mut buffer = std::mem::take(&mut self.buffer);
                unsafe {
                    let ptr = buffer.as_mut_ptr();
                    let capacity = buffer.capacity() * self.row_size;
                    let len = buffer.len() * self.row_size;

                    std::mem::forget(buffer);

                    Vec::from_raw_parts(ptr.cast(), len, capacity)
                }
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
            let len = type_array.len.to_token_stream().to_string().parse::<usize>().map_err(|e| {
                syn::Error::new_spanned(&type_array.len, format!("Failed to parse array length: {}", e))
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
            let len = type_array.len.to_token_stream().to_string().parse::<usize>().map_err(|e| {
                syn::Error::new_spanned(&type_array.len, format!("Failed to parse array length: {}", e))
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

#[test]
fn test_import_asm_impl() {
    let input = quote!(goldilocks_base_field_scalar_add);
    let output = import_asm_impl(input.into()).unwrap();
    println!("{}", output.to_string());
    panic!("done");
}
