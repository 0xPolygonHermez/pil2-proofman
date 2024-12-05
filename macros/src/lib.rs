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
    let generics = parsed_input.generics.params;
    let fields = parsed_input.fields;
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

            pub fn as_slice(&self) -> &[#generics] {
                unsafe {
                    std::slice::from_raw_parts(
                        self as *const #row_struct_name<#generics> as *const #generics,
                        #row_size,
                    )
                }
            }
        }
    };

    // Generate trace struct
    let trace_struct = quote! {
        pub struct #trace_struct_name<'a, #generics> {
            pub buffer: Option<Vec<#generics>>,
            pub slice_trace: &'a mut [#row_struct_name<#generics>],
            pub num_rows: usize,
            pub commit_id: Option<usize>,
        }

        impl<'a, #generics: Default + Clone + Copy> #trace_struct_name<'a, #generics> {
            const NUM_ROWS: usize = #num_rows;

            pub fn new() -> Self {
                let num_rows = Self::NUM_ROWS;
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let buffer = vec![#generics::default(); num_rows * #row_struct_name::<#generics>::ROW_SIZE];
                let slice_trace = unsafe {
                    std::slice::from_raw_parts_mut(
                        buffer.as_ptr() as *mut #row_struct_name<#generics>,
                        num_rows,
                    )
                };

                #trace_struct_name {
                    buffer: Some(buffer),
                    slice_trace,
                    num_rows,
                    commit_id: #commit_id
                }
            }

            pub fn map_buffer(
                external_buffer: &'a mut [#generics],
                num_rows: usize,
                offset: usize,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let start = offset;
                let end = start + num_rows * #row_struct_name::<#generics>::ROW_SIZE;

                if end > external_buffer.len() {
                    return Err("Buffer is too small to fit the trace".into());
                }

                let slice_trace = unsafe {
                    std::slice::from_raw_parts_mut(
                        external_buffer[start..end].as_ptr() as *mut #row_struct_name<#generics>,
                        num_rows,
                    )
                };

                Ok(#trace_struct_name {
                    buffer: None,
                    slice_trace,
                    num_rows,
                })
            }

            pub fn from_row_vec(
                external_buffer: Vec<#row_struct_name<#generics>>,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let mut num_f = external_buffer.len() * #row_struct_name::<#generics>::ROW_SIZE;

                let slice_trace = unsafe {
                    let ptr = external_buffer.as_ptr() as *mut #row_struct_name<#generics>;
                    std::slice::from_raw_parts_mut(
                        ptr,
                        external_buffer.len(),
                    )
                };

                let buffer_f = unsafe {
                    let mut vec = Vec::from_raw_parts(
                        external_buffer.as_ptr() as *mut #generics,
                        num_f,
                        num_f,
                    );
                    vec.resize(num_f, #generics::default());
                    vec
                };

                std::mem::forget(external_buffer);

                Ok(#trace_struct_name {
                    buffer: Some(buffer_f),
                    slice_trace,
                    num_rows: num_f / #row_struct_name::<#generics>::ROW_SIZE,
                })
            }

            pub fn num_rows(&self) -> usize {
                self.num_rows
            }
        }

        impl<'a, #generics> std::ops::Index<usize> for #trace_struct_name<'a, #generics> {
            type Output = #row_struct_name<#generics>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.slice_trace[index]
            }
        }

        impl<'a, #generics> std::ops::IndexMut<usize> for #trace_struct_name<'a, #generics> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.slice_trace[index]
            }
        }

        impl<'a, #generics: Send> common::trace::Trace for #trace_struct_name<'a, #generics> {
            fn num_rows(&self) -> usize {
                self.num_rows
            }

            fn get_buffer_ptr(&mut self) -> *mut u8 {
                let buffer = self.buffer.as_mut().expect("Buffer is not available");
                buffer.as_mut_ptr() as *mut u8
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
    num_rows: LitInt,
    commit_id: Option<LitInt>,
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
        let num_rows = input.parse::<LitInt>()?;
        let commit_id: TokenStream2 = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let commit_id = input.parse::<LitInt>()?;
            quote!(Some(#commit_id))
        } else {
            quote!(None)
        };

        Ok(ParsedTraceInput { row_struct_name, struct_name, generics, fields, num_rows, commit_id })
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

    let struct_name = parsed_input.struct_name;
    let generic_param = parsed_input.generic_param;
    let dimensions = parsed_input.dimensions;
    let fields = parsed_input.fields;

    // Calculate ROW_SIZE based on the field types
    let row_size = fields
        .named
        .iter()
        .map(|field| calculate_field_size_literal(&field.ty))
        .collect::<Result<Vec<usize>>>()?
        .into_iter()
        .sum::<usize>()
        * dimensions;

    // Generate row struct
    let field_definitions = fields.named.iter().map(|field| {
        let Field { ident, ty, .. } = field;
        quote! { pub #ident: #ty, }
    });

    let row_struct = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct #struct_name<#generic_param> {
            #(#field_definitions)*
        }

        impl<#generic_param: Copy> #struct_name<#generic_param> {
            pub const ROW_SIZE: usize = #row_size;

            pub fn as_slice(&self) -> &[#generic_param] {
                unsafe {
                    std::slice::from_raw_parts(
                        self as *const #struct_name<#generic_param> as *const #generic_param,
                        #row_size,
                    )
                }
            }
        }
    };

    Ok(quote! {
        #row_struct
    })
}

struct ParsedValuesInput {
    pub struct_name: Ident,
    pub generic_param: Ident,
    pub dimensions: usize,
    pub fields: FieldsNamed,
}

impl Parse for ParsedValuesInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let struct_name: Ident = input.parse()?;
        input.parse::<Token![<]>()?;
        let generic_param: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let dimensions: LitInt = input.parse()?;
        input.parse::<Token![>]>()?;
        let fields: FieldsNamed = input.parse()?;
        Ok(ParsedValuesInput { struct_name, generic_param, dimensions: dimensions.base10_parse()?, fields })
    }
}

#[test]
fn test_parse_values_01() {
    let input = quote! {
        Values<F, 3> { a: F, b: F }
    };
    let parsed: ParsedValuesInput = parse2(input).unwrap();
    assert_eq!(parsed.struct_name, "Values");
    assert_eq!(parsed.generic_param, "F");
    assert_eq!(parsed.dimensions, 3);
}

#[test]
fn test_parse_values_02() {
    let input = quote! {
        Something<G, 2> { a: G }
    };
    let parsed: ParsedValuesInput = parse2(input).unwrap();
    assert_eq!(parsed.struct_name, "Something");
    assert_eq!(parsed.generic_param, "G");
    assert_eq!(parsed.dimensions, 2);
}

#[test]
fn test_parse_values_03() {
    let input = quote! {
        Something<G, 189_432> { a: G, b: [G; 4] }
    };
    let parsed: ParsedValuesInput = parse2(input).unwrap();
    assert_eq!(parsed.struct_name, "Something");
    assert_eq!(parsed.generic_param, "G");
    assert_eq!(parsed.dimensions, 189_432);
}

#[test]
fn test_trace_macro_generates_default_row_struct() {
    let input = quote! {
        Simple<F> { a: F, b: F }, 2, 788
    };

    let _generated = trace_impl(input).unwrap();
}

#[test]
fn test_trace_macro_with_explicit_row_struct_name() {
    let input = quote! {
        SimpleRow, Simple<F> { a: F, b: F }, 4
    };

    let _generated = trace_impl(input).unwrap();
}

#[test]
fn test_parsing_01() {
    let input = quote! {
        TraceRow, MyTrace<F> { a: F, b: F }, 34, 38
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "TraceRow");
    assert_eq!(parsed.struct_name, "MyTrace");
    assert_eq!(parsed.num_rows.base10_parse::<usize>().unwrap(), 34);
    assert_eq!(parsed.commit_id.unwrap().base10_parse::<usize>().unwrap(), 38);
}

#[test]
fn test_parsing_02() {
    let input = quote! {
        SimpleRow, Simple<F> { a: F }, 127_456
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "SimpleRow");
    assert_eq!(parsed.struct_name, "Simple");
    assert_eq!(parsed.num_rows.base10_parse::<usize>().unwrap(), 127_456);
}

#[test]
fn test_parsing_03() {
    let input = quote! {
        Simple<F> { a: F }, 2
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
