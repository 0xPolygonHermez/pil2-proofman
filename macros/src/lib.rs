use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, format_ident, ToTokens};
use syn::{
    parse2,
    parse::{Parse, ParseStream},
    Ident, Generics, FieldsNamed, Result, Field, Token,
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
        pub struct #trace_struct_name<#generics> {
            pub buffer: Vec<#row_struct_name<#generics>>,
            num_rows: usize,
        }

        impl<#generics: Default + Clone + Copy> #trace_struct_name<#generics> {
            pub fn new(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let mut buff_uninit: Vec<std::mem::MaybeUninit<#row_struct_name<#generics>>> = Vec::with_capacity(num_rows);
                unsafe {
                    buff_uninit.set_len(num_rows);
                }
                let buffer: Vec<#row_struct_name<#generics>> = unsafe { std::mem::transmute(buff_uninit) };

                #trace_struct_name { buffer, num_rows}
            }

            pub fn new_zeroes(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);

                let buffer = vec![#row_struct_name::<#generics>::default(); num_rows];

                #trace_struct_name { buffer, num_rows}
            }

            pub fn num_rows(&self) -> usize {
                self.num_rows
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

        impl<#generics: Send> common::trace::Trace for #trace_struct_name<#generics> {
            fn num_rows(&self) -> usize {
                self.num_rows
            }

            fn get_buffer_ptr(&mut self) -> *mut u8 {
                self.buffer.as_mut_ptr() as *mut u8
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

        Ok(ParsedTraceInput { row_struct_name, struct_name, generics, fields })
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

#[test]
fn test_trace_macro_generates_default_row_struct() {
    let input = quote! {
        Simple<F> { a: F, b: F }
    };

    let _expected = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct SimpleRow<F> {
            pub a: F,
            pub b: F,
        }
        impl<F: Copy> SimpleRow<F> {
            pub const ROW_SIZE: usize = 2usize;
        }
        pub struct Simple<'a, F> {
            pub buffer: Vec<F>,
            pub slice_trace: &'a mut [SimpleRow<F>],
            num_rows: usize,
        }
        impl<'a, F: Default + Clone + Copy> Simple<'a, F> {
            pub fn new(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);
                let buffer = vec![F::default(); num_rows * SimpleRow::<F>::ROW_SIZE];
                let slice_trace = unsafe {
                    std::slice::from_raw_parts_mut(
                        buffer.as_ptr() as *mut SimpleRow<F>,
                        num_rows,
                    )
                };
                Simple {
                    buffer: Some(buffer),
                    slice_trace,
                    num_rows,
                }
            }
            pub fn map_buffer(
                external_buffer: &'a mut [F],
                num_rows: usize,
                offset: usize,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);
                let start = offset;
                let end = start + num_rows * SimpleRow::<F>::ROW_SIZE;
                if end > external_buffer.len() {
                    return Err("Buffer is too small to fit the trace".into());
                }
                let slice_trace = unsafe {
                    std::slice::from_raw_parts_mut(
                        external_buffer[start..end].as_ptr() as *mut SimpleRow<F>,
                        num_rows,
                    )
                };
                Ok(Simple {
                    buffer: None,
                    slice_trace,
                    num_rows,
                })
            }
            pub fn from_row_vec(
                external_buffer: Vec<SimpleRow<F>>,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let num_rows = external_buffer.len().next_power_of_two();
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);
                let slice_trace = unsafe {
                    let ptr = external_buffer.as_ptr() as *mut SimpleRow<F>;
                    std::slice::from_raw_parts_mut(
                        ptr,
                        num_rows,
                    )
                };
                let buffer_f = unsafe {
                    Vec::from_raw_parts(
                        external_buffer.as_ptr() as *mut F,
                        num_rows * SimpleRow::<F>::ROW_SIZE,
                        num_rows * SimpleRow::<F>::ROW_SIZE,
                    )
                };
                std::mem::forget(external_buffer);
                Ok(Simple {
                    buffer: Some(buffer_f),
                    slice_trace,
                    num_rows,
                })
            }
            pub fn num_rows(&self) -> usize {
                self.num_rows
            }
        }
        impl<'a, F> std::ops::Index<usize> for Simple<'a, F> {
            type Output = SimpleRow<F>;
            fn index(&self, index: usize) -> &Self::Output {
                &self.slice_trace[index]
            }
        }
        impl<'a, F> std::ops::IndexMut<usize> for Simple<'a, F> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.slice_trace[index]
            }
        }
        impl<'a, F: Send> common::trace::Trace for Simple<'a, F> {
            fn num_rows(&self) -> usize {
                self.num_rows
            }
            fn get_buffer_ptr(&mut self) -> *mut u8 {
                let buffer = self.buffer.as_mut().expect("Buffer is not available");
                buffer.as_mut_ptr() as *mut u8
            }
        }


    };
    let _generated = trace_impl(input).unwrap();
    // assert_eq!(generated.to_string(), expected.into_token_stream().to_string());
}

#[test]
fn test_trace_macro_with_explicit_row_struct_name() {
    let input = quote! {
        SimpleRow, Simple<F> { a: F, b: F }
    };

    let _expected = quote! {
        #[repr(C)]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct SimpleRow<F> {
            pub a: F,
            pub b: F,
        }

        impl<F: Copy> SimpleRow<F> {
            pub const ROW_SIZE: usize = 2usize;
        }

        pub struct Simple<'a, F> {
            pub buffer: Option<Vec<F>>,
            pub slice_trace: &'a mut [SimpleRow<F>],
            num_rows: usize,
        }

        impl<'a, F: Default + Clone + Copy> Simple<'a, F> {
            pub fn new(num_rows: usize) -> Self {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);
                let buffer = vec![F::default(); num_rows * SimpleRow::<F>::ROW_SIZE];
                let slice_trace = unsafe {
                    std::slice::from_raw_parts_mut(
                        buffer.as_ptr() as *mut SimpleRow<F>,
                        num_rows,
                    )
                };
                Simple {
                    buffer: Some(buffer),
                    slice_trace,
                    num_rows,
                }
            }

            pub fn map_buffer(
                external_buffer: &'a mut [F],
                num_rows: usize,
                offset: usize,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);
                let start = offset;
                let end = start + num_rows * SimpleRow::<F>::ROW_SIZE;
                if end > external_buffer.len() {
                    return Err("Buffer is too small to fit the trace".into());
                }
                let slice_trace = unsafe {
                    std::slice::from_raw_parts_mut(
                        external_buffer[start..end].as_ptr() as *mut SimpleRow<F>,
                        num_rows,
                    )
                };
                Ok(Simple {
                    buffer: None,
                    slice_trace,
                    num_rows,
                })
            }

            pub fn from_row_vec(
                external_buffer: Vec<SimpleRow<F>>,
            ) -> Result<Self, Box<dyn std::error::Error>> {
                let num_rows = external_buffer.len().next_power_of_two();
                assert!(num_rows >= 2);
                assert!(num_rows & (num_rows - 1) == 0);
                let slice_trace = unsafe {
                    let ptr = external_buffer.as_ptr() as *mut SimpleRow<F>;
                    std::slice::from_raw_parts_mut(
                        ptr,
                        num_rows,
                    )
                };
                let buffer_f = unsafe {
                    Vec::from_raw_parts(
                        external_buffer.as_ptr() as *mut F,
                        num_rows * SimpleRow::<F>::ROW_SIZE, num_rows * SimpleRow::<F>::ROW_SIZE,
                    )
                };
                std::mem::forget(external_buffer);
                Ok(Simple {
                    buffer: Some(buffer_f),
                    slice_trace, num_rows,
                })
            }

            pub fn num_rows(&self) -> usize {
                self.num_rows
            }
        }

        impl<'a, F> std::ops::Index<usize> for Simple<'a, F> {
            type Output = SimpleRow<F>;

            fn index(&self, index: usize) -> &Self::Output {
                &self.slice_trace[index]
            }
        }

        impl<'a, F> std::ops::IndexMut<usize> for Simple<'a, F> {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.slice_trace[index]
            }
        }

        impl<'a, F: Send> common::trace::Trace for Simple<'a, F> {
            fn num_rows(&self) -> usize {
                self.num_rows
            }

            fn get_buffer_ptr(&mut self) -> *mut u8 {
                let buffer = self.buffer.as_mut().expect("Buffer is not available");
                buffer.as_mut_ptr() as *mut u8
            }
        }
    };

    let _generated = trace_impl(input).unwrap();
    // assert_eq!(generated.to_string(), expected.into_token_stream().to_string());
}

#[test]
fn test_parsing_01() {
    let input = quote! {
        TraceRow, MyTrace<F> { a: F, b: F }
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "TraceRow");
    assert_eq!(parsed.struct_name, "MyTrace");
}

#[test]
fn test_parsing_02() {
    let input = quote! {
        SimpleRow, Simple<F> { a: F }
    };
    let parsed: ParsedTraceInput = parse2(input).unwrap();
    assert_eq!(parsed.row_struct_name, "SimpleRow");
    assert_eq!(parsed.struct_name, "Simple");
}

#[test]
fn test_parsing_03() {
    let input = quote! {
        Simple<F> { a: F }
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