// trace_row.rs - Main entry point macro

use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Result, Token, braced, parse_macro_input, token};

use crate::packed_row::packed_row_impl;
use crate::unpacked_row::unpacked_row_impl;

/// This struct represents the input for the trace_row macro.
pub struct TraceRowInput {
    pub name: Ident,
    pub generic: Option<Ident>,
    pub fields: Vec<TraceField>,
}

/// This struct represents a field in the trace row.
#[derive(Clone)]
pub struct TraceField {
    pub name: Ident,
    pub ty: BitType,
}

/// This enum represents the type of a field in the trace row.
#[derive(Clone)]
pub enum BitType {
    Bit(usize),
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

pub fn parse_bit_type(input: ParseStream, generic: Option<&Ident>) -> Result<BitType> {
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
            "bit" => Ok(BitType::Bit(1)),
            "ubit" => {
                if input.peek(token::Paren) {
                    let bit_count = get_bit_count(input, "ubit", 1, 64)?;
                    Ok(BitType::Bit(bit_count))
                } else {
                    Err(input.error("Expected parentheses after `ubit`, like `ubit(5)`"))
                }
            }
            "ibit" => {
                if input.peek(token::Paren) {
                    let bit_count = get_bit_count(input, "ibit", 2, 64)?;
                    Ok(BitType::Bit(bit_count))
                } else {
                    Err(input.error("Expected parentheses after `ibit`, like `ibit(5)`"))
                }
            }
            "u8" => Ok(BitType::Bit(8)),
            "u16" => Ok(BitType::Bit(16)),
            "u32" => Ok(BitType::Bit(32)),
            "u64" => Ok(BitType::Bit(64)),
            "i8" => Ok(BitType::Bit(8)),
            "i16" => Ok(BitType::Bit(16)),
            "i32" => Ok(BitType::Bit(32)),
            "i64" => Ok(BitType::Bit(64)),
            _ => Ok(BitType::Generic),
        }
    } else {
        Err(input.error("Expected `bit`, `ubit(N)`, `ibit(N)`, `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, generic or array"))
    }
}

pub fn get_bit_count(input: &syn::parse::ParseBuffer<'_>, field_type: &str, min: usize, max: usize) -> Result<usize> {
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

/// Main trace_row macro function
pub fn trace_row_entrypoint(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let trace_input = parse_macro_input!(input as TraceRowInput);

    // Generate packed struct name
    let packed_name = format_ident!("{}Packed", trace_input.name);
    let unpacked_name = format_ident!("{}", trace_input.name);

    // Generate both packed and unpacked versions
    let packed_tokens = packed_row_impl(&packed_name, &trace_input.generic, &trace_input.fields);
    let unpacked_tokens = unpacked_row_impl(&unpacked_name, &trace_input.generic, &trace_input.fields);

    let combined = quote! {
        #packed_tokens
        #unpacked_tokens
    };

    proc_macro::TokenStream::from(combined)
}

/// Utility functions shared by both implementations
pub fn contains_generic(ty: &BitType) -> bool {
    match ty {
        BitType::Generic => true,
        BitType::Array(inner, _) => contains_generic(inner),
        _ => false,
    }
}

pub fn compute_total_bits(ty: &BitType) -> usize {
    match ty {
        BitType::Bit(n) => *n,
        BitType::Generic => 0,
        BitType::Array(inner, len) => compute_total_bits(inner) * len,
    }
}

pub fn is_array(ty: &BitType) -> bool {
    matches!(ty, BitType::Array(_, _))
}

pub fn collect_dimensions(mut ty: &BitType) -> (usize, Vec<usize>) {
    let mut dims = vec![];
    while let BitType::Array(inner, len) = ty {
        dims.push(*len);
        ty = inner;
    }
    let mut accumulated_dims = vec![1; dims.len()];
    for i in 0..dims.len() - 1 {
        for j in i + 1..dims.len() {
            accumulated_dims[i] *= dims[j];
        }
    }
    accumulated_dims[dims.len() - 1] = dims[0];
    (compute_total_bits(ty), accumulated_dims)
}
