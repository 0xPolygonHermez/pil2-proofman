// trace_row.rs - Main entry point macro

use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Ident, Result, Token, braced, parse_macro_input, token};

use crate::packed_row::packed_row_impl;
use crate::trait_row::trait_impl;
use crate::unpacked_row::unpacked_row_impl;

/// This struct represents the input for the trace_row macro.
pub struct TraceRowInput {
    pub name: Ident,
    pub generic: Option<Ident>,
    pub items: Vec<RowItem>,
}

impl TraceRowInput {
    /// The fields of a row without lanes.
    pub fn fields(&self) -> Option<Vec<TraceField>> {
        self.items
            .iter()
            .map(|item| match item {
                RowItem::Field(f) => Some(f.clone()),
                RowItem::Lane(_) => None,
            })
            .collect()
    }
}

/// A field of a row, or an inlined template instance: `x: lane { .. }` or `x: [lane { .. }; n]`.
#[derive(Clone)]
pub enum RowItem {
    Field(TraceField),
    Lane(LaneItem),
}

/// One lane's fields; `count` is `None` for a single inline.
#[derive(Clone)]
pub struct LaneItem {
    pub name: Ident,
    pub count: Option<usize>,
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
        let (items, _) = parse_items(&content, generic.as_ref(), false)?;
        Ok(TraceRowInput { name, generic, items })
    }
}

/// The name a lane's field has in its row: `{lane}_{field}`.
pub fn flat_name(lane: &Ident, field: &Ident) -> Ident {
    format_ident!("{}_{}", lane, field)
}

/// A row's items and the row-level names tagged `@instr` (only with `tags`).
pub fn parse_items(
    content: ParseStream,
    generic: Option<&Ident>,
    tags: bool,
) -> Result<(Vec<RowItem>, std::collections::HashSet<String>)> {
    let mut items = vec![];
    let mut instr = std::collections::HashSet::new();
    while !content.is_empty() {
        let name: Ident = content.parse()?;
        let _colon_token: Token![:] = content.parse()?;
        if let Some((count, lane)) = parse_lane(content)? {
            let (inner, lane_instr) = parse_items(&lane, generic, tags)?;
            let fields = inner
                .into_iter()
                .map(|item| match item {
                    RowItem::Field(f) => Ok(f),
                    RowItem::Lane(l) => Err(syn::Error::new_spanned(l.name, "lanes do not nest")),
                })
                .collect::<Result<Vec<_>>>()?;
            if fields.is_empty() {
                return Err(syn::Error::new_spanned(name, "a lane has at least one field"));
            }
            instr.extend(lane_instr.iter().map(|f| format!("{name}_{f}")));
            items.push(RowItem::Lane(LaneItem { name, count, fields }));
        } else {
            let ty = parse_bit_type(content, generic)?;
            if parse_tag(content, tags)? {
                instr.insert(name.to_string());
            }
            items.push(RowItem::Field(TraceField { name, ty }));
        }
        if content.peek(Token![,]) {
            let _comma: Token![,] = content.parse()?;
        }
    }
    Ok((items, instr))
}

fn parse_tag(content: ParseStream, tags: bool) -> Result<bool> {
    if !content.peek(Token![@]) {
        return Ok(false);
    }
    let at: Token![@] = content.parse()?;
    let tag: Ident = content.parse()?;
    if !tags {
        return Err(syn::Error::new_spanned(at, "tags are for `indexed_trace_row!`"));
    }
    if tag != "instr" {
        return Err(syn::Error::new_spanned(tag, "unknown tag; expected `@instr`"));
    }
    Ok(true)
}

/// `lane { .. }` or `[lane { .. }; n]`: the lanes (`None` for one) and the lane's fields.
fn parse_lane<'a>(content: ParseStream<'a>) -> Result<Option<(Option<usize>, syn::parse::ParseBuffer<'a>)>> {
    let is_lane = |input: ParseStream| input.peek(Ident) && input.fork().parse::<Ident>().is_ok_and(|i| i == "lane");
    if is_lane(content) {
        let _: Ident = content.parse()?;
        let fields;
        braced!(fields in content);
        return Ok(Some((None, fields)));
    }
    if content.peek(token::Bracket) {
        let fork = content.fork();
        let inner;
        syn::bracketed!(inner in fork);
        if !is_lane(&inner) {
            return Ok(None);
        }
        let array;
        syn::bracketed!(array in content);
        let _: Ident = array.parse()?;
        let fields;
        braced!(fields in array);
        let _semi: Token![;] = array.parse()?;
        let n: syn::LitInt = array.parse()?;
        return match n.base10_parse::<usize>()? {
            0 => Err(syn::Error::new_spanned(n, "an array of lanes has at least 1 lane")),
            n => Ok(Some((Some(n), fields))),
        };
    }
    Ok(None)
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
            format!("`{field_type}` fields must be between {min} and {max} bits wide"),
        ));
    }
    Ok(bit_count)
}

/// Main trace_row macro function
pub fn trace_row_entrypoint(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let trace_input = parse_macro_input!(input as TraceRowInput);
    let Some(fields) = trace_input.fields() else {
        return proc_macro::TokenStream::from(crate::lanes::lane_row_impl(&trace_input));
    };

    // Generate struct names and trait name
    let packed_name = format_ident!("{}Packed", trace_input.name);
    let unpacked_name = format_ident!("{}", trace_input.name);
    let trait_name = format_ident!("{}Ops", trace_input.name);

    // Generate both packed and unpacked structs, plus the shared trait
    let packed_tokens = packed_row_impl(&packed_name, &trace_input.generic, &fields, &[], 0, true);
    let unpacked_tokens = unpacked_row_impl(&unpacked_name, &trace_input.generic, &fields);
    let trait_tokens =
        trait_impl(&trait_name, &unpacked_name, Some(&packed_name), &trace_input.generic, &fields, quote! {});

    let combined = quote! {
        #packed_tokens
        #unpacked_tokens
        #trait_tokens
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

pub fn collect_dimensions(mut ty: &BitType) -> (usize, Vec<usize>, Vec<usize>) {
    let mut dims = vec![];
    while let BitType::Array(inner, len) = ty {
        dims.push(*len);
        ty = inner;
    }
    let mut accumulated_dims = vec![1; dims.len()];
    for (i, val) in accumulated_dims.iter_mut().enumerate().take(dims.len() - 1) {
        for dim in dims.iter().skip(i + 1) {
            *val *= *dim;
        }
    }
    if let Some(last) = dims.first() {
        accumulated_dims[dims.len() - 1] = *last;
    }
    (compute_total_bits(ty), dims, accumulated_dims)
}
