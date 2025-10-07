// unpacked_row.rs - Unpacked row implementation

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::trace_row::{TraceField, BitType, contains_generic, compute_total_bits, is_array, collect_dimensions};

pub fn unpacked_row_impl(name: &Ident, generic: &Option<Ident>, fields: &[TraceField]) -> TokenStream {
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

    let unpacked_fields = get_unpacked_fields(fields);
    let setter_getters = get_unpacked_setters_getters(fields);

    // Calculate the total number of F elements in the row
    let row_size = calculate_row_size(fields);

    let default_field_exprs = get_default_field_exprs(fields);

    quote! {
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct #name #generics_with_bounds {
            #(#unpacked_fields,)*
        }

        impl #generics_with_bounds Default for #name #generics {
            fn default() -> Self {
                Self {
                    #(#default_field_exprs,)*
                }
            }
        }

        impl #generics_with_bounds #name #generics {
            #(#setter_getters)*
        }

        impl #generics_with_bounds proofman_common::trace::TraceRow for #name #generics {
            const ROW_SIZE: usize = #row_size; // Total number of F elements
            const IS_PACKED: bool = false;
            const PACKED_WORDS: usize = 0;
            const UNPACK_INFO: &'static [usize] = &[];
        }
    }
}

fn get_unpacked_fields(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut unpacked_fields = vec![];

    for f in fields.iter() {
        let name = &f.name;
        if contains_generic(&f.ty) {
            // Expand generic fields: arrays become arrays of F, not just F
            let field_type = generate_f_field_type(&f.ty);
            unpacked_fields.push(quote! { pub #name: #field_type });
        } else {
            // Non-generic fields become F with the appropriate array structure
            let field_type = generate_f_field_type(&f.ty);
            unpacked_fields.push(quote! { pub #name: #field_type });
        }
    }

    unpacked_fields
}

fn get_unpacked_setters_getters(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut setter_getters = vec![];

    for f in fields.iter() {
        if contains_generic(&f.ty) {
            // For generic fields, only generate setters/getters for non-array fields
            // Array fields can be accessed directly
            if !is_array(&f.ty) {
                add_unpacked_generic_setter_getter(&f.name, &mut setter_getters);
            }
        } else {
            // For non-generic fields, generate F field accessors with conversion
            if is_array(&f.ty) {
                add_unpacked_array_setter_getter(&f.name, &f.ty, &mut setter_getters);
            } else {
                add_unpacked_setter_getter(&f.name, &f.ty, &mut setter_getters);
            }
        }
    }

    setter_getters
}

fn add_unpacked_generic_setter_getter(field_name: &Ident, setter_getters: &mut Vec<TokenStream>) {
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

fn add_unpacked_setter_getter(field_name: &Ident, field_type: &BitType, setter_getters: &mut Vec<TokenStream>) {
    let bit_width = compute_total_bits(field_type);
    let rust_type = type_for_bitwidth(bit_width);
    let from_method = method_name_for_bitwidth(bit_width);

    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);

    let conversion = if bit_width == 1 {
        quote! { self.#field_name.as_canonical_u64() != 0 }
    } else {
        quote! { self.#field_name.as_canonical_u64() as #rust_type }
    };

    setter_getters.push(quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, value: #rust_type) {
            self.#field_name = F::#from_method(value);
        }

        #[inline(always)]
        pub fn #getter_name(&self) -> #rust_type {
            #conversion
        }
    });
}

fn add_unpacked_array_setter_getter(field_name: &Ident, field_type: &BitType, setter_getters: &mut Vec<TokenStream>) {
    let (bit_width, dims) = collect_dimensions(field_type);
    let rust_type = type_for_bitwidth(bit_width);
    let from_method = method_name_for_bitwidth(bit_width);
    let args = dimension_args(&dims);
    let array_access = generate_array_access(&args);

    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);

    let conversion = if bit_width == 1 {
        quote! { self.#field_name #array_access.as_canonical_u64() != 0 }
    } else {
        quote! { self.#field_name #array_access.as_canonical_u64() as #rust_type }
    };

    setter_getters.push(quote! {
        #[inline(always)]
        pub fn #setter_name(&mut self, #(#args: usize,)* value: #rust_type) {
            self.#field_name #array_access = F::#from_method(value);
        }

        #[inline(always)]
        pub fn #getter_name(&self, #(#args: usize),*) -> #rust_type {
            #conversion
        }
    });
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

fn method_name_for_bitwidth(width: usize) -> Ident {
    match width {
        1 => format_ident!("from_bool"),
        2..=8 => format_ident!("from_u8"),
        9..=16 => format_ident!("from_u16"),
        17..=32 => format_ident!("from_u32"),
        33..=64 => format_ident!("from_u64"),
        _ => format_ident!("from_u128"),
    }
}

fn dimension_args(dims: &[usize]) -> Vec<Ident> {
    dims.iter().enumerate().map(|(i, _)| format_ident!("i{}", i)).collect()
}

fn generate_array_access(idents: &[Ident]) -> TokenStream {
    let mut access = quote! {};
    for id in idents {
        access = quote! { #access[#id] };
    }
    access
}

fn calculate_row_size(fields: &[TraceField]) -> usize {
    let mut size = 0;
    for field in fields {
        size += calculate_field_size(&field.ty);
    }
    size
}

fn calculate_field_size(ty: &BitType) -> usize {
    match ty {
        BitType::Bit(_) => 1,  // Each bit field is stored as one F element
        BitType::Generic => 1, // Generic F field is one F element
        BitType::Array(inner, len) => {
            calculate_field_size(inner) * len // Recursively calculate array size
        }
    }
}

fn get_default_field_exprs(fields: &[TraceField]) -> Vec<TokenStream> {
    let mut default_exprs = vec![];

    for f in fields.iter() {
        let name = &f.name;
        let init = default_expr(&f.ty);
        default_exprs.push(quote! { #name: #init });
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
