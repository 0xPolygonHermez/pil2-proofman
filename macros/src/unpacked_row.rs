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

    let mut unpacked_fields = get_unpacked_fields(fields);
    let setter_getters = get_unpacked_setters_getters(fields);

    // Calculate the total number of F elements in the row
    let row_size = calculate_row_size(fields);

    let mut default_field_exprs = get_default_field_exprs(fields);

    // A fixed-column row can have no fields at all.
    if let Some(g) = generic {
        if fields.is_empty() {
            unpacked_fields.push(quote! { _phantom: std::marker::PhantomData<#g> });
            default_field_exprs.push(quote! { _phantom: std::marker::PhantomData });
        }
    }

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
    let (bit_width, dims, _acc_dims) = collect_dimensions(field_type);
    let rust_type = type_for_bitwidth(bit_width);
    let from_method = method_name_for_bitwidth(bit_width);

    // Runtime params: i0: usize, ...
    let runtime_idents: Vec<Ident> = dims.iter().enumerate().map(|(i, _)| format_ident!("i{}", i)).collect();
    let runtime_access = generate_array_access(&runtime_idents);

    let setter_name = format_ident!("set_{}", field_name);
    let getter_name = format_ident!("get_{}", field_name);
    let setter_name_all = format_ident!("set_all_{}", field_name);
    let getter_name_all = format_ident!("get_all_{}", field_name);

    let runtime_conversion = if bit_width == 1 {
        quote! { self.#field_name #runtime_access.as_canonical_u64() != 0 }
    } else {
        quote! { self.#field_name #runtime_access.as_canonical_u64() as #rust_type }
    };

    // Whole-array type: dims is outermost-first, wrap innermost-first
    let mut nested_type = rust_type.clone();
    for &len in dims.iter().rev() {
        nested_type = quote! { [#nested_type; #len] };
    }

    // self.field[i0][i1]... and values[i0][i1]...
    let all_field_access = {
        let mut acc = quote! { self.#field_name };
        for id in &runtime_idents {
            acc = quote! { #acc[#id] };
        }
        acc
    };
    let all_values_access = {
        let mut acc = quote! { values };
        for id in &runtime_idents {
            acc = quote! { #acc[#id] };
        }
        acc
    };

    // Setter: nested for-loops (unrolled by optimizer)
    let inner_setter_stmt = quote! { #all_field_access = F::#from_method(#all_values_access); };
    let mut all_setter_body = inner_setter_stmt;
    for (i, &len) in dims.iter().enumerate().rev() {
        let id = &runtime_idents[i];
        all_setter_body = quote! { for #id in 0..#len { #all_setter_body } };
    }

    // Getter: nested std::array::from_fn
    let inner_getter_expr = if bit_width == 1 {
        quote! { #all_field_access.as_canonical_u64() != 0 }
    } else {
        let rt = rust_type.clone();
        quote! { #all_field_access.as_canonical_u64() as #rt }
    };
    let mut all_getter_expr = inner_getter_expr;
    for i in (0..dims.len()).rev() {
        let id = &runtime_idents[i];
        all_getter_expr = quote! { std::array::from_fn(|#id| #all_getter_expr) };
    }

    setter_getters.push(quote! {
        // Runtime-indexed version
        #[inline(always)]
        pub fn #setter_name(&mut self, #(#runtime_idents: usize,)* value: #rust_type) {
            self.#field_name #runtime_access = F::#from_method(value);
        }

        #[inline(always)]
        pub fn #getter_name(&self, #(#runtime_idents: usize),*) -> #rust_type {
            #runtime_conversion
        }

        // Whole-array version
        #[inline(always)]
        pub fn #setter_name_all(&mut self, values: &#nested_type) {
            #all_setter_body
        }

        #[inline(always)]
        pub fn #getter_name_all(&self) -> #nested_type {
            #all_getter_expr
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
