// lanes.rs - rows with lanes: `x: lane { .. }` and `x: [lane { .. }; n]` fields, laid out in
// order. Per lane: struct `RowX<F>`, view `RowXView` (`RowOps::x(lane)`) and its `RowXOps`;
// the row's own accessors flatten field `a` of `x` to `x_a`, the lane first for an array.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::packed_row::{packed_row_impl, PackedGroup};
use crate::trace_row::{
    collect_dimensions, compute_total_bits, contains_generic, flat_name, is_array, BitType, LaneItem, RowItem,
    TraceField, TraceRowInput,
};
use crate::trait_row::{rust_type_for_bits, trait_impl};
use crate::unpacked_row::{
    calculate_row_size, get_default_field_exprs, get_unpacked_fields, get_unpacked_setters_getters, unpacked_row_impl,
};

/// `snake_case` -> `SnakeCase`.
fn pascal(name: &Ident) -> String {
    name.to_string()
        .split('_')
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().chain(c).collect::<String>()).unwrap_or_default()
        })
        .collect()
}

fn lane_struct(row: &Ident, lane: &LaneItem) -> Ident {
    format_ident!("{}{}", row, pascal(&lane.name))
}

pub fn lane_items(items: &[RowItem]) -> Vec<&LaneItem> {
    items
        .iter()
        .filter_map(|i| match i {
            RowItem::Lane(l) => Some(l),
            RowItem::Field(_) => None,
        })
        .collect()
}

/// The row's fields with each lane's flattened (`x_a: [T; n]` for an array), and the arrays' groups.
pub fn flatten(items: &[RowItem]) -> (Vec<TraceField>, Vec<PackedGroup>) {
    let mut fields = vec![];
    let mut groups = vec![];
    for item in items {
        match item {
            RowItem::Field(f) => fields.push(f.clone()),
            RowItem::Lane(lane) => {
                if let Some(count) = lane.count {
                    groups.push(PackedGroup {
                        name: lane.name.clone(),
                        first: fields.len(),
                        len: lane.fields.len(),
                        count,
                    });
                }
                fields.extend(lane.fields.iter().map(|f| flat_field(lane, f)));
            }
        }
    }
    (fields, groups)
}

/// Field `f` of `lane` as a field of the row.
pub fn flat_field(lane: &LaneItem, f: &TraceField) -> TraceField {
    let ty = match lane.count {
        Some(n) => BitType::Array(Box::new(f.ty.clone()), n),
        None => f.ty.clone(),
    };
    TraceField { name: flat_name(&lane.name, &f.name), ty }
}

pub fn lane_row_impl(input: &TraceRowInput) -> TokenStream {
    let Some(g) = &input.generic else {
        return quote! { compile_error!("a row with lanes takes its field generic, like `Row<F> { .. }`"); };
    };
    let name = &input.name;
    let packed_name = format_ident!("{}Packed", name);
    let trait_name = format_ident!("{}Ops", name);
    let lanes = lane_items(&input.items);

    let (flat, groups) = flatten(&input.items);
    // A packed row is of bit-width columns only; a row with `F` columns has no packed variant.
    let packed = flat.iter().all(|f| !contains_generic(&f.ty)).then_some(&packed_name);
    let packed_struct = packed.map(|p| packed_row_impl(p, &input.generic, &flat, &groups, 0, true));

    let lane_structs = lanes.iter().map(|l| unpacked_row_impl(&lane_struct(name, l), &input.generic, &l.fields));
    let row_struct = unpacked_lanes_row(name, g, &input.items);
    let views = lanes.iter().map(|l| lane_view(name, &trait_name, g, l));
    let provided = lanes.iter().map(|l| {
        let method = &l.name;
        let view = format_ident!("{}View", lane_struct(name, l));
        match l.count {
            Some(n) => quote! {
                fn #method(&mut self, lane: usize) -> #view<'_, Self> where Self: Sized {
                    debug_assert!(lane < #n, "lane {lane} out of the {} of `{}`", #n, stringify!(#method));
                    #view { row: self, lane }
                }
            },
            None => quote! {
                fn #method(&mut self) -> #view<'_, Self> where Self: Sized {
                    #view { row: self, lane: 0 }
                }
            },
        }
    });
    let ops = trait_impl(&trait_name, name, packed, &input.generic, &flat, quote! { #(#provided)* });

    quote! {
        #(#lane_structs)*
        #row_struct
        #packed_struct
        #ops
        #(#views)*
    }
}

/// The unpacked row, with the row's accessors of the lanes' fields.
fn unpacked_lanes_row(name: &Ident, g: &Ident, items: &[RowItem]) -> TokenStream {
    let mut fields = vec![];
    let mut defaults = vec![];
    let mut sizes = vec![];
    let mut plain = vec![];
    let mut accessors = vec![];
    let mut untyped_lane = vec![];
    for item in items {
        match item {
            RowItem::Field(f) => {
                fields.extend(get_unpacked_fields(std::slice::from_ref(f)));
                defaults.extend(get_default_field_exprs(std::slice::from_ref(f)));
                let size = calculate_row_size(std::slice::from_ref(f));
                sizes.push(quote! { #size });
                plain.push(f.clone());
            }
            RowItem::Lane(lane) => {
                let field = &lane.name;
                let ty = lane_struct(name, lane);
                let (decl, default, size) = match lane.count {
                    Some(n) => (
                        quote! { [#ty<#g>; #n] },
                        quote! { [#ty::default(); #n] },
                        quote! { #n * <#ty<#g> as proofman_common::trace::TraceRow>::ROW_SIZE },
                    ),
                    None => (
                        quote! { #ty<#g> },
                        quote! { #ty::default() },
                        quote! { <#ty<#g> as proofman_common::trace::TraceRow>::ROW_SIZE },
                    ),
                };
                fields.push(quote! { pub #field: #decl });
                defaults.push(quote! { #field: #default });
                sizes.push(size);
                accessors
                    .extend(lane.fields.iter().filter(|f| !contains_generic(&f.ty)).map(|f| lane_accessors(lane, f)));
                // A single lane's scalar `F` fields, which the row trait forwards to, as for a plain field.
                if lane.count.is_none() {
                    for f in lane.fields.iter().filter(|f| contains_generic(&f.ty) && !is_array(&f.ty)) {
                        let flat = flat_name(&lane.name, &f.name);
                        let (setter, getter) = (format_ident!("set_{}", flat), format_ident!("get_{}", flat));
                        let name = &f.name;
                        untyped_lane.push(quote! {
                            #[inline(always)]
                            pub fn #setter(&mut self, value: #g) { self.#field.#name = value; }
                            #[inline(always)]
                            pub fn #getter(&self) -> #g { self.#field.#name }
                        });
                    }
                }
            }
        }
    }
    let (untyped, typed) = get_unpacked_setters_getters(&plain);

    quote! {
        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct #name<#g: Copy + Default + Send> {
            #(#fields,)*
        }

        impl<#g: Copy + Default + Send> Default for #name<#g> {
            fn default() -> Self {
                Self { #(#defaults,)* }
            }
        }

        impl<#g: Copy + Default + Send> #name<#g> {
            #(#untyped)*
            #(#untyped_lane)*
        }

        impl<#g: PrimeField64 + Copy + Default + Send> #name<#g> {
            #(#typed)*
            #(#accessors)*
        }

        impl<#g: Copy + Default + Send> proofman_common::trace::TraceRow for #name<#g> {
            const ROW_SIZE: usize = 0 #(+ #sizes)*;
            const IS_PACKED: bool = false;
        }
    }
}

/// The row's accessors of field `f` of `lane`, forwarding to the lane's own.
fn lane_accessors(lane: &LaneItem, f: &TraceField) -> TokenStream {
    let flat = flat_name(&lane.name, &f.name);
    let (setter, getter) = (format_ident!("set_{}", flat), format_ident!("get_{}", flat));
    let (setter_all, getter_all) = (format_ident!("set_all_{}", flat), format_ident!("get_all_{}", flat));
    let (lane_setter, lane_getter) = (format_ident!("set_{}", f.name), format_ident!("get_{}", f.name));
    let (lane_setter_all, lane_getter_all) = (format_ident!("set_all_{}", f.name), format_ident!("get_all_{}", f.name));
    let (rust_ty, idx, nested) = lane_shape(&f.ty);
    let field = &lane.name;
    let (lane_arg, at) = match lane.count {
        Some(_) => (quote! { lane: usize, }, quote! { [lane] }),
        None => (quote! {}, quote! {}),
    };
    let one = quote! {
        #[inline(always)]
        pub fn #setter(&mut self, #lane_arg #(#idx: usize,)* value: #rust_ty) {
            self.#field #at.#lane_setter(#(#idx,)* value)
        }
        #[inline(always)]
        pub fn #getter(&self, #lane_arg #(#idx: usize),*) -> #rust_ty {
            self.#field #at.#lane_getter(#(#idx),*)
        }
    };
    // Whole arrays: a single lane's own, or one value per lane of an array.
    let all = match (lane.count, is_array(&f.ty)) {
        (None, false) => quote! {},
        (None, true) => quote! {
            #[inline]
            pub fn #setter_all(&mut self, values: &#nested) { self.#field.#lane_setter_all(values) }
            #[inline]
            pub fn #getter_all(&self) -> #nested { self.#field.#lane_getter_all() }
        },
        (Some(n), array) => {
            let (set, get) = if array {
                (quote! { lane.#lane_setter_all(v) }, quote! { self.#field[l].#lane_getter_all() })
            } else {
                (quote! { lane.#lane_setter(*v) }, quote! { self.#field[l].#lane_getter() })
            };
            quote! {
                #[inline]
                pub fn #setter_all(&mut self, values: &[#nested; #n]) {
                    for (lane, v) in self.#field.iter_mut().zip(values) {
                        #set;
                    }
                }
                #[inline]
                pub fn #getter_all(&self) -> [#nested; #n] {
                    std::array::from_fn(|l| #get)
                }
            }
        }
    };
    quote! { #one #all }
}

/// `RowXView`, one lane of a row, and its `RowXOps` accessors, which forward to the row's.
fn lane_view(row: &Ident, row_trait: &Ident, g: &Ident, lane: &LaneItem) -> TokenStream {
    let lane_name = lane_struct(row, lane);
    let lane_trait = format_ident!("{}Ops", lane_name);
    let view = format_ident!("{}View", lane_name);
    let lane_arg = lane.count.map(|_| quote! { self.lane, });

    // `F` fields have no accessors, as in a row without lanes.
    let (signatures, methods): (Vec<_>, Vec<_>) = lane
        .fields
        .iter()
        .filter(|f| !contains_generic(&f.ty))
        .map(|f| {
            let flat = flat_name(&lane.name, &f.name);
            let (row_setter, row_getter) = (format_ident!("set_{}", flat), format_ident!("get_{}", flat));
            let (setter, getter) = (format_ident!("set_{}", f.name), format_ident!("get_{}", f.name));
            let (rust_ty, idx, _) = lane_shape(&f.ty);
            let set = quote! { fn #setter(&mut self, #(#idx: usize,)* value: #rust_ty) where #g: PrimeField64 };
            let get = quote! { fn #getter(&self, #(#idx: usize),*) -> #rust_ty where #g: PrimeField64 };
            (
                quote! { #set; #get; },
                quote! {
                    #[inline(always)]
                    #set { #row_trait::#row_setter(self.row, #lane_arg #(#idx,)* value) }
                    #[inline(always)]
                    #get { #row_trait::#row_getter(&*self.row, #lane_arg #(#idx),*) }
                },
            )
        })
        .unzip();

    let steps = lane.count.map(|n| {
        quote! {
            impl<'a, R> #view<'a, R> {
                pub const LANES: usize = #n;

                /// The lane of `rows` holding `step`, a row's lanes filled before the next row.
                pub fn step(rows: &'a mut [R], step: usize) -> Self {
                    Self { row: &mut rows[step / #n], lane: step % #n }
                }
            }
        }
    });

    quote! {
        pub struct #view<'a, R> {
            pub row: &'a mut R,
            pub lane: usize,
        }

        #steps

        pub trait #lane_trait<#g: Copy + Default + Send + 'static> {
            #(#signatures)*
        }

        impl<'a, #g: Copy + Default + Send + 'static, R: #row_trait<#g>> #lane_trait<#g> for #view<'a, R> {
            #(#methods)*
        }
    }
}

/// A typed lane field's value type, index arguments `i1..` (after the lane) and array type.
fn lane_shape(ty: &BitType) -> (TokenStream, Vec<Ident>, TokenStream) {
    if !is_array(ty) {
        let rust_ty = rust_type_for_bits(compute_total_bits(ty));
        return (rust_ty.clone(), vec![], rust_ty);
    }
    let (bits, dims, _) = collect_dimensions(ty);
    let rust_ty = rust_type_for_bits(bits);
    let idx = (1..=dims.len()).map(|i| format_ident!("i{}", i)).collect();
    let nested = dims.iter().rev().fold(rust_ty.clone(), |acc, len| quote! { [#acc; #len] });
    (rust_ty, idx, nested)
}
