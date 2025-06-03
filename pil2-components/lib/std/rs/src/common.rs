use std::sync::Arc;

use fields::PrimeField64;

use proofman_common::{ProofCtx, SetupCtx};
use proofman_hints::{
    get_hint_field_constant_gc, get_hint_field_constant, get_hint_field_constant_a, HintFieldOptions, HintFieldOutput,
    HintFieldValue,
};

pub trait AirComponent<F: PrimeField64> {
    fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, airgroup_id: Option<usize>, air_id: Option<usize>) -> Arc<Self>;
}

// Helper to extract hint fields
pub fn get_global_hint_field_constant_as<T, F>(sctx: &SetupCtx<F>, hint_id: u64, field_name: &str) -> T
where
    T: TryFrom<u64>,
    T::Error: std::fmt::Debug,
    F: PrimeField64,
{
    let HintFieldValue::Field(field_value) = get_hint_field_constant_gc(sctx, hint_id, field_name, false) else {
        panic!("Hint '{}' for field '{}' must be a field element", hint_id, field_name);
    };

    let biguint_value = field_value.as_canonical_u64();

    biguint_value.try_into().unwrap_or_else(|_| panic!("Cannot convert value to {}", std::any::type_name::<T>()))
}

pub fn get_hint_field_constant_as_field<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> F {
    match get_hint_field_constant(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options) {
        HintFieldValue::Field(value) => value,
        _ => panic!("Hint '{}' for field '{}' must be a field element", hint_id, field_name),
    }
}

pub fn validate_binary_field<F: PrimeField64>(value: F, field_name: &str) -> bool {
    if value.is_zero() {
        false
    } else if value.is_one() {
        true
    } else {
        tracing::error!("{} hint must be either 0 or 1", field_name);
        panic!();
    }
}

pub fn get_hint_field_constant_as_u64<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> u64 {
    let value = match get_hint_field_constant::<F>(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options) {
        HintFieldValue::Field(value) => value,
        _ => panic!("Hint '{}' for field '{}' must be a field element", hint_id, field_name),
    };

    value.as_canonical_u64()
}

pub fn get_hint_field_constant_a_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> Vec<String> {
    let hint_fields = get_hint_field_constant_a(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options);

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::String(value) => return_values.push(value.clone()),
            _ => panic!("Hint '{}' for field '{}' at position '{}' must be a string", hint_id, field_name, i),
        }
    }

    return_values
}

pub fn get_hint_field_constant_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> String {
    match get_hint_field_constant(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options) {
        HintFieldValue::String(value) => value,
        _ => panic!("Hint '{}' for field '{}' must be a string", hint_id, field_name),
    }
}

// Helper to extract a single field element as usize
pub fn extract_field_element_as_usize<F: PrimeField64>(field: &HintFieldValue<F>, name: &str) -> usize {
    let HintFieldValue::Field(field_value) = field else {
        panic!("'{}' hint must be a field element", name);
    };
    field_value.as_canonical_u64() as usize
}

pub fn get_row_field_value<F: PrimeField64>(field_value: &HintFieldValue<F>, row: usize, name: &str) -> F {
    match field_value.get(row) {
        HintFieldOutput::Field(value) => value,
        _ => panic!("'{}' must be a field element", name),
    }
}
