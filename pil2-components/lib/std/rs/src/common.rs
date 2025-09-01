use std::sync::Arc;

use fields::PrimeField64;

use proofman_common::{ProofCtx, SetupCtx};
use proofman_hints::{
    get_hint_field_constant, get_hint_field_constant_a, get_hint_field_constant_gc, get_hint_field_gc_constant_a,
    HintFieldOptions, HintFieldOutput, HintFieldValue,
};

pub const STD_MODE_DEFAULT: usize = 0;
pub const STD_MODE_ONE_INSTANCE: usize = 1;

pub trait AirComponent<F: PrimeField64> {
    fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, airgroup_id: usize, air_id: usize) -> Arc<Self>;
}

// TODO: Remove aliases if they exist!
/// Normalize the values.
pub fn normalize_vals<F: PrimeField64>(vals: &[HintFieldOutput<F>]) -> Vec<HintFieldOutput<F>> {
    let is_zero = |v: &HintFieldOutput<F>| match v {
        HintFieldOutput::Field(x) => *x == F::ZERO,
        HintFieldOutput::FieldExtended(ext) => ext.is_zero(),
    };

    // Find the index of the last non-zero entry
    let last_non_zero = vals.iter().rposition(|v| !is_zero(v)).unwrap_or(0);

    // Keep everything from index 0 to last_non_zero
    vals[..=last_non_zero].to_vec()
}

// Helper to extract hint fields
pub fn get_global_hint_field<F: PrimeField64>(sctx: &SetupCtx<F>, hint_id: u64, field_name: &str) -> F {
    match get_hint_field_constant_gc(sctx, hint_id, field_name, false) {
        HintFieldValue::Field(value) => value,
        _ => panic!("Hint '{hint_id}' for field '{field_name}' must be a field element"),
    }
}

pub fn get_global_hint_field_constant_as<T, F>(sctx: &SetupCtx<F>, hint_id: u64, field_name: &str) -> T
where
    T: TryFrom<u64>,
    T::Error: std::fmt::Debug,
    F: PrimeField64,
{
    let HintFieldValue::Field(field_value) = get_hint_field_constant_gc(sctx, hint_id, field_name, false) else {
        panic!("Hint '{hint_id}' for field '{field_name}' must be a field element");
    };

    let biguint_value = field_value.as_canonical_u64();

    biguint_value.try_into().unwrap_or_else(|_| panic!("Cannot convert value to {}", std::any::type_name::<T>()))
}

pub fn get_global_hint_field_constant_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    hint_id: u64,
    field_name: &str,
) -> String {
    let hint_field = get_hint_field_constant_gc(sctx, hint_id, field_name, false);

    match hint_field {
        HintFieldValue::String(value) => value,
        _ => panic!("Hint '{hint_id}' for field '{field_name}' must be a string"),
    }
}

pub fn get_global_hint_field_constant_a_as<T, F>(sctx: &SetupCtx<F>, hint_id: u64, field_name: &str) -> Vec<T>
where
    T: TryFrom<u64>,
    F: PrimeField64,
{
    let hint_fields = get_hint_field_gc_constant_a(sctx, hint_id, field_name, false);

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::Field(value) => {
                let converted = T::try_from(value.as_canonical_u64()).unwrap_or_else(|_| {
                    panic!("Cannot convert value at position {i} to {}", std::any::type_name::<T>())
                });
                return_values.push(converted);
            }
            _ => panic!("Hint '{hint_id}' for field '{field_name}' at position '{i}' must be a field element"),
        }
    }

    return_values
}

pub fn get_global_hint_field_constant_a_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    hint_id: u64,
    field_name: &str,
) -> Vec<String> {
    let hint_fields = get_hint_field_gc_constant_a(sctx, hint_id, field_name, false);

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::String(value) => return_values.push(value.clone()),
            _ => panic!("Hint '{hint_id}' for field '{field_name}' at position '{i}' must be a string"),
        }
    }

    return_values
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
        _ => panic!("Hint '{hint_id}' for field '{field_name}' must be a field element"),
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

pub fn get_hint_field_constant_as<T, F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> T
where
    T: TryFrom<u64>,
{
    let value = match get_hint_field_constant::<F>(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options) {
        HintFieldValue::Field(value) => value.as_canonical_u64(),
        _ => panic!("Hint '{hint_id}' for field '{field_name}' must be a field element"),
    };

    T::try_from(value).unwrap_or_else(|_| panic!("Cannot convert value to {}", std::any::type_name::<T>()))
}

pub fn get_hint_field_constant_a_as<T, F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> Vec<T>
where
    T: TryFrom<u64>,
{
    let hint_fields = get_hint_field_constant_a(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options);

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::Field(value) => {
                let converted = T::try_from(value.as_canonical_u64()).unwrap_or_else(|_| {
                    panic!("Cannot convert value at position {i} to {}", std::any::type_name::<T>())
                });
                return_values.push(converted);
            }
            _ => panic!("Hint '{hint_id}' for field '{field_name}' at position '{i}' must be a field element"),
        }
    }

    return_values
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
            _ => panic!("Hint '{hint_id}' for field '{field_name}' at position '{i}' must be a string"),
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
        _ => panic!("Hint '{hint_id}' for field '{field_name}' must be a string"),
    }
}

// Helper to extract a single field element as usize
pub fn extract_field_element_as_usize<F: PrimeField64>(field: &HintFieldValue<F>, name: &str) -> usize {
    let HintFieldValue::Field(field_value) = field else {
        panic!("'{name}' hint must be a field element");
    };
    field_value.as_canonical_u64() as usize
}

pub fn get_row_field_value<F: PrimeField64>(field_value: &HintFieldValue<F>, row: usize, name: &str) -> F {
    match field_value.get(row) {
        HintFieldOutput::Field(value) => value,
        _ => panic!("'{name}' must be a field element"),
    }
}
