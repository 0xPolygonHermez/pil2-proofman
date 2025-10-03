use std::sync::Arc;

use fields::PrimeField64;

use proofman_common::{ProofCtx, ProofmanError, ProofmanResult, SetupCtx};
use proofman_hints::{
    get_hint_field_constant, get_hint_field_constant_a, get_hint_field_constant_gc, get_hint_field_gc_constant_a,
    HintFieldOptions, HintFieldOutput, HintFieldValue,
};

pub const STD_MODE_DEFAULT: usize = 0;
pub const STD_MODE_ONE_INSTANCE: usize = 1;

pub trait AirComponent<F: PrimeField64> {
    fn new(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        airgroup_id: usize,
        air_id: usize,
        shared_tables: bool,
    ) -> ProofmanResult<Arc<Self>>;
}

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
pub fn get_global_hint_field<F: PrimeField64>(sctx: &SetupCtx<F>, hint_id: u64, field_name: &str) -> ProofmanResult<F> {
    match get_hint_field_constant_gc(sctx, hint_id, field_name, false)? {
        HintFieldValue::Field(value) => Ok(value),
        _ => Err(ProofmanError::InvalidHints(format!(
            "Hint '{hint_id}' for field '{field_name}' must be a field element"
        ))),
    }
}

pub fn get_global_hint_field_constant_as<T, F>(sctx: &SetupCtx<F>, hint_id: u64, field_name: &str) -> ProofmanResult<T>
where
    T: TryFrom<u64>,
    T::Error: std::fmt::Debug,
    F: PrimeField64,
{
    let field_value = match get_hint_field_constant_gc(sctx, hint_id, field_name, false)? {
        HintFieldValue::Field(field_value) => field_value,
        _ => {
            return Err(ProofmanError::InvalidHints(format!(
                "Hint '{hint_id}' for field '{field_name}' must be a field element"
            )))
        }
    };

    let biguint_value = field_value.as_canonical_u64();

    let value: T = biguint_value.try_into().map_err(|_| {
        ProofmanError::InvalidAssignation(format!("Cannot convert value to {}", std::any::type_name::<T>()))
    })?;

    Ok(value)
}

pub fn get_global_hint_field_constant_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    hint_id: u64,
    field_name: &str,
) -> ProofmanResult<String> {
    let hint_field = get_hint_field_constant_gc(sctx, hint_id, field_name, false)?;

    match hint_field {
        HintFieldValue::String(value) => Ok(value),
        _ => Err(ProofmanError::InvalidHints(format!("Hint '{hint_id}' for field '{field_name}' must be a string"))),
    }
}

pub fn get_global_hint_field_constant_a_as<T, F>(
    sctx: &SetupCtx<F>,
    hint_id: u64,
    field_name: &str,
) -> ProofmanResult<Vec<T>>
where
    T: TryFrom<u64>,
    F: PrimeField64,
{
    let hint_fields = get_hint_field_gc_constant_a(sctx, hint_id, field_name, false)?;

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::Field(value) => {
                let converted = T::try_from(value.as_canonical_u64()).map_err(|_| {
                    ProofmanError::InvalidHints(format!(
                        "Cannot convert value at position {} to {}",
                        i,
                        std::any::type_name::<T>()
                    ))
                })?;
                return_values.push(converted);
            }
            _ => {
                return Err(ProofmanError::InvalidHints(format!(
                    "Hint '{}' for field '{}' at position '{}' must be a field element",
                    hint_id, field_name, i
                )));
            }
        }
    }

    Ok(return_values)
}

pub fn get_global_hint_field_constant_a_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    hint_id: u64,
    field_name: &str,
) -> ProofmanResult<Vec<String>> {
    let hint_fields = get_hint_field_gc_constant_a(sctx, hint_id, field_name, false)?;

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::String(value) => return_values.push(value.clone()),
            _ => {
                return Err(ProofmanError::InvalidHints(format!(
                    "Hint '{hint_id}' for field '{field_name}' at position '{i}' must be a string"
                )));
            }
        }
    }

    Ok(return_values)
}

pub fn get_hint_field_constant_as_field<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> ProofmanResult<F> {
    match get_hint_field_constant(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options)? {
        HintFieldValue::Field(value) => Ok(value),
        _ => Err(ProofmanError::InvalidHints(format!(
            "Hint '{hint_id}' for field '{field_name}' must be a field element"
        ))),
    }
}

pub fn validate_binary_field<F: PrimeField64>(value: F, field_name: &str) -> ProofmanResult<bool> {
    if value.is_zero() {
        Ok(false)
    } else if value.is_one() {
        Ok(true)
    } else {
        Err(ProofmanError::InvalidHints(format!("{} hint must be either 0 or 1", field_name)))
    }
}

pub fn get_hint_field_constant_as<T, F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> ProofmanResult<T>
where
    T: TryFrom<u64>,
{
    let value = match get_hint_field_constant::<F>(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options)?
    {
        HintFieldValue::Field(value) => value.as_canonical_u64(),
        _ => {
            return Err(ProofmanError::InvalidHints(format!(
                "Hint '{hint_id}' for field '{field_name}' must be a field element"
            )))
        }
    };

    T::try_from(value)
        .map_err(|_| ProofmanError::InvalidHints(format!("Cannot convert value to {}", std::any::type_name::<T>())))
}

pub fn get_hint_field_constant_a_as<T, F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> ProofmanResult<Vec<T>>
where
    T: TryFrom<u64>,
{
    let hint_fields = get_hint_field_constant_a(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options)?;

    let mut return_values = Vec::with_capacity(hint_fields.values.len());

    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        let field_value = match hint_field {
            HintFieldValue::Field(v) => v,
            _ => {
                return Err(ProofmanError::InvalidHints(format!(
                    "Hint '{hint_id}' for field '{field_name}' at position {i} must be a field element"
                )));
            }
        };

        let converted = T::try_from(field_value.as_canonical_u64()).map_err(|_| {
            ProofmanError::InvalidHints(format!(
                "Cannot convert value at position {i} to {}",
                std::any::type_name::<T>()
            ))
        })?;

        return_values.push(converted);
    }

    Ok(return_values)
}

pub fn get_hint_field_constant_a_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> ProofmanResult<Vec<String>> {
    let hint_fields = get_hint_field_constant_a(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options)?;

    let mut return_values = Vec::new();
    for (i, hint_field) in hint_fields.values.iter().enumerate() {
        match hint_field {
            HintFieldValue::String(value) => return_values.push(value.clone()),
            _ => {
                return Err(ProofmanError::InvalidHints(format!(
                    "Hint '{hint_id}' for field '{field_name}' at position '{i}' must be a string"
                )))
            }
        }
    }

    Ok(return_values)
}

pub fn get_hint_field_constant_as_string<F: PrimeField64>(
    sctx: &SetupCtx<F>,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> ProofmanResult<String> {
    match get_hint_field_constant(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options)? {
        HintFieldValue::String(value) => Ok(value),
        _ => Err(ProofmanError::InvalidHints(format!("Hint '{hint_id}' for field '{field_name}' must be a string"))),
    }
}

// Helper to extract a single field element as usize
pub fn extract_field_element_as_usize<F: PrimeField64>(field: &HintFieldValue<F>, name: &str) -> ProofmanResult<usize> {
    let HintFieldValue::Field(field_value) = field else {
        return Err(ProofmanError::InvalidHints(format!("'{name}' hint must be a field element")));
    };
    Ok(field_value.as_canonical_u64() as usize)
}

pub fn get_row_field_value<F: PrimeField64>(
    field_value: &HintFieldValue<F>,
    row: usize,
    name: &str,
) -> ProofmanResult<F> {
    match field_value.get(row) {
        HintFieldOutput::Field(value) => Ok(value),
        _ => Err(ProofmanError::InvalidHints(format!("'{name}' must be a field element"))),
    }
}
