use std::sync::Arc;

use p3_field::PrimeField;
use num_traits::ToPrimitive;
use proofman::{get_hint_field_constant_gc, WitnessManager};
use proofman_common::{AirInstance, ProofCtx, SetupCtx, StdMode};
use proofman_hints::{get_hint_field_constant, HintFieldOptions, HintFieldOutput, HintFieldValue};

pub trait AirComponent<F> {
    const MY_NAME: &'static str;

    fn new(
        wcm: Arc<WitnessManager<F>>,
        mode: Option<StdMode>,
        airgroup_id: Option<usize>,
        air_id: Option<usize>,
    ) -> Arc<Self>;

    fn debug(
        &self,
        _pctx: &ProofCtx<F>,
        _sctx: &SetupCtx,
        _air_instance: &mut AirInstance<F>,
        _num_rows: usize,
        _debug_data_hints: Vec<u64>,
    ) {
    }
}

// Helper to extract hint fields
pub fn get_global_hint_field_constant_as<T, F>(sctx: Arc<SetupCtx>, hint_id: u64, field_name: &str) -> T
where
    T: TryFrom<u64>,
    T::Error: std::fmt::Debug,
    F: PrimeField,
{
    let HintFieldValue::Field(field_value) = get_hint_field_constant_gc::<F>(sctx.clone(), hint_id, field_name, false)
    else {
        panic!("Hint '{}' for field '{}' must be a field element", hint_id, field_name);
    };

    let biguint_value = field_value.as_canonical_biguint();

    biguint_value
        .to_u64()
        .expect("Cannot convert to u64")
        .try_into()
        .expect(&format!("Cannot convert value to {}", std::any::type_name::<T>()))
}

pub fn get_hint_field_constant_as_field<F: PrimeField>(
    sctx: &SetupCtx,
    airgroup_id: usize,
    air_id: usize,
    hint_id: usize,
    field_name: &str,
    hint_field_options: HintFieldOptions,
) -> F {
    match get_hint_field_constant::<F>(sctx, airgroup_id, air_id, hint_id, field_name, hint_field_options) {
        HintFieldValue::Field(value) => value,
        _ => panic!("Hint '{}' for field '{}' must be a field element", hint_id, field_name),
    }
}

// Helper to extract a single field element as usize
pub fn extract_field_element_as_usize<F: PrimeField>(field: &HintFieldValue<F>, name: &str) -> usize {
    let HintFieldValue::Field(field_value) = field else {
        panic!("'{}' hint must be a field element", name);
    };
    field_value.as_canonical_biguint().to_usize().expect("Cannot convert to usize")
}

pub fn get_row_field_value<F: PrimeField>(field_value: &HintFieldValue<F>, row: usize, name: &str) -> F {
    match field_value.get(row) {
        HintFieldOutput::Field(value) => value,
        _ => panic!("'{}' must be a field element", name),
    }
}
