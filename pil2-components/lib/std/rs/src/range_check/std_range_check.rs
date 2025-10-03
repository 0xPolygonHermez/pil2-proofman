use std::{fmt::Debug, sync::Arc};

use fields::PrimeField64;

use witness::WitnessComponent;
use proofman_common::{BufferPool, ProofCtx, ProofmanError, ProofmanResult, SetupCtx};
use proofman_hints::{
    get_hint_field_constant, get_hint_field_gc_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue,
};

use crate::{
    extract_field_element_as_usize, get_global_hint_field_constant_as, get_hint_field_constant_as,
    get_hint_field_constant_as_field, validate_binary_field, AirComponent, SpecifiedRanges, StdVirtualTable, U16Air,
    U8Air,
};

pub struct StdRangeCheck<F: PrimeField64> {
    _phantom: std::marker::PhantomData<F>,
    ranges: Vec<StdRange>,
    pub u8air: Option<Arc<U8Air>>,
    pub u16air: Option<Arc<U16Air>>,
    pub specified_ranges_air: Option<Arc<SpecifiedRanges>>,
    virtual_table: Arc<StdVirtualTable<F>>,
}

#[derive(Debug, Clone)]
struct StdRange {
    rc_type: StdRangeType,
    is_virtual: bool,
    virtual_id: usize,
    data: RangeData,
}

#[derive(Debug, Clone)]
enum StdRangeType {
    U8Air,
    U16Air,
    U8AirDouble,
    U16AirDouble,
    SpecifiedRanges,
}

#[derive(Debug, PartialEq, Clone)]
struct RangeData {
    min: i64,
    max: i64,
    predefined: bool,
}

struct HintCache {
    opid: u64,
    predefined: bool,
    min: i64,
    max: i64,
    rc_type: StdRangeType,
    is_virtual: bool,
}

impl<F: PrimeField64> StdRangeCheck<F> {
    pub fn new(
        pctx: Arc<ProofCtx<F>>,
        sctx: &SetupCtx<F>,
        virtual_table: Arc<StdVirtualTable<F>>,
        shared_tables: bool,
    ) -> ProofmanResult<Arc<Self>> {
        // Find which range check related AIRs need to be instantiated
        let u8air_hint = get_hint_ids_by_name(sctx.get_global_bin(), "u8air");
        let u16air_hint = get_hint_ids_by_name(sctx.get_global_bin(), "u16air");
        let specified_ranges_air_hint = get_hint_ids_by_name(sctx.get_global_bin(), "specified_ranges");

        // Instantiate the AIRs
        let u8air = Self::create_air::<U8Air>(&pctx, sctx, shared_tables, &u8air_hint)?;
        let u16air = Self::create_air::<U16Air>(&pctx, sctx, shared_tables, &u16air_hint)?;
        let specified_ranges_air =
            Self::create_air::<SpecifiedRanges>(&pctx, sctx, shared_tables, &specified_ranges_air_hint)?;

        // Early return if no range check users
        let std_rc_users = get_hint_ids_by_name(sctx.get_global_bin(), "std_rc_users");
        let Some(std_rc_users) = std_rc_users.first() else {
            return Ok(Arc::new(Self {
                _phantom: std::marker::PhantomData,
                ranges: Vec::new(),
                u8air,
                u16air,
                specified_ranges_air,
                virtual_table,
            }));
        };

        let num_users = get_global_hint_field_constant_as::<usize, F>(sctx, *std_rc_users, "num_users")?;
        let airgroup_ids = get_hint_field_gc_constant_a(sctx, *std_rc_users, "airgroup_ids", false)?;
        let air_ids = get_hint_field_gc_constant_a(sctx, *std_rc_users, "air_ids", false)?;
        let opids_count = get_global_hint_field_constant_as::<usize, F>(sctx, *std_rc_users, "opids_count")?;
        let spec_opids_count = get_global_hint_field_constant_as::<usize, F>(sctx, *std_rc_users, "spec_opids_count")?;

        let mut ranges = vec![
            StdRange {
                rc_type: StdRangeType::U8Air,
                is_virtual: false,
                virtual_id: 0,
                data: RangeData { min: 0, max: 0, predefined: false },
            };
            opids_count
        ];

        let mut processed_predefined_ranges = spec_opids_count;
        let mut processed_specified_ranges = 0;
        for i in 0..num_users {
            let airgroup_id = extract_field_element_as_usize(&airgroup_ids.values[i], "airgroup_id")?;
            let air_id = extract_field_element_as_usize(&air_ids.values[i], "air_id")?;

            Self::register_ranges(
                sctx,
                virtual_table.clone(),
                airgroup_id,
                air_id,
                &mut ranges,
                &mut processed_predefined_ranges,
                &mut processed_specified_ranges,
            )?;
        }

        Ok(Arc::new(Self {
            _phantom: std::marker::PhantomData,
            ranges,
            u8air,
            u16air,
            specified_ranges_air,
            virtual_table,
        }))
    }

    // Helper function to instantiate AIRs
    fn create_air<T>(
        pctx: &ProofCtx<F>,
        sctx: &SetupCtx<F>,
        shared_tables: bool,
        hints: &[u64],
    ) -> ProofmanResult<Option<Arc<T>>>
    where
        T: AirComponent<F>,
    {
        if hints.is_empty() {
            return Ok(None);
        }
        let airgroup_id = get_global_hint_field_constant_as(sctx, hints[0], "airgroup_id")?;
        let air_id = get_global_hint_field_constant_as(sctx, hints[0], "air_id")?;
        if (airgroup_id as u64 == F::NEG_ONE.as_canonical_u64()) || (air_id as u64 == F::NEG_ONE.as_canonical_u64()) {
            // The AIR is virtual, so we do not instantiate it
            return Ok(None);
        }

        Ok(Some(T::new(pctx, sctx, airgroup_id, air_id, shared_tables)?))
    }

    // Helper function to register ranges
    fn register_ranges(
        sctx: &SetupCtx<F>,
        virtual_table: Arc<StdVirtualTable<F>>,
        airgroup_id: usize,
        air_id: usize,
        ranges: &mut [StdRange],
        processed_predefined_ranges: &mut usize,
        processed_specified_ranges: &mut usize,
    ) -> ProofmanResult<()> {
        let setup = sctx.get_setup(airgroup_id, air_id)?;

        // Obtain info from the range hints
        let rc_hints = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "range_def");

        for hint in rc_hints {
            let hint_data = Self::parse_range_hint(sctx, airgroup_id, air_id, hint)?;

            let data = RangeData { min: hint_data.min, max: hint_data.max, predefined: hint_data.predefined };

            // If the range is already defined, skip
            if ranges.iter().any(|r| r.data == data) {
                continue;
            }

            // Otherwise, define the range
            let (rc_type, idx) = match hint_data.rc_type {
                StdRangeType::U8Air | StdRangeType::U16Air | StdRangeType::U8AirDouble | StdRangeType::U16AirDouble => {
                    let idx = *processed_predefined_ranges;
                    *processed_predefined_ranges += 1;
                    (hint_data.rc_type, idx)
                }
                StdRangeType::SpecifiedRanges => {
                    let idx = *processed_specified_ranges;
                    *processed_specified_ranges += 1;
                    (hint_data.rc_type, idx)
                }
            };

            let is_virtual = hint_data.is_virtual;
            let virtual_id = if is_virtual {
                // Get the virtual table ID
                virtual_table.get_global_id(hint_data.opid as usize)?
            } else {
                0
            };
            ranges[idx] = StdRange { rc_type, is_virtual: hint_data.is_virtual, virtual_id, data };
        }
        Ok(())
    }

    fn parse_range_hint(sctx: &SetupCtx<F>, airgroup_id: usize, air_id: usize, hint: u64) -> ProofmanResult<HintCache> {
        let options = HintFieldOptions::default();

        let opid =
            get_hint_field_constant_as::<u64, F>(sctx, airgroup_id, air_id, hint as usize, "opid", options.clone())?;

        let predefined = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "predefined",
                options.clone(),
            )?,
            "Predefined",
        )?;

        let min_val =
            get_hint_field_constant_as::<u64, F>(sctx, airgroup_id, air_id, hint as usize, "min", options.clone())?;
        let min_neg = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "min_neg",
                options.clone(),
            )?,
            "Min neg",
        )?;

        let max_val =
            get_hint_field_constant_as::<u64, F>(sctx, airgroup_id, air_id, hint as usize, "max", options.clone())?;
        let max_neg = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "max_neg",
                options.clone(),
            )?,
            "Max neg",
        )?;

        let HintFieldValue::String(rc_type_str) =
            get_hint_field_constant::<F>(sctx, airgroup_id, air_id, hint as usize, "type", options.clone())?
        else {
            return Err(ProofmanError::StdError("Type hint must be a string".to_string()));
        };

        let is_virtual = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "is_virtual",
                options.clone(),
            )?,
            "Is virtual",
        )?;

        let min = if min_neg { min_val as i128 - F::ORDER_U64 as i128 } else { min_val as i128 };

        let max = if max_neg { max_val as i128 - F::ORDER_U64 as i128 } else { max_val as i128 };

        // Check that min or max does not overflow 63 bits
        if min > i64::MAX as i128 || max > i64::MAX as i128 {
            return Err(ProofmanError::StdError("Min/Max value is too large".to_string()));
        }

        // Use match with string literals for better optimization
        let rc_type = match rc_type_str.as_str() {
            "U8" => StdRangeType::U8Air,
            "U16" => StdRangeType::U16Air,
            "U8Double" => StdRangeType::U8AirDouble,
            "U16Double" => StdRangeType::U16AirDouble,
            "Specified" => StdRangeType::SpecifiedRanges,
            _ => return Err(ProofmanError::StdError("Invalid range check type: {rc_type_str}".to_string())),
        };

        Ok(HintCache { opid, predefined, min: min as i64, max: max as i64, rc_type, is_virtual })
    }

    pub fn get_range_id(&self, min: i64, max: i64, predefined: Option<bool>) -> ProofmanResult<usize> {
        // Default predefined value in STD is false
        let predefined = predefined.unwrap_or(false);

        // Find the range with the given [min,max] values, return its id
        let received_range_data = RangeData { min, max, predefined };
        if let Some(i) = self.ranges.iter().position(|r| r.data == received_range_data) {
            Ok(i)
        } else {
            Err(ProofmanError::StdError(format!(
                "Range not found: [min,max] = [{min},{max}] (predefined: {predefined})"
            )))
        }
    }

    pub fn assign_value(&self, id: usize, value: i64, multiplicity: u64) {
        // Find the range with the given id
        let range_item = &self.ranges[id];

        // Check that the value is contained within the range
        #[cfg(all(debug_assertions, feature = "verify-rc-values"))]
        check_value_in_range(range_item, value);

        // Update the multiplicity of the corresponding AIR
        match range_item.rc_type {
            StdRangeType::U8Air => {
                // Here, we can safely assume that value ∊ [0,2⁸-1]
                // Therefore, we can safely cast value to u8
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let row = U8Air::get_global_row(value as u8);

                    // Increment the virtual row
                    self.virtual_table.inc_virtual_row(range_item.virtual_id, row, multiplicity);
                } else {
                    self.u8air.as_ref().unwrap().update_input(value as u8, multiplicity);
                }
            }
            StdRangeType::U16Air => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1]
                // Therefore, we can safely cast value to u16
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let row = U16Air::get_global_row(value as u16);

                    // Increment the virtual row
                    self.virtual_table.inc_virtual_row(range_item.virtual_id, row, multiplicity);
                } else {
                    self.u16air.as_ref().unwrap().update_input(value as u16, multiplicity);
                }
            }
            StdRangeType::U8AirDouble => {
                // Here, we can safely assume that value ∊ [0,2⁸-1], min >= 0 and max <= 2⁸-1
                // Therefore, we can safely cast value to u8
                let range_data = &range_item.data;
                let lower_value = (value - range_data.min) as u8;
                let upper_value = (range_data.max - value) as u8;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = vec![U8Air::get_global_row(lower_value), U8Air::get_global_row(upper_value)];

                    // Increment the virtual row
                    self.virtual_table.inc_virtual_rows_same_mul(range_item.virtual_id, &rows, multiplicity);
                } else {
                    let u8_air = self.u8air.as_ref().unwrap();
                    u8_air.update_input(lower_value, multiplicity);
                    u8_air.update_input(upper_value, multiplicity);
                }
            }
            StdRangeType::U16AirDouble => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1], min >= 0 and max <= 2¹⁶-1
                // Therefore, we can safely cast value to u16
                let range_data = &range_item.data;
                let lower_value = (value - range_data.min) as u16;
                let upper_value = (range_data.max - value) as u16;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = vec![U16Air::get_global_row(lower_value), U16Air::get_global_row(upper_value)];

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_rows_same_mul(range_item.virtual_id, &rows, multiplicity);
                } else {
                    let u16_air = self.u16air.as_ref().unwrap();
                    u16_air.update_input(lower_value, multiplicity);
                    u16_air.update_input(upper_value, multiplicity);
                }
            }
            StdRangeType::SpecifiedRanges => {
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let row = SpecifiedRanges::get_global_row(range_item.data.min, value);

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_row(range_item.virtual_id, row, multiplicity);
                } else {
                    self.specified_ranges_air.as_ref().unwrap().update_input(id, value, multiplicity);
                }
            }
        }
    }

    pub fn assign_values(&self, id: usize, values: Vec<u32>) {
        // Find the range with the given id
        let range_item = &self.ranges[id];

        // Check that the value is contained within the range
        #[cfg(all(debug_assertions, feature = "verify-rc-values"))]
        for (value, _) in values.iter().enumerate() {
            check_value_in_range(range_item, value);
        }

        // Update the multiplicity of the corresponding AIR
        match range_item.rc_type {
            StdRangeType::U8Air => {
                // Here, we can safely assume that value ∊ [0,2⁸-1]
                // Therefore, we can safely cast value to u8
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let vals: Vec<u8> = (0..values.len()).map(|v| v as u8).collect();
                    let rows = U8Air::get_global_rows(&vals);

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_rows(range_item.virtual_id, &rows, &values);
                } else {
                    self.u8air.as_ref().unwrap().update_inputs(values);
                }
            }
            StdRangeType::U16Air => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1]
                // Therefore, we can safely cast value to u16
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let vals: Vec<u16> = (0..values.len()).map(|v| v as u16).collect();
                    let rows = U16Air::get_global_rows(&vals);

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_rows(range_item.virtual_id, &rows, &values);
                } else {
                    self.u16air.as_ref().unwrap().update_inputs(values);
                }
            }
            StdRangeType::SpecifiedRanges => {
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let vals: Vec<i64> = (0..values.len()).map(|v| v as i64).collect();
                    let rows = SpecifiedRanges::get_global_rows(range_item.data.min, &vals);

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_rows(range_item.virtual_id, &rows, &values);
                } else {
                    self.specified_ranges_air.as_ref().unwrap().update_inputs(id, values);
                }
            }
            StdRangeType::U8AirDouble | StdRangeType::U16AirDouble => unreachable!()
        }
    }

    #[cfg(all(debug_assertions, feature = "verify-rc-values"))]
    fn check_value_in_range(range: &StdRange, value: i64) {
        let min = range.data.min;
        let max = range.data.max;
        if value < min || value > max {
            log::error!("Value {} is not in the range [min,max] = [{},{}]", value, min, max);
            panic!("Range check failed");
        }
    }
}

impl<F: PrimeField64> WitnessComponent<F> for StdRangeCheck<F> {
    fn pre_calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        Ok(())
    }
}
