use std::{fmt::Debug, sync::Arc};

use proofman_fields::PrimeField64;

use proofman_witness::WitnessComponent;
use proofman_common::{BufferPool, ProofCtx, ProofmanError, ProofmanResult, SetupCtx};
use proofman_hints::{
    get_hint_field_constant, get_hint_field_gc_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue,
};

use crate::{
    extract_field_element_as_usize, get_global_hint_field_constant_as, get_hint_field_constant_as,
    get_hint_field_constant_as_field, validate_binary_field, AirComponent, RCMultiplicity, RCValue, SpecifiedRanges,
    StdVirtualTable, U16Air, U8Air,
};

pub struct StdRangeCheck<F: PrimeField64> {
    _phantom: std::marker::PhantomData<F>,
    ranges: Vec<StdRange>,
    pub u8air: Option<Arc<U8Air<F>>>,
    pub u16air: Option<Arc<U16Air<F>>>,
    pub specified_ranges_air: Option<Arc<SpecifiedRanges<F>>>,
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
        let u8air = Self::create_air::<U8Air<F>>(&pctx, sctx, shared_tables, &u8air_hint)?;
        let u16air = Self::create_air::<U16Air<F>>(&pctx, sctx, shared_tables, &u16air_hint)?;
        let specified_ranges_air =
            Self::create_air::<SpecifiedRanges<F>>(&pctx, sctx, shared_tables, &specified_ranges_air_hint)?;

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
                &pctx,
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
    #[allow(clippy::too_many_arguments)]
    fn register_ranges(
        sctx: &SetupCtx<F>,
        pctx: &ProofCtx<F>,
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
            let hint_data = Self::parse_range_hint(sctx, pctx, airgroup_id, air_id, hint)?;

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

    fn parse_range_hint(
        sctx: &SetupCtx<F>,
        pctx: &ProofCtx<F>,
        airgroup_id: usize,
        air_id: usize,
        hint: u64,
    ) -> ProofmanResult<HintCache> {
        let options = HintFieldOptions::default();

        let setup = sctx.get_setup(airgroup_id, air_id)?;

        let opid = get_hint_field_constant_as::<u64, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint as usize,
            "opid",
            options.clone(),
        )?;

        let predefined = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "predefined",
                options.clone(),
            )?,
            "Predefined",
        )?;

        let min_val = get_hint_field_constant_as::<u64, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint as usize,
            "min",
            options.clone(),
        )?;
        let min_neg = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "min_neg",
                options.clone(),
            )?,
            "Min neg",
        )?;

        let max_val = get_hint_field_constant_as::<u64, F>(
            pctx,
            setup,
            airgroup_id,
            air_id,
            hint as usize,
            "max",
            options.clone(),
        )?;
        let max_neg = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                pctx,
                setup,
                airgroup_id,
                air_id,
                hint as usize,
                "max_neg",
                options.clone(),
            )?,
            "Max neg",
        )?;

        let HintFieldValue::String(rc_type_str) =
            get_hint_field_constant::<F>(pctx, setup, airgroup_id, air_id, hint as usize, "type", options.clone())?
        else {
            return Err(ProofmanError::StdError("Type hint must be a string".to_string()));
        };

        let is_virtual = validate_binary_field(
            get_hint_field_constant_as_field::<F>(
                pctx,
                setup,
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
        #[cfg(debug_assertions)]
        Self::check_value_in_range(range_item, value);

        // Update the multiplicity of the corresponding AIR
        match range_item.rc_type {
            StdRangeType::U8Air => {
                // Here, we can safely assume that value ∊ [0,2⁸-1]
                // Therefore, we can safely cast value to u8
                let value = value as u8;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let row = U8Air::<F>::get_global_row(value);

                    // Increment the virtual row
                    self.virtual_table.inc_virtual_row(range_item.virtual_id, row, multiplicity);
                } else {
                    self.u8air.as_ref().unwrap().update_value(value, multiplicity);
                }
            }
            StdRangeType::U16Air => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1]
                // Therefore, we can safely cast value to u16
                let value = value as u16;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let row = U16Air::<F>::get_global_row(value);

                    // Increment the virtual row
                    self.virtual_table.inc_virtual_row(range_item.virtual_id, row, multiplicity);
                } else {
                    self.u16air.as_ref().unwrap().update_value(value, multiplicity);
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
                    let rows = [U8Air::<F>::get_global_row(lower_value), U8Air::<F>::get_global_row(upper_value)];

                    // Increment the virtual row
                    self.virtual_table.inc_virtual_rows_same_mul(range_item.virtual_id, &rows, multiplicity);
                } else {
                    let u8_air = self.u8air.as_ref().unwrap();
                    u8_air.update_pairs([lower_value, upper_value].into_iter().map(|v| (v, multiplicity)));
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
                    let rows = [U16Air::<F>::get_global_row(lower_value), U16Air::<F>::get_global_row(upper_value)];

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_rows_same_mul(range_item.virtual_id, &rows, multiplicity);
                } else {
                    let u16_air = self.u16air.as_ref().unwrap();
                    u16_air.update_pairs([lower_value, upper_value].into_iter().map(|v| (v, multiplicity)));
                }
            }
            StdRangeType::SpecifiedRanges => {
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let row = SpecifiedRanges::<F>::get_global_row(range_item.data.min, value);

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_row(range_item.virtual_id, row, multiplicity);
                } else {
                    let specified_ranges_air = self.specified_ranges_air.as_ref().unwrap();
                    specified_ranges_air.update_value(id, value, multiplicity);
                }
            }
        }
    }

    /// Increments the multiplicities for value/multiplicity pairs.
    ///
    /// Generic over both widths so the caller's slices are consumed as-is: each element is
    /// widened inside the iterator chain, so nothing is materialized on the way down.
    pub fn assign_values<V: RCValue, M: RCMultiplicity>(&self, id: usize, values: &[V], multiplicities: &[M]) {
        // Find the range with the given id
        let range_item = &self.ranges[id];

        // Check that the value is contained within the range
        #[cfg(debug_assertions)]
        {
            assert_eq!(values.len(), multiplicities.len(), "Rows and multiplicities must have the same length");

            for &value in values {
                Self::check_value_in_range(range_item, value.to_i64());
            }
        }

        let pairs =
            || values.iter().copied().zip(multiplicities.iter().copied()).map(|(v, m)| (v.to_i64(), m.to_u64()));

        // Update the multiplicity of the corresponding AIR
        match range_item.rc_type {
            StdRangeType::U8Air => {
                // Here, we can safely assume that value ∊ [0,2⁸-1]
                // Therefore, we can safely cast value to u8
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = pairs().map(|(v, m)| (U8Air::<F>::get_global_row(v as u8), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.u8air.as_ref().unwrap().update_pairs(pairs().map(|(v, m)| (v as u8, m)));
                }
            }
            StdRangeType::U16Air => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1]
                // Therefore, we can safely cast value to u16
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = pairs().map(|(v, m)| (U16Air::<F>::get_global_row(v as u16), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.u16air.as_ref().unwrap().update_pairs(pairs().map(|(v, m)| (v as u16, m)));
                }
            }
            StdRangeType::U8AirDouble => {
                // Here, we can safely assume that value ∊ [0,2⁸-1], min >= 0 and max <= 2⁸-1
                // Therefore, we can safely cast value to u8
                let range_data = &range_item.data;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let lower_rows = pairs().map(|(v, m)| (U8Air::<F>::get_global_row((v - range_data.min) as u8), m));
                    let upper_rows = pairs().map(|(v, m)| (U8Air::<F>::get_global_row((range_data.max - v) as u8), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, lower_rows);
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, upper_rows);
                } else {
                    let u8_air = self.u8air.as_ref().unwrap();
                    u8_air.update_pairs(pairs().map(|(v, m)| ((v - range_data.min) as u8, m)));
                    u8_air.update_pairs(pairs().map(|(v, m)| ((range_data.max - v) as u8, m)));
                }
            }
            StdRangeType::U16AirDouble => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1], min >= 0 and max <= 2¹⁶-1
                // Therefore, we can safely cast value to u16
                let range_data = &range_item.data;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let lower_rows =
                        pairs().map(|(v, m)| (U16Air::<F>::get_global_row((v - range_data.min) as u16), m));
                    let upper_rows =
                        pairs().map(|(v, m)| (U16Air::<F>::get_global_row((range_data.max - v) as u16), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, lower_rows);
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, upper_rows);
                } else {
                    let u16_air = self.u16air.as_ref().unwrap();
                    u16_air.update_pairs(pairs().map(|(v, m)| ((v - range_data.min) as u16, m)));
                    u16_air.update_pairs(pairs().map(|(v, m)| ((range_data.max - v) as u16, m)));
                }
            }
            StdRangeType::SpecifiedRanges => {
                let range_min = range_item.data.min;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = pairs().map(|(v, m)| (SpecifiedRanges::<F>::get_global_row(range_min, v), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.specified_ranges_air.as_ref().unwrap().update_pairs(id, pairs());
                }
            }
        }
    }

    /// Increments the multiplicities of several values by the same amount.
    pub fn assign_values_same_mul<V: RCValue>(&self, id: usize, values: &[V], multiplicity: u64) {
        // Find the range with the given id
        let range_item = &self.ranges[id];

        // Check that all values are contained within the range
        #[cfg(debug_assertions)]
        {
            for &value in values {
                Self::check_value_in_range(range_item, value.to_i64());
            }
        }

        let pairs = || values.iter().copied().map(move |v| (v.to_i64(), multiplicity));

        // Update the multiplicity of the corresponding AIR
        match range_item.rc_type {
            StdRangeType::U8Air => {
                // Here, we can safely assume that value ∊ [0,2⁸-1]
                // Therefore, we can safely cast value to u8
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = pairs().map(|(v, m)| (U8Air::<F>::get_global_row(v as u8), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.u8air.as_ref().unwrap().update_pairs(pairs().map(|(v, m)| (v as u8, m)));
                }
            }
            StdRangeType::U16Air => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1]
                // Therefore, we can safely cast value to u16
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = pairs().map(|(v, m)| (U16Air::<F>::get_global_row(v as u16), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.u16air.as_ref().unwrap().update_pairs(pairs().map(|(v, m)| (v as u16, m)));
                }
            }
            StdRangeType::U8AirDouble => {
                // Here, we can safely assume that value ∊ [0,2⁸-1], min >= 0 and max <= 2⁸-1
                // Therefore, we can safely cast value to u8
                let range_data = &range_item.data;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let lower_rows = pairs().map(|(v, m)| (U8Air::<F>::get_global_row((v - range_data.min) as u8), m));
                    let upper_rows = pairs().map(|(v, m)| (U8Air::<F>::get_global_row((range_data.max - v) as u8), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, lower_rows);
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, upper_rows);
                } else {
                    let u8_air = self.u8air.as_ref().unwrap();
                    u8_air.update_pairs(pairs().map(|(v, m)| ((v - range_data.min) as u8, m)));
                    u8_air.update_pairs(pairs().map(|(v, m)| ((range_data.max - v) as u8, m)));
                }
            }
            StdRangeType::U16AirDouble => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1], min >= 0 and max <= 2¹⁶-1
                // Therefore, we can safely cast value to u16
                let range_data = &range_item.data;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let lower_rows =
                        pairs().map(|(v, m)| (U16Air::<F>::get_global_row((v - range_data.min) as u16), m));
                    let upper_rows =
                        pairs().map(|(v, m)| (U16Air::<F>::get_global_row((range_data.max - v) as u16), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, lower_rows);
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, upper_rows);
                } else {
                    let u16_air = self.u16air.as_ref().unwrap();
                    u16_air.update_pairs(pairs().map(|(v, m)| ((v - range_data.min) as u16, m)));
                    u16_air.update_pairs(pairs().map(|(v, m)| ((range_data.max - v) as u16, m)));
                }
            }
            StdRangeType::SpecifiedRanges => {
                let range_min = range_item.data.min;
                if range_item.is_virtual {
                    // Get the rows corresponding to the values
                    let rows = pairs().map(|(v, m)| (SpecifiedRanges::<F>::get_global_row(range_min, v), m));

                    // Increment the virtual rows
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.specified_ranges_air.as_ref().unwrap().update_pairs(id, pairs());
                }
            }
        }
    }

    /// Increments the multiplicities of the consecutive values `[start, start + muls.len())`.
    pub fn assign_values_ranged<M: RCMultiplicity>(&self, id: usize, start: Option<i64>, multiplicities: &[M]) {
        // Find the range with the given id
        let range_item = &self.ranges[id];

        // Initialize the starting point
        let start = start.unwrap_or(range_item.data.min);

        // Check that the value is contained within the range
        #[cfg(debug_assertions)]
        {
            let end = start + multiplicities.len() as i64 - 1;
            let min = range_item.data.min;
            let max = range_item.data.max;
            if start < min {
                panic!("Range check failed: Start value {} is less than the range minimum {}", start, min);
            }
            if end > max {
                panic!(
                    "Range check failed: End value {} (start {} + count {} - 1) exceeds the range maximum {}",
                    end,
                    start,
                    multiplicities.len(),
                    max
                );
            }
        }

        // Values come from a synthetic range — pass them straight through as iterator
        // pairs, no intermediate Vec allocation. (For U8/U16, get_global_row is a no-op
        // cast `v as u64`, so the virtual path's "row" is just the synthetic value
        // cast to u64.)
        let pairs = || multiplicities.iter().copied().enumerate().map(move |(i, m)| (start + i as i64, m.to_u64()));

        match range_item.rc_type {
            StdRangeType::U8Air => {
                // Here, we can safely assume that value ∊ [0,2⁸-1]
                // Therefore, we can safely cast value to u8
                if range_item.is_virtual {
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, pairs().map(|(v, m)| (v as u64, m)));
                } else {
                    self.u8air.as_ref().unwrap().update_pairs(pairs().map(|(v, m)| (v as u8, m)));
                }
            }
            StdRangeType::U16Air => {
                // Here, we can safely assume that value ∊ [0,2¹⁶-1]
                // Therefore, we can safely cast value to u16
                if range_item.is_virtual {
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, pairs().map(|(v, m)| (v as u64, m)));
                } else {
                    self.u16air.as_ref().unwrap().update_pairs(pairs().map(|(v, m)| (v as u16, m)));
                }
            }
            StdRangeType::U8AirDouble => {
                // Here, we can safely assume that (val - min) ∊ [0,2⁸-1] and (max - val) ∊ [0,2⁸-1]
                // (max - min <= 2⁸-1 by construction). Therefore, we can safely cast both to u8.
                let range_data = &range_item.data;
                if range_item.is_virtual {
                    let lower_rows = pairs().map(|(v, m)| ((v - range_data.min) as u64, m));
                    let upper_rows = pairs().map(|(v, m)| ((range_data.max - v) as u64, m));
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, lower_rows);
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, upper_rows);
                } else {
                    let air = self.u8air.as_ref().unwrap();
                    air.update_pairs(pairs().map(|(v, m)| ((v - range_data.min) as u8, m)));
                    air.update_pairs(pairs().map(|(v, m)| ((range_data.max - v) as u8, m)));
                }
            }
            StdRangeType::U16AirDouble => {
                // Here, we can safely assume that (val - min) ∊ [0,2¹⁶-1] and (max - val) ∊ [0,2¹⁶-1]
                // (max - min <= 2¹⁶-1 by construction). Therefore, we can safely cast both to u16.
                let range_data = &range_item.data;
                if range_item.is_virtual {
                    let lower_rows = pairs().map(|(v, m)| ((v - range_data.min) as u64, m));
                    let upper_rows = pairs().map(|(v, m)| ((range_data.max - v) as u64, m));
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, lower_rows);
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, upper_rows);
                } else {
                    let air = self.u16air.as_ref().unwrap();
                    air.update_pairs(pairs().map(|(v, m)| ((v - range_data.min) as u16, m)));
                    air.update_pairs(pairs().map(|(v, m)| ((range_data.max - v) as u16, m)));
                }
            }
            StdRangeType::SpecifiedRanges => {
                let range_min = range_item.data.min;
                if range_item.is_virtual {
                    let rows = pairs().map(|(v, m)| ((v - range_min) as u64, m));
                    self.virtual_table.inc_virtual_pairs(range_item.virtual_id, rows);
                } else {
                    self.specified_ranges_air.as_ref().unwrap().update_pairs(id, pairs());
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    fn check_value_in_range(range: &StdRange, value: i64) {
        let min = range.data.min;
        let max = range.data.max;
        if value < min || value > max {
            panic!("Range check failed: Value {} is not in the range [min,max] = [{},{}]", value, min, max);
        }
    }
}

#[cfg(test)]
impl<F: PrimeField64> StdRangeCheck<F> {
    fn for_test(
        ranges: Vec<StdRange>,
        u8air: Option<Arc<U8Air<F>>>,
        u16air: Option<Arc<U16Air<F>>>,
        specified_ranges_air: Option<Arc<SpecifiedRanges<F>>>,
        virtual_table: Arc<StdVirtualTable<F>>,
    ) -> Self {
        Self { _phantom: std::marker::PhantomData, ranges, u8air, u16air, specified_ranges_air, virtual_table }
    }
}

impl<F: PrimeField64 + Send + Sync + 'static> WitnessComponent<F> for StdRangeCheck<F> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::Goldilocks as F;

    const U8_ROWS: usize = 64;
    const U8_COLS: usize = 256 / U8_ROWS;
    const U16_ROWS: usize = 1024;
    const U16_COLS: usize = 65536 / U16_ROWS;
    const SR_ROWS: usize = 64;
    const SR_COLS: usize = 2;

    /// One range-check holding a single range, wired to freshly zeroed tables.
    struct Fixture {
        rc: StdRangeCheck<F>,
        u8air: Option<Arc<U8Air<F>>>,
        u16air: Option<Arc<U16Air<F>>>,
        sr: Option<Arc<SpecifiedRanges<F>>>,
        vt: Arc<StdVirtualTable<F>>,
    }

    impl Fixture {
        fn new(rc_type: StdRangeType, is_virtual: bool, min: i64, max: i64) -> Self {
            // The virtual table must be able to hold whichever rows this range emits.
            let (vt_rows, vt_cols) = match rc_type {
                StdRangeType::U16Air | StdRangeType::U16AirDouble => (U16_ROWS, U16_COLS),
                _ => (U8_ROWS, U8_COLS),
            };
            let vt = StdVirtualTable::<F>::for_test(vt_rows, vt_cols, vec![(0, 0)]);

            let u8air = matches!(rc_type, StdRangeType::U8Air | StdRangeType::U8AirDouble)
                .then(|| U8Air::<F>::for_test(U8_ROWS));
            let u16air = matches!(rc_type, StdRangeType::U16Air | StdRangeType::U16AirDouble)
                .then(|| U16Air::<F>::for_test(U16_ROWS));
            let sr = matches!(rc_type, StdRangeType::SpecifiedRanges)
                .then(|| SpecifiedRanges::<F>::for_test(SR_ROWS, &[(min, 0)], SR_COLS));

            let range =
                StdRange { rc_type, is_virtual, virtual_id: 0, data: RangeData { min, max, predefined: false } };
            let rc = StdRangeCheck::for_test(vec![range], u8air.clone(), u16air.clone(), sr.clone(), vt.clone());
            Fixture { rc, u8air, u16air, sr, vt }
        }

        /// Every table this fixture owns, concatenated in a fixed order, so a comparison
        /// covers the whole observable state rather than one guessed-at table.
        fn snapshot(&self) -> Vec<u64> {
            let mut out = Vec::new();
            for part in [
                self.u8air.as_ref().map(|a| a.snapshot()),
                self.u16air.as_ref().map(|a| a.snapshot()),
                self.sr.as_ref().map(|a| a.snapshot()),
                Some(self.vt.snapshot()),
            ]
            .into_iter()
            .flatten()
            {
                out.extend(part);
            }
            out
        }
    }

    /// Guards every differential test against passing on two all-zero snapshots.
    fn assert_wrote_something(snapshot: &[u64], what: &str) {
        assert!(snapshot.iter().any(|&m| m != 0), "{what} wrote nothing — the comparison would be vacuous");
    }

    /// Every rc_type, both virtual and not, with a [min,max] each one can actually serve.
    fn scenarios() -> Vec<(StdRangeType, i64, i64)> {
        vec![
            (StdRangeType::U8Air, 0, 255),
            (StdRangeType::U16Air, 0, 65535),
            (StdRangeType::U8AirDouble, 20, 200),
            (StdRangeType::U16AirDouble, 1000, 50000),
            (StdRangeType::SpecifiedRanges, 100, 199),
        ]
    }

    fn sample_values(min: i64, max: i64) -> Vec<i64> {
        // Endpoints, a few interior points, and a repeat so multiplicities accumulate.
        let mid = min + (max - min) / 2;
        vec![min, max, mid, min + 1, max - 1, mid, min]
    }

    /// The batch API must land exactly where N single-value calls would.
    #[test]
    fn assign_values_matches_repeated_assign_value() {
        for (rc_type, min, max) in scenarios() {
            for is_virtual in [false, true] {
                let vals = sample_values(min, max);
                let muls: Vec<u32> = (0..vals.len() as u32).map(|i| i + 1).collect();

                let reference = Fixture::new(rc_type.clone(), is_virtual, min, max);
                for (&v, &m) in vals.iter().zip(muls.iter()) {
                    reference.rc.assign_value(0, v, m as u64);
                }

                let batched = Fixture::new(rc_type.clone(), is_virtual, min, max);
                batched.rc.assign_values(0, &vals, &muls);

                let expected = reference.snapshot();
                assert_wrote_something(&expected, &format!("assign_values {rc_type:?} (virtual: {is_virtual})"));
                assert_eq!(
                    expected,
                    batched.snapshot(),
                    "assign_values diverged for {rc_type:?} (virtual: {is_virtual})"
                );
            }
        }
    }

    #[test]
    fn assign_values_same_mul_matches_repeated_assign_value() {
        for (rc_type, min, max) in scenarios() {
            for is_virtual in [false, true] {
                let vals = sample_values(min, max);
                let mul = 7u64;

                let reference = Fixture::new(rc_type.clone(), is_virtual, min, max);
                for &v in vals.iter() {
                    reference.rc.assign_value(0, v, mul);
                }

                let batched = Fixture::new(rc_type.clone(), is_virtual, min, max);
                batched.rc.assign_values_same_mul(0, &vals, mul);

                let expected = reference.snapshot();
                assert_wrote_something(&expected, &format!("same_mul {rc_type:?} (virtual: {is_virtual})"));
                assert_eq!(
                    expected,
                    batched.snapshot(),
                    "assign_values_same_mul diverged for {rc_type:?} (virtual: {is_virtual})"
                );
            }
        }
    }

    #[test]
    fn assign_values_ranged_matches_repeated_assign_value() {
        for (rc_type, min, max) in scenarios() {
            for is_virtual in [false, true] {
                // A window inside the range, so `start` is exercised as Some(..).
                let start = min + 3;
                let muls: Vec<u32> = (0..17u32).map(|i| i % 5).collect();
                assert!(start + muls.len() as i64 - 1 <= max);

                let reference = Fixture::new(rc_type.clone(), is_virtual, min, max);
                for (i, &m) in muls.iter().enumerate() {
                    reference.rc.assign_value(0, start + i as i64, m as u64);
                }

                let batched = Fixture::new(rc_type.clone(), is_virtual, min, max);
                batched.rc.assign_values_ranged(0, Some(start), &muls);

                let expected = reference.snapshot();
                assert_wrote_something(&expected, &format!("ranged {rc_type:?} (virtual: {is_virtual})"));
                assert_eq!(
                    expected,
                    batched.snapshot(),
                    "assign_values_ranged diverged for {rc_type:?} (virtual: {is_virtual})"
                );
            }
        }
    }

    /// `start: None` must mean "from the range minimum".
    #[test]
    fn assign_values_ranged_defaults_start_to_range_min() {
        for (rc_type, min, max) in scenarios() {
            for is_virtual in [false, true] {
                let muls: Vec<u32> = (1..9u32).collect();

                let explicit = Fixture::new(rc_type.clone(), is_virtual, min, max);
                explicit.rc.assign_values_ranged(0, Some(min), &muls);

                let defaulted = Fixture::new(rc_type.clone(), is_virtual, min, max);
                defaulted.rc.assign_values_ranged(0, None, &muls);

                let expected = explicit.snapshot();
                assert_wrote_something(&expected, &format!("ranged-default {rc_type:?} (virtual: {is_virtual})"));
                assert_eq!(
                    expected,
                    defaulted.snapshot(),
                    "ranged default start diverged for {rc_type:?} (virtual: {is_virtual})"
                );
            }
        }
    }

    /// The multiplicity width must not change the result: u16/u32/u64/usize all widen the same.
    #[test]
    fn multiplicity_width_is_irrelevant() {
        for (rc_type, min, max) in scenarios() {
            for is_virtual in [false, true] {
                let vals = sample_values(min, max);
                let m64: Vec<u64> = (1..=vals.len() as u64).collect();
                let m16: Vec<u16> = m64.iter().map(|&m| m as u16).collect();
                let m32: Vec<u32> = m64.iter().map(|&m| m as u32).collect();
                let msz: Vec<usize> = m64.iter().map(|&m| m as usize).collect();

                let wide = Fixture::new(rc_type.clone(), is_virtual, min, max);
                wide.rc.assign_values(0, &vals, &m64);
                let expected = wide.snapshot();
                assert_wrote_something(&expected, &format!("width baseline {rc_type:?}"));

                for (name, actual) in [
                    ("u16", {
                        let f = Fixture::new(rc_type.clone(), is_virtual, min, max);
                        f.rc.assign_values(0, &vals, &m16);
                        f.snapshot()
                    }),
                    ("u32", {
                        let f = Fixture::new(rc_type.clone(), is_virtual, min, max);
                        f.rc.assign_values(0, &vals, &m32);
                        f.snapshot()
                    }),
                    ("usize", {
                        let f = Fixture::new(rc_type.clone(), is_virtual, min, max);
                        f.rc.assign_values(0, &vals, &msz);
                        f.snapshot()
                    }),
                ] {
                    assert_eq!(expected, actual, "{name} multiplicities diverged for {rc_type:?}");
                }
            }
        }
    }

    /// Likewise for the value width on the way in.
    #[test]
    fn value_width_is_irrelevant() {
        let (min, max) = (0i64, 255i64);
        for is_virtual in [false, true] {
            let vals64 = sample_values(min, max);
            let vals32: Vec<u32> = vals64.iter().map(|&v| v as u32).collect();
            let vals8: Vec<u8> = vals64.iter().map(|&v| v as u8).collect();
            let muls: Vec<u32> = vec![3; vals64.len()];

            let wide = Fixture::new(StdRangeType::U8Air, is_virtual, min, max);
            wide.rc.assign_values(0, &vals64, &muls);
            let expected = wide.snapshot();
            assert_wrote_something(&expected, "value-width baseline");

            let narrow32 = Fixture::new(StdRangeType::U8Air, is_virtual, min, max);
            narrow32.rc.assign_values(0, &vals32, &muls);
            assert_eq!(expected, narrow32.snapshot(), "u32 values diverged");

            let narrow8 = Fixture::new(StdRangeType::U8Air, is_virtual, min, max);
            narrow8.rc.assign_values(0, &vals8, &muls);
            assert_eq!(expected, narrow8.snapshot(), "u8 values diverged");
        }
    }

    /// Zero multiplicities are skipped, not written.
    #[test]
    fn zero_multiplicities_are_noops() {
        for (rc_type, min, max) in scenarios() {
            for is_virtual in [false, true] {
                let vals = sample_values(min, max);
                let zeros: Vec<u32> = vec![0; vals.len()];

                let f = Fixture::new(rc_type.clone(), is_virtual, min, max);
                f.rc.assign_values(0, &vals, &zeros);
                assert!(f.snapshot().iter().all(|&m| m == 0), "{rc_type:?} wrote on zero multiplicity");
            }
        }
    }
}
