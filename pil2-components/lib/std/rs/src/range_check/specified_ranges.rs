use core::panic;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use rayon::prelude::*;

use num_traits::ToPrimitive;
use p3_field::PrimeField;

use witness::WitnessComponent;
use proofman_common::{AirInstance, ProofCtx, SetupCtx, TraceInfo};
use proofman_hints::{get_hint_field_constant, get_hint_ids_by_name, HintFieldOptions, HintFieldValue};
use proofman_util::create_buffer_fast;

use crate::{AirComponent, Range};

pub struct SpecifiedRanges<F: PrimeField> {
    // Parameters
    airgroup_id: usize,
    air_id: usize,
    instance_id: AtomicU64,
    num_rows: usize,
    num_cols: usize,
    ranges: Vec<Range<F>>,
    multiplicities: Vec<Vec<AtomicU64>>,
    calculated: AtomicBool,
}

impl<F: PrimeField> AirComponent<F> for SpecifiedRanges<F> {
    const MY_NAME: &'static str = "SpecRang";

    fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, airgroup_id: Option<usize>, air_id: Option<usize>) -> Arc<Self> {
        let airgroup_id = airgroup_id.expect("Airgroup ID must be provided");
        let air_id = air_id.expect("Air ID must be provided");

        let setup = sctx.get_setup(airgroup_id, air_id);
        let specified_hints = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_ranges");
        let mut ranges = Vec::new();
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;
        let num_cols = if specified_hints.is_empty() { 0 } else { specified_hints.len() - 1 };
        let mut multiplicities = Vec::new();

        if !specified_hints.is_empty() {
            for hint in specified_hints[1..].iter() {
                let predefined = get_hint_field_constant(
                    sctx,
                    airgroup_id,
                    air_id,
                    *hint as usize,
                    "predefined",
                    HintFieldOptions::default(),
                );
                let min = get_hint_field_constant(
                    sctx,
                    airgroup_id,
                    air_id,
                    *hint as usize,
                    "min",
                    HintFieldOptions::default(),
                );
                let min_neg = get_hint_field_constant(
                    sctx,
                    airgroup_id,
                    air_id,
                    *hint as usize,
                    "min_neg",
                    HintFieldOptions::default(),
                );
                let max = get_hint_field_constant(
                    sctx,
                    airgroup_id,
                    air_id,
                    *hint as usize,
                    "max",
                    HintFieldOptions::default(),
                );
                let max_neg = get_hint_field_constant(
                    sctx,
                    airgroup_id,
                    air_id,
                    *hint as usize,
                    "max_neg",
                    HintFieldOptions::default(),
                );

                let HintFieldValue::Field(predefined) = predefined else {
                    log::error!("Predefined hint must be a field element");
                    panic!();
                };
                let predefined = {
                    if !predefined.is_zero() && !predefined.is_one() {
                        log::error!("Predefined hint must be either 0 or 1");
                        panic!();
                    }
                    predefined.is_one()
                };
                let HintFieldValue::Field(min) = min else {
                    log::error!("Min hint must be a field element");
                    panic!();
                };
                let min_neg = match min_neg {
                    HintFieldValue::Field(value) => {
                        if value.is_zero() {
                            false
                        } else if value.is_one() {
                            true
                        } else {
                            log::error!("Predefined hint must be either 0 or 1");
                            panic!("Invalid predefined hint value"); // Or return Err if you prefer error handling
                        }
                    }
                    _ => {
                        log::error!("Max_neg hint must be a field element");
                        panic!("Invalid hint type"); // Or return Err if you prefer error handling
                    }
                };
                let HintFieldValue::Field(max) = max else {
                    log::error!("Max hint must be a field element");
                    panic!();
                };
                let max_neg = match max_neg {
                    HintFieldValue::Field(value) => {
                        if value.is_zero() {
                            false
                        } else if value.is_one() {
                            true
                        } else {
                            log::error!("Predefined hint must be either 0 or 1");
                            panic!("Invalid predefined hint value"); // Or return Err if you prefer error handling
                        }
                    }
                    _ => {
                        log::error!("Max_neg hint must be a field element");
                        panic!("Invalid hint type"); // Or return Err if you prefer error handling
                    }
                };

                ranges.push(Range(min, max, min_neg, max_neg, predefined));
            }

            multiplicities = (0..num_cols)
                .into_par_iter()
                .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
                .collect();
        }

        Arc::new(Self {
            airgroup_id,
            air_id,
            instance_id: AtomicU64::new(0),
            num_cols,
            num_rows,
            ranges,
            multiplicities,
            calculated: AtomicBool::new(false),
        })
    }
}

impl<F: PrimeField> SpecifiedRanges<F> {
    #[inline(always)]
    pub fn update_inputs(&self, value: F, range: Range<F>, multiplicity: F) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }
        let val = (value - range.0)
            .as_canonical_biguint()
            .to_usize()
            .unwrap_or_else(|| panic!("Cannot convert to usize: {:?}", value));

        let range_index =
            self.ranges.iter().position(|r| *r == range).unwrap_or_else(|| panic!("Range {:?} not found", range));

        let index = val % self.num_rows;
        let mul = multiplicity.as_canonical_biguint();
        self.multiplicities[range_index][index].fetch_add(mul.to_u64().unwrap(), Ordering::Relaxed);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField> WitnessComponent<F> for SpecifiedRanges<F> {
    fn execute(&self, pctx: Arc<ProofCtx<F>>) -> Vec<usize> {
        let (instance_found, mut instance_id) = pctx.dctx_find_instance(self.airgroup_id, self.air_id);

        if !instance_found {
            instance_id = pctx.add_instance_all(self.airgroup_id, self.air_id);
        }

        self.instance_id.store(instance_id as u64, Ordering::SeqCst);
        Vec::new()
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, _instance_ids: &[usize]) {
        if stage == 1 {
            let instance_id = self.instance_id.load(Ordering::Relaxed) as usize;

            if !_instance_ids.contains(&instance_id) {
                return;
            }

            self.calculated.store(true, Ordering::Relaxed);

            pctx.dctx_distribute_multiplicities(&self.multiplicities, instance_id);

            if pctx.dctx_is_my_instance(instance_id) {
                let buffer_size = self.num_cols * self.num_rows;
                let mut buffer = create_buffer_fast(buffer_size);
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, vec) in self.multiplicities.iter().enumerate() {
                        chunk[col] = F::from_u64(vec[row].load(Ordering::Relaxed));
                    }
                });

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
