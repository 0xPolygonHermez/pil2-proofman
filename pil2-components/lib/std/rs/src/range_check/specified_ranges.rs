use core::panic;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use rayon::prelude::*;

use p3_field::PrimeField64;

use witness::WitnessComponent;
use proofman_common::{AirInstance, ProofCtx, SetupCtx, TraceInfo};
use proofman_hints::{
    get_hint_field_constant, get_hint_field_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue,
};
use proofman_util::create_buffer_fast;

use crate::AirComponent;

use super::RangeData;

#[derive(Debug)]
pub struct SpecifiedRange {
    id: usize,
    mul_idx: usize,
    min: i64,
}

pub struct SpecifiedRanges {
    // Parameters
    airgroup_id: usize,
    air_id: usize,

    // Inputs
    num_rows: usize,
    num_cols: usize,
    ranges: Vec<SpecifiedRange>,
    multiplicities: Vec<Vec<AtomicU64>>,
}

impl<F: PrimeField64> AirComponent<F> for SpecifiedRanges {
    const MY_NAME: &'static str = "SpecRang";

    fn new(
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        airgroup_id: Option<usize>,
        air_id: Option<usize>,
    ) -> Arc<Self> {
        let airgroup_id = airgroup_id.expect("Airgroup ID must be provided");
        let air_id = air_id.expect("Air ID must be provided");

        let setup = sctx.get_setup(airgroup_id, air_id);
        let specified_data_hint = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_ranges_data");
        let specified_hints = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_range");

        let mut ranges = Vec::new();
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        if !specified_hints.is_empty() {
            let hint = specified_data_hint[0];
            let col_num = get_hint_field_constant::<F>(
                &sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "col_num",
                HintFieldOptions::default(),
            );

            let opids_count = get_hint_field_constant::<F>(
                &sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "opids_count",
                HintFieldOptions::default(),
            );

            let opids = get_hint_field_constant_a::<F>(
                &sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "opids",
                HintFieldOptions::default(),
            )
            .values;

            let opids_len = get_hint_field_constant_a::<F>(
                &sctx,
                airgroup_id,
                air_id,
                hint as usize,
                "opids_len",
                HintFieldOptions::default(),
            )
            .values;

            let HintFieldValue::Field(opids_count) = opids_count else {
                log::error!("Opid hint must be a field element");
                panic!();
            };
            let opids_count = opids_count.as_canonical_u64() as usize;

            let mut offset = 0;
            for i in 0..opids_count {
                let opid = &opids[i];
                let opid_len = &opids_len[i];

                // Convert to the correct type
                let HintFieldValue::Field(opid) = opid else {
                    log::error!("Opid hint must be a field element");
                    panic!();
                };
                let opid = opid.as_canonical_u64() as usize;

                let HintFieldValue::Field(opid_len) = opid_len else {
                    log::error!("Opid hint must be a field element");
                    panic!();
                };
                let opid_len = opid_len.as_canonical_u64() as usize;

                for j in 0..opid_len {
                    let hint = specified_hints[offset + j];

                    let idx = get_hint_field_constant::<F>(
                        &sctx,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "idx",
                        HintFieldOptions::default(),
                    );

                    let min = get_hint_field_constant::<F>(
                        &sctx,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "min",
                        HintFieldOptions::default(),
                    );

                    let min_neg = get_hint_field_constant::<F>(
                        &sctx,
                        airgroup_id,
                        air_id,
                        hint as usize,
                        "min_neg",
                        HintFieldOptions::default(),
                    );

                    let HintFieldValue::Field(idx) = idx else {
                        log::error!("Min hint must be a field element");
                        panic!();
                    };
                    let idx = idx.as_canonical_u64() as usize;

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
                                panic!("Invalid predefined hint value");
                            }
                        }
                        _ => {
                            log::error!("Max_neg hint must be a field element");
                            panic!("Invalid hint type");
                        }
                    };
                    let min = if min_neg {
                        (min.as_canonical_u64() as i128 - RangeData::<F>::ORDER) as i64
                    } else {
                        // In this conversion we assume that min is at most of 63 bits
                        min.as_canonical_u64() as i64
                    };

                    ranges.push(SpecifiedRange { id: opid, mul_idx: idx, min });
                }

                offset += opid_len;
            }

            let HintFieldValue::Field(num_cols) = col_num else {
                log::error!("Opid hint must be a field element");
                panic!();
            };
            let num_cols = num_cols.as_canonical_u64() as usize;
            let multiplicities = (0..num_cols)
                .into_par_iter()
                .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
                .collect();

            Arc::new(Self { airgroup_id, air_id, num_cols, num_rows, ranges, multiplicities })
        } else {
            panic!("No specified ranges found");
        }
    }
}

impl SpecifiedRanges {
    #[inline(always)]
    pub fn update_inputs(&self, id: usize, value: i64, multiplicity: u64) {
        // Get the ranges for the given id
        let ranges = self.ranges.iter().filter(|r| r.id == id).collect::<Vec<_>>();

        // Identify to which sub-range the value belongs
        let min_global = ranges[0].min;
        let range_idx = (value - min_global) as usize / self.num_rows;
        let range = ranges[range_idx];

        // Get the row index
        let min_local = range.min;
        let row_idx = (value - min_local) as usize;

        // Update the multiplicity
        self.multiplicities[range.mul_idx][row_idx].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64> WitnessComponent<F> for SpecifiedRanges {
    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        let (instance_found, _) = pctx.dctx_find_instance(self.airgroup_id, self.air_id);

        if !instance_found {
            pctx.dctx_add_instance_no_assign(
                self.airgroup_id,
                self.air_id,
                pctx.get_weight(self.airgroup_id, self.air_id),
            );
        }
    }

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>) {
        if stage == 1 {
            let (_, instance_id) = pctx.dctx_find_instance(self.airgroup_id, self.air_id);

            pctx.dctx_distribute_multiplicities(&self.multiplicities, instance_id);

            if pctx.dctx_is_my_instance(instance_id) {
                let buffer_size = self.num_cols * self.num_rows;
                let mut buffer = create_buffer_fast::<F>(buffer_size);
                buffer.par_chunks_mut(self.num_cols).enumerate().for_each(|(row, chunk)| {
                    for (col, vec) in self.multiplicities.iter().enumerate() {
                        chunk[col] = F::from_canonical_u64(vec[row].load(Ordering::Relaxed));
                    }
                });

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
