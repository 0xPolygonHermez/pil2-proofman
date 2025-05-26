use core::panic;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use rayon::prelude::*;

use p3_field::PrimeField64;
use std::path::PathBuf;

use witness::WitnessComponent;
use proofman_common::{AirInstance, ProofCtx, SetupCtx, TraceInfo};
use proofman_hints::{get_hint_field_constant_a, get_hint_ids_by_name, HintFieldOptions, HintFieldValue};
use proofman_util::create_buffer_fast;

use crate::{get_hint_field_constant_as_field, get_hint_field_constant_as_u64, validate_binary_field, AirComponent};

#[derive(Debug, Clone)]
pub struct SpecifiedRange {
    id: usize,
    mul_idx: usize,
    min: i64,
}

pub struct SpecifiedRanges {
    // Parameters
    airgroup_id: usize,
    air_id: usize,
    instance_id: AtomicU64,
    num_rows: usize,
    num_cols: usize,
    ranges: Vec<SpecifiedRange>,
    multiplicities: Vec<Vec<AtomicU64>>,
    calculated: AtomicBool,
}

impl<F: PrimeField64> AirComponent<F> for SpecifiedRanges {
    const MY_NAME: &'static str = "SpecRang";

    fn new(pctx: &ProofCtx<F>, sctx: &SetupCtx<F>, airgroup_id: Option<usize>, air_id: Option<usize>) -> Arc<Self> {
        let airgroup_id = airgroup_id.expect("Airgroup ID must be provided");
        let air_id = air_id.expect("Air ID must be provided");
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        let setup = sctx.get_setup(airgroup_id, air_id);

        // Get the relevant data
        let specified_data_hint =
            get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_ranges_data")[0] as usize;
        let col_num = get_hint_field_constant_as_u64::<F>(
            sctx,
            airgroup_id,
            air_id,
            specified_data_hint,
            "col_num",
            HintFieldOptions::default(),
        );

        let opids_count = get_hint_field_constant_as_u64::<F>(
            sctx,
            airgroup_id,
            air_id,
            specified_data_hint,
            "opids_count",
            HintFieldOptions::default(),
        );

        let opids = get_hint_field_constant_a::<F>(
            sctx,
            airgroup_id,
            air_id,
            specified_data_hint,
            "opids",
            HintFieldOptions::default(),
        )
        .values;

        let opids_len = get_hint_field_constant_a::<F>(
            sctx,
            airgroup_id,
            air_id,
            specified_data_hint,
            "opids_len",
            HintFieldOptions::default(),
        )
        .values;

        // Get and store the ranges
        let specified_hints = get_hint_ids_by_name(setup.p_setup.p_expressions_bin, "specified_range");
        let mut ranges = Vec::new();
        let mut offset = 0;
        for i in 0..opids_count as usize {
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
                let idx = offset + j;
                let hint = specified_hints[idx];

                let min = get_hint_field_constant_as_u64::<F>(
                    sctx,
                    airgroup_id,
                    air_id,
                    hint as usize,
                    "min",
                    HintFieldOptions::default(),
                );

                let min_neg = get_hint_field_constant_as_field::<F>(
                    sctx,
                    airgroup_id,
                    air_id,
                    hint as usize,
                    "min_neg",
                    HintFieldOptions::default(),
                );
                let min_neg = validate_binary_field(min_neg, "Min neg");

                let min = if min_neg { min as i128 - F::ORDER_U64 as i128 } else { min as i128 };

                // In this conversion we assume that min is at most of 63 bits
                // We can safely assume it because we have already check this minimum before
                ranges.push(SpecifiedRange { id: opid, mul_idx: idx, min: min as i64 });
            }

            offset += opid_len;
        }

        let num_cols = col_num as usize;
        let multiplicities = (0..num_cols)
            .into_par_iter()
            .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
            .collect();

        Arc::new(Self {
            airgroup_id,
            air_id,
            num_cols,
            num_rows,
            ranges,
            multiplicities,
            instance_id: AtomicU64::new(0),
            calculated: AtomicBool::new(false),
        })
    }
}

impl SpecifiedRanges {
    #[inline(always)]
    pub fn update_inputs(&self, id: usize, value: i64, multiplicity: u64) {
        if self.calculated.load(Ordering::Relaxed) {
            return;
        }
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
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        let (instance_found, mut instance_id) = pctx.dctx_find_instance(self.airgroup_id, self.air_id);

        if !instance_found {
            instance_id = pctx.add_instance_all(self.airgroup_id, self.air_id);
        }

        self.calculated.store(false, Ordering::Relaxed);
        self.instance_id.store(instance_id as u64, Ordering::SeqCst);
        Vec::new()
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _core_id: usize,
        _n_cores: usize,
    ) {
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
                        chunk[col] = F::from_u64(vec[row].swap(0, Ordering::Relaxed));
                    }
                });

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
