use std::sync::{atomic::AtomicU64, Arc};

use p3_field::PrimeField64;

use proofman_util::create_buffer_fast;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use witness::WitnessComponent;
use proofman_common::{TraceInfo, AirInstance, ProofCtx, SetupCtx};
use std::sync::atomic::Ordering;

use crate::AirComponent;

const P2_16: usize = 65536;

type Min = u16;

pub struct U16Air {
    airgroup_id: usize,
    air_id: usize,

    num_rows: usize,
    num_cols: usize,
    mins: Vec<Min>,
    multiplicities: Vec<Vec<AtomicU64>>,
}

impl<F: PrimeField64> AirComponent<F> for U16Air {
    const MY_NAME: &'static str = "U16Air   ";

    fn new(
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        airgroup_id: Option<usize>,
        air_id: Option<usize>,
    ) -> Arc<Self> {
        let airgroup_id = airgroup_id.expect("Airgroup ID must be provided");
        let air_id = air_id.expect("Air ID must be provided");
        let num_rows = pctx.global_info.airs[airgroup_id][air_id].num_rows;

        // Get and store the ranges
        let num_cols: usize = (P2_16 + num_rows - 1) / num_rows;
        let mins = (0..num_cols).into_par_iter().map(|i| (i * num_rows) as Min).collect();

        let multiplicities = (0..num_cols)
            .into_par_iter()
            .map(|_| (0..num_rows).into_par_iter().map(|_| AtomicU64::new(0)).collect())
            .collect();

        Arc::new(Self { airgroup_id, air_id, num_rows, num_cols, mins, multiplicities })
    }
}

impl U16Air {
    #[inline(always)]
    pub fn update_inputs(&self, value: u16, multiplicity: u64) {
        let mins = &self.mins;

        // Identify to which sub-range the value belongs
        let range_idx = value as usize / self.num_rows;

        // Get the row index
        let min_local = mins[range_idx];
        let row_idx = (value - min_local) as usize;

        // Update the multiplicity
        self.multiplicities[range_idx][row_idx].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64> WitnessComponent<F> for U16Air {
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
