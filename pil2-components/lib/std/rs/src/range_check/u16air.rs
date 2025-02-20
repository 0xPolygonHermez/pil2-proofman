use std::sync::{atomic::AtomicU64, Arc};

use p3_field::PrimeField64;

use witness::WitnessComponent;
use proofman_common::{TraceInfo, AirInstance, ProofCtx, SetupCtx};
use std::sync::atomic::Ordering;

use crate::AirComponent;

pub struct U16Air<F: PrimeField64> {
    airgroup_id: usize,
    air_id: usize,
    multiplicity: Vec<AtomicU64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField64> AirComponent<F> for U16Air<F> {
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

        Arc::new(Self {
            airgroup_id,
            air_id,
            multiplicity: (0..num_rows).map(|_| AtomicU64::new(0)).collect(),
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F: PrimeField64> U16Air<F> {
    #[inline(always)]
    pub fn update_inputs(&self, value: u16, multiplicity: u64) {
        // Get the row index
        let row_idx = value as usize;

        // Update the multiplicity
        self.multiplicity[row_idx].fetch_add(multiplicity, Ordering::Relaxed);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField64> WitnessComponent<F> for U16Air<F> {
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

            pctx.dctx_distribute_multiplicity(&self.multiplicity, instance_id);

            if pctx.dctx_is_my_instance(instance_id) {
                let buffer = self
                    .multiplicity
                    .iter()
                    .map(|x| F::from_canonical_u64(x.load(Ordering::Relaxed)))
                    .collect::<Vec<F>>();

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
