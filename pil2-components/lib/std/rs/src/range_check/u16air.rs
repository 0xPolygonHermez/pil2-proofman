use std::sync::{atomic::AtomicU64, Arc};
use num_traits::ToPrimitive;
use p3_field::{Field, PrimeField};

use witness::WitnessComponent;
use proofman_common::{TraceInfo, AirInstance, ProofCtx, SetupCtx};
use std::sync::atomic::Ordering;

use crate::AirComponent;

const NUM_ROWS: usize = 1 << 16;

pub struct U16Air<F: Field> {
    airgroup_id: usize,
    air_id: usize,
    multiplicity: Vec<AtomicU64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField> AirComponent<F> for U16Air<F> {
    const MY_NAME: &'static str = "U16Air   ";

    fn new(
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        airgroup_id: Option<usize>,
        air_id: Option<usize>,
    ) -> Arc<Self> {
        let airgroup_id = airgroup_id.expect("Airgroup ID must be provided");
        let air_id = air_id.expect("Air ID must be provided");
        Arc::new(Self {
            airgroup_id,
            air_id,
            multiplicity: (0..NUM_ROWS).map(|_| AtomicU64::new(0)).collect(),
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<F: PrimeField> U16Air<F> {
    #[inline(always)]
    pub fn update_inputs(&self, value: F, multiplicity: F) {
        let value = value.as_canonical_biguint().to_usize().expect("Cannot convert to usize");
        let index = value % NUM_ROWS;
        let mul = multiplicity.as_canonical_biguint();
        self.multiplicity[index].fetch_add(mul.to_u64().unwrap(), Ordering::Relaxed);
    }

    pub fn airgroup_id(&self) -> usize {
        self.airgroup_id
    }

    pub fn air_id(&self) -> usize {
        self.air_id
    }
}

impl<F: PrimeField> WitnessComponent<F> for U16Air<F> {
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

            let mut multiplicity_u64 =
                self.multiplicity.iter().map(|x| x.load(Ordering::Relaxed)).collect::<Vec<u64>>();

            pctx.dctx_distribute_multiplicity(&mut multiplicity_u64, instance_id);

            if pctx.dctx_is_my_instance(instance_id) {
                let buffer = multiplicity_u64.iter().map(|x| F::from_canonical_u64(*x)).collect::<Vec<F>>();

                let air_instance = AirInstance::new(TraceInfo::new(self.airgroup_id, self.air_id, buffer));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }
}
