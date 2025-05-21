use std::sync::{Arc, RwLock};

use proofman_common::{write_custom_commit_trace, AirInstance, FromTrace, ProofCtx, SetupCtx};
use witness::WitnessComponent;
use std::path::PathBuf;
use p3_field::PrimeField64;

use crate::{BuildProofValues, BuildPublicValues, FibonacciSquareAirValues, FibonacciSquareRomTrace, FibonacciSquareTrace};

pub struct FibonacciSquare {
    instance_ids: RwLock<Vec<usize>>,
}

impl FibonacciSquare {
    const MY_NAME: &'static str = "FiboSqre";

    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for FibonacciSquare {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        let instance_ids =
            vec![pctx
                .add_instance_all(FibonacciSquareTrace::<usize>::AIRGROUP_ID, FibonacciSquareTrace::<usize>::AIR_ID)];
        *self.instance_ids.write().unwrap() = instance_ids.clone();
        instance_ids
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _core_id: usize,
        _n_cores: usize,
    ) {
        if stage == 1 {
            let instance_id = instance_ids[0];

            log::debug!("{} ··· Starting witness computation stage {}", Self::MY_NAME, 1);

            let mut publics = BuildPublicValues::from_vec_guard(pctx.get_publics());

            let module = F::as_canonical_u64(&publics.module);
            let mut a = F::as_canonical_u64(&publics.in1);
            let mut b = F::as_canonical_u64(&publics.in2);

            let mut trace = FibonacciSquareTrace::new();

            trace[0].a = F::from_u64(a);
            trace[0].b = F::from_u64(b);

            for i in 1..trace.num_rows() {
                let tmp = b;
                let result = (a.pow(2) + b.pow(2)) % module;
                (a, b) = (tmp, result);

                trace[i].a = F::from_u64(a);
                trace[i].b = F::from_u64(b);
            }

            publics.out = trace[trace.num_rows() - 1].b;

            let mut proof_values = BuildProofValues::from_vec_guard(pctx.get_proof_values());
            proof_values.value1 = F::from_u64(5);
            proof_values.value2 = F::from_u64(125);

            if pctx.dctx_is_my_instance(instance_id) {
                let mut air_values = FibonacciSquareAirValues::<F>::new();
                air_values.fibo1[0] = F::from_u64(1);
                air_values.fibo1[1] = F::from_u64(2);
                air_values.fibo3 = [F::from_u64(5), F::from_u64(5), F::from_u64(5)];

                let air_instance =
                    AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));
                pctx.add_air_instance(air_instance, instance_id);
            }
        }
    }

    fn gen_custom_commits_fixed(
        &self,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        check: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut trace_rom = FibonacciSquareRomTrace::new_zeroes();

        for i in 0..trace_rom.num_rows() {
            trace_rom[i].line = F::from_u64(3 + i as u64);
            trace_rom[i].flags = F::from_u64(2 + i as u64);
        }

        let file_name = pctx.get_custom_commits_fixed_buffer("rom")?;

        let setup = sctx.get_setup(trace_rom.airgroup_id(), trace_rom.air_id());
        let blowup_factor = 1 << (setup.stark_info.stark_struct.n_bits_ext - setup.stark_info.stark_struct.n_bits);
        write_custom_commit_trace(&mut trace_rom, blowup_factor, file_name, check)?;
        Ok(())
    }

    fn debug(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, _instance_ids: &[usize]) {
        // let trace = FibonacciSquareTrace::from_vec(_pctx.get_air_instance_trace(0, 0, 0));
        // let fixed = FibonacciSquareFixed::from_vec(_sctx.get_fixed(0, 0));
        // let air_values = FibonacciSquareAirValues::from_vec(pctx.get_air_instance_air_values(0, 0, 0));
        // let airgroup_values = FibonacciSquareAirGroupValues::from_vec(pctx.get_air_instance_airgroup_values(0, 0, 0));

        // let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());
        // let proof_values = BuildProofValues::from_vec_guard(pctx.get_proof_values());

        // log::info!("{}    First row 1: {:?}", Self::MY_NAME, trace[1]);
        // log::info!("{}    Air values: {:?}", Self::MY_NAME, air_values);
        // log::info!("{}    Airgroup values: {:?}", Self::MY_NAME, airgroup_values);
        // log::info!("{}    Publics: {:?}", Self::MY_NAME, publics);
        // log::info!("{}    Proof values: {:?}", Self::MY_NAME, proof_values);
    }
}
