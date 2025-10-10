use std::sync::{Arc, RwLock};

use proofman_common::{BufferPool, write_custom_commit_trace, AirInstance, FromTrace, ProofCtx, SetupCtx};
use witness::WitnessComponent;
use std::path::PathBuf;
use fields::PrimeField64;

use crate::{BuildPublicValues, FibonacciSquareAirValues, FibonacciSquareRomTrace, FibonacciSquareTrace};

pub struct FibonacciSquare {
    instance_ids: RwLock<Vec<usize>>,
}

impl FibonacciSquare {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for FibonacciSquare {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, global_ids: &RwLock<Vec<usize>>, _input_data_path: Option<PathBuf>) {
        let global_id = pctx.add_instance(FibonacciSquareTrace::<F>::AIRGROUP_ID, FibonacciSquareTrace::<F>::AIR_ID, 1);
        let instance_ids = vec![global_id];
        *self.instance_ids.write().unwrap() = instance_ids.clone();
        global_ids.write().unwrap().push(global_id);
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let instance_id = instance_ids[0];

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let mut publics = BuildPublicValues::from_vec_guard(pctx.get_publics());

            let module = F::as_canonical_u64(&publics.module);
            let mut a = F::as_canonical_u64(&publics.in1);
            let mut b = F::as_canonical_u64(&publics.in2);

            let mut trace = FibonacciSquareTrace::new_from_vec_zeroes(buffer_pool.take_buffer());

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

            let mut air_values = FibonacciSquareAirValues::<F>::new();
            air_values.fibo1[0] = F::from_u64(1);
            air_values.fibo1[1] = F::from_u64(2);
            air_values.fibo3 = [F::from_u64(5), F::from_u64(5), F::from_u64(5)];

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));
            pctx.add_air_instance(air_instance, instance_id);
        }
    }

    fn gen_custom_commits_fixed(
        &self,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        check: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = vec![F::ZERO; FibonacciSquareRomTrace::<F>::ROW_SIZE * FibonacciSquareRomTrace::<F>::NUM_ROWS];
        let mut trace_rom = FibonacciSquareRomTrace::new_from_vec_zeroes(buffer);

        for i in 0..trace_rom.num_rows() {
            trace_rom[i].line = F::from_u64(3 + i as u64);
            trace_rom[i].flags = F::from_u64(2 + i as u64);
        }

        let file_name = pctx.get_custom_commits_fixed_buffer("rom", true)?;

        let setup = sctx.get_setup(trace_rom.airgroup_id(), trace_rom.air_id());
        let blowup_factor = 1 << (setup.stark_info.stark_struct.n_bits_ext - setup.stark_info.stark_struct.n_bits);
        write_custom_commit_trace::<F>(&mut trace_rom, blowup_factor, &file_name, check)?;
        Ok(())
    }

    fn debug(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, _instance_ids: &[usize]) {
        // let trace = FibonacciSquareTrace::from_vec(_pctx.get_air_instance_trace(0, 0, 0));
        // let fixed = FibonacciSquareFixed::from_vec(_sctx.get_fixed(0, 0));
        // let air_values = FibonacciSquareAirValues::from_vec(pctx.get_air_instance_air_values(0, 0, 0));
        // let airgroup_values = FibonacciSquareAirGroupValues::from_vec(pctx.get_air_instance_airgroup_values(0, 0, 0));

        // let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());
        // let proof_values = BuildProofValues::from_vec_guard(pctx.get_proof_values());

        // tracing::info!("  First row 1: {:?}", trace[1]);
        // tracing::info!("  Air values: {:?}", air_values);
        // tracing::info!("  Airgroup values: {:?}", airgroup_values);
        // tracing::info!("  Publics: {:?}", publics);
        // tracing::info!("  Proof values: {:?}", proof_values);
    }
}
