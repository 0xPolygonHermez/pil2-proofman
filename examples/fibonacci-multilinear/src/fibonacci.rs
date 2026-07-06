use std::sync::{Arc, RwLock};

use fields::PrimeField64;
use proofman_common::{
    write_custom_commit_trace, AirInstance, BufferPool, FromTrace, ProofCtx, ProofmanResult, SetupCtx, init_gpu_setup,
};
use witness::WitnessComponent;

use crate::{FibonacciTrace, BuildPublicValues, MERKLE_TREE_ARITY, FibonacciAirValues, FibonacciRomTrace};

pub struct Fibonacci {
    instance_ids: RwLock<Vec<usize>>,
}

impl Fibonacci {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { instance_ids: RwLock::new(Vec::new()) })
    }
}

impl<F: PrimeField64> WitnessComponent<F> for Fibonacci {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        let global_id = pctx.add_instance(FibonacciTrace::<F>::AIRGROUP_ID, FibonacciTrace::<F>::AIR_ID)?;
        *self.instance_ids.write().unwrap() = vec![global_id];
        global_ids.write().unwrap().push(global_id);
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let instance_id = instance_ids[0];
            tracing::debug!("··· Starting Fibonacci witness computation stage 1");

            let publics = BuildPublicValues::from_vec_guard(pctx.get_publics());

            let module = F::as_canonical_u64(&publics.module);
            let mut a = F::as_canonical_u64(&publics.in1);
            let mut b = F::as_canonical_u64(&publics.in2);

            let mut trace = FibonacciTrace::new_from_vec_zeroes(buffer_pool.take_buffer())?;

            trace[0].a = publics.in1;
            trace[0].b = publics.in2;
            for i in 1..trace.num_rows() {
                let tmp = b;
                let result = (a + b) % module;
                (a, b) = (tmp, result);

                trace[i].a = F::from_u64(a);
                trace[i].b = F::from_u64(b);
            }

            let mut air_values = FibonacciAirValues::<F>::new();
            air_values.fibo1[0] = F::from_u64(1);
            air_values.fibo1[1] = F::from_u64(2);
            air_values.fibo3 = [F::from_u64(2), F::from_u64(0), F::from_u64(0)];

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace).with_air_values(&mut air_values));
            pctx.add_air_instance(air_instance, instance_id);
        }
        Ok(())
    }

    fn gen_custom_commits_fixed(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx<F>>) -> ProofmanResult<()> {
        proofman_starks_lib_c::set_hash_family_c(&pctx.global_info.hash);

        let buffer = vec![F::ZERO; FibonacciRomTrace::<F>::ROW_SIZE * FibonacciRomTrace::<F>::NUM_ROWS];
        let mut trace_rom = FibonacciRomTrace::new_from_vec_zeroes(buffer)?;

        for i in 0..trace_rom.num_rows() {
            trace_rom[i].line = F::from_u64(3 + i as u64);
            trace_rom[i].flags = F::from_u64(2 + i as u64);
        }

        let file_name = pctx.get_custom_commits_fixed_buffer("rom", true)?;

        let setup = sctx.get_setup(trace_rom.airgroup_id(), trace_rom.air_id())?;
        let blowup_factor = 1 << (setup.stark_info.stark_struct.n_bits_ext - setup.stark_info.stark_struct.n_bits);
        init_gpu_setup(setup.stark_info.stark_struct.n_bits_ext, pctx.gpu)?;
        write_custom_commit_trace::<F>(&pctx, &mut trace_rom, blowup_factor, MERKLE_TREE_ARITY, &file_name)?;
        Ok(())
    }
}
