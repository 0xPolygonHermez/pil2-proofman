use proofman::{
    executor::{BufferManager, Executor},
    trace, ProofCtx,
};
use proofman::executor;

use goldilocks::{Goldilocks, AbstractField};

use log::debug;

executor!(FibonacciExecutor);

impl Executor<Goldilocks> for FibonacciExecutor {
    fn witness_computation(
        &self,
        stage_id: u32,
        proof_ctx: &mut ProofCtx<Goldilocks>,
        _buffer_manager: Option<&Box<dyn BufferManager<Goldilocks>>>,
    ) {
        if stage_id != 1 {
            debug!("Nothing to do for stage_id {}", stage_id);
            return;
        }

        let subproof_id = proof_ctx.pilout.find_subproof_id_by_name("Fibonacci").expect("Subproof not found");
        let air_id = 0;
        let num_rows = proof_ctx.pilout.subproofs[subproof_id].airs[air_id].num_rows.unwrap() as usize;

        trace!(Fibonacci { a: Goldilocks, b: Goldilocks });
        let mut fib = Fibonacci::new(num_rows);

        fib.a[0] = Goldilocks::one();
        fib.b[0] = Goldilocks::one();

        for i in 1..num_rows {
            fib.a[i] = fib.b[i - 1];
            fib.b[i] = fib.a[i - 1] + fib.b[i - 1];
        }

        let mocked_buffer = vec![0u8; num_rows];
        proof_ctx.add_instance(subproof_id, air_id, mocked_buffer, fib).expect("Error adding trace to air instance");
    }
}