use proofman::{
    executor,
    executor::Executor,
    channel::{SenderB, ReceiverB},
    message::Message,
    proof_ctx::ProofCtx,
    task::TasksTable,
    trace,
};
use math::{fields::f64::BaseElement, FieldElement};
use pilout::find_subproof_id_by_name;

use log::debug;

executor!(FibonacciExecutor: BaseElement);

impl Executor<BaseElement> for FibonacciExecutor {
    fn witness_computation(&self, stage_id: u32, proof_ctx: &ProofCtx<BaseElement>, _tasks: &TasksTable, _tx: &SenderB<Message>, _rx: &ReceiverB<Message>) {
        if stage_id != 1 {
            debug!("Nothing to do for stage_id {}", stage_id);
            return;
        }

        let subproof_id = find_subproof_id_by_name(&proof_ctx.pilout, "Fibonacci").expect("Subproof not found");
        let air_id = 0;
        let num_rows = proof_ctx.pilout.subproofs[subproof_id].airs[air_id].num_rows.unwrap() as usize;

        trace!(Fibonacci {
            a: BaseElement,
            b: BaseElement
        });
        let mut fib = Fibonacci::new(num_rows);

        fib.a[0] = BaseElement::ONE;
        fib.b[0] = BaseElement::ONE;

        for i in 1..num_rows {
            fib.a[i] = fib.b[i - 1];
            fib.b[i] = fib.a[i - 1] + fib.b[i - 1];
        }
        
        proof_ctx.add_trace_to_air_instance(subproof_id, air_id, fib).expect("Error adding trace to air instance");
    }
}