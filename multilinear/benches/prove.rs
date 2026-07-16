//! Baseline microbenchmark for `prove_air` over synthetic Fibonacci AIRs.
//!
//! Run (with the per-phase split):
//! ```
//! RUST_LOG=proofman_multilinear=debug \
//!   cargo bench -p proofman-multilinear --features testutil
//! ```

use std::time::Instant;

use proofman_multilinear::test_air::{fib_ir, fib_trace};
use proofman_multilinear::{prove_air, MlParams};

fn main() {
    let _ = env_logger::builder().format_timestamp(None).try_init();

    let sizes = [8u32, 10, 12, 16, 20, 22];
    let reps = 2;

    eprintln!("{:>4}  {:>12}", "n", "prove(ms)");
    for &n in &sizes {
        let ir = fib_ir(n, MlParams::default());
        let (witness, consts, publics) = fib_trace(n);

        // Report the best of a few reps (steadier than the mean under noise).
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let proof = prove_air(&ir, &witness, &consts, None, &[], &publics, &[], &[], &[]).expect("prove_air");
            std::hint::black_box(&proof);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("{n:>4}  {best:>12.1}");
    }
}
