use std::sync::Arc;

use witness::{WitnessComponent, execute, define_wc};
use proofman_common::{FromTrace, AirInstance, ProofCtx, SetupCtx};

use fields::PrimeField64;
use rand::{
    distr::{StandardUniform, Distribution},
    Rng, SeedableRng,
    rngs::StdRng,
};

use crate::ConnectionNewTrace;

define_wc!(ConnectionNew, "Connct_N");

impl<F: PrimeField64> WitnessComponent<F> for ConnectionNew
where
    StandardUniform: Distribution<F>,
{
    execute!(ConnectionNewTrace, 1);

    fn calculate_witness(&self, stage: u32, pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, instance_ids: &[usize]) {
        if stage == 1 {
            let seed = if cfg!(feature = "debug") { 0 } else { rand::rng().random::<u64>() };
            let mut rng = StdRng::seed_from_u64(seed);
            let mut trace = ConnectionNewTrace::new();
            let num_rows = trace.num_rows();

            tracing::debug!("··· Starting witness computation stage {}", 1);

            let mut frame = [0; 6];
            let mut conn_len = [0; 6];
            for i in 0..num_rows {
                for j in 0..6 {
                    trace[i].d[j] = F::ZERO;
                }

                // Start connection
                trace[i].a[0] = rng.random();
                trace[i].b[0] = rng.random();
                trace[i].c[0] = rng.random();

                // Start connection
                trace[i].a[1] = rng.random();
                trace[i].b[1] = rng.random();
                trace[i].c[1] = rng.random();
                if i == 3 + frame[1] {
                    trace[i - 1].c[1] = trace[i].c[1];
                    frame[1] += num_rows / 2;
                }

                trace[i].a[2] = F::ZERO;
                trace[i].b[2] = F::ZERO;
                trace[i].c[2] = F::ZERO;

                // TODO: Finish!
                // // Start connection
                // trace[i].a[2] = rng.random();
                // trace[i].b[2] = rng.random();
                // trace[i].c[2] = rng.random();
                // if i == 3 + frame[2] {
                //     trace[i - 1].c[2] = trace[i].c[2];

                //     trace[0 + frame[2]].c[2] = trace[i].b[2];
                //     trace[1 + frame[2]].a[2] = trace[i].b[2];
                //     conn_len[2] += 2;
                // }

                // if i == 3 + frame[2] {
                //     trace[i - 1].c[2] = trace[i].c[2];

                //     trace[0 + frame[2]].c[2] = trace[i].b[2];
                //     trace[1 + frame[2]].a[2] = trace[i].b[2];
                //     conn_len[2] += 2;
                // }

                // if conn_len[2] == 3 {
                //     frame[2] += num_rows / 4;
                //     conn_len[2] = 0;
                // }

                // Start connection
                trace[i].a[3] = rng.random();
                trace[i].b[3] = rng.random();
                trace[i].c[3] = rng.random();
                if i == 2 + frame[3] {
                    trace[i - 1].c[3] = trace[i].a[3];
                    frame[3] += num_rows / 2;
                }

                if i == 3 {
                    trace[i - 3].c[3] = trace[i].b[3];
                    trace[i - 2].a[3] = trace[i - 3].c[3];
                }

                // Start connection
                trace[i].a[4] = rng.random();
                trace[i].b[4] = rng.random();
                trace[i].c[4] = rng.random();

                if i == 2 + frame[4] {
                    trace[i - 1].d[4] = trace[i - 1].b[4];
                    trace[i - 1].a[4] = trace[i].c[4];
                    conn_len[4] += 1;
                }

                if i == 3 + frame[4] {
                    trace[i - 1].b[4] = trace[i].a[4];
                    trace[i].c[4] = trace[i - 1].b[4];
                    conn_len[4] += 1;
                }

                if conn_len[4] == 2 {
                    frame[4] += num_rows / 2;
                    conn_len[4] = 0;
                }

                // Start connection
                trace[i].a[5] = rng.random();
                trace[i].b[5] = rng.random();
                trace[i].c[5] = rng.random();
                if i == 3 + frame[5] {
                    trace[i - 1].d[5] = trace[i].d[5];
                    trace[i - 3].b[5] = trace[i].d[5];
                    conn_len[5] += 2;
                }

                if i == 8 {
                    trace[5].b[5] = trace[i].c[5];
                    trace[1].a[5] = trace[i].c[5];
                }

                if conn_len[5] == 2 {
                    frame[5] += num_rows / 2;
                    conn_len[5] = 0;
                }
            }

            let air_instance = AirInstance::new_from_trace(FromTrace::new(&mut trace));
            pctx.add_air_instance(air_instance, instance_ids[0]);
        }
    }
}
