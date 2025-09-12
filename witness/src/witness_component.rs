use std::sync::Arc;

use fields::PrimeField64;
use proofman_common::{BufferPool, ProofCtx, SetupCtx, DebugInfo};
use std::path::PathBuf;

pub trait WitnessComponent<F: PrimeField64>: Send + Sync {
    fn execute(&self, _pctx: Arc<ProofCtx<F>>, _input_data_path: Option<PathBuf>) -> Vec<usize> {
        Vec::new()
    }

    fn debug(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, _instance_ids: &[usize]) {}

    fn calculate_witness(
        &self,
        _stage: u32,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
    }

    fn pre_calculate_witness(
        &self,
        _stage: u32,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
        for instance_id in instance_ids {
            pctx.set_witness_ready(*instance_id, false);
        }
    }

    fn end(&self, _pctx: Arc<ProofCtx<F>>, _sctx: Arc<SetupCtx<F>>, _debug_info: &DebugInfo) {}

    fn gen_custom_commits_fixed(
        &self,
        _pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        _check: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

#[macro_export]
macro_rules! execute {
    ($Trace:ident, $num_instances: expr) => {
        fn execute(&self, pctx: Arc<ProofCtx<F>>, _input_data_path: Option<std::path::PathBuf>) -> Vec<usize> {
            let mut instance_ids = Vec::new();
            for _ in 0..$num_instances {
                instance_ids.push(pctx.add_instance($Trace::<usize>::AIRGROUP_ID, $Trace::<usize>::AIR_ID, 1));
            }
            *self.instance_ids.write().unwrap() = instance_ids.clone();
            instance_ids
        }
    };
}

#[macro_export]
macro_rules! define_wc {
    ($StructName:ident, $name:expr) => {
        use std::sync::atomic::{AtomicU64, Ordering};
        pub struct $StructName {
            instance_ids: std::sync::RwLock<Vec<usize>>,
            seed: AtomicU64,
        }

        impl $StructName {
            pub fn new() -> std::sync::Arc<Self> {
                std::sync::Arc::new(Self { instance_ids: std::sync::RwLock::new(Vec::new()), seed: AtomicU64::new(0) })
            }

            pub fn set_seed(&self, seed: u64) {
                self.seed.store(seed, Ordering::Relaxed);
            }
        }
    };
}

#[macro_export]
macro_rules! define_wc_with_std {
    ($StructName:ident, $name:expr) => {
        use pil_std_lib::Std;
        use std::sync::atomic::{AtomicU64, Ordering};
        pub struct $StructName<F: PrimeField64> {
            std_lib: Arc<Std<F>>,
            instance_ids: std::sync::RwLock<Vec<usize>>,
            seed: AtomicU64,
        }

        impl<F: PrimeField64> $StructName<F> {
            pub fn new(std_lib: Arc<Std<F>>) -> std::sync::Arc<Self> {
                std::sync::Arc::new(Self {
                    std_lib,
                    instance_ids: std::sync::RwLock::new(Vec::new()),
                    seed: AtomicU64::new(0),
                })
            }

            pub fn set_seed(&self, seed: u64) {
                self.seed.store(seed, Ordering::Relaxed);
            }
        }
    };
}
