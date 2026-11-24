//! Declarative proving-key pipeline: a list of `Step`s per circuit kind, one executor.

pub mod kinds;
pub mod steps;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use pil2_pilout::pilout::PilOut;
use pil2_stark_recurser::plonk2pil::PlonkResult;
use pil2_stark_recurser::stark2circom::CircomGenOptions;
use pil_info::PrepareOptions;
use proofman_common::hash_family::HashFamily;
use serde_json::Value;

use crate::output::witness_gen::WitnessTracker;
use crate::pil::stark_setup::KeyIdentity;
use crate::types::stark_struct::{StarkSettings, StarkStruct};

pub use steps::Step;

/// (starkinfo, verifierinfo, expressionsinfo) as JSON.
pub type PilInfoJson = (Value, Value, Value);
/// (base field, extended field) intermediate polynomials of a pil_info run.
pub type ImPolsInfo = (Vec<String>, Vec<String>);

/// External tools and include paths the steps call.
#[derive(Clone, Debug, Default)]
pub struct Tools {
    pub circom_exec: String,
    /// circom `-l` paths, in order.
    pub libs: Vec<String>,
    pub std_pil_path: String,
    pub recurser_pil_path: String,
    pub circom_helpers_dir: String,
}

/// One verifier circom the prologue writes to `circom/<file>`.
#[derive(Clone, Debug)]
pub struct VerifierInput {
    pub file: String,
    pub const_root: [String; 4],
    pub stark_info: Value,
    pub verifier_info: Value,
}

/// The data of `gen_circom_circuit` besides the template; `opts.batch_size` is taken from the ctx.
#[derive(Debug)]
pub struct CircuitInput {
    pub stark_infos: Vec<Value>,
    pub vadcop_info: Value,
    pub basic_vk: Vec<Vec<Vec<String>>>,
    pub agg_vk: Vec<Vec<String>>,
    pub publics: Vec<Value>,
    pub opts: CircomGenOptions,
}

/// State the steps hand to each other.
pub struct PipelineCtx {
    /// Holds `circom/`, `build/` and `pil/`.
    pub build_dir: PathBuf,
    pub files_dir: PathBuf,
    /// File stem inside `files_dir`.
    pub name: String,
    /// Stem of the generated circom and of the `build/` and `pil/` intermediates.
    pub circuit: String,
    /// Log prefix, e.g. `Air 'X' recursive1`.
    pub label: String,
    pub family: HashFamily,
    pub tools: Tools,
    pub witness: Option<WitnessTracker>,
    pub verifiers: Vec<VerifierInput>,
    pub circuit_input: Option<CircuitInput>,
    /// A circom file compiled as is (no generator).
    pub circom_file: Option<PathBuf>,
    /// Stem circom compiled (`build/<stem>.r1cs`), set by the prologue.
    pub compiled: Option<String>,
    pub batch_size: usize,
    pub plonk: Option<PlonkResult>,
    pub pilout: Option<Arc<PilOut>>,
    /// (airgroup, air) of the key's air inside `pilout`.
    pub air: (usize, usize),
    /// `<fixed_dir>/<name>.fixed` supplies a basic air's `.const`.
    pub fixed_dir: Option<PathBuf>,
    /// Derived into `stark_struct` at the tail from the air's row count, unless already set.
    pub settings: StarkSettings,
    pub stark_struct: Option<StarkStruct>,
    pub prepare: PrepareOptions,
    pub key: KeyIdentity,
    /// (starkinfo, verifierinfo, expressionsinfo) to reuse instead of running pil_info.
    pub reuse_pil_info: Option<PilInfoJson>,
    pub pil_info: Option<PilInfoJson>,
    /// (summary, im pols info) of the pil_info run.
    pub stats: Option<(String, ImPolsInfo)>,
    pub n_bits: Option<usize>,
    pub const_root: Option<[u64; 4]>,
}

impl PipelineCtx {
    pub fn new(build_dir: PathBuf, files_dir: PathBuf, name: &str, family: HashFamily) -> Self {
        Self {
            build_dir,
            files_dir,
            name: name.to_string(),
            circuit: name.to_string(),
            label: format!("Circuit '{name}'"),
            family,
            tools: Tools::default(),
            witness: None,
            verifiers: vec![],
            circuit_input: None,
            circom_file: None,
            compiled: None,
            batch_size: 1,
            plonk: None,
            pilout: None,
            air: (0, 0),
            fixed_dir: None,
            settings: StarkSettings::default(),
            stark_struct: None,
            prepare: PrepareOptions::default(),
            key: KeyIdentity::default(),
            reuse_pil_info: None,
            pil_info: None,
            stats: None,
            n_bits: None,
            const_root: None,
        }
    }

    pub fn hash(&self) -> &'static str {
        self.family.as_str()
    }

    pub fn file(&self, ext: &str) -> PathBuf {
        self.files_dir.join(format!("{}.{ext}", self.name))
    }

    pub fn plonk(&self) -> Result<&PlonkResult> {
        self.plonk.as_ref().ok_or_else(|| anyhow::anyhow!("{}: no plonk2pil result yet", self.circuit))
    }

    pub fn pilout(&self) -> Result<Arc<PilOut>> {
        self.pilout.clone().ok_or_else(|| anyhow::anyhow!("{}: no pilout yet", self.circuit))
    }

    /// The `(starkinfo, verifierinfo, expressionsinfo)` the tail produced or reused.
    pub fn pil_info(&self) -> Result<&PilInfoJson> {
        self.pil_info.as_ref().ok_or_else(|| anyhow::anyhow!("{}: no starkinfo yet", self.circuit))
    }
}

/// Run `steps` in order; the first error stops the run and is returned unchanged (never inspected).
pub fn run(steps: &[Step], ctx: &mut PipelineCtx) -> Result<()> {
    steps.iter().try_for_each(|s| s.run(ctx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn ctx() -> PipelineCtx {
        PipelineCtx::new("/tmp/b".into(), "/tmp/b/files".into(), "Circ", HashFamily::Blake3)
    }

    fn recording(log: &Arc<Mutex<Vec<&'static str>>>, tag: &'static str, fail: bool) -> Step {
        let log = log.clone();
        Step::Custom {
            name: tag,
            run: Box::new(move |_| {
                log.lock().unwrap().push(tag);
                if fail {
                    anyhow::bail!("{tag} failed")
                }
                Ok(())
            }),
        }
    }

    #[test]
    fn executor_runs_steps_in_order_and_stops_at_first_error() {
        let log = Arc::new(Mutex::new(vec![]));
        let steps = [
            recording(&log, "a", false),
            recording(&log, "b", false),
            recording(&log, "c", true),
            recording(&log, "d", false),
        ];
        let err = run(&steps, &mut ctx()).unwrap_err();
        assert_eq!(*log.lock().unwrap(), vec!["a", "b", "c"], "d must not run after c fails");
        assert!(err.to_string().contains("c failed"), "{err}");
    }

    #[test]
    fn executor_with_no_steps_is_ok() {
        assert!(run(&[], &mut ctx()).is_ok());
    }

    #[test]
    fn steps_see_and_update_the_shared_context() {
        let steps = [
            Step::Custom {
                name: "set",
                run: Box::new(|c| {
                    c.n_bits = Some(17);
                    Ok(())
                }),
            },
            Step::Custom {
                name: "read",
                run: Box::new(|c| {
                    anyhow::ensure!(c.n_bits == Some(17), "n_bits not handed over");
                    Ok(())
                }),
            },
        ];
        run(&steps, &mut ctx()).unwrap();
    }

    #[derive(Debug)]
    struct TestError(u32);
    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "test error {}", self.0)
        }
    }
    impl std::error::Error for TestError {}

    #[test]
    fn custom_step_error_is_propagated_unchanged() {
        let steps = [Step::Custom { name: "boom", run: Box::new(|_| Err(TestError(7).into())) }];
        let err = run(&steps, &mut ctx()).unwrap_err();
        assert_eq!(err.downcast_ref::<TestError>().map(|e| e.0), Some(7), "{err:#}");
    }

    #[test]
    fn needs_compressor_error_survives_run() {
        use crate::proving_key::recursive::NeedsCompressorError;
        let steps =
            [Step::Custom { name: "check", run: Box::new(|_| Err(NeedsCompressorError { n_bits: 18 }.into())) }];
        let err = run(&steps, &mut ctx()).unwrap_err();
        assert!(err.is::<NeedsCompressorError>(), "commands/recursive_setup.rs relies on this downcast");
    }
}
