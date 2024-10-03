use std::{error::Error, sync::Arc};

use pil_std_lib::Std;
use proofman::{WitnessLibrary, WitnessManager};
use proofman_common::{initialize_logger, ExecutionCtx, ProofCtx, SetupCtx, WitnessPilout};

use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{Pilout, SimpleLeft, SimpleRight};

pub struct SimpleWitness<F: PrimeField> {
    pub wcm: Option<Arc<WitnessManager<F>>>,
    pub simple_left: Option<Arc<SimpleLeft<F>>>,
    pub simple_right: Option<Arc<SimpleRight<F>>>,
    pub std_lib: Option<Arc<Std<F>>>,
}

impl<F: PrimeField> Default for SimpleWitness<F>
where
    Standard: Distribution<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField> SimpleWitness<F>
where
    Standard: Distribution<F>,
{
    pub fn new() -> Self {
        Self { wcm: None, simple_left: None, simple_right: None, std_lib: None }
    }

    pub fn initialize(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx, ectx, sctx));

        let std_lib = Std::new(wcm.clone(), None);
        let simple_left = SimpleLeft::new(wcm.clone());
        let simple_right = SimpleRight::new(wcm.clone());

        self.wcm = Some(wcm);
        self.std_lib = Some(std_lib);
        self.simple_left = Some(simple_left);
        self.simple_right = Some(simple_right);
    }
}

impl<F: PrimeField> WitnessLibrary<F> for SimpleWitness<F>
where
    Standard: Distribution<F>,
{
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.initialize(pctx.clone(), ectx.clone(), sctx.clone());

        self.wcm.as_ref().unwrap().start_proof(pctx, ectx, sctx);
    }

    fn end_proof(&mut self) {
        self.wcm.as_ref().unwrap().end_proof();
    }

    fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        // Execute those components that need to be executed
        self.simple_left.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.simple_right.as_ref().unwrap().execute(pctx, ectx, sctx);
    }

    fn calculate_witness(&mut self, stage: u32, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.wcm.as_ref().unwrap().calculate_witness(stage, pctx, ectx, sctx);
    }

    fn pilout(&self) -> WitnessPilout {
        Pilout::pilout()
    }
}

#[no_mangle]
pub extern "Rust" fn init_library(
    ectx: Arc<ExecutionCtx>,
    _: Option<PathBuf>,
) -> Result<Box<dyn WitnessLibrary<Goldilocks>>, Box<dyn Error>> {
    initialize_logger(ectx.verbose_mode);

    let simple_witness = SimpleWitness::new();
    Ok(Box::new(simple_witness))
}

#[cfg(test)]
mod tests {
    use proofman_cli::commands::verify_constraints::VerifyConstraintsCmd;
    use proofman_cli::commands::prove::ProveCmd;
    use proofman_cli::commands::field::Field;
    use std::path::{Path, PathBuf};
    use std::env;
    use std::fs;

    fn get_root_path() -> PathBuf {
        std::fs::canonicalize(std::env::current_dir().expect("Failed to get current directory").join("../../../../"))
            .expect("Failed to canonicalize root path")
    }

    fn get_witness_lib_path() -> PathBuf {
        get_root_path().join("target/debug/libsimple.so")
    }

    fn get_proving_key_path() -> PathBuf {
        get_root_path().join("pil2-components/test/simple/build/provingKey")
    }

    #[test]
    fn test_verify_constraints() {
        let verify_constraints = VerifyConstraintsCmd {
            witness_lib: get_witness_lib_path(),
            rom: None,
            public_inputs: None,
            proving_key: get_proving_key_path(),
            field: Field::Goldilocks,
            debug: 1,
            verbose: 0,
        };

        if let Err(e) = verify_constraints.run() {
            eprintln!("Failed to verify constraints: {:?}", e);
            std::process::exit(1);
        }
    }

    #[test]
    fn test_gen_proof() {
        let proof_dir = get_root_path().join("pil2-components/test/simple/build/proofs");

        if proof_dir.exists() {
            std::fs::remove_dir_all(&proof_dir).expect("Failed to remove proof directory");
        }

        let gen_proof = ProveCmd {
            witness_lib: get_witness_lib_path(),
            rom: None,
            public_inputs: None,
            proving_key: get_proving_key_path(),
            field: Field::Goldilocks,
            debug: true,
            aggregation: false,
            output_dir: proof_dir.clone(),
            verbose: 0,
        };

        if let Err(e) = gen_proof.run() {
            eprintln!("Failed to verify constraints: {:?}", e);
            std::process::exit(1);
        }

        let pil2_proofman_js_path = if let Ok(path) = env::var("PIL_PROOFMAN_JS") {
            // If PIL_PROOFMAN_JS is set, use its value
            fs::canonicalize(Path::new(&path).parent().unwrap_or_else(|| Path::new("."))).expect("Failed")
        } else {
            // Fallback if PIL_PROOFMAN_JS is not set
            get_root_path().join("../pil2-proofman-js")
        };

        let proof_verification = std::process::Command::new("node")
            .arg(pil2_proofman_js_path.join("src/main_verify.js"))
            .arg("-k")
            .arg(get_proving_key_path())
            .arg("-p")
            .arg(proof_dir.clone())
            .status()
            .expect("Failed to execute proof verification command");

        if !proof_verification.success() {
            eprintln!("Error: Proof verification failed.");
            std::process::exit(1);
        }
    }
}
