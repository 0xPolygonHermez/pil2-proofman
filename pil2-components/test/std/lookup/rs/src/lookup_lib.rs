use std::{error::Error, sync::Arc};

use pil_std_lib::Std;
use proofman::{WitnessLibrary, WitnessManager};
use proofman_common::{initialize_logger, ExecutionCtx, ProofCtx, SetupCtx, WitnessPilout};

use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{Lookup0, Lookup1, Lookup2_12, Lookup2_13, Lookup2_15, Lookup3, Pilout};

pub struct LookupWitness<F: PrimeField> {
    pub wcm: Option<Arc<WitnessManager<F>>>,
    pub lookup0: Option<Arc<Lookup0<F>>>,
    pub lookup1: Option<Arc<Lookup1<F>>>,
    pub lookup2_12: Option<Arc<Lookup2_12<F>>>,
    pub lookup2_13: Option<Arc<Lookup2_13<F>>>,
    pub lookup2_15: Option<Arc<Lookup2_15<F>>>,
    pub lookup3: Option<Arc<Lookup3<F>>>,
    pub std_lib: Option<Arc<Std<F>>>,
}

impl<F: PrimeField> Default for LookupWitness<F>
where
    Standard: Distribution<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField> LookupWitness<F>
where
    Standard: Distribution<F>,
{
    pub fn new() -> Self {
        LookupWitness {
            wcm: None,
            lookup0: None,
            lookup1: None,
            lookup2_12: None,
            lookup2_13: None,
            lookup2_15: None,
            lookup3: None,
            std_lib: None,
        }
    }

    pub fn initialize(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx, ectx, sctx));

        let std_lib = Std::new(wcm.clone(), None);
        let lookup0 = Lookup0::new(wcm.clone());
        let lookup1 = Lookup1::new(wcm.clone());
        let lookup2_12 = Lookup2_12::new(wcm.clone());
        let lookup2_13 = Lookup2_13::new(wcm.clone());
        let lookup2_15 = Lookup2_15::new(wcm.clone());
        let lookup3 = Lookup3::new(wcm.clone());

        self.wcm = Some(wcm);
        self.lookup0 = Some(lookup0);
        self.lookup1 = Some(lookup1);
        self.lookup2_12 = Some(lookup2_12);
        self.lookup2_13 = Some(lookup2_13);
        self.lookup2_15 = Some(lookup2_15);
        self.lookup3 = Some(lookup3);
        self.std_lib = Some(std_lib);
    }
}

impl<F: PrimeField> WitnessLibrary<F> for LookupWitness<F>
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
        self.lookup0.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.lookup1.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.lookup2_12.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.lookup2_13.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.lookup2_15.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.lookup3.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
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

    let lookup_witness = LookupWitness::new();
    Ok(Box::new(lookup_witness))
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
        std::fs::canonicalize(std::env::current_dir().expect("Failed to get current directory").join("../../../../../"))
            .expect("Failed to canonicalize root path")
    }

    fn get_witness_lib_path() -> PathBuf {
        get_root_path().join("target/debug/liblookup.so")
    }

    fn get_proving_key_path() -> PathBuf {
        get_root_path().join("pil2-components/test/std/lookup/build/provingKey")
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
        let proof_dir = get_root_path().join("pil2-components/test/std/lookup/build/proofs");

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
