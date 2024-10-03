use std::io::Read;
use std::{fs::File, sync::Arc};

use proofman_common::{initialize_logger, ExecutionCtx, ProofCtx, SetupCtx, WitnessPilout};
use proofman::{WitnessLibrary, WitnessManager};
use pil_std_lib::{RCAirData, RangeCheckAir, Std};
use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;

use std::error::Error;
use std::path::PathBuf;
use crate::FibonacciSquarePublics;

use crate::{FibonacciSquare, Pilout, Module, U_8_AIR_AIRGROUP_ID, U_8_AIR_AIR_IDS};

pub struct FibonacciWitness<F: PrimeField> {
    public_inputs_path: Option<PathBuf>,
    wcm: Option<Arc<WitnessManager<F>>>,
    fibonacci: Option<Arc<FibonacciSquare<F>>>,
    module: Option<Arc<Module<F>>>,
    std_lib: Option<Arc<Std<F>>>,
}

impl<F: PrimeField> FibonacciWitness<F> {
    pub fn new(public_inputs_path: Option<PathBuf>) -> Self {
        Self { public_inputs_path, wcm: None, fibonacci: None, module: None, std_lib: None }
    }
}

impl<F: PrimeField> WitnessLibrary<F> for FibonacciWitness<F> {
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx.clone(), ectx.clone(), sctx.clone()));

        let rc_air_data = vec![RCAirData {
            air_name: RangeCheckAir::U8Air,
            airgroup_id: U_8_AIR_AIRGROUP_ID,
            air_id: U_8_AIR_AIR_IDS[0],
        }];

        let std_lib = Std::new(wcm.clone(), Some(rc_air_data));
        let module = Module::new(wcm.clone(), std_lib.clone());
        let fibonacci = FibonacciSquare::new(wcm.clone(), module.clone());

        self.wcm = Some(wcm.clone());
        self.fibonacci = Some(fibonacci);
        self.module = Some(module);
        self.std_lib = Some(std_lib);

        let public_inputs: FibonacciSquarePublics = if let Some(path) = &self.public_inputs_path {
            let mut file = File::open(path).unwrap();

            if !file.metadata().unwrap().is_file() {
                panic!("Public inputs file not found");
            }

            let mut contents = String::new();

            let _ =
                file.read_to_string(&mut contents).map_err(|err| format!("Failed to read public inputs file: {}", err));

            serde_json::from_str(&contents).unwrap()
        } else {
            FibonacciSquarePublics::default()
        };

        let pi: Vec<u8> = public_inputs.into();
        *pctx.public_inputs.inputs.write().unwrap() = pi;

        wcm.start_proof(pctx, ectx, sctx);
    }

    fn end_proof(&mut self) {
        self.wcm.as_ref().unwrap().end_proof();
    }

    fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.fibonacci.as_ref().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.module.as_ref().unwrap().execute(pctx, ectx, sctx);
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
    public_inputs_path: Option<PathBuf>,
) -> Result<Box<dyn WitnessLibrary<Goldilocks>>, Box<dyn Error>> {
    initialize_logger(ectx.verbose_mode);

    let fibonacci_witness = FibonacciWitness::new(public_inputs_path);
    Ok(Box::new(fibonacci_witness))
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
        std::fs::canonicalize(std::env::current_dir().expect("Failed to get current directory").join("../../"))
            .expect("Failed to canonicalize root path")
    }

    fn get_witness_lib_path() -> PathBuf {
        get_root_path().join("target/debug/libfibonacci_square.so")
    }

    fn get_proving_key_path() -> PathBuf {
        get_root_path().join("examples/fibonacci-square/build/provingKey")
    }

    #[test]
    fn test_verify_constraints() {
        let verify_constraints = VerifyConstraintsCmd {
            witness_lib: get_witness_lib_path(),
            rom: None,
            public_inputs: Some(get_root_path().join("examples/fibonacci-square/src/inputs.json")),
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
        let proof_dir = get_root_path().join("examples/fibonacci-square/build/proofs");

        if proof_dir.exists() {
            std::fs::remove_dir_all(&proof_dir).expect("Failed to remove proof directory");
        }

        let gen_proof = ProveCmd {
            witness_lib: get_witness_lib_path(),
            rom: None,
            public_inputs: Some(get_root_path().join("examples/fibonacci-square/src/inputs.json")),
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
