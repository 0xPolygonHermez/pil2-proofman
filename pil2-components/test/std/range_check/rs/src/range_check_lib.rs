use std::{cell::OnceCell, error::Error, sync::Arc};

use pil_std_lib::{RCAirData, RangeCheckAir, Std};
use proofman::{WitnessLibrary, WitnessManager};
use proofman_common::{initialize_logger, ExecutionCtx, ProofCtx, SetupCtx, WitnessPilout};

use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    MultiRangeCheck1, MultiRangeCheck2, Pilout, RangeCheck1, RangeCheck2, RangeCheck3, RangeCheck4,
    SPECIFIED_RANGES_AIRGROUP_ID, SPECIFIED_RANGES_AIR_IDS, U_16_AIR_AIRGROUP_ID, U_16_AIR_AIR_IDS,
    U_8_AIR_AIRGROUP_ID, U_8_AIR_AIR_IDS,
};

pub struct RangeCheckWitness<F: PrimeField> {
    pub wcm: OnceCell<Arc<WitnessManager<F>>>,
    pub range_check1: OnceCell<Arc<RangeCheck1<F>>>,
    pub range_check2: OnceCell<Arc<RangeCheck2<F>>>,
    pub range_check3: OnceCell<Arc<RangeCheck3<F>>>,
    pub range_check4: OnceCell<Arc<RangeCheck4<F>>>,
    pub multi_range_check1: OnceCell<Arc<MultiRangeCheck1<F>>>,
    pub multi_range_check2: OnceCell<Arc<MultiRangeCheck2<F>>>,
    pub std_lib: OnceCell<Arc<Std<F>>>,
}

impl<F: PrimeField> Default for RangeCheckWitness<F>
where
    Standard: Distribution<F>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField> RangeCheckWitness<F>
where
    Standard: Distribution<F>,
{
    pub fn new() -> Self {
        RangeCheckWitness {
            wcm: OnceCell::new(),
            range_check1: OnceCell::new(),
            range_check2: OnceCell::new(),
            range_check3: OnceCell::new(),
            range_check4: OnceCell::new(),
            multi_range_check1: OnceCell::new(),
            multi_range_check2: OnceCell::new(),
            std_lib: OnceCell::new(),
        }
    }

    fn initialize(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx, ectx, sctx));

        // TODO: Ad macro data into RCAIRData: SpecifiedRanges0Trace.
        // In fact, I only need to pass the length of mul of Specified...
        // Anyways, this solution would be very very specific
        let rc_air_data = vec![
            RCAirData { air_name: RangeCheckAir::U8Air, airgroup_id: U_8_AIR_AIRGROUP_ID, air_id: U_8_AIR_AIR_IDS[0] },
            RCAirData {
                air_name: RangeCheckAir::U16Air,
                airgroup_id: U_16_AIR_AIRGROUP_ID,
                air_id: U_16_AIR_AIR_IDS[0],
            },
            RCAirData {
                air_name: RangeCheckAir::SpecifiedRanges,
                airgroup_id: SPECIFIED_RANGES_AIRGROUP_ID,
                air_id: SPECIFIED_RANGES_AIR_IDS[0],
            },
        ];

        let std_lib = Std::new(wcm.clone(), Some(rc_air_data));
        let range_check1 = RangeCheck1::new(wcm.clone(), std_lib.clone());
        let range_check2 = RangeCheck2::new(wcm.clone(), std_lib.clone());
        let range_check3 = RangeCheck3::new(wcm.clone(), std_lib.clone());
        let range_check4 = RangeCheck4::new(wcm.clone(), std_lib.clone());
        let multi_range_check1 = MultiRangeCheck1::new(wcm.clone(), std_lib.clone());
        let multi_range_check2 = MultiRangeCheck2::new(wcm.clone(), std_lib.clone());

        let _ = self.wcm.set(wcm);
        let _ = self.range_check1.set(range_check1);
        let _ = self.range_check2.set(range_check2);
        let _ = self.range_check3.set(range_check3);
        let _ = self.range_check4.set(range_check4);
        let _ = self.multi_range_check1.set(multi_range_check1);
        let _ = self.multi_range_check2.set(multi_range_check2);
        let _ = self.std_lib.set(std_lib);
    }
}

impl<F: PrimeField> WitnessLibrary<F> for RangeCheckWitness<F>
where
    Standard: Distribution<F>,
{
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.initialize(pctx.clone(), ectx.clone(), sctx.clone());

        self.wcm.get().unwrap().start_proof(pctx, ectx, sctx);
    }

    fn end_proof(&mut self) {
        self.wcm.get().unwrap().end_proof();
    }

    fn execute(&self, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        // Execute those components that need to be executed
        self.range_check1.get().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.range_check2.get().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.range_check3.get().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.range_check4.get().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.multi_range_check1.get().unwrap().execute(pctx.clone(), ectx.clone(), sctx.clone());
        self.multi_range_check2.get().unwrap().execute(pctx, ectx, sctx);
    }

    fn calculate_witness(&mut self, stage: u32, pctx: Arc<ProofCtx<F>>, ectx: Arc<ExecutionCtx>, sctx: Arc<SetupCtx>) {
        self.wcm.get().unwrap().calculate_witness(stage, pctx, ectx, sctx);
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

    let range_check_witness = RangeCheckWitness::new();
    Ok(Box::new(range_check_witness))
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
        get_root_path().join("target/debug/librange_check.so")
    }

    fn get_proving_key_path() -> PathBuf {
        get_root_path().join("pil2-components/test/std/range_check/build/provingKey")
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
        let proof_dir = get_root_path().join("pil2-components/test/std/range_check/build/proofs");

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
