use std::{cell::OnceCell, error::Error, path::PathBuf, sync::Arc};

use pil_std_lib::Std;
use proofman::{WitnessLibrary, WitnessManager};
use proofman_common::{initialize_logger, ProofCtx, SetupCtx, VerboseMode};

use p3_field::PrimeField;
use p3_goldilocks::Goldilocks;
use rand::{distributions::Standard, prelude::Distribution};

use crate::{
    RangeCheckMix, RangeCheckDynamic1, RangeCheckDynamic2, MultiRangeCheck1, MultiRangeCheck2, RangeCheck1,
    RangeCheck2, RangeCheck3, RangeCheck4,
};

pub struct RangeCheckWitness<F: PrimeField> {
    pub wcm: OnceCell<Arc<WitnessManager<F>>>,
    pub range_check1: OnceCell<Arc<RangeCheck1<F>>>,
    pub range_check2: OnceCell<Arc<RangeCheck2<F>>>,
    pub range_check3: OnceCell<Arc<RangeCheck3<F>>>,
    pub range_check4: OnceCell<Arc<RangeCheck4<F>>>,
    pub multi_range_check1: OnceCell<Arc<MultiRangeCheck1<F>>>,
    pub multi_range_check2: OnceCell<Arc<MultiRangeCheck2<F>>>,
    pub range_check_dynamic1: OnceCell<Arc<RangeCheckDynamic1<F>>>,
    pub range_check_dynamic2: OnceCell<Arc<RangeCheckDynamic2<F>>>,
    pub range_check_mix: OnceCell<Arc<RangeCheckMix<F>>>,
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
            range_check_dynamic1: OnceCell::new(),
            range_check_dynamic2: OnceCell::new(),
            range_check_mix: OnceCell::new(),
            std_lib: OnceCell::new(),
        }
    }

    fn initialize(&self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        let wcm = Arc::new(WitnessManager::new(pctx, sctx));

        let std_lib = Std::new(wcm.clone());
        let range_check1 = RangeCheck1::new(wcm.clone(), std_lib.clone());
        let range_check2 = RangeCheck2::new(wcm.clone(), std_lib.clone());
        let range_check3 = RangeCheck3::new(wcm.clone(), std_lib.clone());
        let range_check4 = RangeCheck4::new(wcm.clone(), std_lib.clone());
        let multi_range_check1 = MultiRangeCheck1::new(wcm.clone(), std_lib.clone());
        let multi_range_check2 = MultiRangeCheck2::new(wcm.clone(), std_lib.clone());
        let range_check_dynamic1 = RangeCheckDynamic1::new(wcm.clone(), std_lib.clone());
        let range_check_dynamic2 = RangeCheckDynamic2::new(wcm.clone(), std_lib.clone());
        let range_check_mix = RangeCheckMix::new(wcm.clone(), std_lib.clone());

        let _ = self.wcm.set(wcm);
        let _ = self.range_check1.set(range_check1);
        let _ = self.range_check2.set(range_check2);
        let _ = self.range_check3.set(range_check3);
        let _ = self.range_check4.set(range_check4);
        let _ = self.multi_range_check1.set(multi_range_check1);
        let _ = self.multi_range_check2.set(multi_range_check2);
        let _ = self.range_check_dynamic1.set(range_check_dynamic1);
        let _ = self.range_check_dynamic2.set(range_check_dynamic2);
        let _ = self.range_check_mix.set(range_check_mix);
        let _ = self.std_lib.set(std_lib);
    }
}

impl<F: PrimeField> WitnessLibrary<F> for RangeCheckWitness<F>
where
    Standard: Distribution<F>,
{
    fn start_proof(&mut self, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        self.initialize(pctx.clone(), sctx.clone());

        self.wcm.get().unwrap().start_proof(pctx, sctx);
    }

    fn end_proof(&mut self) {
        self.wcm.get().unwrap().end_proof();
    }

    fn execute(&self, pctx: Arc<ProofCtx<F>>) {
        // Execute those components that need to be executed
        self.range_check1.get().unwrap().execute(pctx.clone());
        self.range_check2.get().unwrap().execute(pctx.clone());
        self.range_check3.get().unwrap().execute(pctx.clone());
        self.range_check4.get().unwrap().execute(pctx.clone());
        self.multi_range_check1.get().unwrap().execute(pctx.clone());
        self.multi_range_check2.get().unwrap().execute(pctx.clone());
        self.range_check_dynamic1.get().unwrap().execute(pctx.clone());
        self.range_check_dynamic2.get().unwrap().execute(pctx.clone());
        self.range_check_mix.get().unwrap().execute(pctx.clone());
    }

    fn calculate_witness(&mut self, stage: u32, pctx: Arc<ProofCtx<F>>, sctx: Arc<SetupCtx>) {
        self.wcm.get().unwrap().calculate_witness(stage, pctx, sctx);
    }
}

#[no_mangle]
pub extern "Rust" fn init_library(
    _: Option<PathBuf>,
    _: Option<PathBuf>,
    verbose_mode: VerboseMode,
) -> Result<Box<dyn WitnessLibrary<Goldilocks>>, Box<dyn Error>> {
    initialize_logger(verbose_mode);
    let range_check_witness = RangeCheckWitness::new();
    Ok(Box::new(range_check_witness))
}

// #[cfg(test)]
// mod tests {
//     use proofman_cli::commands::verify_constraints::{Field, VerifyConstraintsCmd};

//     #[test]
//     fn test_verify_constraints() {
//         let root_path = std::env::current_dir().expect("Failed to get current directory").join("../../../../");
//         let root_path = std::fs::canonicalize(root_path).expect("Failed to canonicalize root path");

//         let verify_constraints = VerifyConstraintsCmd {
//             witness_lib: root_path.join("target/debug/librange_check.so"),
//             rom: None,
//             public_inputs: None,
//             proving_key: root_path.join("test/std/range_check/build/provingKey"),
//             field: Field::Goldilocks,
//             verbose: 0,
//         };

//         if let Err(e) = verify_constraints.run() {
//             eprintln!("Failed to verify constraints: {:?}", e);
//             std::process::exit(1);
//         }
//     }
// }
