use pil_std_lib::Std;
use proofman_common::ProofmanResult;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use fields::PrimeField64;
use fields::Goldilocks;

use crate::{Connection1, Connection2, ConnectionNew};
use proofman::register_std;

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std = Std::new(wcm.get_pctx(), wcm.get_sctx(), false)?;
        let connection1 = Connection1::new();
        let connection2 = Connection2::new();
        let connection_new = ConnectionNew::new();

        register_std(wcm, &std);
        wcm.register_component(connection1.clone());
        wcm.register_component(connection2.clone());
        wcm.register_component(connection_new.clone());
        Ok(())
    }
}
