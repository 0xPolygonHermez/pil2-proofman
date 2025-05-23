use std::sync::Arc;

use pil_std_lib::Std;
use witness::{witness_library, WitnessLibrary, WitnessManager};

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks;
use rand::distr::{StandardUniform, Distribution};

use crate::{Connection1, Connection2, ConnectionNew};

witness_library!(WitnessLib, Goldilocks);

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib
where
    StandardUniform: Distribution<F>,
{
    fn register_witness(&mut self, wcm: Arc<WitnessManager<F>>) {
        Std::new(wcm.clone());
        let connection1 = Connection1::new();
        let connection2 = Connection2::new();
        let connection_new = ConnectionNew::new();

        wcm.register_component(connection1.clone());
        wcm.register_component(connection2.clone());
        wcm.register_component(connection_new.clone());
    }
}
