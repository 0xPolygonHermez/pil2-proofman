#[allow(unused_imports)] // generated for every air shape
mod pil_helpers;
mod component;
mod dom_first;
mod dom_last;
mod dom_not_first;
mod dom_not_last;
mod dom_odd;
mod dom_limits_16;
mod dom_not_quarter;
mod dom_mixed;
mod tamper;

pub use pil_helpers::*;

use std::sync::Arc;

use pil2_std_lib::Std;
use proofman::register_std;
use proofman_common::{load_from_json, ProofmanResult};
use proofman_fields::{Goldilocks, PrimeField64};
use proofman_witness::{witness_library, WitnessComponent, WitnessLibrary, WitnessManager};
use serde::Deserialize;

witness_library!(WitnessLib, Goldilocks);

/// The public inputs: which AIRs get an instance (all of them when absent).
#[derive(Default, Deserialize)]
struct DomainsInputs {
    airs: Option<Vec<String>>,
}

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), true)?;
        register_std(wcm, &std_lib);
        let inputs: DomainsInputs = load_from_json(&wcm.get_public_inputs_path());
        let components: [(&str, Arc<dyn WitnessComponent<F>>); 8] = [
            ("DomFirst", dom_first::DomFirst::new()),
            ("DomLast", dom_last::DomLast::new()),
            ("DomNotFirst", dom_not_first::DomNotFirst::new()),
            ("DomNotLast", dom_not_last::DomNotLast::new()),
            ("DomOdd", dom_odd::DomOdd::new()),
            ("DomLimits16", dom_limits_16::DomLimits16::new()),
            ("DomNotQuarter", dom_not_quarter::DomNotQuarter::new()),
            ("DomMixed", dom_mixed::DomMixed::new()),
        ];
        if let Some(airs) = &inputs.airs {
            if let Some(unknown) = airs.iter().find(|a| !components.iter().any(|(name, _)| name == a)) {
                panic!("inputs name an unknown air {unknown:?}");
            }
        }
        for (name, component) in components {
            if inputs.airs.as_ref().is_none_or(|airs| airs.iter().any(|a| a == name)) {
                wcm.register_component(component);
            }
        }
        Ok(())
    }
}
