#[allow(unused_imports)] // generated for every air shape
mod pil_helpers;
mod component;
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

/// The public inputs: which AIRs get instances (all of them when absent), and how many each (1 when absent).
#[derive(Default, Deserialize)]
struct DomainsInputs {
    airs: Option<Vec<String>>,
    instances: Option<usize>,
}

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), true)?;
        register_std(wcm, &std_lib);
        let inputs: DomainsInputs = load_from_json(&wcm.get_public_inputs_path());
        let n = inputs.instances.unwrap_or(1);
        assert!(n > 0, "\"instances\" must be at least 1");
        let components: [(&str, Arc<dyn WitnessComponent<F>>); 9] = [
            ("DomFirst", dom_first::DomFirst::new(std_lib.clone(), n)),
            ("DomFirstTwin", dom_first_twin::DomFirstTwin::new(std_lib.clone(), n)),
            ("DomLast", dom_last::DomLast::new(std_lib.clone(), n)),
            ("DomNotFirst", dom_not_first::DomNotFirst::new(std_lib.clone(), n)),
            ("DomNotLast", dom_not_last::DomNotLast::new(std_lib.clone(), n)),
            ("DomOdd", dom_odd::DomOdd::new(std_lib.clone(), n)),
            ("DomLimits16", dom_limits_16::DomLimits16::new(std_lib.clone(), n)),
            ("DomNotQuarter", dom_not_quarter::DomNotQuarter::new(std_lib.clone(), n)),
            ("DomMixed", dom_mixed::DomMixed::new(std_lib.clone(), n)),
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

// a === 1 domain FIRST: rows past 0 are free.
crate::domain_component!(dom_first, DomFirst, DomFirstTrace, "DomFirst", [a], |t, _| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 1);
    }
    Ok(())
});

// DomFirst's twin: same PIL, same witness.
crate::domain_component!(dom_first_twin, DomFirstTwin, DomFirstTwinTrace, "DomFirstTwin", [a], |t, _| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 1);
    }
    Ok(())
});

// b === 0 domain LAST: b = (i + 1) mod N is 0 only on the last row.
crate::domain_component!(dom_last, DomLast, DomLastTrace, "DomLast", [b], |t, _| {
    let n = t.num_rows();
    for i in 0..n {
        t[i].b = F::from_u64(((i + 1) % n) as u64);
    }
    Ok(())
});

// (a - 'a - 1)^2 === 0 domain !FIRST: a = i + 5 breaks the wrap read by row 0, which is outside !FIRST.
crate::domain_component!(dom_not_first, DomNotFirst, DomNotFirstTrace, "DomNotFirst", [a], |t, _| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 5);
    }
    Ok(())
});

// a' === a + 1 domain !LAST: a = i + 5 breaks the wrap on row N-1, which is outside !LAST.
crate::domain_component!(dom_not_last, DomNotLast, DomNotLastTrace, "DomNotLast", [a], |t, _| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 5);
    }
    Ok(())
});

// t === 0 domain ODD: the even rows hold 3.
crate::domain_component!(dom_odd, DomOdd, DomOddTrace, "DomOdd", [t], |tr, _| {
    for i in 0..tr.num_rows() {
        tr[i].t = F::from_u64(if i % 2 == 1 { 0 } else { 3 });
    }
    Ok(())
});

// s * s * (1 - s) === 0 domain LIMITS_16: s is boolean only on rows 0 and 15 of each 16.
crate::domain_component!(dom_limits_16, DomLimits16, DomLimits16Trace, "DomLimits16", [s], |t, _| {
    for i in 0..t.num_rows() {
        t[i].s = F::from_u64(match i % 16 {
            0 => 0,
            15 => 1,
            _ => 3,
        });
    }
    Ok(())
});

// u === 7 domain !Q1 (Q1 = [1,0,0,0]): u is 7 except on rows 0 mod 4.
crate::domain_component!(dom_not_quarter, DomNotQuarter, DomNotQuarterTrace, "DomNotQuarter", [u], |t, _| {
    for i in 0..t.num_rows() {
        t[i].u = F::from_u64(if i % 4 == 0 { i as u64 } else { 7 });
    }
    Ok(())
});

// x^3 === 8 on FIRST, x^3 === w on !FIRST (w[0] = 0: row 0 is excluded), e, f zero on even rows, v === x + w;
// x is a byte, range-checked through the std table.
crate::domain_component!(dom_mixed, DomMixed, DomMixedTrace, "DomMixed", [x, w, e, f, v], |t, std| {
    let range = std.get_range_id(0, 255, None)?;
    for i in 0..t.num_rows() {
        let x = if i == 0 { 2 } else { (i % 250) as u64 + 3 };
        std.range_check_one(range, x);
        let w = if i == 0 { 0 } else { x * x * x };
        t[i].x = F::from_u64(x);
        t[i].w = F::from_u64(w);
        t[i].e = F::from_u64(if i % 2 == 0 { 0 } else { 1 });
        t[i].f = F::from_u64(if i % 2 == 0 { 0 } else { 2 });
        t[i].v = F::from_u64(x + w);
    }
    Ok(())
});
