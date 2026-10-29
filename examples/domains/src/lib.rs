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

// a === 1 domain FIRST: rows past 0 are free.
crate::domain_component!(dom_first, DomFirst, DomFirstTrace, "DomFirst", [a], |t| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 1);
    }
});

// b === 0 domain LAST: b = (i + 1) mod N is 0 only on the last row.
crate::domain_component!(dom_last, DomLast, DomLastTrace, "DomLast", [b], |t| {
    let n = t.num_rows();
    for i in 0..n {
        t[i].b = F::from_u64(((i + 1) % n) as u64);
    }
});

// (a - 'a - 1)^2 === 0 domain !FIRST: a = i + 5 breaks the wrap read by row 0, which is outside !FIRST.
crate::domain_component!(dom_not_first, DomNotFirst, DomNotFirstTrace, "DomNotFirst", [a], |t| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 5);
    }
});

// a' === a + 1 domain !LAST: a = i + 5 breaks the wrap on row N-1, which is outside !LAST.
crate::domain_component!(dom_not_last, DomNotLast, DomNotLastTrace, "DomNotLast", [a], |t| {
    for i in 0..t.num_rows() {
        t[i].a = F::from_u64(i as u64 + 5);
    }
});

// t === 0 domain ODD: the even rows hold 3.
crate::domain_component!(dom_odd, DomOdd, DomOddTrace, "DomOdd", [t], |tr| {
    for i in 0..tr.num_rows() {
        tr[i].t = F::from_u64(if i % 2 == 1 { 0 } else { 3 });
    }
});

// s * s * (1 - s) === 0 domain LIMITS_16: s is boolean only on rows 0 and 15 of each 16.
crate::domain_component!(dom_limits_16, DomLimits16, DomLimits16Trace, "DomLimits16", [s], |t| {
    for i in 0..t.num_rows() {
        t[i].s = F::from_u64(match i % 16 {
            0 => 0,
            15 => 1,
            _ => 3,
        });
    }
});

// u === 7 domain !Q1 (Q1 = [1,0,0,0]): u is 7 except on rows 0 mod 4.
crate::domain_component!(dom_not_quarter, DomNotQuarter, DomNotQuarterTrace, "DomNotQuarter", [u], |t| {
    for i in 0..t.num_rows() {
        t[i].u = F::from_u64(if i % 4 == 0 { i as u64 } else { 7 });
    }
});

// x^3 === 8 on FIRST, x^3 === w on !FIRST (w[0] = 0: row 0 is excluded), e, f zero on even rows, v === x + w.
crate::domain_component!(dom_mixed, DomMixed, DomMixedTrace, "DomMixed", [x, w, e, f, v], |t| {
    for i in 0..t.num_rows() {
        let x = if i == 0 { 2 } else { i as u64 + 3 };
        let w = if i == 0 { 0 } else { x * x * x };
        t[i].x = F::from_u64(x);
        t[i].w = F::from_u64(w);
        t[i].e = F::from_u64(if i % 2 == 0 { 0 } else { 1 });
        t[i].f = F::from_u64(if i % 2 == 0 { 0 } else { 2 });
        t[i].v = F::from_u64(x + w);
    }
});
