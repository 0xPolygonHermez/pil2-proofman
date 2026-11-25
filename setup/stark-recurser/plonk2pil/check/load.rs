//! Adapters from the artefacts as the setup holds them to the checker's inputs (no file I/O).

use std::collections::BTreeMap;

use num_bigint::BigUint;
use pil_info::core::ir::ConstraintSystem;

use super::fixed::FixedSource;
use super::{CheckError, Template};
use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::{FixedPol, FixedValue};

/// The template a plonk2pil PIL requires and the parameters it instantiates it with.
pub fn template(pil: &str) -> Result<Template, CheckError> {
    let bad = |why: &str| CheckError::Artefact(format!("pilout: PIL text: {why}"));
    let file = pil
        .lines()
        .find_map(|l| l.trim().strip_prefix("require \"")?.strip_suffix(".pil\";").map(str::to_string))
        .ok_or_else(|| bad("no required template"))?;
    let insts: Vec<&str> = pil.lines().filter(|l| l.contains(" alias ") && l.contains('(')).collect();
    let [inst] = insts[..] else { return Err(bad(&format!("{} instantiations, not one", insts.len()))) };
    let args = inst.split_once('(').and_then(|(_, r)| r.rsplit_once(')')).ok_or_else(|| bad("no arguments"))?.0;
    let mut params = BTreeMap::new();
    for a in args.split(',').filter(|a| !a.trim().is_empty()) {
        let (k, v) = a.split_once(':').ok_or_else(|| bad("argument without a value"))?;
        let v = v.trim();
        let v = match v.split_once("**") {
            Some((b, e)) => {
                b.trim().parse::<u64>().ok().zip(e.trim().parse::<u32>().ok()).and_then(|(b, e)| b.checked_pow(e))
            }
            None => v.parse().ok(),
        };
        params.insert(k.trim().to_string(), v.ok_or_else(|| bad("a non-integer argument"))?);
    }
    Ok(Template { file, params })
}

/// An external column's values by (name, index).
pub type External<'a, F> = Box<dyn Fn(&str, u32) -> Option<Vec<F>> + 'a>;

/// The pilout's fixed columns: inline values (big-endian bytes) or, for an external one, `external`.
pub struct PiloutFixed<'a, F> {
    pub cs: &'a ConstraintSystem,
    pub cols: &'a [pil2_pilout::pilout::FixedCol],
    /// An external column's values by (name, index).
    pub external: External<'a, F>,
}

impl<F: PlonkField> FixedSource<F> for PiloutFixed<'_, F> {
    fn column(&self, id: u32) -> Result<Vec<F>, CheckError> {
        let meta = self
            .cs
            .fixed
            .get(id as usize)
            .ok_or_else(|| CheckError::Artefact(format!("fixed: no fixed column {id}")))?;
        let col = self
            .cols
            .get(id as usize)
            .ok_or_else(|| CheckError::Artefact(format!("fixed: no values of fixed column {id}")))?;
        if col.values.is_empty() {
            let name = meta.name.rsplit('.').next().unwrap_or(&meta.name);
            let idx = meta.lengths.as_ref().and_then(|l| l.first().copied()).unwrap_or(0);
            return (self.external)(name, idx)
                .ok_or_else(|| CheckError::Artefact(format!("fixed: external column {name}[{idx}] not supplied")));
        }
        Ok(col.values.iter().map(|b| super::pil::fe::<F>(&BigUint::from_bytes_be(b))).collect())
    }
}

/// plonk2pil's columns by (name without the airgroup, index), each converted by `conv`.
pub fn plonk_columns<'a, V, F>(pols: &'a [FixedPol<V>], conv: impl Fn(&V) -> F + 'a) -> External<'a, F> {
    Box::new(move |name, idx| {
        pols.iter()
            .find(|p| p.name.rsplit('.').next() == Some(name) && p.index == idx as usize)
            .map(|p| p.values.iter().map(&conv).collect())
    })
}

/// Checks plonk2pil's result `res` of the r1cs `r1cs_bytes` against the pilout its PIL compiled to.
pub fn check_result<V: FixedValue>(
    r1cs_bytes: &[u8],
    res: &crate::plonk2pil::PlonkResult<V>,
    pilout: &pil2_pilout::pilout::PilOut,
) -> anyhow::Result<super::CheckReport> {
    use crate::plonk2pil::field::R1csPrime;
    use pil_info::cfg::{FieldCfg, Opening};
    let r1cs = crate::plonk2pil::r1cs::types::read_r1cs_from_bytes::<V::Field>(r1cs_bytes)?;
    let (cfg, opening) = match <V::Field as PlonkField>::PRIME {
        R1csPrime::Goldilocks => (FieldCfg::goldilocks(), Opening::Deep),
        R1csPrime::Bn128 => (FieldCfg::bn128(), Opening::Shplonk),
    };
    let cs = pil_info::core::from_pilout::from_pilout(pilout, 0, 0, &cfg, opening)?;
    let exec = proofman_common::exec_format::ExecFile::<V::Field>::from_words(&res.exec)?;
    let template = template(&res.pil_str)?;
    let air = pilout
        .air_groups
        .first()
        .and_then(|g| g.airs.first())
        .ok_or_else(|| anyhow::anyhow!("the pilout has no air"))?;
    let fixed =
        PiloutFixed { cs: &cs, cols: &air.fixed_cols, external: plonk_columns(&res.fixed_pols, |v: &V| v.to_field()) };
    Ok(super::check_plonk(&super::CheckInput {
        r1cs: &r1cs,
        cs: &cs,
        fixed: &fixed,
        exec: &exec,
        template: &template,
    })?)
}
