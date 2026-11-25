//! The part of `pilout.globalInfo.json` that does not depend on the backend: the global proof
//! values and publics maps.

use crate::core::from_pilout::{proof_values, publics, PiloutError};
use crate::core::ir::NamedValue;
use crate::core::pilout_core::PiloutCore;

/// `proofValuesMap` (one entry per cell, no lengths) and `publicsMap`, by id; each declared value named once.
pub fn global_values_maps(core: &PiloutCore) -> Result<(Vec<NamedValue>, Vec<NamedValue>), PiloutError> {
    let pvs = proof_values(core)?.into_iter().map(|v| NamedValue { lengths: None, ..v }).collect();
    Ok((pvs, publics(core)?))
}
