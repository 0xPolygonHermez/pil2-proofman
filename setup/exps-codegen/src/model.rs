//! Serde views over the slices of `*.starkinfo.json` / `*.expressionsinfo.json`
//! that the Q-expression codegen reads.
//!
//! These are intentionally a **minimal, self-contained** view, not a reuse of
//! `proofman_common::StarkInfo`. That struct lives in a crate that (via `fields`)
//! transitively links the C++/CUDA `proofman-starks-lib-c`, so depending on it
//! would drag the whole prover toolchain into this lightweight codegen crate and
//! break standalone builds. Reading our own tiny view keeps the crate to
//! serde + rayon + regex. Only the fields we traverse are typed; serde ignores
//! the rest.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StarkStruct {
    pub n_bits: u64,
    pub n_bits_ext: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CmPol {
    pub stage: u64,
    pub stage_pos: u64,
    pub dim: u64,
}

/// An entry of `airValuesMap` / `airgroupValuesMap`. Only `stage` is read
/// (absent => treated as stage 1).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueMapEntry {
    #[serde(default)]
    pub stage: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StarkInfo {
    pub opening_points: Vec<i64>,
    pub stark_struct: StarkStruct,
    pub n_constants: u64,
    pub n_stages: u64,
    pub map_sections_n: HashMap<String, u64>,
    pub cm_pols_map: Vec<CmPol>,
    pub air_values_map: Vec<ValueMapEntry>,
    #[serde(default)]
    pub airgroup_values_map: Vec<ValueMapEntry>,
    pub c_exp_id: i64,
    pub airgroup_id: i64,
    pub air_id: i64,
}

/// One source operand of a code step.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Src {
    #[serde(rename = "type")]
    pub op_type: String,
    #[serde(default)]
    pub id: Option<u64>,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
    #[serde(default)]
    pub dim: Option<u64>,
    #[serde(default)]
    pub prime: Option<i64>,
    // Note: a `Zi` operand also carries `boundaryId`, but the emitted load uses a
    // single zerofier (`aux[off_zi + row]`) and ignores it.
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Dest {
    #[serde(rename = "type")]
    pub dest_type: String,
    #[serde(default)]
    pub id: Option<u64>,
    pub dim: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Step {
    pub op: String,
    pub src: Vec<Src>,
    pub dest: Dest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExprCode {
    pub exp_id: i64,
    pub code: Vec<Step>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpressionsInfo {
    pub expressions_code: Vec<ExprCode>,
}

impl StarkInfo {
    /// blowup = 1 << (nBitsExt - nBits)
    pub fn blowup(&self) -> i64 {
        1i64 << (self.stark_struct.n_bits_ext - self.stark_struct.n_bits)
    }
}

impl Src {
    /// `int(src["value"])` — the literal of a `number` operand. Accepts a JSON
    /// number or a decimal string (pil2 emits Goldilocks constants as strings).
    pub fn number_value(&self) -> anyhow::Result<u64> {
        match &self.value {
            Some(serde_json::Value::String(s)) => Ok(s.parse::<u64>()?),
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().ok_or_else(|| anyhow::anyhow!("number operand value not a u64: {n}"))
            }
            other => Err(anyhow::anyhow!("number operand missing/invalid value: {other:?}")),
        }
    }
}
