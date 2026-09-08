//! Reading the parts of `StarkInfo` the SHPLONK setup is derived from.
//!
//! pil-fflonk took its `f_i` packing from a pil1 shkey produced by pil-stark.
//! pil2 has no such file: the same information has to be derived from the AIR's
//! `StarkInfo`, which is what this module reads.
//!
//! Only the fields the derivation needs are modelled. `pil2-stark-setup` has a
//! fuller `StarkInfo`, but depending on that crate would create a cycle once it
//! grows a `setup_pilfflonk` command (see the crate docs), and the fields below
//! are a small, stable subset.
//!
//! Two differences from pil-fflonk's shkey are worth stating up front, because
//! they are where a naive translation goes wrong:
//!
//! * **Opening points are signed here.** pil2 writes `[-1, 0, 1]`, meaning
//!   previous, current and next row; pil-fflonk's shkey used unsigned indices.
//! * **`qDim` is 3 for Goldilocks AIRs.** The quotient lives in the cubic
//!   extension there. Over BN254 it is 1, so an AIR carrying `qDim = 3` was
//!   compiled for the wrong field.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// A committed or constant polynomial's placement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolMap {
    pub stage: u32,
    pub name: String,
    pub dim: u32,
    #[serde(rename = "polsMapId")]
    pub pols_map_id: u64,
    #[serde(rename = "stageId")]
    pub stage_id: u64,
    /// Column within the stage's buffer. Absent for constant polynomials, which
    /// are addressed by `polsMapId` instead.
    #[serde(rename = "stagePos", default)]
    pub stage_pos: Option<u64>,
}

/// One entry of the evaluation map: a polynomial opened at a point.
///
/// `prime` is the row offset (-1 previous, 0 current, 1 next) and `opening_pos`
/// indexes `StarkInfo::opening_points`. Together these say which polynomial is
/// opened where, which is what determines each `f_i`'s opening set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvMap {
    #[serde(rename = "type")]
    pub ev_type: String,
    pub id: u64,
    pub prime: i64,
    #[serde(rename = "openingPos")]
    pub opening_pos: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkStruct {
    #[serde(rename = "nBits")]
    pub n_bits: u32,
    #[serde(rename = "nBitsExt")]
    pub n_bits_ext: u32,
}

/// The subset of `StarkInfo` the SHPLONK setup derivation consumes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StarkInfo {
    pub name: String,
    #[serde(rename = "airId")]
    pub air_id: u64,
    #[serde(rename = "airgroupId")]
    pub airgroup_id: u64,

    #[serde(rename = "nStages")]
    pub n_stages: u32,
    #[serde(rename = "nConstants")]
    pub n_constants: u64,
    #[serde(rename = "nPublics")]
    pub n_publics: u64,

    /// Signed row offsets: -1 previous, 0 current, 1 next.
    #[serde(rename = "openingPoints")]
    pub opening_points: Vec<i64>,

    #[serde(rename = "qDeg")]
    pub q_deg: u64,
    #[serde(rename = "qDim")]
    pub q_dim: u32,

    /// Columns per section, keyed `const`, `cm1`, `cm2`, ...
    #[serde(rename = "mapSectionsN")]
    pub map_sections_n: BTreeMap<String, u64>,

    #[serde(rename = "cmPolsMap")]
    pub cm_pols_map: Vec<PolMap>,
    #[serde(rename = "constPolsMap")]
    pub const_pols_map: Vec<PolMap>,

    #[serde(rename = "evMap")]
    pub ev_map: Vec<EvMap>,

    #[serde(rename = "starkStruct")]
    pub stark_struct: StarkStruct,
}

impl StarkInfo {
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("parsing StarkInfo")
    }

    /// The stage that carries the quotient polynomial.
    ///
    /// Committed stages run 1..=n_stages; the quotient is committed one stage
    /// later, which is why `mapSectionsN` has a `cm{n_stages + 1}` section.
    pub fn quotient_stage(&self) -> u32 {
        self.n_stages + 1
    }

    /// Columns in a committed stage's buffer.
    pub fn stage_width(&self, stage: u32) -> u64 {
        self.map_sections_n.get(&format!("cm{stage}")).copied().unwrap_or(0)
    }

    /// Committed polynomials belonging to a stage, in column order.
    pub fn pols_in_stage(&self, stage: u32) -> Vec<&PolMap> {
        let mut v: Vec<&PolMap> = self.cm_pols_map.iter().filter(|p| p.stage == stage).collect();
        v.sort_by_key(|p| p.stage_pos.unwrap_or(p.pols_map_id));
        v
    }

    /// Distinct opening points a polynomial is evaluated at, as row offsets.
    ///
    /// Read from `evMap` rather than assumed: a polynomial need not be opened
    /// at every point the AIR declares.
    pub fn opening_offsets_for(&self, ev_type: &str, id: u64) -> Vec<i64> {
        let mut offsets: Vec<i64> =
            self.ev_map.iter().filter(|e| e.ev_type == ev_type && e.id == id).map(|e| e.prime).collect();
        offsets.sort_unstable();
        offsets.dedup();
        offsets
    }

    /// Check the AIR can be proven with SHPLONK over BN254.
    ///
    /// Rejects rather than adapts: an AIR compiled for Goldilocks puts its
    /// quotient and challenges in a cubic extension, and silently treating
    /// those as dim 1 would misread every buffer.
    pub fn ensure_bn254_compatible(&self) -> Result<()> {
        if self.q_dim != 1 {
            bail!(
                "AIR {:?} has qDim {}, so its quotient lives in an extension field; BN254 expressions are dim 1 \
                 throughout, meaning this AIR was compiled for Goldilocks",
                self.name,
                self.q_dim
            );
        }
        if let Some(p) = self.cm_pols_map.iter().find(|p| p.dim != 1) {
            bail!("committed polynomial {:?} has dim {}, but BN254 AIRs are dim 1 throughout", p.name, p.dim);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RANGE_CHECK_1: &str = include_str!("../tests/fixtures/RangeCheck1.starkinfo.json");

    fn fixture() -> StarkInfo {
        StarkInfo::from_json(RANGE_CHECK_1).expect("fixture parses")
    }

    #[test]
    fn parses_a_real_stark_info() {
        let si = fixture();

        assert_eq!(si.name, "RangeCheck1");
        assert_eq!(si.n_stages, 2);
        assert_eq!(si.n_constants, 1);
        assert_eq!(si.q_deg, 2);
        assert_eq!(si.stark_struct.n_bits, 3);
        assert_eq!(si.stark_struct.n_bits_ext, 4);
    }

    /// pil2 opening points are signed row offsets, unlike pil-fflonk's shkey.
    #[test]
    fn opening_points_are_signed_row_offsets() {
        let si = fixture();
        assert_eq!(si.opening_points, vec![-1, 0, 1]);
        assert!(si.opening_points.contains(&-1), "a previous-row opening must survive parsing");
    }

    #[test]
    fn groups_committed_polynomials_by_stage() {
        let si = fixture();

        // 8 in stage 1, 3 in stage 2, 2 in the quotient stage.
        assert_eq!(si.pols_in_stage(1).len(), 8);
        assert_eq!(si.pols_in_stage(2).len(), 3);
        assert_eq!(si.pols_in_stage(si.quotient_stage()).len(), 2);

        // Widths come from mapSectionsN and must cover the polynomials placed
        // in each stage.
        assert_eq!(si.stage_width(1), 8);
        assert_eq!(si.stage_width(2), 9);
    }

    #[test]
    fn stage_polynomials_are_returned_in_column_order() {
        let si = fixture();
        let pols = si.pols_in_stage(1);
        let positions: Vec<u64> = pols.iter().map(|p| p.stage_pos.unwrap()).collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(positions, sorted, "stage polynomials must come back ordered by column");
    }

    /// Opening sets are read from evMap, not assumed uniform: a polynomial need
    /// not be opened at every point the AIR declares.
    #[test]
    fn opening_offsets_come_from_the_evaluation_map() {
        let si = fixture();

        let mut any_partial = false;
        for pol in &si.cm_pols_map {
            let offsets = si.opening_offsets_for("cm", pol.pols_map_id);
            assert!(offsets.iter().all(|o| si.opening_points.contains(o)), "offset outside the declared points");
            if !offsets.is_empty() && offsets.len() < si.opening_points.len() {
                any_partial = true;
            }
        }
        assert!(any_partial, "expected at least one polynomial opened at fewer than all points");
    }

    #[test]
    fn constant_polynomials_have_no_stage_position() {
        let si = fixture();
        assert_eq!(si.const_pols_map.len(), 1);
        assert_eq!(si.const_pols_map[0].stage, 0);
        assert!(si.const_pols_map[0].stage_pos.is_none(), "constants are addressed by polsMapId, not stagePos");
    }

    /// A Goldilocks AIR must be refused rather than silently misread.
    #[test]
    fn rejects_an_extension_field_air() {
        let si = fixture();

        // The fixture is a Goldilocks AIR: qDim 3 and dim-3 committed columns.
        assert_eq!(si.q_dim, 3);
        let err = si.ensure_bn254_compatible().unwrap_err().to_string();
        assert!(err.contains("qDim 3"), "unexpected error: {err}");
        assert!(err.contains("compiled for Goldilocks"), "unexpected error: {err}");
    }

    #[test]
    fn accepts_a_dim_one_air() {
        let mut si = fixture();
        si.q_dim = 1;
        for p in &mut si.cm_pols_map {
            p.dim = 1;
        }
        assert!(si.ensure_bn254_compatible().is_ok());
    }

    #[test]
    fn round_trips_through_json() {
        let si = fixture();
        let text = serde_json::to_string(&si).unwrap();
        let again = StarkInfo::from_json(&text).unwrap();
        assert_eq!(si, again);
    }
}
