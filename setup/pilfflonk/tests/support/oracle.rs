//! The oracle's names for what a pilfflonkinfo opens, shared by the tests that evaluate `Q(ξ)`.

use pil2_pilout::pilout as pb;
use pil_info::core::from_pilout::pilout_expr_index;
use proofman_pilfflonk::oracle::ColumnRef;
use proofman_pilfflonk::{EvMapEntry, PilfflonkInfo, PolType};

/// The column an evMap entry evaluates: a fixed one by its pilout index, a committed one by stage and
/// `stageId`, an im pol by its `expId` mapped back to the pilout's expression.
pub fn column_of(pilout: &pb::PilOut, info: &PilfflonkInfo, e: &EvMapEntry) -> ColumnRef {
    let pol = info.pol(e.pol_type, e.id).expect("an evMap entry of the pilfflonkinfo");
    match (e.pol_type, pol.exp_id) {
        (PolType::Const, _) => ColumnRef::Fixed(e.id as usize),
        (PolType::Cm, Some(exp_id)) if pol.im_pol => {
            ColumnRef::Im(pilout_expr_index(pilout, 0, 0, exp_id as u32).unwrap())
        }
        (PolType::Cm, _) => ColumnRef::Witness { stage: pol.stage as usize, idx: pol.stage_id as usize },
    }
}

/// The pilout expressions of the im pols, in cmPolsMap order.
pub fn im_pols(pilout: &pb::PilOut, info: &PilfflonkInfo) -> Vec<usize> {
    info.cm_pols_map
        .iter()
        .filter(|p| p.im_pol)
        .map(|p| pilout_expr_index(pilout, 0, 0, p.exp_id.expect("an im pol has an expId") as u32).unwrap())
        .collect()
}
