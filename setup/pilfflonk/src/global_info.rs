//! `pilout.globalInfo.json` of a pilout (pilfflonk/docs/formats.md#globalinfo): the part of the
//! STARK's schema that does not depend on the backend, built as the STARK setup builds it, and
//! pilfflonk's fields.

use pil2_pilout::pilout as pb;
use pil_info::output::global_info::{build_global_proof_values_map, build_global_publics_map};
use proofman_pilfflonk::tag::{Backend, Field, Modulus, Transcript};
use proofman_pilfflonk::{
    AggType, GlobalInfoAir, NameStageEntry, PilfflonkError, PilfflonkGlobalInfo, SetupParams, FORMAT_VERSION,
};

use crate::error::SetupError;

/// The name of the pilout, `<name>` in the `provingKey/` (pilfflonk/docs/formats.md#provingkey),
/// as the STARK setup names it (`commands/setup.rs`).
pub fn pilout_name(pilout: &pb::PilOut) -> String {
    pilout.name.clone().unwrap_or_else(|| "pilout".to_string())
}

/// The name of an airgroup, `<airgroup>` in the `provingKey/`: the STARK setup's name for its
/// directory (`commands/setup.rs`). The STARK's globalInfo calls an airgroup without a name
/// `"unnamed"` instead; here the globalInfo and the directory agree, as the paths of the
/// `provingKey/` are derived from the globalInfo.
fn airgroup_name(airgroup_id: usize, airgroup: &pb::AirGroup) -> String {
    airgroup.name.clone().unwrap_or_else(|| format!("airgroup_{airgroup_id}"))
}

/// The name of an AIR, `<air>` in the `provingKey/`: as [`airgroup_name`].
fn air_name(air_id: usize, air: &pb::Air) -> String {
    air.name.clone().unwrap_or_else(|| format!("air_{air_id}"))
}

/// The entries of a map `pil-info` builds as JSON, as the globalInfo types them.
fn name_stage_entries(values: Vec<serde_json::Value>) -> Result<Vec<NameStageEntry>, SetupError> {
    serde_json::from_value(serde_json::Value::Array(values)).map_err(|e| SetupError::Pilfflonk(PilfflonkError::from(e)))
}

/// The globalInfo of `pilout`, set up with `setup_params`.
///
/// The common part is the STARK setup's (`setup/pil2-stark/src/output/global_info.rs`): the
/// airgroups and AIRs in the pilout's order, the aggregation of each airgroup value,
/// `numChallenges` (`[0]` for a pilout that has none, as the STARK), and the proof values and
/// publics maps, which `pil-info` builds for both setups.
pub fn global_info(pilout: &pb::PilOut, setup_params: SetupParams) -> Result<PilfflonkGlobalInfo, SetupError> {
    let mut airs = Vec::with_capacity(pilout.air_groups.len());
    let mut air_groups = Vec::with_capacity(pilout.air_groups.len());
    let mut agg_types = Vec::with_capacity(pilout.air_groups.len());
    for (airgroup_id, airgroup) in pilout.air_groups.iter().enumerate() {
        air_groups.push(airgroup_name(airgroup_id, airgroup));
        airs.push(
            airgroup
                .airs
                .iter()
                .enumerate()
                .map(|(air_id, air)| GlobalInfoAir {
                    name: air_name(air_id, air),
                    num_rows: u64::from(air.num_rows.unwrap_or(0)),
                })
                .collect(),
        );
        let aggs = airgroup
            .air_group_values
            .iter()
            .map(|v| match u64::try_from(v.agg_type) {
                Ok(agg_type) => Ok(AggType { agg_type, stage: u64::from(v.stage) }),
                Err(_) => Err(SetupError::InvalidPilout(format!(
                    "an airgroup value of airgroup {airgroup_id} aggregates by {}",
                    v.agg_type
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        agg_types.push(aggs);
    }

    let num_challenges = if pilout.num_challenges.is_empty() {
        vec![0]
    } else {
        pilout.num_challenges.iter().map(|&n| u64::from(n)).collect()
    };

    Ok(PilfflonkGlobalInfo {
        name: pilout_name(pilout),
        airs,
        air_groups,
        agg_types,
        backend: Backend,
        format_version: FORMAT_VERSION,
        field: Field,
        modulus: Modulus,
        transcript: Transcript,
        setup_params,
        n_publics: u64::from(pilout.num_public_values),
        num_challenges,
        num_proof_values: pilout.num_proof_values.iter().map(|&n| u64::from(n)).collect(),
        proof_values_map: name_stage_entries(build_global_proof_values_map(&pilout.symbols))?,
        publics_map: name_stage_entries(build_global_publics_map(&pilout.symbols))?,
    })
}
