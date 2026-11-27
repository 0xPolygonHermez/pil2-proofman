//! `pilout.globalInfo.json` of a pilout (pilfflonk/docs/formats.md#globalinfo): the part of the
//! STARK's schema that does not depend on the backend, built as the STARK setup builds it, and
//! pilfflonk's fields.

use pil2_pilout::pilout as pb;
use pil_info::core::ir::NamedValue;
use pil_info::core::pilout_core::{Air, AirGroupValue, PiloutCore, PiloutError};
use pil_info::output::global_info::global_values_maps;
use proofman_pilfflonk::tag::{Backend, Field, Modulus, Transcript};
use proofman_pilfflonk::{AggType, GlobalInfoAir, NameStageEntry, PilfflonkGlobalInfo, SetupParams, FORMAT_VERSION};

use crate::error::SetupError;

/// The name of the pilout, `<name>` in the `provingKey/` (pilfflonk/docs/formats.md#provingkey),
/// as the STARK setup names it (`commands/setup.rs`).
pub fn pilout_name(pilout: &pb::PilOut) -> String {
    pilout.name.clone().unwrap_or_else(|| "pilout".to_string())
}

/// The name of an airgroup or AIR, `<airgroup>`/`<air>` in the `provingKey/`: the STARK setup's name for its
/// directory (`commands/setup.rs`). The STARK's globalInfo calls an unnamed one `"unnamed"` instead; here the
/// globalInfo and the directory agree, as the paths of the `provingKey/` are derived from the globalInfo.
fn dir_name(name: &Option<String>, prefix: &str, id: usize) -> String {
    name.clone().unwrap_or_else(|| format!("{prefix}_{id}"))
}

/// The entries of a map `pil-info` builds as JSON, as the globalInfo types them.
fn name_stage_entries(values: Vec<NamedValue>) -> Vec<NameStageEntry> {
    let entry = |v: NamedValue| NameStageEntry {
        name: v.name,
        stage: u64::from(v.stage),
        lengths: v.lengths.unwrap_or_default().into_iter().map(u64::from).collect(),
    };
    values.into_iter().map(entry).collect()
}

/// The globalInfo of `pilout`, set up with `setup_params`.
///
/// The common part is the STARK setup's (`setup/pil2-stark/src/output/global_info.rs`): the
/// airgroups and AIRs in the pilout's order, the aggregation of each airgroup value,
/// `numChallenges` (`[0]` for a pilout that has none, as the STARK), and the proof values and
/// publics maps, which `pil-info` builds for both setups.
pub fn global_info(pilout: &pb::PilOut, setup_params: SetupParams) -> Result<PilfflonkGlobalInfo, SetupError> {
    let invalid = |e: PiloutError| SetupError::InvalidPilout(e.to_string());
    let core = PiloutCore::header_of(pilout).map_err(invalid)?;
    let (proof_values, publics) = global_values_maps(&core).map_err(invalid)?;
    let mut airs = Vec::with_capacity(core.air_groups.len());
    let mut air_groups = Vec::with_capacity(core.air_groups.len());
    let mut agg_types = Vec::with_capacity(core.air_groups.len());
    for (airgroup_id, airgroup) in core.air_groups.iter().enumerate() {
        air_groups.push(dir_name(&airgroup.name, "airgroup", airgroup_id));
        let air = |(air_id, air): (usize, &Air)| GlobalInfoAir {
            name: dir_name(&air.name, "air", air_id),
            num_rows: u64::from(air.num_rows),
        };
        airs.push(airgroup.airs.iter().enumerate().map(air).collect());
        let agg = |v: &AirGroupValue| AggType { agg_type: v.agg_type as u64, stage: u64::from(v.stage) };
        agg_types.push(airgroup.air_group_values.iter().map(agg).collect());
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
        proof_values_map: name_stage_entries(proof_values),
        publics_map: name_stage_entries(publics),
    })
}
