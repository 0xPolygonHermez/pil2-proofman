//! `<air>.pilfflonkinfo.json` from what the passes return
//! (pilfflonk/docs/formats.md#pilfflonkinfo): the maps of the STARK's starkinfo that pilfflonk
//! keeps, the committed polynomials and the layout.
//!
//! **From `pil-info` to `PilfflonkInfo`.** The maps, the evMap, the opening points, the
//! boundaries (the ones the code's `Zi` operands index), `qDeg` and `cExpId` are `pil-info`'s
//! as they are, with every dimension 1 (the `stagePos` of a fixed column, which `pil-info` does
//! not set, is its index), but for these:
//!
//! - **`Q`.** `pil-info` ends `cmPolsMap` with the STARK's pieces of the quotient, `Q0 …
//!   Q{qDeg−1}` at stage `nStages + 1` (`qDeg` of them, of `N` coefficients each: what the STARK
//!   commits), and none if `qDeg = 0`. pilfflonk commits `Q` whole unless it is split
//!   (pilfflonk/docs/protocol.md#q-pieces): the pieces are replaced by
//!   `layout::q_pieces(qDeg, maxQDegree)` entries, `Q0 … Q<m−1>`, piece `i` at stageId and
//!   stagePos `i`, which is one, `Q0`, when `Q` is not split. They are the last entries of
//!   `cmPolsMap`, after every column and im pol, and no operand of the code refers to them: no
//!   index the code uses changes. `mapSectionsN.cm{nStages+1}` counts the new entries. The evMap
//!   has no piece of `Q` (`Opening::Shplonk`), and the layout has the pieces in the last `f` (or
//!   `f`, if `--extra-muls` splits their group: pilfflonk/docs/protocol.md#grouping-rules), opened
//!   at `ξ` only. `maxQDegree` is the `--max-q-degree` that splits `Q`, and 0 if it does not
//!   (`layout::split_max_q_degree`).
//! - **The names of the im pols.** `pil-info` names every im pol of an AIR `<air>.ImPol`, and
//!   their evaluations would share a name in the proof (`names`). Here the im pols of an AIR are
//!   the array `<air>.ImPol`: the `k`-th of `cmPolsMap` has `lengths: [k]`, so its name in the
//!   proof and in the layout is `<air>.ImPol[k]`.
//! - **The names of the columns named alike** (pilfflonk/docs/formats.md#proof-names). The std
//!   declares some columns in a loop, under one name and without an index (the sum bus's
//!   `im_cluster` and `im_single`, `std_sum.pil`), and a pilout can have several columns of that
//!   name. As the im pols, the columns of a pol map that share a name, none of them with
//!   `lengths`, are the array of that name: the `k`-th of them in the map has `lengths: [k]`,
//!   `im_cluster[0]`, `im_cluster[1]`, … ([`index_names_alike`]). A name that a column with
//!   `lengths` has is left as it is: the elements of an array share their name and differ in
//!   their indices.
//! - **The evMap.** `pil-info`'s, followed by the pairs the fusions of the grouping add
//!   (`layout::ev_map_of`): the indices of `pil-info`'s entries, which the `qVerifier` refers to,
//!   do not change.
//!
//! Every column of the AIR, fixed or committed, im pols and pieces of `Q` included, then has a
//! name of its own (`names::column_name`), or the AIR is refused ([`check_names`]): two arrays
//! of the same name, or a column named as the setup names another.
//!
//! The stage-1 columns keep `pil-info`'s order: the pilout's columns at `stageId 0 … C−1`, and
//! the im pols after them, which is what the witness files hold (`WitnessShape`).

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use pil2_pilout::pilout as pb;
use pil_info::pil::constraint_poly::Boundary as PassesBoundary;
use pil_info::types::pilout_info::SymbolInfo;
use pil_info::PilInfoResult;
use proofman_pilfflonk::layout::{q_pieces, split_max_q_degree};
use proofman_pilfflonk::names::column_name;
pub use proofman_pilfflonk::names::q_piece_name;
use proofman_pilfflonk::{
    Boundary, ChallengeMapEntry, EvMapEntry, JsonFile, NameStageEntry, PilfflonkInfo, PolMapEntry, PolType,
};

use crate::error::SetupError;
use crate::layout::{committed_pols, ev_map_of, Committed, Packing, QShape};

/// The AIR the pilfflonkinfo describes, as the globalInfo names it.
#[derive(Clone, Copy, Debug)]
pub struct AirRef<'a> {
    pub name: &'a str,
    pub airgroup_id: u64,
    pub air_id: u64,
}

/// What the setup knows of an AIR once the passes have run: its pilfflonkinfo, validated, and
/// its committed polynomials.
#[derive(Clone, Debug)]
pub struct AirSetup {
    pub info: PilfflonkInfo,
    pub committed: Committed,
}

fn passes_output<T>(what: String) -> Result<T, SetupError> {
    Err(SetupError::PassesOutput(what))
}

fn lengths(symbol: &SymbolInfo) -> Vec<u64> {
    symbol.lengths.iter().flatten().map(|&l| l as u64).collect()
}

/// The columns of `map` that share a name, none of them with `lengths` (see [the module](self)):
/// each gets `lengths: [k]`, `k` its position among them in the map. A name that only one column
/// has, or that a column with `lengths` has, does not change.
fn index_names_alike(map: &mut [PolMapEntry]) {
    // For each name, how many columns have it, and whether one of them has lengths.
    let mut names: BTreeMap<String, (usize, bool)> = BTreeMap::new();
    for p in map.iter() {
        let (count, indexed) = names.entry(p.name.clone()).or_default();
        *count += 1;
        *indexed |= !p.lengths.is_empty();
    }
    let mut next: BTreeMap<String, u64> = BTreeMap::new();
    for p in map.iter_mut() {
        if names.get(&p.name).is_some_and(|&(count, indexed)| count > 1 && !indexed) {
            let k = next.entry(p.name.clone()).or_default();
            p.lengths = vec![*k];
            *k += 1;
        }
    }
}

/// Refuses two columns of `air` that the proof and the layout would name alike (see [the
/// module](self)): every entry of the pol maps, pieces of `Q` included, must have a name of its
/// own (`names::column_name`).
fn check_names(air: &str, const_pols_map: &[PolMapEntry], cm_pols_map: &[PolMapEntry]) -> Result<(), SetupError> {
    let mut named: BTreeMap<String, String> = BTreeMap::new();
    for (map_name, map) in [("constPolsMap", const_pols_map), ("cmPolsMap", cm_pols_map)] {
        for (i, p) in map.iter().enumerate() {
            let entry = format!("{map_name}[{i}]");
            match named.entry(column_name(&p.name, &p.lengths)) {
                Entry::Vacant(vacant) => {
                    vacant.insert(entry);
                }
                Entry::Occupied(first) => {
                    return Err(SetupError::ColumnName {
                        air: air.to_string(),
                        name: first.key().clone(),
                        first: first.get().clone(),
                        second: entry,
                    })
                }
            }
        }
    }
    Ok(())
}

/// The stage and the `stageId` of a column of a pol map of `pil-info`, which must have dimension
/// 1, or an error naming it.
fn column(symbol: &SymbolInfo, what: &str, i: usize) -> Result<(u64, u64), SetupError> {
    let (Some(stage), Some(stage_id)) = (symbol.stage, symbol.stage_id) else {
        return passes_output(format!("{what}[{i}] ({}) has no stage or stageId", symbol.name));
    };
    if symbol.dim != 1 {
        return passes_output(format!("{what}[{i}] ({}) has dim {}, not 1", symbol.name, symbol.dim));
    }
    Ok((stage as u64, stage_id as u64))
}

/// `constPolsMap`: every fixed column of `air`, each with its symbol.
fn const_pols_map(result: &PilInfoResult, air: &pb::Air) -> Result<Vec<PolMapEntry>, SetupError> {
    let map = &result.setup.const_pols_map;
    if map.len() != air.fixed_cols.len() || map.iter().any(|s| s.sym_type != "fixed") {
        return Err(SetupError::InvalidPilout(format!(
            "the AIR has {} fixed columns, and symbols for {} of them: every column needs one",
            air.fixed_cols.len(),
            map.iter().filter(|s| s.sym_type == "fixed").count()
        )));
    }
    let mut const_pols_map = map
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (stage, stage_id) = column(s, "constPolsMap", i)?;
            if stage != 0 {
                return passes_output(format!("constPolsMap[{i}] ({}) is of stage {stage}", s.name));
            }
            Ok(PolMapEntry {
                stage: 0,
                name: s.name.clone(),
                dim: 1,
                pols_map_id: i as u64,
                stage_id,
                lengths: lengths(s),
                im_pol: false,
                exp_id: None,
                stage_pos: i as u64,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    index_names_alike(&mut const_pols_map);
    Ok(const_pols_map)
}

/// `cmPolsMap`: the columns and im pols of `pil-info`'s, the columns named alike indexed, then the
/// pieces of `Q` (see [the module](self)).
fn cm_pols_map(result: &PilInfoResult, air: &pb::Air, q_pieces: u64) -> Result<Vec<PolMapEntry>, SetupError> {
    let setup = &result.setup;
    let q_stage = setup.n_stages + 1;
    let n_columns = setup.cm_pols_map.iter().take_while(|s| s.stage != Some(q_stage)).count();
    if let Some(i) = setup.cm_pols_map[n_columns..].iter().position(|s| s.stage != Some(q_stage)) {
        return passes_output(format!("cmPolsMap[{}] is not a piece of Q, and comes after one", n_columns + i));
    }

    let mut map = Vec::with_capacity(n_columns + q_pieces as usize);
    let mut n_im_pols = 0u64;
    for (i, s) in setup.cm_pols_map[..n_columns].iter().enumerate() {
        if s.sym_type != "witness" {
            return Err(SetupError::InvalidPilout(format!("witness column {i} of the AIR has no symbol")));
        }
        let (stage, stage_id) = column(s, "cmPolsMap", i)?;
        if !(1..=setup.n_stages as u64).contains(&stage) {
            return passes_output(format!("cmPolsMap[{i}] ({}) is of stage {stage}", s.name));
        }
        let Some(stage_pos) = s.stage_pos else {
            return passes_output(format!("cmPolsMap[{i}] ({}) has no stagePos", s.name));
        };
        let (lengths, exp_id) = if s.im_pol {
            let Some(exp_id) = s.exp_id else {
                return passes_output(format!("cmPolsMap[{i}] is an im pol without its expression"));
            };
            n_im_pols += 1;
            (vec![n_im_pols - 1], Some(exp_id as u64))
        } else {
            (lengths(s), None)
        };
        map.push(PolMapEntry {
            stage,
            name: s.name.clone(),
            dim: 1,
            pols_map_id: i as u64,
            stage_id,
            lengths,
            im_pol: s.im_pol,
            exp_id,
            stage_pos: stage_pos as u64,
        });
    }
    // The pilout's columns of each stage, the ones its symbols name, are those the trace has.
    for (s, &width) in air.stage_widths.iter().enumerate() {
        let stage = s as u64 + 1;
        let named = map.iter().filter(|p| p.stage == stage && !p.im_pol).count();
        if named != width as usize {
            return Err(SetupError::InvalidPilout(format!(
                "stage {stage} of the AIR has {width} witness columns, and symbols for {named}: every column needs one"
            )));
        }
    }
    // The im pols have lengths already: a name of theirs does not change.
    index_names_alike(&mut map);
    for i in 0..q_pieces {
        map.push(PolMapEntry {
            stage: q_stage as u64,
            name: q_piece_name(i),
            dim: 1,
            pols_map_id: map.len() as u64,
            stage_id: i,
            lengths: vec![],
            im_pol: false,
            exp_id: None,
            stage_pos: i,
        });
    }
    Ok(map)
}

fn name_stage_entries(map: &[SymbolInfo], what: &str) -> Result<Vec<NameStageEntry>, SetupError> {
    map.iter()
        .enumerate()
        .map(|(i, s)| match s.stage {
            Some(stage) if s.dim == 1 => {
                Ok(NameStageEntry { name: s.name.clone(), stage: stage as u64, lengths: lengths(s) })
            }
            _ => passes_output(format!("{what}[{i}] ({}) must have a stage and dim 1", s.name)),
        })
        .collect()
}

fn challenges_map(result: &PilInfoResult) -> Result<Vec<ChallengeMapEntry>, SetupError> {
    if !result.pil_code.challenges_map.is_empty() {
        return passes_output("the code added challenges, as FRI does".to_string());
    }
    result
        .setup
        .challenges_map
        .iter()
        .enumerate()
        .map(|(i, c)| match (c.stage, c.stage_id) {
            (Some(stage), Some(stage_id)) if c.dim == 1 && !c.name.is_empty() => {
                Ok(ChallengeMapEntry { name: c.name.clone(), stage: stage as u64, dim: 1, stage_id: stage_id as u64 })
            }
            _ => passes_output(format!(
                "challengesMap[{i}] ({:?}) must have a name, a stage, a stageId and dim 1",
                c.name
            )),
        })
        .collect()
}

fn boundary(b: &PassesBoundary) -> Result<Boundary, SetupError> {
    match (b.name.as_str(), b.offset_min, b.offset_max) {
        ("everyRow", _, _) => Ok(Boundary::EveryRow),
        ("firstRow", _, _) => Ok(Boundary::FirstRow),
        ("lastRow", _, _) => Ok(Boundary::LastRow),
        ("everyFrame", Some(min), Some(max)) => {
            Ok(Boundary::EveryFrame { offset_min: u64::from(min), offset_max: u64::from(max) })
        }
        (name, min, max) => passes_output(format!("boundary {name} (offsets {min:?}, {max:?})")),
    }
}

fn ev_map(result: &PilInfoResult) -> Result<Vec<EvMapEntry>, SetupError> {
    result
        .pil_code
        .ev_map
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let pol_type = match (e.entry_type.as_str(), e.commit_id) {
                ("cm", None) => PolType::Cm,
                ("const", None) => PolType::Const,
                (other, _) => return passes_output(format!("evMap[{i}] is of type {other}")),
            };
            Ok(EvMapEntry { pol_type, id: e.id as u64, prime: e.prime, opening_pos: e.opening_pos as u64 })
        })
        .collect()
}

/// The pilfflonkinfo of `air` (of the pilout, the one [`crate::validate::validate`] returned) from
/// the result of the passes on it, with the layout `packing` says (`layout::committed_pols`) and
/// `Q` split in pieces of degree `max_q_degree` if its degree is above it
/// (pilfflonk/docs/protocol.md#q-pieces; 0 does not split it). It is validated
/// (`JsonFile::validate`) before it is returned, so that nothing is written for a pilfflonkinfo
/// that cannot be.
pub fn air_setup(
    result: &PilInfoResult,
    air_ref: AirRef,
    air: &pb::Air,
    max_q_degree: u64,
    packing: Packing,
) -> Result<AirSetup, SetupError> {
    if result.fri_exp_id.is_some() {
        return passes_output("a FRI polynomial: the passes must run with PilInfoCfg::bn254()".to_string());
    }
    let setup = &result.setup;
    let q_deg = u64::try_from(result.q_deg).map_err(|_| SetupError::QDegree(result.q_deg))?;
    let n_stages = setup.n_stages as u64;
    let q_stage = n_stages + 1;
    let n_bits = u64::from(setup.pil_power);
    let max_q_degree = split_max_q_degree(q_deg, max_q_degree);
    let q = QShape { stage: q_stage, q_deg, max_q_degree };

    let const_pols_map = const_pols_map(result, air)?;
    let cm_pols_map = cm_pols_map(result, air, q_pieces(q_deg, max_q_degree))?;
    check_names(air_ref.name, &const_pols_map, &cm_pols_map)?;
    let ev_map = ev_map(result)?;
    let committed = committed_pols(n_bits, q, &const_pols_map, &cm_pols_map, &ev_map, packing)?;
    let ev_map = ev_map_of(&ev_map, &committed.layout, q_stage, &setup.opening_points)?;

    let mut map_sections_n = std::collections::BTreeMap::new();
    map_sections_n.insert("const".to_string(), const_pols_map.len() as u64);
    for stage in 1..=q_stage {
        map_sections_n.insert(format!("cm{stage}"), cm_pols_map.iter().filter(|p| p.stage == stage).count() as u64);
    }

    let info = PilfflonkInfo {
        name: air_ref.name.to_string(),
        airgroup_id: air_ref.airgroup_id,
        air_id: air_ref.air_id,
        n_bits,
        n_stages,
        n_constants: const_pols_map.len() as u64,
        cm_pols_map,
        const_pols_map,
        challenges_map: challenges_map(result)?,
        air_values_map: name_stage_entries(&setup.air_values_map, "airValuesMap")?,
        airgroup_values_map: name_stage_entries(&setup.airgroup_values_map, "airgroupValuesMap")?,
        map_sections_n,
        opening_points: setup.opening_points.clone(),
        boundaries: result.boundaries.iter().map(boundary).collect::<Result<_, _>>()?,
        ev_map,
        q_deg,
        q_dim: 1,
        max_q_degree,
        c_exp_id: result.c_exp_id as u64,
        layout: committed.layout.clone(),
    };
    info.validate()?;
    Ok(AirSetup { info, committed })
}
