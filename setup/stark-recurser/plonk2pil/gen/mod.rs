//! The one plonk2pil generator: a declarative [`spec::AirLayout`] per (kind, hash family) from
//! [`layouts`], applied by the one placer, [`place`], between the shared head and tail of
//! [`finish`]. Over Goldilocks for the STARK recursion's airs, over BN128 for the SNARK wraps.

pub mod blocks;
mod finish;
pub mod layouts;
mod place;
mod pil_str;
pub mod spec;
#[cfg(test)]
mod wrap_tests;

use anyhow::{bail, ensure, Result};
use proofman_common::hash_family::{lookup_gate, recursive_bits_threshold, GateRole, HashFamily};
use proofman_fields::QuotientMap;

use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::rand_hex;
use blocks::{
    plan_compressor_geometry, Blake3Blocks, CompressorDemand, HashBlocks, DEFAULT_LANES, MAX_LANES, PLONK_GATES_PER_ROW,
};
use finish::{finish, prepare, size, Prepared};
use place::{place, plan_groups, select, Placed, Sheet};
use spec::{AirLayout, Blocks, Gates, Param};
pub use spec::SetupKind;

/// Sets `r1cs` up as a `kind` of `opts.hash`: over Goldilocks a compressor or aggregation air, over
/// BN128 the SNARK wrap.
pub fn setup<F: PlonkField + QuotientMap<u64>>(
    r1cs: &R1csFile<F>,
    kind: SetupKind,
    opts: &PlonkOptions,
) -> Result<SetupResult<F>> {
    let hash = opts.hash;
    let wrap = if kind == SetupKind::Wrap { Some(layouts::wrap(hash, r1cs)?) } else { None };
    if wrap.is_none() {
        refuse_bn128_gates(r1cs)?;
    }
    let prep = prepare(r1cs, opts);
    let mut floor = opts.min_n_bits.unwrap_or(0);
    let layout = match wrap {
        Some(layout) => layout,
        None => {
            let mut lanes = opts.blake3_lanes.unwrap_or(DEFAULT_LANES);
            // blake3's compressor is its aggregator at a geometry of its own.
            if (kind, hash) == (SetupKind::Compressor, HashFamily::Blake3) {
                floor = floor.max(recursive_bits_threshold(hash.as_str()));
                if opts.blake3_lanes.is_none() {
                    let geom = plan_compressor_geometry(&compressor_demand(&prep), floor);
                    (lanes, floor) = (geom.lanes, geom.n_bits);
                }
            }
            assert!(
                (1..=MAX_LANES).contains(&lanes),
                "LANES must be in 1..={MAX_LANES} (see aggregator.pil), got {lanes}"
            );
            layouts::stark(kind, hash, lanes)
        }
    };

    let blocks = hash_blocks(&layout, &prep);
    let dry = place(&layout, &prep, &blocks, None);
    ensure!(
        dry.uses == r1cs.custom_gates_uses.len(),
        "plonk2pil: the {hash} {kind} places {} of the r1cs's {} custom-gate uses",
        dry.uses,
        r1cs.custom_gates_uses.len()
    );
    let sizing = size::<F>(&layout, dry.first_public_row, prep.n_publics, floor)?;
    if dry.band_blocks > blocks.n_blocks() {
        tracing::warn!(
            "plonk-dominated: the band needs {} block interiors against the hashing's {}; a compressor fixes this if \
             the air is too tall",
            dry.band_blocks,
            blocks.n_blocks()
        );
    }
    tracing::info!(
        "NUsed: {}, nBits: {}, plonk rows: {}, hash blocks: {}",
        sizing.n_used,
        sizing.n_bits,
        dry.plonk_rows,
        blocks.n_blocks()
    );

    let mut sheet = Sheet::new(&layout, 1 << sizing.n_bits);
    let placed = place(&layout, &prep, &blocks, Some(&mut sheet));
    assert_eq!(placed, dry, "the placement disagrees with its dry run");

    let name = opts.airgroup_name.clone().unwrap_or_else(|| {
        if layout.stamped {
            format!("{}{}", layout.default_name, rand_hex())
        } else {
            layout.default_name.to_string()
        }
    });
    let degree =
        if layout.degree_fixed { layout.max_degree } else { opts.max_constraint_degree.unwrap_or(layout.max_degree) };
    let params = pil_params(&layout, &prep, &blocks, &placed);
    let pil = pil_str::gen_pil_str(&layout, &name, sizing.n_bits, prep.n_publics, degree, &params);
    Ok(finish(&layout, prep, sheet, sizing, name, pil))
}

fn hash_blocks<'a, F: PlonkField + QuotientMap<u64>>(layout: &AirLayout, prep: &Prepared<'a, F>) -> HashBlocks<'a, F> {
    match layout.blocks {
        Blocks::None => HashBlocks::None,
        Blocks::Blake3 { lanes } => HashBlocks::Blake3(Blake3Blocks::new(prep.r1cs, &prep.cgi, lanes)),
        Blocks::Blake3Wrap => HashBlocks::Wrap(Blake3Blocks::new(prep.r1cs, &prep.cgi, 1).uses()),
    }
}

/// The blake3 compressor's demand: its hashing, and its band circuits as the aggregator lays them.
fn compressor_demand<F: PlonkField>(prep: &Prepared<'_, F>) -> CompressorDemand {
    let layout = layouts::stark(SetupKind::Compressor, HashFamily::Blake3, 1);
    let mut band: Vec<(usize, usize)> =
        layout.sections.iter().map(|s| (select(prep, s.gates).len().div_ceil(s.per_row), s.rows)).collect();
    let (groups, _) = plan_groups(&prep.constraints, PLONK_GATES_PER_ROW, true, 0);
    band.push((groups.iter().map(|g| g.len().div_ceil(PLONK_GATES_PER_ROW)).sum(), 1));
    CompressorDemand::new(prep.r1cs, &prep.cgi, band)
}

/// The airtemplate's parameters.
fn pil_params<F: PlonkField + QuotientMap<u64>>(
    layout: &AirLayout,
    prep: &Prepared<'_, F>,
    blocks: &HashBlocks<'_, F>,
    placed: &Placed,
) -> Vec<(&'static str, usize)> {
    let b3 = match blocks {
        HashBlocks::Blake3(b) => Some(b),
        _ => None,
    };
    let b3 = |f: fn(&Blake3Blocks<'_, F>) -> usize| b3.map_or(0, f);
    let value = |p: Param| match p {
        Param::PlonkRows => placed.plonk_rows,
        Param::Uses(role) => prep.cgi.n(role),
        Param::Rows(role) => {
            let s = layout.sections.iter().find(|s| s.gates == Gates::Role(role)).expect("the role's section");
            select(prep, s.gates).len().div_ceil(s.per_row)
        }
        Param::PublicsStart => placed.first_public_row,
        Param::NodeBlocks => b3(|b| b.n_node_blocks),
        Param::ChunkBlocks => b3(|b| b.n_chunk_blocks),
        Param::ParentBlocks => b3(|b| b.n_parent_blocks),
        Param::Lanes => b3(|b| b.lanes),
    };
    layout.pil_params.iter().map(|&(name, p)| (name, value(p))).collect()
}

/// Refuses a Goldilocks r1cs with a gate of the BN128 wraps, `PoseidonT` or `Num2Bytes`: the STARK
/// layouts would leave it unplaced.
pub fn refuse_bn128_gates<F>(r1cs: &R1csFile<F>) -> Result<()> {
    let of_the_wrap = |name: &str| {
        lookup_gate(name).is_some_and(|(role, _)| matches!(role, GateRole::PoseidonT | GateRole::RangeCheck))
    };
    if let Some(gate) = r1cs.custom_gates.iter().find(|g| of_the_wrap(&g.template_name)) {
        bail!(
            "plonk2pil: the r1cs is over Goldilocks and uses {}, a gate of the BN128 wraps, which are over BN128",
            gate.template_name
        );
    }
    Ok(())
}
