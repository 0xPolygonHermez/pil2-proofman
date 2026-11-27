//! The head and tail every layout shares: the merged plonk conversion, the sizing, and the
//! publics, connection and fixed columns of the result.

use anyhow::{ensure, Result};
use proofman_fields::QuotientMap;

use super::blocks::CLOCK_WRAP_ROWS;
use super::place::Sheet;
use super::spec::{AirLayout, Blocks, Init};
use crate::plonk2pil::field::PlonkField;
use crate::plonk2pil::merge_copies::{apply_remap_to_s_map, r1cs2plonk_merged, CopyMerge};
use crate::plonk2pil::r1cs::to_plonk::{get_custom_gates_info, CustomGatesInfo, PlonkAddition, PlonkConstraint};
use crate::plonk2pil::r1cs::types::{FixedPol, PlonkOptions, R1csFile, SetupResult};
use crate::plonk2pil::utils::{bind_public_signals, build_s_polynomials, public_rows};

/// An r1cs converted to plonk, with its custom gates.
pub struct Prepared<'a, F> {
    pub r1cs: &'a R1csFile<F>,
    pub constraints: Vec<PlonkConstraint<F>>,
    pub additions: Vec<PlonkAddition<F>>,
    pub merge: CopyMerge,
    pub cgi: CustomGatesInfo<F>,
    pub n_publics: u32,
}

pub fn prepare<'a, F: PlonkField>(r1cs: &'a R1csFile<F>, opts: &PlonkOptions) -> Prepared<'a, F> {
    let (constraints, additions, merge) = r1cs2plonk_merged(r1cs, opts.merge_copies);
    tracing::info!("Number of plonk constraints: {}", constraints.len());
    let cgi = get_custom_gates_info(r1cs);
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    Prepared { r1cs, constraints, additions, merge, cgi, n_publics }
}

/// The AIR's height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizing {
    pub n_bits: usize,
    /// Before the caller's floor: whether the circuit itself is too big.
    pub n_bits_natural: usize,
    pub n_used: usize,
    pub first_public_row: usize,
    pub n_public_rows: usize,
}

/// Sizes an AIR whose publics start at `first_public_row`, never below `floor`.
pub fn size<F: PlonkField>(
    layout: &AirLayout,
    first_public_row: usize,
    n_publics: u32,
    floor: usize,
) -> Result<Sizing> {
    let n_public_rows = public_rows(n_publics, layout.conn_cols);
    let n_used = first_public_row + n_public_rows;
    // blake3's clock selectors read backwards and wrap: the last CLOCKS-1 rows must stay padding.
    let window = if matches!(layout.blocks, Blocks::Blake3 { .. }) { CLOCK_WRAP_ROWS } else { 0 };
    let hold = first_public_row + window.max(n_public_rows);
    let n_bits_natural = (hold.max(2).next_power_of_two().trailing_zeros() as usize).max(layout.min_bits);
    let n_bits = n_bits_natural.max(floor);
    let max_bits = F::ROOTS_OF_UNITY.len() - 1;
    ensure!(
        n_bits <= max_bits,
        "plonk2pil: the {} needs 2^{n_bits} rows ({n_used} used), and {:?} has domains of at most 2^{max_bits}",
        layout.kind,
        F::PRIME
    );
    Ok(Sizing { n_bits, n_bits_natural, n_used, first_public_row, n_public_rows })
}

/// The result: publics bound, the copy merge swept and checked, `S` built, the fixed columns in the
/// layout's order.
pub fn finish<F: PlonkField + QuotientMap<u64>>(
    layout: &AirLayout,
    prep: Prepared<'_, F>,
    mut sheet: Sheet<F>,
    sizing: Sizing,
    name: String,
    pil_str: String,
) -> SetupResult<F> {
    let Sizing { n_bits, n_bits_natural, n_used, first_public_row, n_public_rows } = sizing;
    let n = 1usize << n_bits;
    bind_public_signals(&mut sheet.s_map, first_public_row, prep.n_publics, layout.conn_cols);
    let mut fixed = layout.fixed.clone();
    if layout.publics_row_fixed && n_public_rows > 0 {
        for k in 0..n_public_rows {
            sheet.set("PUBLICS_ROW", k, first_public_row + k, F::ONE);
        }
        fixed.push(("PUBLICS_ROW", n_public_rows, Init::Zero));
    }
    apply_remap_to_s_map(&mut sheet.s_map, &prep.merge.remap);
    let mut sv = build_s_polynomials::<F>(layout.conn_cols, n, n_bits, n_used, &sheet.s_map).into_iter();

    let fixed_pols = fixed
        .iter()
        .flat_map(|&(col, count, init)| (0..count).map(move |index| (col, index, init)))
        .map(|(col, index, init)| {
            let values = if col == "S" { sv.next().expect("one S column each") } else { sheet.take(col, index, init) };
            FixedPol { name: format!("{name}.{col}"), index, values }
        })
        .collect();

    SetupResult {
        fixed_pols,
        pil_str,
        n_bits,
        n_bits_natural,
        n_used,
        s_map: sheet.s_map,
        gate_bands: sheet.gate_bands,
        plonk_additions: prep.additions,
        airgroup_name: name.clone(),
        air_name: name,
        band_aux: layout.band_aux,
    }
}
