//! The SRS and the commitments of the fixed `f_i` (spec §4.2.5, A.6): `pilfflonk.srs.bin`,
//! `<air>.verkey.json` and the `[τ]₂` of the vkey, all computed by the C++ core (M6).

use std::path::Path;

use proofman_pilfflonk::field::FIELD_BYTES;
use proofman_pilfflonk::{AirVerkey, FqBytes, G1Affine, G2Affine, Layout};
use proofman_starks_lib_c::{pilfflonk_srs_from_ptau_c, PilFflonkSrs, PILFFLONK_G2_BYTES};

use crate::error::SetupError;
use crate::fixed::FixedColumns;

/// Writes `pilfflonk.srs.bin` at `srs_path`: the first `n_g1` powers `[τ^i]₁` of the ptau at
/// `ptau`, and `[1]₂`, `[τ]₂` (spec §4.2.5, A.6), replacing any file there. `n_g1` is the largest
/// `degree` of the layout, the coefficients of its largest `f_i`, which takes exactly that many
/// powers (M12).
///
/// This is where the last check of spec §4.2.1 happens: the C++ reader refuses a ptau with fewer
/// than `n_g1` powers before it reads a point, and the error says how many it has. Only sections
/// 1 to 3 of the ptau are read.
pub fn write_srs(ptau: &Path, n_g1: u64, srs_path: &Path) -> Result<(), SetupError> {
    pilfflonk_srs_from_ptau_c(ptau, n_g1, srs_path).map_err(SetupError::native(format!(
        "cannot take the {n_g1} powers [τ^i]₁ of the largest f of the layout from the ptau {}",
        ptau.display()
    )))
}

/// Loads the `pilfflonk.srs.bin` at `srs_path`.
pub fn load_srs(srs_path: &Path) -> Result<PilFflonkSrs, SetupError> {
    PilFflonkSrs::load(srs_path).map_err(SetupError::native(format!("cannot load the SRS {}", srs_path.display())))
}

/// `[τ]₂` of the SRS: the vkey's `X_2` (A.6).
pub fn x_2(srs: &PilFflonkSrs) -> Result<G2Affine, SetupError> {
    let bytes = srs.g2(1).map_err(SetupError::native("cannot read [τ]₂ of the SRS"))?;
    let mut coordinates = [FqBytes::ZERO; PILFFLONK_G2_BYTES / FIELD_BYTES];
    for (coordinate, chunk) in coordinates.iter_mut().zip(bytes.chunks_exact(FIELD_BYTES)) {
        let mut le = [0u8; FIELD_BYTES];
        le.copy_from_slice(chunk);
        *coordinate = FqBytes::from_le_bytes(le)?;
    }
    let [x_c0, x_c1, y_c0, y_c1] = coordinates;
    Ok(G2Affine { x: [x_c0, x_c1], y: [y_c0, y_c1] })
}

/// The commitment `[f(τ)]₁` of the fixed `f(X) = Σ_j p_j(X^k)·X^j` whose `p_j` interpolates the
/// fixed column `columns[j]` (A.2, rule 4), `k` = `columns.len()`: one `pilfflonk_commit_fixed`
/// call (M6).
pub fn commit_fixed_f(srs: &PilFflonkSrs, fixed: &FixedColumns, columns: &[u64]) -> Result<G1Affine, SetupError> {
    let mut evals = Vec::with_capacity(columns.len() * fixed.n_rows());
    for &id in columns {
        let column = usize::try_from(id).ok().and_then(|i| fixed.column(i)).ok_or_else(|| {
            SetupError::Layout(format!("a fixed f packs column {id}, and the AIR has {}", fixed.n_columns()))
        })?;
        evals.extend(column.iter().map(|v| v.to_le_bytes()));
    }
    let k = columns.len() as u64;
    let commitment = srs
        .commit_fixed(fixed.n_bits(), k, &evals)
        .map_err(SetupError::native(format!("cannot commit to the fixed f of the columns {columns:?}")))?;
    Ok(G1Affine::from_le_bytes(&commitment)?)
}

/// `<air>.verkey.json`: the commitments of the fixed `f_i` of `layout`, its entries of stage 0,
/// in its order (A.6), each packing the columns its `pols` name (their `constPolsMap` index,
/// which is the column's index in `fixed`).
pub fn air_verkey(srs: &PilFflonkSrs, fixed: &FixedColumns, layout: &Layout) -> Result<AirVerkey, SetupError> {
    let mut commitments = Vec::with_capacity(layout.n_fixed());
    for (i, f) in layout.0.iter().enumerate().filter(|(_, f)| f.stage == 0) {
        if f.k != f.pols.len() as u64 {
            return Err(SetupError::Layout(format!("f{i} has k = {} and packs {} columns", f.k, f.pols.len())));
        }
        let columns: Vec<u64> = f.pols.iter().map(|p| p.id).collect();
        commitments.push(commit_fixed_f(srs, fixed, &columns)?);
    }
    Ok(AirVerkey(commitments))
}
