//! Which columns each combined polynomial packs, for one stage.
//!
//! A proof is produced stage by stage: each computes some polynomials, commits
//! them, and the commitments feed the transcript that draws the next stage's
//! challenges (see [`crate::air::air_challenges`]).
//!
//! This module reads the key and works out *what* to pack -- which columns of
//! the stage's buffer belong to each `f_i`, in slot order, and how many
//! coefficients to take from each. The packing itself is rapidsnark's
//! `CPolynomial`, reached through `proofman-fflonk-lib-c::combine`, the same
//! class the fflonk prover builds its `C0`/`C1`/`C2` with.
//!
//! Splitting it this way keeps the key-reading in Rust and the field arithmetic
//! in the C++ that already implements it, rather than having a second
//! implementation of the interleave to keep in step.

use anyhow::{Context, Result, bail};

use crate::proof::commitment_key;
use crate::zkey::ZKey;

/// Bytes per coefficient, matching the key's `FrElement`.
pub const FR_BYTES: usize = 32;

/// One combined polynomial's recipe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagePlan {
    /// `f0`, `f1`, ... -- the key a commitment is stored under.
    pub name: String,
    /// The stage columns this packs, in slot order, with how many coefficients
    /// to take from each.
    pub columns: Vec<(usize, usize)>,
}

/// Work out what each `f_i` drawing on `stage` packs.
///
/// Returned in `f_i` index order, which is also the order their commitments are
/// absorbed into the transcript.
pub fn stage_plan(zkey: &ZKey, stage: u32) -> Result<Vec<StagePlan>> {
    let names =
        zkey.pols_names_stage.get(&stage).with_context(|| format!("the key names no polynomials for stage {stage}"))?;

    let mut out = Vec::new();
    for f in &zkey.f {
        let Some(entry) = f.stages.iter().find(|s| s.stage == stage) else { continue };

        // A combined polynomial spanning several stages is filled in across
        // them, one stage's contribution at a time. Nothing in the reference
        // key does that, so rather than implement an accumulation that cannot
        // be tested, say so.
        if f.stages.len() != 1 {
            bail!(
                "f{} draws on {} stages; building a combined polynomial across stages is not implemented",
                f.index,
                f.stages.len()
            );
        }

        let mut columns = Vec::with_capacity(f.pols.len());
        for name in &f.pols {
            let id = names
                .iter()
                .position(|n| n == name)
                .with_context(|| format!("f{}: {name} is not among stage {stage}'s polynomials", f.index))?;

            let degree = entry
                .pols
                .iter()
                .find(|p| &p.name == name)
                .with_context(|| format!("f{}: {name} has no degree in stage {stage}", f.index))?
                .degree;

            columns.push((id, degree as usize));
        }

        out.push(StagePlan { name: commitment_key(f.index), columns });
    }

    Ok(out)
}

/// The number of columns a stage's buffer has.
pub fn stage_width(zkey: &ZKey, stage: u32) -> Result<usize> {
    Ok(zkey
        .pols_names_stage
        .get(&stage)
        .with_context(|| format!("the key names no polynomials for stage {stage}"))?
        .len())
}

/// Grow a coefficient buffer to `to_rows`, leaving the new rows zero.
///
/// A committed stage's polynomials are taller than its trace: interpolating `N`
/// evaluations gives `N` coefficients, but the stage reserves `N + openings + 1`
/// so blinding has somewhere to write -- it adds `b·(X^(j+N) - X^j)`, which is
/// why the extra rows sit just above the domain. They are zero when blinding is
/// disabled, which is what makes a run reproducible.
pub fn pad_rows(buf: &[u8], n_cols: usize, to_rows: usize) -> Result<Vec<u8>> {
    if n_cols == 0 {
        bail!("cannot pad a buffer with no columns");
    }
    if !buf.len().is_multiple_of(n_cols * FR_BYTES) {
        bail!("a buffer of {} bytes is not a whole number of {n_cols}-column rows", buf.len());
    }

    let from_rows = buf.len() / (n_cols * FR_BYTES);
    if to_rows < from_rows {
        bail!("cannot pad {from_rows} rows down to {to_rows}");
    }

    let mut out = vec![0u8; to_rows * n_cols * FR_BYTES];
    out[..buf.len()].copy_from_slice(buf);
    Ok(out)
}

/// The coefficients a stage reserves per column, which is the tallest degree it
/// declares.
pub fn reserved_rows(zkey: &ZKey, stage: u32) -> Result<usize> {
    zkey.f
        .iter()
        .flat_map(|f| f.stages.iter().filter(|s| s.stage == stage))
        .flat_map(|s| s.pols.iter().map(|p| p.degree as usize))
        .max()
        .with_context(|| format!("stage {stage} declares no degrees"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZKEY: &[u8] = include_bytes!("../tests/fixtures/reference/pilfflonk.zkey");

    fn key() -> ZKey {
        ZKey::from_bytes(ZKEY).expect("the vendored key parses")
    }

    /// The plan names the constant stage's two combined polynomials, in the
    /// order the transcript absorbs them.
    #[test]
    fn plans_the_constant_stage() {
        let zkey = key();
        let plan = stage_plan(&zkey, 0).unwrap();

        let names: Vec<&str> = plan.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["f0", "f1"]);

        // f0 packs six constants, each the full domain.
        assert_eq!(plan[0].columns.len(), 6);
        assert!(plan[0].columns.iter().all(|&(_, len)| len == 256));
    }

    /// Slot order is the key's `pols` order, not the stage's column order --
    /// they differ, and swapping them would commit to a different polynomial.
    #[test]
    fn slot_order_follows_the_combined_polynomial_not_the_stage() {
        let zkey = key();
        let plan = stage_plan(&zkey, 0).unwrap();

        let ids: Vec<usize> = plan[0].columns.iter().map(|&(id, _)| id).collect();
        assert_eq!(ids, vec![5, 4, 3, 2, 1, 0], "f0's slots run down the stage's columns");
    }

    /// Stage 1 reserves more coefficients than the domain has rows, for
    /// blinding.
    #[test]
    fn a_committed_stage_reserves_rows_for_blinding() {
        let zkey = key();
        assert!(reserved_rows(&zkey, 1).unwrap() > 256);
        assert_eq!(stage_width(&zkey, 1).unwrap(), 15);
    }

    #[test]
    fn rejects_a_stage_the_key_does_not_know() {
        assert!(stage_plan(&key(), 99).is_err());
        assert!(stage_width(&key(), 99).is_err());
    }

    #[test]
    fn pads_rows_with_zeroes() {
        let buf = vec![7u8; FR_BYTES * 4];
        let padded = pad_rows(&buf, 2, 4).unwrap();

        assert_eq!(padded.len(), FR_BYTES * 8);
        assert_eq!(&padded[..buf.len()], &buf[..]);
        assert!(padded[buf.len()..].iter().all(|&b| b == 0));
        assert_eq!(pad_rows(&buf, 2, 2).unwrap(), buf, "padding to the same height is a copy");
    }

    #[test]
    fn rejects_padding_that_would_shrink_or_misalign() {
        let buf = vec![0u8; FR_BYTES * 4];
        assert!(pad_rows(&buf, 2, 1).is_err());
        assert!(pad_rows(&buf, 3, 4).is_err());
        assert!(pad_rows(&buf, 0, 4).is_err());
    }
}
