//! Building a combined polynomial from the polynomials it packs.
//!
//! A commitment covers several polynomials at once by interleaving their
//! coefficients:
//!
//! ```text
//! f(X) = Σ_j X^j · pol_j(X^nPols)
//! ```
//!
//! which in coefficient terms puts slot `j`'s coefficient `i` at index
//! `i · nPols + j`. That is what lets one commitment answer `nPols` opening
//! claims, and it is why the opening set for a point is a coset of the
//! `nPols`-th roots of unity -- see [`crate::roots`].
//!
//! Two different layouts meet here and they are easy to confuse. A stage's
//! coefficient buffer is *coefficient-major*: coefficient `i` of polynomial `p`
//! sits at `p + nPolsStage · i`, striding by the number of polynomials in the
//! whole stage. The combined polynomial is *slot-major* over a different width,
//! `nPols`, which counts only the polynomials this `f_i` packs. Reading with
//! the wrong stride still produces a well-formed buffer and a plausible
//! commitment, so the two are named apart throughout.
//!
//! This is data movement, not arithmetic: coefficients are copied as opaque
//! field elements in whatever representation the caller holds, so the same code
//! serves the key's stored values and anything computed later.

use anyhow::{Result, bail};

/// Bytes per coefficient, matching the key's `FrElement`.
pub const FR_BYTES: usize = 32;

/// Read one polynomial's coefficients out of a stage's buffer.
///
/// `pol_id` is the polynomial's column, `n_pols_stage` the number of columns,
/// and `len` how many coefficients to take.
pub fn read_column(buf: &[u8], pol_id: usize, n_pols_stage: usize, len: usize) -> Result<Vec<u8>> {
    if n_pols_stage == 0 {
        bail!("a stage with no polynomials has no columns to read");
    }
    if pol_id >= n_pols_stage {
        bail!("column {pol_id} is out of range for a stage {n_pols_stage} polynomials wide");
    }

    let mut out = Vec::with_capacity(len * FR_BYTES);
    for i in 0..len {
        let at = (pol_id + n_pols_stage * i) * FR_BYTES;
        let end = at + FR_BYTES;
        if end > buf.len() {
            bail!(
                "coefficient {i} of column {pol_id} runs past the stage buffer ({} coefficients available)",
                buf.len() / FR_BYTES
            );
        }
        out.extend_from_slice(&buf[at..end]);
    }
    Ok(out)
}

/// Interleave slot polynomials into one combined polynomial.
///
/// Slot `j`'s coefficient `i` lands at `i · slots.len() + j`. Slots may be
/// shorter than each other; the gaps stay zero, which is what a polynomial of
/// lower degree contributes.
///
/// `len` is the combined polynomial's length in coefficients, from the key.
/// It is given rather than inferred because a combined polynomial is padded to
/// the degree the setup planned for, which can exceed what the slots fill.
pub fn interleave(slots: &[Vec<u8>], len: usize) -> Result<Vec<u8>> {
    let n_pols = slots.len();
    if n_pols == 0 {
        bail!("a combined polynomial packs no slots");
    }

    let mut out = vec![0u8; len * FR_BYTES];
    for (j, slot) in slots.iter().enumerate() {
        if !slot.len().is_multiple_of(FR_BYTES) {
            bail!("slot {j} is {} bytes, not a whole number of coefficients", slot.len());
        }

        for i in 0..slot.len() / FR_BYTES {
            let at = i * n_pols + j;
            if at >= len {
                bail!(
                    "slot {j} coefficient {i} lands at index {at}, past the combined length of {len} -- \
                     the slot is longer than the planned degree allows"
                );
            }
            out[at * FR_BYTES..(at + 1) * FR_BYTES].copy_from_slice(&slot[i * FR_BYTES..(i + 1) * FR_BYTES]);
        }
    }
    Ok(out)
}

/// Grow a coefficient buffer to `to_rows`, leaving the new rows zero.
///
/// A committed stage's polynomials are taller than its trace: interpolating `N`
/// evaluations gives `N` coefficients, but the stage reserves `N + openings + 1`
/// so blinding has somewhere to write. Blinding adds a multiple of the
/// vanishing polynomial -- `b·(X^(j+N) - X^j)` -- which is why the extra rows
/// sit just above the domain.
///
/// The rows are zero when blinding is disabled, which is what makes a run
/// reproducible.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Coefficients as single recognisable bytes, so a layout mistake is
    /// visible rather than a wall of zeroes.
    fn coeff(tag: u8) -> Vec<u8> {
        let mut c = vec![0u8; FR_BYTES];
        c[0] = tag;
        c
    }

    fn buffer(tags: &[u8]) -> Vec<u8> {
        tags.iter().flat_map(|&t| coeff(t)).collect()
    }

    fn tags(buf: &[u8]) -> Vec<u8> {
        buf.chunks(FR_BYTES).map(|c| c[0]).collect()
    }

    /// A stage buffer strides by the stage's width, not by the number of
    /// polynomials a given f_i happens to pack.
    #[test]
    fn reads_a_column_by_the_stage_stride() {
        // Three polynomials, two coefficients each: [a0 b0 c0 a1 b1 c1].
        let buf = buffer(&[10, 20, 30, 11, 21, 31]);

        assert_eq!(tags(&read_column(&buf, 0, 3, 2).unwrap()), vec![10, 11]);
        assert_eq!(tags(&read_column(&buf, 1, 3, 2).unwrap()), vec![20, 21]);
        assert_eq!(tags(&read_column(&buf, 2, 3, 2).unwrap()), vec![30, 31]);
    }

    #[test]
    fn rejects_a_column_outside_the_stage() {
        let buf = buffer(&[1, 2, 3, 4]);
        assert!(read_column(&buf, 2, 2, 2).is_err());
        assert!(read_column(&buf, 0, 0, 1).is_err());
    }

    /// Running off the end is reported rather than silently truncated: a short
    /// buffer means the caller's idea of the degree disagrees with the key's.
    #[test]
    fn rejects_a_column_that_runs_past_the_buffer() {
        let buf = buffer(&[1, 2, 3, 4]);
        assert!(read_column(&buf, 0, 2, 3).is_err());
    }

    /// The defining layout: slot j's coefficient i at i*nPols + j.
    #[test]
    fn interleaves_slots_into_the_combined_layout() {
        let a = buffer(&[10, 11, 12]);
        let b = buffer(&[20, 21, 22]);

        let packed = interleave(&[a, b], 6).unwrap();
        assert_eq!(tags(&packed), vec![10, 20, 11, 21, 12, 22]);
    }

    /// A shorter slot leaves zeroes where its higher coefficients would be,
    /// which is exactly a polynomial of lower degree.
    #[test]
    fn a_shorter_slot_leaves_zeroes() {
        let a = buffer(&[10, 11, 12]);
        let b = buffer(&[20]);

        let packed = interleave(&[a, b], 6).unwrap();
        assert_eq!(tags(&packed), vec![10, 20, 11, 0, 12, 0]);
    }

    /// The combined length comes from the key and may exceed what the slots
    /// fill; the tail stays zero.
    #[test]
    fn pads_to_the_planned_length() {
        let a = buffer(&[10, 11]);
        let packed = interleave(&[a], 5).unwrap();
        assert_eq!(tags(&packed), vec![10, 11, 0, 0, 0]);
    }

    /// A slot longer than the plan allows is an error. Truncating instead
    /// would commit to a polynomial quietly missing its top coefficients.
    #[test]
    fn rejects_a_slot_longer_than_the_planned_degree() {
        let a = buffer(&[1, 2, 3, 4]);
        assert!(interleave(&[a], 3).is_err());
    }

    #[test]
    fn rejects_a_ragged_slot() {
        assert!(interleave(&[vec![0u8; FR_BYTES + 1]], 4).is_err());
        assert!(interleave(&[], 4).is_err());
    }

    #[test]
    fn pads_rows_with_zeroes() {
        // Two columns, two rows: [a0 b0 a1 b1].
        let buf = buffer(&[10, 20, 11, 21]);
        let padded = pad_rows(&buf, 2, 4).unwrap();

        assert_eq!(tags(&padded), vec![10, 20, 11, 21, 0, 0, 0, 0]);
        assert_eq!(pad_rows(&buf, 2, 2).unwrap(), buf, "padding to the same height is a copy");
    }

    #[test]
    fn rejects_padding_that_would_shrink_or_misalign() {
        let buf = buffer(&[10, 20, 11, 21]);
        assert!(pad_rows(&buf, 2, 1).is_err());
        assert!(pad_rows(&buf, 3, 4).is_err());
        assert!(pad_rows(&buf, 0, 4).is_err());
    }

    /// Reading columns out and interleaving them back is the identity when the
    /// widths agree -- the two layouts coincide only in that case, which is why
    /// they are kept apart.
    #[test]
    fn column_then_interleave_round_trips_at_equal_width() {
        let buf = buffer(&[10, 20, 11, 21, 12, 22]);

        let slots: Vec<Vec<u8>> = (0..2).map(|p| read_column(&buf, p, 2, 3).unwrap()).collect();
        assert_eq!(interleave(&slots, 6).unwrap(), buf);
    }
}
