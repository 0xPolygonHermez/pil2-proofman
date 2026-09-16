//! The combined polynomials a stage contributes, ready to be committed.
//!
//! A proof is produced stage by stage: each one computes some polynomials,
//! commits them, and the commitments go into the transcript that draws the next
//! stage's challenges (see [`crate::air::air_challenges`]). This module covers
//! the middle step -- turning a stage's coefficient buffer into the combined
//! polynomials that stage commits.
//!
//! It stops short of committing. Multiplying by the powers of tau needs the
//! curve arithmetic in `proofman-fflonk-lib-c`, and depending on that here
//! would put a C++ toolchain in the way of building a crate that is otherwise
//! pure Rust. So this returns what to commit and the caller commits it, the
//! same division [`crate::pairing`] uses for the verifier's final check.

use anyhow::{Context, Result, bail};

use crate::packing::{FR_BYTES, interleave, read_column};
use crate::proof::commitment_key;
use crate::zkey::ZKey;

/// One combined polynomial, named as its commitment will be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Combined {
    /// `f0`, `f1`, ... -- the key a commitment is stored under.
    pub name: String,
    /// Interleaved coefficients, in the key's representation.
    pub coefficients: Vec<u8>,
}

/// The length a combined polynomial needs: the highest index any slot reaches,
/// plus one.
///
/// Not the `degree` the key records. That is computed from each slot's length
/// rather than its top index, so it overshoots by `nPols` -- harmless where it
/// is used as an upper bound, wrong as a buffer size.
fn combined_len(slots: &[Vec<u8>]) -> usize {
    let n = slots.len();
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .map(|(j, s)| (s.len() / FR_BYTES - 1) * n + j + 1)
        .max()
        .unwrap_or(0)
}

/// Build every combined polynomial that draws on `stage`.
///
/// `coefficients` is that stage's buffer, coefficient-major over the stage's
/// full width. Returns them in `f_i` index order, which is also the order their
/// commitments are absorbed into the transcript.
pub fn combined_for(zkey: &ZKey, stage: u32, coefficients: &[u8]) -> Result<Vec<Combined>> {
    let names =
        zkey.pols_names_stage.get(&stage).with_context(|| format!("the key names no polynomials for stage {stage}"))?;
    let n_pols_stage = names.len();

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

        let mut slots = Vec::with_capacity(f.pols.len());
        for name in &f.pols {
            let pol_id = names
                .iter()
                .position(|n| n == name)
                .with_context(|| format!("f{}: {name} is not among stage {stage}'s polynomials", f.index))?;

            let degree = entry
                .pols
                .iter()
                .find(|p| &p.name == name)
                .with_context(|| format!("f{}: {name} has no degree in stage {stage}", f.index))?
                .degree;

            slots.push(
                read_column(coefficients, pol_id, n_pols_stage, degree as usize)
                    .with_context(|| format!("f{}: reading {name}", f.index))?,
            );
        }

        let len = combined_len(&slots);
        let coefficients = interleave(&slots, len).with_context(|| format!("f{}: interleaving its slots", f.index))?;

        out.push(Combined { name: commitment_key(f.index), coefficients });
    }

    Ok(out)
}

/// Drop trailing zero coefficients.
///
/// The setup stores polynomials trimmed this way -- it is what the C++
/// `fixDegree()` does. Trailing zeros contribute nothing to a commitment, so
/// this matters for comparing against the key, not for committing.
pub fn trim(coefficients: &[u8]) -> &[u8] {
    let mut end = coefficients.len();
    while end >= FR_BYTES && coefficients[end - FR_BYTES..end].iter().all(|&b| b == 0) {
        end -= FR_BYTES;
    }
    &coefficients[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZKEY: &[u8] = include_bytes!("../tests/fixtures/reference/pilfflonk.zkey");

    fn key() -> ZKey {
        ZKey::from_bytes(ZKEY).expect("the vendored key parses")
    }

    /// Stage 0 is the one a key can be checked against on its own: it holds the
    /// constant polynomials, and the key records both their coefficients and
    /// the combined polynomials built from them.
    #[test]
    fn rebuilds_the_constant_stage_from_the_key() {
        let zkey = key();
        let coefs = zkey.bulk.get(&crate::zkey::SECTION_CONST_POLS_COEFS).expect("constant coefficients");

        let built = combined_for(&zkey, 0, coefs).unwrap();
        assert_eq!(built.len(), 2, "the reference key has two constant-only combined polynomials");

        for c in &built {
            let recorded = zkey
                .f_commitments
                .iter()
                .find(|r| r.name == c.name)
                .unwrap_or_else(|| panic!("{} has no recorded polynomial", c.name));

            assert_eq!(trim(&c.coefficients), recorded.pol.as_slice(), "{}", c.name);
        }
    }

    /// Order is the transcript's: commitments are absorbed by f_i index, so a
    /// reordering here would change every challenge that follows.
    #[test]
    fn returns_them_in_index_order() {
        let zkey = key();
        let coefs = zkey.bulk.get(&crate::zkey::SECTION_CONST_POLS_COEFS).unwrap();

        let names: Vec<String> = combined_for(&zkey, 0, coefs).unwrap().into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["f0", "f1"]);
    }

    /// A stage the key does not describe is an error rather than an empty list,
    /// which would look like a stage that legitimately commits nothing.
    #[test]
    fn rejects_a_stage_the_key_does_not_know() {
        let zkey = key();
        assert!(combined_for(&zkey, 99, &[]).is_err());
    }

    /// A truncated buffer is reported, not silently short-read.
    #[test]
    fn rejects_a_short_coefficient_buffer() {
        let zkey = key();
        let coefs = zkey.bulk.get(&crate::zkey::SECTION_CONST_POLS_COEFS).unwrap();
        assert!(combined_for(&zkey, 0, &coefs[..coefs.len() / 2]).is_err());
    }

    #[test]
    fn trim_drops_only_whole_trailing_zero_coefficients() {
        let mut buf = vec![0u8; FR_BYTES * 3];
        buf[0] = 7;
        assert_eq!(trim(&buf).len(), FR_BYTES);

        buf[FR_BYTES * 2] = 9;
        assert_eq!(trim(&buf).len(), FR_BYTES * 3);

        assert_eq!(trim(&[0u8; FR_BYTES * 2]).len(), 0);
        assert_eq!(trim(&[]).len(), 0);
    }
}
