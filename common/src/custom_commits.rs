use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use proofman_fields::PrimeField64;
use proofman_starks_lib_c::write_custom_commit_c;

use crate::trace::Trace;
use crate::ProofmanError;
use crate::ProofmanResult;
use crate::ProofCtx;

pub fn write_custom_commit_trace<F: PrimeField64>(
    pctx: &ProofCtx<F>,
    custom_trace: &mut dyn Trace<F>,
    blowup_factor: u64,
    merkle_tree_arity: u64,
    file_name: &Path,
) -> ProofmanResult<Vec<F>> {
    let buffer = custom_trace.get_buffer();
    let arity = merkle_tree_arity;
    let n = custom_trace.num_rows() as u64;
    let n_extended = blowup_factor * custom_trace.num_rows() as u64;
    let n_bits = n.trailing_zeros() as u64;
    let n_bits_ext = n_extended.trailing_zeros() as u64;
    let n_cols = custom_trace.num_cols() as u64;
    let mut root = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO];

    write_custom_commit_c(
        root.as_mut_ptr() as *mut u8,
        arity,
        n_bits,
        n_bits_ext,
        n_cols,
        pctx.get_device_buffers_ptr(),
        buffer.as_ptr() as *mut u8,
        file_name.to_str().expect("Invalid file name"),
    );

    Ok(root)
}

fn num_nodes_mt(height: u64, arity: u64) -> u64 {
    const HASH_SIZE: u64 = 4;
    let mut num_nodes = height;
    let mut nodes_level = height;
    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }
    num_nodes * HASH_SIZE
}

pub fn custom_commit_num_elements(n: u64, n_extended: u64, n_cols: u64, arity: u64) -> u64 {
    (n + n_extended) * n_cols + num_nodes_mt(n_extended, arity)
}

/// Legacy (pre-packing) file size -- kept only to recognise such files and reject them by name.
pub fn custom_commit_file_size_bytes(n: u64, n_extended: u64, n_cols: u64, arity: u64) -> u64 {
    (custom_commit_num_elements(n, n_extended, n_cols, arity) + 4) * 8
}

/// How hard `initialize_custom_commits` checks the files it registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomCommitValidation {
    /// Paths only -- for the flow that is about to generate the files.
    Skip,
    Lenient,
    Strict,
}

/// Words to reserve in the const-pols buffer for an air's packed custom commits. Sized for
/// words_per_row == n_cols: the real value lives in the commit file, unknown when the buffer
/// is sized.
pub fn custom_commit_reserved_words(n_bits: u32, stage_widths: &[u64]) -> usize {
    let n = 1u64 << n_bits;
    stage_widths.iter().copied().filter(|w| *w > 0).map(|w| (1 + w + n * w) as usize).sum()
}

/// A registered custom commit: file, Merkle root, and its packed `words_per_row`.
pub type CustomCommitEntry = (PathBuf, Vec<u8>, u64);

/// Byte size of a packed file: root, header word, one width per column, then the packed rows.
pub fn custom_commit_packed_file_size_bytes(n: u64, n_cols: u64, words_per_row: u64) -> u64 {
    32 + (1 + n_cols + n * words_per_row) * 8
}

/// `words_per_row` from a packed file's header, validated against the proving key. The legacy
/// layout is unsupported, and recognised only to say so: every pre-change file has that size.
pub fn custom_commit_words_per_row(
    path: &Path,
    n: u64,
    n_extended: u64,
    n_cols: u64,
    arity: u64,
) -> ProofmanResult<u64> {
    let actual = fs::metadata(path)
        .map_err(|e| ProofmanError::ProofmanError(format!("Failed to open {}: {e}", path.display())))?
        .len();

    if actual == custom_commit_file_size_bytes(n, n_extended, n_cols, arity) {
        return Err(ProofmanError::ProofmanError(format!(
            "custom commit file '{}' is in the legacy (unpacked) layout, which is no longer supported",
            path.display()
        )));
    }

    if actual > 40 {
        let mut file = File::open(path)?;
        let mut header = [0u8; 8];
        file.seek(SeekFrom::Start(32))?;
        file.read_exact(&mut header)?;
        let words_per_row = u64::from_le_bytes(header);
        if words_per_row > 0
            && words_per_row <= n_cols
            && actual == custom_commit_packed_file_size_bytes(n, n_cols, words_per_row)
        {
            return Ok(words_per_row);
        }
    }

    Err(ProofmanError::ProofmanError(format!(
        "custom commit file '{}' is {actual} bytes, which is not the packed layout for this proving key",
        path.display()
    )))
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use std::fs;

    const N: u64 = 64;
    const NE: u64 = 128;
    const W: u64 = 5;
    const ARITY: u64 = 4;

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("custom_commit_fmt_{tag}"));
        fs::create_dir_all(&dir).unwrap();
        dir.join("rom.bin")
    }

    #[test]
    fn legacy_layout_is_rejected_as_legacy() {
        let p = tmp("legacy");
        fs::write(&p, vec![0u8; custom_commit_file_size_bytes(N, NE, W, ARITY) as usize]).unwrap();
        let err = custom_commit_words_per_row(&p, N, NE, W, ARITY).unwrap_err().to_string();
        assert!(err.contains("legacy"), "message should name the legacy layout: {err}");
    }

    #[test]
    fn packed_size_is_detected_with_its_words_per_row() {
        let p = tmp("packed");
        let wpr = 3u64;
        let mut body = vec![0u8; 32];
        body.extend_from_slice(&wpr.to_le_bytes());
        body.extend(std::iter::repeat(0u8).take(((W + N * wpr) * 8) as usize));
        assert_eq!(body.len() as u64, custom_commit_packed_file_size_bytes(N, W, wpr));
        fs::write(&p, &body).unwrap();
        assert_eq!(custom_commit_words_per_row(&p, N, NE, W, ARITY).unwrap(), wpr);
    }

    #[test]
    fn truncated_file_is_an_error() {
        let p = tmp("short");
        let mut body = vec![0u8; 32];
        body.extend_from_slice(&2u64.to_le_bytes()); // words_per_row = 2
        body.extend(std::iter::repeat(0u8).take(64)); // far short of (W + N*2)*8
        fs::write(&p, &body).unwrap();
        assert!(custom_commit_words_per_row(&p, N, NE, W, ARITY).is_err());
    }

    #[test]
    fn impossible_words_per_row_is_an_error() {
        let p = tmp("wpr");
        let mut body = vec![0u8; 32];
        body.extend_from_slice(&(W + 1).to_le_bytes()); // wpr can never exceed nCols
        body.extend(std::iter::repeat(0u8).take(((W + N * (W + 1)) * 8) as usize));
        assert_eq!(body.len() as u64, custom_commit_packed_file_size_bytes(N, W, W + 1));
        fs::write(&p, &body).unwrap();
        assert!(custom_commit_words_per_row(&p, N, NE, W, ARITY).is_err());
    }

    #[test]
    fn zisk_rom_reserves_the_raw_small_domain() {
        // N = 2^22, w = 12 -> 384 MB reserved, vs 256 MB actually uploaded at words_per_row = 8.
        let words = custom_commit_reserved_words(22, &[12]);
        assert_eq!(words, 1 + 12 + (1usize << 22) * 12);
        assert_eq!(words * 8 / (1024 * 1024), 384);
    }

    #[test]
    fn airs_without_a_custom_commit_reserve_nothing() {
        assert_eq!(custom_commit_reserved_words(22, &[0]), 0);
        assert_eq!(custom_commit_reserved_words(22, &[]), 0);
    }

    #[test]
    fn several_commits_sum() {
        let n = 1usize << 10;
        assert_eq!(custom_commit_reserved_words(10, &[3, 0, 5]), (1 + 3 + n * 3) + (1 + 5 + n * 5));
    }

    #[test]
    fn missing_file_is_an_error() {
        let p = tmp("absent").with_file_name("does_not_exist.bin");
        let _ = fs::remove_file(&p);
        assert!(custom_commit_words_per_row(&p, N, NE, W, ARITY).is_err());
    }
}
