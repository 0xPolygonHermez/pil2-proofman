//! The wrap's trace in the bytes pilfflonk takes it in (each value 32 bytes, canonical, little
//! endian), built straight from the circom witness's bytes, which are in that form already: the map
//! gathers a cell as a copy, and only the exec's additions compute in `Fr`, in parallel, level by
//! level (an addition reads the wires of earlier levels only), as rapidsnark's GPU prover does.

use proofman_common::exec_format::ExecFile;
use proofman_common::ProofmanError;
use proofman_fields::Bn128;
use proofman_pilfflonk::FrBytes;
use rayon::prelude::*;

use crate::error::{WrapWitnessError, WrapWitnessResult};

/// Bytes of a value.
pub(crate) const VALUE_BYTES: usize = 32;

/// The value at `cell` of `trace` if it is below `2^64`.
pub(crate) fn word(trace: &[u8], cell: usize) -> Option<u64> {
    let bytes = &trace[cell * VALUE_BYTES..(cell + 1) * VALUE_BYTES];
    bytes[8..].iter().all(|&b| b == 0).then(|| u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")))
}

/// Writes `v` at `cell` of `trace`.
pub(crate) fn put_word(trace: &mut [u8], cell: usize, v: u64) {
    let bytes = &mut trace[cell * VALUE_BYTES..(cell + 1) * VALUE_BYTES];
    bytes[..8].copy_from_slice(&v.to_le_bytes());
    bytes[8..].fill(0);
}

/// The value at `cell` of `trace`, for a message: its decimal.
pub(crate) fn show(trace: &[u8], cell: usize) -> String {
    let mut bytes = [0u8; VALUE_BYTES];
    bytes.copy_from_slice(&trace[cell * VALUE_BYTES..(cell + 1) * VALUE_BYTES]);
    Bn128::from_le_bytes(bytes).map_or_else(|| "a value not below r".to_string(), |v| v.to_string())
}

/// The publics and the trace, `n_rows x n_cols` row after row, of `wires`, the circom witness's
/// bytes (each below `r`): what `ExecFile::committed_pols` computes over `Fr`, with its checks
/// ([`ExecFile::check_witness`]).
pub(crate) fn trace_of(
    exec: &ExecFile<Bn128>,
    mut wires: Vec<u8>,
    n_publics: usize,
    n_rows: usize,
    n_cols: usize,
) -> WrapWitnessResult<(Vec<FrBytes>, Vec<u8>)> {
    let n_witness = wires.len() / VALUE_BYTES;
    let cells = exec.check_witness(n_witness, n_publics, n_rows, n_cols).map_err(WrapWitnessError::Exec)?;
    let len = cells.checked_mul(VALUE_BYTES).ok_or_else(|| {
        WrapWitnessError::Exec(ProofmanError::InvalidSetup(format!(
            "exec: a trace of {cells} cells does not fit in memory"
        )))
    })?;
    let (map_rows, map_cols) = (exec.layout.map_rows(), exec.layout.map_cols());
    let n_adds = exec.additions.len();
    let n_wires = n_witness + n_adds;
    let publics = (1..=n_publics)
        .map(|w| {
            let mut bytes = [0u8; VALUE_BYTES];
            bytes.copy_from_slice(&wires[w * VALUE_BYTES..(w + 1) * VALUE_BYTES]);
            FrBytes::from_le_bytes(bytes).expect("the circom witness is below r")
        })
        .collect();

    let (level, n_levels) = addition_levels(exec, n_witness)?;
    let mut by_level: Vec<Vec<u32>> = vec![Vec::new(); n_levels as usize];
    for (i, &l) in level.iter().enumerate() {
        by_level[l as usize].push(i as u32);
    }
    tracing::debug!("exec: {n_adds} additions in {n_levels} levels");

    wires.resize(n_wires * VALUE_BYTES, 0);
    let value = |wires: &[u8], w: u32| {
        let w = w as usize;
        let mut bytes = [0u8; VALUE_BYTES];
        bytes.copy_from_slice(&wires[w * VALUE_BYTES..(w + 1) * VALUE_BYTES]);
        Bn128::from_le_bytes(bytes).expect("every wire is below r")
    };
    for adds in &by_level {
        let sums: Vec<(u32, [u8; VALUE_BYTES])> = adds
            .par_iter()
            .map(|&i| {
                let addition = &exec.additions[i as usize];
                let [l, r] = addition.wires.map(|w| value(&wires, w));
                (i, (l * addition.coeffs[0] + r * addition.coeffs[1]).to_le_bytes())
            })
            .collect();
        for (i, bytes) in sums {
            let w = n_witness + i as usize;
            wires[w * VALUE_BYTES..(w + 1) * VALUE_BYTES].copy_from_slice(&bytes);
        }
    }

    // Zeroed pages from the allocator; the rows the map does not reach stay so.
    let mut trace = vec![0u8; len];
    let row_bytes = n_cols * VALUE_BYTES;
    if map_cols > 0 {
        trace[..map_rows * row_bytes].par_chunks_exact_mut(row_bytes).enumerate().for_each(|(row, cells)| {
            for (col, &wire) in exec.map[row * map_cols..(row + 1) * map_cols].iter().enumerate() {
                if wire != 0 {
                    let w = wire as usize;
                    cells[col * VALUE_BYTES..(col + 1) * VALUE_BYTES]
                        .copy_from_slice(&wires[w * VALUE_BYTES..(w + 1) * VALUE_BYTES]);
                }
            }
        });
    }
    Ok((publics, trace))
}

/// Each addition's level, one past the deepest addition it reads (0 if it reads only the circom
/// witness's `n_witness` wires), and the number of levels; or why an addition reads a wire not
/// defined before it.
pub(crate) fn addition_levels(exec: &ExecFile<Bn128>, n_witness: usize) -> WrapWitnessResult<(Vec<u32>, u32)> {
    let mut level = vec![0u32; exec.additions.len()];
    let mut n_levels = 0u32;
    for (i, addition) in exec.additions.iter().enumerate() {
        let wire = n_witness + i;
        let mut l = 0;
        for &read in &addition.wires {
            let read = read as usize;
            if read >= wire {
                return Err(WrapWitnessError::Exec(ProofmanError::InvalidSetup(format!(
                    "exec: addition {i}, wire {wire}, reads wire {read}, which is not defined before it"
                ))));
            }
            if read >= n_witness {
                l = l.max(level[read - n_witness] + 1);
            }
        }
        level[i] = l;
        n_levels = n_levels.max(l + 1);
    }
    Ok((level, n_levels))
}
