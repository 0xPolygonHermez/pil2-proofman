//! Fixed column I/O: read and write fixed polynomial binary files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};

use anyhow::{bail, Result};

use crate::io::bin_file_writer::BinFileWriter;

/// Reorder plonk fixed polynomials to match the pilout `fixed_cols` order.
///
/// The JS pil2-compiler writes `fixed_cols` in PIL declaration order (S[27] before C[10]).
/// `build_fixed_pols` outputs C values first then S values (mirroring JS `plonk2pil`).
/// Without reordering, C values would be written into S slots and vice versa in the `.const` file.
///
/// This function mirrors the JS `generateMultiArrayIndexes` expansion and produces
/// `plonk_pol_values` in the order each empty `fixed_col` entry expects.
///
/// SymbolType::FIXED_COL == 1 in the pilout proto.
pub fn reorder_plonk_pols_for_pilout(
    fixed_pols: &[stark_recurser::plonk2pil::FixedPol],
    symbols: &[pilout::pilout::Symbol],
    air_group_id: u32,
    air_id: u32,
) -> Vec<Vec<u64>> {
    // Build lookup: (name, element_index) → values
    let pol_map: HashMap<(String, usize), &Vec<u64>> =
        fixed_pols.iter().map(|fp| ((fp.name.clone(), fp.index), &fp.values)).collect();

    // Filter to fixed cols for this air, sort by id (= pilout fixed_cols order)
    let mut fixed_syms: Vec<&pilout::pilout::Symbol> = symbols
        .iter()
        .filter(|s| {
            s.r#type == 1 // FIXED_COL
                && s.air_group_id == Some(air_group_id)
                && s.air_id == Some(air_id)
        })
        .collect();
    fixed_syms.sort_by_key(|s| s.id);

    // Expand array symbols to individual columns (matching JS generateMultiArrayIndexes)
    // Skip scalar symbols — they have inline values in the pilout, not from plonk.
    let mut result = Vec::new();
    for sym in fixed_syms {
        if sym.lengths.is_empty() {
            // Scalar fixed col: has inline values in pilout, no entry in plonk_pols.
            continue;
        }
        let total: usize = sym.lengths.iter().map(|&l| l as usize).product();
        for idx in 0..total {
            if let Some(vals) = pol_map.get(&(sym.name.clone(), idx)) {
                result.push((*vals).clone());
            }
        }
    }

    result
}

/// Metadata for a single fixed polynomial column within a binary file.
#[derive(Debug, Clone)]
pub struct FixedPolInfo {
    pub lengths: Vec<u32>,
    pub values: Vec<u64>,
}

/// Read a fixed polynomial binary file (.cnst format).
///
/// Returns a map from `"{airgroupName}_{airName}"` to a map of
/// polynomial name -> vector of `FixedPolInfo` entries.
///
/// File layout:
///   - 4-byte magic "cnst"
///   - u32 LE version
///   - u32 LE number of sections (1)
///   - Section 1:
///     - u32 LE section_id, u64 LE section_size
///     - string: airgroup_name
///     - string: air_name
///     - u64 LE: N (number of rows)
///     - u32 LE: nFixedPols
///     - For each fixed pol:
///       - string: name
///       - u32 LE: n_lengths
///       - u32 LE * n_lengths: lengths
///       - u64 LE * N: values
pub fn read_fixed_pols_bin(
    fixed_info: &mut HashMap<String, HashMap<String, Vec<FixedPolInfo>>>,
    bin_filename: &str,
) -> Result<()> {
    let file = File::open(bin_filename)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"cnst" {
        bail!("Invalid magic in fixed pols file: expected 'cnst'");
    }

    let _version = read_u32_le(&mut reader)?;
    let _n_sections = read_u32_le(&mut reader)?;

    // Read section header
    let _section_id = read_u32_le(&mut reader)?;
    let _section_size = read_u64_le(&mut reader)?;

    // Read data
    let airgroup_name = read_string(&mut reader)?;
    let air_name = read_string(&mut reader)?;
    let n = read_u64_le(&mut reader)?;
    let n_fixed_pols = read_u32_le(&mut reader)?;

    let mut pols_info: HashMap<String, Vec<FixedPolInfo>> = HashMap::new();

    for _ in 0..n_fixed_pols {
        let name = read_string(&mut reader)?;
        let n_lengths = read_u32_le(&mut reader)?;
        let mut lengths = Vec::with_capacity(n_lengths as usize);
        for _ in 0..n_lengths {
            lengths.push(read_u32_le(&mut reader)?);
        }

        let mut values = Vec::with_capacity(n as usize);
        let mut buf = vec![0u8; n as usize * 8];
        reader.read_exact(&mut buf)?;
        for i in 0..n as usize {
            let val = u64::from_le_bytes([
                buf[i * 8],
                buf[i * 8 + 1],
                buf[i * 8 + 2],
                buf[i * 8 + 3],
                buf[i * 8 + 4],
                buf[i * 8 + 5],
                buf[i * 8 + 6],
                buf[i * 8 + 7],
            ]);
            values.push(val);
        }

        pols_info.entry(name).or_default().push(FixedPolInfo { lengths, values });
    }

    let key = format!("{}_{}", airgroup_name, air_name);
    fixed_info.insert(key, pols_info);

    Ok(())
}

/// Write a fixed polynomial binary file (.cnst format).
///
/// Ports `writeFixedPolsBin` from `fixed_cols.js` (recurser-js).
/// Uses `BinFileWriter` to produce the same sectioned binary as `@iden3/binfileutils`.
///
/// `fixed_info` is a list of (name, lengths, values) tuples.
pub fn write_fixed_pols_bin(
    bin_filename: &str,
    airgroup_name: &str,
    air_name: &str,
    n: u64,
    fixed_info: &[(String, Vec<u32>, Vec<u64>)],
) -> Result<()> {
    let mut writer = BinFileWriter::new(bin_filename, "cnst", 1, 1)?;
    writer.start_write_section(1)?;

    writer.write_string(airgroup_name)?;
    writer.write_string(air_name)?;
    writer.write_u64(n)?;
    writer.write_u32(fixed_info.len() as u32)?;

    for (name, lengths, values) in fixed_info {
        writer.write_string(name)?;
        writer.write_u32(lengths.len() as u32)?;
        for &len in lengths {
            writer.write_u32(len)?;
        }
        // Write all values as u64 LE bytes in one shot
        let mut buf = vec![0u8; values.len() * 8];
        for (i, &v) in values.iter().enumerate() {
            buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        writer.write_bytes(&buf)?;
    }

    writer.end_write_section()?;
    writer.close()
}

/// Write the `.const` file from the pilout and plonk fixed polynomial values.
///
/// Ports the JS logic: `getFixedPolsPil2(...)` followed by `fixedCols.saveToFile(...)`.
///
/// For each fixed column defined in the pilout `air.fixed_cols`:
/// - If the column has **no** inline values (`values.is_empty()`), the values come from
///   the plonk2pil output polynomials (`plonk_pol_values`), taken in order.
/// - If the column has inline values, those values from the pilout are used directly.
///   These are the selector polynomials computed by pil2-stark.
///
/// The output is a flat row-major interleaved buffer:
///   [col0_row0, col1_row0, ..., colN_row0, col0_row1, ..., colN_rowNrows]
/// Written as raw u64 LE values — matching `fixedCols.saveToFile()` in JS.
pub fn write_const_file(path: &str, air: &pilout::pilout::Air, plonk_pol_values: &[Vec<u64>]) -> Result<()> {
    // num_rows is the actual row count (not log2)
    let n_rows = air.num_rows.unwrap_or(0) as usize;
    // Fall back to plonk polynomial length if pilout doesn't specify rows
    let n_rows = if n_rows > 0 {
        n_rows
    } else if let Some(first) = plonk_pol_values.first() {
        first.len()
    } else {
        0
    };
    let n_constants = air.fixed_cols.len();

    let mut flat_buffer = vec![0u64; n_rows * n_constants];
    let mut plonk_idx = 0usize;

    for (col_idx, fixed_col) in air.fixed_cols.iter().enumerate() {
        if fixed_col.values.is_empty() {
            // Values come from plonk2pil (C or S polynomial)
            if let Some(vals) = plonk_pol_values.get(plonk_idx) {
                for (row, &val) in vals.iter().enumerate() {
                    if row < n_rows {
                        flat_buffer[row * n_constants + col_idx] = val;
                    }
                }
                plonk_idx += 1;
            }
        } else {
            // Inline values from the pilout (selector polynomials)
            for (row, val_bytes) in fixed_col.values.iter().enumerate() {
                if row >= n_rows {
                    break;
                }
                flat_buffer[row * n_constants + col_idx] = bytes_to_u64_be(val_bytes);
            }
        }
    }

    write_fixed_cols_raw(path, &flat_buffer)
}

/// Write a flat fixed-column buffer to a raw binary file (row-major u64 LE).
///
/// Equivalent to `fixedCols.saveToFile()` in JS.
/// Format: n_cols * N u64 LE values, interleaved row by row.
pub fn write_fixed_cols_raw(path: &str, buffer: &[u64]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &val in buffer {
        writer.write_all(&val.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

// ------ Internal helpers ------

fn read_u32_le(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(reader: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(reader: &mut impl Read) -> io::Result<String> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        reader.read_exact(&mut byte)?;
        if byte[0] == 0 {
            break;
        }
        buf.push(byte[0]);
    }
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn bytes_to_u64_be(bytes: &[u8]) -> u64 {
    // Pilout stores fixed col values big-endian (JS bint2buf uses writeBigUInt64BE).
    // buf2bint reads them back with readBigUInt64BE.
    let mut val = 0u64;
    for &b in bytes.iter().take(8) {
        val = (val << 8) | (b as u64);
    }
    val
}
