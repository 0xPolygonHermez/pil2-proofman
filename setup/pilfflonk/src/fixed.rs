//! The fixed columns of an AIR, and `<air>.const` (pilfflonk/docs/formats.md#fixed-columns).
//!
//! `<air>.const` holds the fixed columns row by row, each value a canonical `Fr` of 32 bytes
//! little-endian, with no header: the value of column `c` at row `i` is at byte `(i·C + c)·32`,
//! and the file has exactly `N·C·32` bytes. It is the layout of the STARK's `.const` (8 bytes a
//! value there) and of the witness files (pilfflonk/docs/formats.md#witness-directory).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use num_bigint::BigUint;
use pil2_pilout::pilout as pb;
use proofman_pilfflonk::field::FIELD_BYTES;
use proofman_pilfflonk::global_info::MAX_NBITS;
use proofman_pilfflonk::FrBytes;

use crate::error::SetupError;

/// The fixed columns of an AIR of `2^n_bits` rows: every value canonical (`< r`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedColumns {
    n_bits: u64,
    /// One vector of `N` values per column, in the order of the pilout's `fixedCols`, which is
    /// the order of `constPolsMap`.
    columns: Vec<Vec<FrBytes>>,
}

impl FixedColumns {
    /// `columns` on a domain of `2^n_bits` rows: each must have `N` values.
    pub fn new(n_bits: u64, columns: Vec<Vec<FrBytes>>) -> Result<Self, SetupError> {
        if n_bits > MAX_NBITS {
            return Err(SetupError::NBits { n_bits });
        }
        let n_rows = 1usize << n_bits;
        if let Some((column, values)) = columns.iter().enumerate().find(|(_, c)| c.len() != n_rows) {
            return Err(SetupError::FixedValues { column, n_values: values.len(), n_rows });
        }
        Ok(Self { n_bits, columns })
    }

    /// The fixed columns of `air`, decoded from the pilout's values (big-endian bytes of any
    /// length, none for 0). Refuses a value that is not below `r`
    /// (pilfflonk/docs/README.md#what-the-setup-refuses): a pilout over BN254 has none, and
    /// reducing it would hide a compiler bug. `air` must have a power of two of rows, of at most
    /// `2^28` ([`crate::validate::validate`] checks it).
    pub fn from_air(air: &pb::Air) -> Result<Self, SetupError> {
        let num_rows = air.num_rows.unwrap_or(0);
        if !num_rows.is_power_of_two() || u64::from(num_rows.trailing_zeros()) > MAX_NBITS {
            let label = air.name.clone().unwrap_or_default();
            return Err(SetupError::NumRows { air: label, num_rows });
        }
        let n_rows = num_rows as usize;
        let mut columns = Vec::with_capacity(air.fixed_cols.len());
        for (column, col) in air.fixed_cols.iter().enumerate() {
            if col.values.len() != n_rows {
                return Err(SetupError::FixedValues { column, n_values: col.values.len(), n_rows });
            }
            let values = col
                .values
                .iter()
                .enumerate()
                .map(|(row, bytes)| {
                    decode_pilout_value(bytes).ok_or_else(|| SetupError::ConstantNotBelowR {
                        location: format!("row {row} of fixed column {column}"),
                        value: BigUint::from_bytes_be(bytes).to_str_radix(10),
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            columns.push(values);
        }
        Self::new(u64::from(num_rows.trailing_zeros()), columns)
    }

    pub fn n_bits(&self) -> u64 {
        self.n_bits
    }

    /// `N = 2^n_bits`.
    pub fn n_rows(&self) -> usize {
        1 << self.n_bits
    }

    pub fn n_columns(&self) -> usize {
        self.columns.len()
    }

    /// The `N` values of column `i`, if there is one.
    pub fn column(&self, i: usize) -> Option<&[FrBytes]> {
        self.columns.get(i).map(Vec::as_slice)
    }

    /// Writes `<air>.const` (see [the module](self)) at `path`, replacing any file there.
    pub fn write_const(&self, path: &Path) -> Result<(), SetupError> {
        let io = SetupError::io(path);
        let file = File::create(path).map_err(SetupError::io(path))?;
        let mut writer = BufWriter::new(file);
        let mut row = Vec::with_capacity(self.n_columns() * FIELD_BYTES);
        let result = (0..self.n_rows())
            .try_for_each(|i| {
                row.clear();
                for column in &self.columns {
                    row.extend_from_slice(&column[i].to_le_bytes());
                }
                writer.write_all(&row)
            })
            .and_then(|()| writer.flush());
        result.map_err(io)
    }

    /// Reads the `<air>.const` at `path` of `n_columns` columns of `2^n_bits` rows. Refuses a file
    /// of any other size, and a value that is not below `r`.
    pub fn read_const(path: &Path, n_bits: u64, n_columns: usize) -> Result<Self, SetupError> {
        if n_bits > MAX_NBITS {
            return Err(SetupError::NBits { n_bits });
        }
        let n_rows = 1usize << n_bits;
        let bytes = std::fs::read(path).map_err(SetupError::io(path))?;
        let expected = n_rows.checked_mul(n_columns).and_then(|n| n.checked_mul(FIELD_BYTES));
        let Some(expected) = expected else {
            let reason = format!("{n_columns} columns of {n_rows} rows do not fit in memory");
            return Err(SetupError::ConstFile { path: path.to_path_buf(), reason });
        };
        if bytes.len() != expected {
            let reason = format!(
                "{} bytes, not the {expected} of {n_columns} columns of {n_rows} rows of {FIELD_BYTES} bytes",
                bytes.len()
            );
            return Err(SetupError::ConstFile { path: path.to_path_buf(), reason });
        }
        let mut columns = vec![Vec::with_capacity(n_rows); n_columns];
        for (index, value) in bytes.chunks_exact(FIELD_BYTES).enumerate() {
            let mut le = [0u8; FIELD_BYTES];
            le.copy_from_slice(value);
            let value = FrBytes::from_le_bytes(le).map_err(|_| SetupError::ConstFile {
                path: path.to_path_buf(),
                reason: format!(
                    "the value of row {} of column {} is not below r",
                    index / n_columns,
                    index % n_columns
                ),
            })?;
            columns[index % n_columns].push(value);
        }
        Self::new(n_bits, columns)
    }
}

/// A value of a pilout, big-endian bytes of any length (none for 0), as a canonical `Fr`; `None`
/// if it is not below `r`.
fn decode_pilout_value(bytes: &[u8]) -> Option<FrBytes> {
    let significant = &bytes[bytes.iter().position(|&b| b != 0).unwrap_or(bytes.len())..];
    if significant.len() > FIELD_BYTES {
        return None;
    }
    let mut le = [0u8; FIELD_BYTES];
    for (to, from) in le.iter_mut().zip(significant.iter().rev()) {
        *to = *from;
    }
    FrBytes::from_le_bytes(le).ok()
}
