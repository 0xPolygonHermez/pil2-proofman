//! The fixed columns of an AIR, and `<air>.const` (pilfflonk/docs/formats.md#fixed-columns).
//!
//! `<air>.const` holds the fixed columns row by row, each value a canonical `Fr` of 32 bytes
//! little-endian, with no header: the value of column `c` at row `i` is at byte `(i·C + c)·32`,
//! and the file has exactly `N·C·32` bytes. It is the layout of the STARK's `.const` (8 bytes a
//! value there) and of the witness files (pilfflonk/docs/formats.md#witness-directory).
//!
//! The values are the pilout's, but for the columns it declares `#pragma fixed_external`, which it
//! has none of: a caller of the library gives those ([`ExternalFixedColumn`],
//! [`FixedColumns::from_air_with_external`]), as the STARK's recursive setup gives `plonk2pil`'s
//! `S` and `C` to `write_const_file` (`setup/pil2-stark/src/io/fixed_cols.rs`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, SymbolType};
use pil_info::types::pilout_info::format_symbols;
use pil_info::FieldCfg;
use proofman_pilfflonk::field::FIELD_BYTES;
use proofman_pilfflonk::global_info::MAX_NBITS;
use proofman_pilfflonk::FrBytes;

use crate::error::SetupError;
use crate::validate::{air_label, ValidAir};

/// The values of a fixed column that the pilout declares `#pragma fixed_external`, and so has none
/// of (pilfflonk/docs/formats.md#fixed-columns): computed by the caller, as `plonk2pil` computes
/// the `S` and `C` of an AIR that verifies a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFixedColumn {
    /// The name of the column's symbol in the pilout, which `constPolsMap` has too:
    /// `Connection.S1`.
    pub name: String,
    /// Its position among the fixed columns of the AIR of that name, in the pilout's order: its
    /// index in the array (row-major if the array has several dimensions), `plonk2pil`'s
    /// `FixedPol::index`, and 0 for a column that is not an array.
    pub index: usize,
    /// Its `N` values, row by row. They are below `r`, as every `FrBytes` is.
    pub values: Vec<FrBytes>,
}

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
    /// length, none for 0). Refuses a column without a value per row, and a value that is not
    /// below `r` (pilfflonk/docs/README.md#what-the-setup-refuses): a pilout over BN254 has none,
    /// and reducing it would hide a compiler bug. `air` must have a power of two of rows, of at
    /// most `2^28` ([`crate::validate::validate`] checks it).
    pub fn from_air(air: &pb::Air) -> Result<Self, SetupError> {
        let n_bits = n_bits_of(air)?;
        let n_rows = 1usize << n_bits;
        let columns = air
            .fixed_cols
            .iter()
            .enumerate()
            .map(|(column, col)| decode_column(column, col, n_rows))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(n_bits, columns)
    }

    /// The fixed columns of `air`, the AIR of `pilout`, in the order of `fixedCols`: the pilout's
    /// values, decoded as [`Self::from_air`] decodes them, and in each column the pilout has none
    /// of (one it declares `#pragma fixed_external`) those of the external column that names it,
    /// given in any order. Refuses an external column of no fixed column of the AIR, of one the
    /// pilout has values of, of a column named before, or without a value per row; and a column
    /// without values that no external column fills. Without external columns, it is
    /// [`Self::from_air`], which refuses a column without values as any other without a value per
    /// row.
    pub fn from_air_with_external(
        pilout: &pb::PilOut,
        air: ValidAir<'_>,
        external: Vec<ExternalFixedColumn>,
    ) -> Result<Self, SetupError> {
        if external.is_empty() {
            return Self::from_air(air.air);
        }
        let n_bits = n_bits_of(air.air)?;
        let n_rows = 1usize << n_bits;
        let names = column_names(pilout, air)?;
        let placed = place_external(air, &names, external, n_rows)?;
        let columns = air
            .air
            .fixed_cols
            .iter()
            .zip(placed)
            .enumerate()
            .map(|(column, (col, placed))| match (placed, &names[column]) {
                (Some(values), _) => Ok(values),
                (None, Some((name, index))) if col.values.is_empty() => {
                    Err(SetupError::ExternalFixedMissing { column, name: name.clone(), index: *index })
                }
                // A column with values; or one without values or a symbol, which no external column
                // can name, refused as from_air refuses it.
                (None, _) => decode_column(column, col, n_rows),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(n_bits, columns)
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

/// `log2(N)` of `air`, which must have a power of two of rows, of at most `2^28`.
fn n_bits_of(air: &pb::Air) -> Result<u64, SetupError> {
    let num_rows = air.num_rows.unwrap_or(0);
    if !num_rows.is_power_of_two() || u64::from(num_rows.trailing_zeros()) > MAX_NBITS {
        let label = air.name.clone().unwrap_or_default();
        return Err(SetupError::NumRows { air: label, num_rows });
    }
    Ok(u64::from(num_rows.trailing_zeros()))
}

/// The `n_rows` values of `col`, fixed column `column` of its AIR, decoded from the pilout (see
/// [`FixedColumns::from_air`]).
fn decode_column(column: usize, col: &pb::FixedCol, n_rows: usize) -> Result<Vec<FrBytes>, SetupError> {
    if col.values.len() != n_rows {
        return Err(SetupError::FixedValues { column, n_values: col.values.len(), n_rows });
    }
    col.values
        .iter()
        .enumerate()
        .map(|(row, bytes)| {
            decode_pilout_value(bytes).ok_or_else(|| SetupError::ConstantNotBelowR {
                location: format!("row {row} of fixed column {column}"),
                value: BigUint::from_bytes_be(bytes).to_str_radix(10),
            })
        })
        .collect()
}

/// The name and the index ([`ExternalFixedColumn`]) of each fixed column of `air`, the AIR of
/// `pilout`, from the symbols as `pil-info` expands them into `constPolsMap`; `None` for a column
/// without a symbol. Refuses a symbol of a column the AIR does not have, and two of one column.
fn column_names(pilout: &pb::PilOut, air: ValidAir<'_>) -> Result<Vec<Option<(String, usize)>>, SetupError> {
    let of_air = |s: &&pb::Symbol| {
        s.r#type == SymbolType::FixedCol as i32
            && s.air_group_id == Some(air.airgroup_id as u32)
            && s.air_id == Some(air.air_id as u32)
    };
    let symbols: Vec<pb::Symbol> = pilout.symbols.iter().filter(of_air).cloned().collect();
    // One per column, at its position in fixedCols (its stageId); the field only sets the
    // dimension of the columns of stage 2 or above.
    let mut columns = format_symbols(&symbols, &[], &[], &[], &FieldCfg::bn254()).map_err(SetupError::Passes)?;
    columns.sort_by_key(|s| s.stage_id);

    let n_columns = air.air.fixed_cols.len();
    let mut names = vec![None; n_columns];
    let mut next_index: BTreeMap<&str, usize> = BTreeMap::new();
    for symbol in &columns {
        let Some((column, slot)) = symbol.stage_id.and_then(|c| names.get_mut(c).map(|slot| (c, slot))) else {
            return Err(SetupError::InvalidPilout(format!(
                "the symbol {} is of a fixed column the AIR does not have: it has {n_columns}",
                symbol.name
            )));
        };
        if let Some((other, _)) = slot {
            return Err(SetupError::InvalidPilout(format!(
                "fixed column {column} has two symbols, {other} and {}",
                symbol.name
            )));
        }
        let index = next_index.entry(&symbol.name).or_default();
        *slot = Some((symbol.name.clone(), *index));
        *index += 1;
    }
    Ok(names)
}

/// The values of each external column at the fixed column of `air` it names (`names`, of
/// [`column_names`]), checked: a column of the AIR, which the pilout has no values of, named once,
/// and `n_rows` values.
fn place_external(
    air: ValidAir<'_>,
    names: &[Option<(String, usize)>],
    external: Vec<ExternalFixedColumn>,
    n_rows: usize,
) -> Result<Vec<Option<Vec<FrBytes>>>, SetupError> {
    let columns: BTreeMap<(&str, usize), usize> = names
        .iter()
        .enumerate()
        .filter_map(|(column, name)| name.as_ref().map(|(name, index)| ((name.as_str(), *index), column)))
        .collect();
    let mut placed = vec![None; names.len()];
    for ExternalFixedColumn { name, index, values } in external {
        let Some(&column) = columns.get(&(name.as_str(), index)) else {
            let air = air_label(air.air, air.airgroup_id, air.air_id);
            return Err(SetupError::ExternalFixedUnknown { air, name, index });
        };
        if !air.air.fixed_cols[column].values.is_empty() {
            return Err(SetupError::ExternalFixedNotEmpty { name, index, column });
        }
        if placed[column].is_some() {
            return Err(SetupError::ExternalFixedTwice { name, index });
        }
        if values.len() != n_rows {
            return Err(SetupError::ExternalFixedValues { name, index, n_values: values.len(), n_rows });
        }
        placed[column] = Some(values);
    }
    Ok(placed)
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
