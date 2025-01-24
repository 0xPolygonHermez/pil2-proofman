use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};
use num_bigint::BigUint;
use num_traits::Zero;

/// Represents a symbol in the computation process.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub id: usize,
    pub stage: usize,
    pub lengths: Vec<usize>,
}

/// Generates multi-array indexes recursively.
pub fn generate_multi_array_indexes(
    symbols: &mut Vec<Symbol>,
    name: &str,
    lengths: &[usize],
    mut pol_id: usize,
    stage: usize,
    indexes: Vec<usize>,
) -> usize {
    if indexes.len() == lengths.len() {
        symbols.push(Symbol { name: name.to_string(), id: pol_id, stage, lengths: indexes });
        return pol_id + 1;
    }

    for i in 0..lengths[indexes.len()] {
        pol_id =
            generate_multi_array_indexes(symbols, name, lengths, pol_id, stage, [indexes.clone(), vec![i]].concat());
    }

    pol_id
}

/// Represents the columnar data structure used in PIL computations.
pub struct ColsPil2 {
    pub symbols: Vec<Symbol>,
    pub n: usize,
    pub n_cols: usize,
    pub buffer: Vec<BigUint>,
    pub field_mod: BigUint,
}

impl ColsPil2 {
    pub fn new(symbols: Vec<Symbol>, degree: usize, field_mod: BigUint) -> Self {
        let n_cols = symbols.len();
        let buffer = vec![BigUint::zero(); degree * n_cols];
        Self { symbols, n: degree, n_cols, buffer, field_mod }
    }

    /// Sets a value in the multi-dimensional array structure.
    pub fn set_value_multi_array(arr: &mut Vec<VecDeque<BigUint>>, indexes: &[usize], value: BigUint) {
        if indexes.len() == 1 {
            // If at the last index, insert the value directly into the VecDeque
            while arr.len() <= indexes[0] {
                arr.push(VecDeque::new());
            }
            arr[indexes[0]].push_back(value);
        } else {
            // Ensure the nested structure exists
            while arr.len() <= indexes[0] {
                arr.push(VecDeque::new());
            }
            let next_index = indexes[0];
            let next_arr = &mut arr[next_index];

            // Ensure the next level is correctly initialized as a Vec<VecDeque<BigUint>>
            let mut nested_arr = Vec::new();
            nested_arr.push(VecDeque::new());

            ColsPil2::set_value_multi_array(&mut nested_arr, &indexes[1..], value);

            // Replace the existing VecDeque with a new structure (if needed)
            if next_arr.is_empty() {
                *next_arr = nested_arr[0].clone();
            }
        }
    }

    /// Saves the buffer to a file.
    pub fn save_to_file(&self, file_name: &str) -> std::io::Result<()> {
        let mut file = File::create(file_name)?;
        for value in &self.buffer {
            let bytes = value.to_bytes_be();
            file.write_all(&bytes)?;
        }
        Ok(())
    }

    /// Loads the buffer from a file.
    pub fn load_from_file(&mut self, file_name: &str) -> std::io::Result<()> {
        let mut file = File::open(file_name)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        for chunk in buffer.chunks(8) {
            let value = BigUint::from_bytes_be(chunk);
            self.buffer.push(value);
        }
        Ok(())
    }
}

/// Generates fixed columns for the computation process.
pub fn generate_fixed_cols(symbols: Vec<Symbol>, degree: usize, field_mod: BigUint) -> ColsPil2 {
    let mut fixed_symbols = Vec::new();

    for symbol in symbols.iter() {
        if symbol.stage != 0 {
            continue;
        }
        if symbol.lengths.is_empty() {
            fixed_symbols.push(symbol.clone());
        } else {
            generate_multi_array_indexes(
                &mut fixed_symbols,
                &symbol.name,
                &symbol.lengths,
                symbol.id,
                symbol.stage,
                vec![],
            );
        }
    }

    ColsPil2::new(fixed_symbols, degree, field_mod)
}

/// Generates witness columns for the computation process.
pub fn generate_wtns_cols(symbols: Vec<Symbol>, degree: usize, field_mod: BigUint) -> ColsPil2 {
    let mut witness_symbols = Vec::new();

    for symbol in symbols.iter() {
        if symbol.stage != 1 {
            continue;
        }
        if symbol.lengths.is_empty() {
            witness_symbols.push(symbol.clone());
        } else {
            generate_multi_array_indexes(
                &mut witness_symbols,
                &symbol.name,
                &symbol.lengths,
                symbol.id,
                symbol.stage,
                vec![],
            );
        }
    }

    ColsPil2::new(witness_symbols, degree, field_mod)
}
