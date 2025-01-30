use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Write};
use num_bigint::BigUint;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,              // Name of the symbol
    pub id: usize,                 // Unique identifier
    pub stage: usize,              // Stage of the symbol in computation
    pub dim: usize,                // Dimension of the symbol (default 1)
    pub air_group_id: usize,       // AIR Group ID (equivalent to `stage_id`)
    pub pol_deg: Option<usize>,    // Polynomial degree (optional)
    pub values: Option<Vec<u128>>, // Polynomial values (used for "fixed" symbols)
    pub symbol_type: String,       // "challenge", "fixed", "witness", "custom"
    pub pol_id: Option<usize>,     // ID of polynomial (if applicable)
    pub commit_id: Option<usize>,  // Commitment ID (if applicable)
    pub lengths: Vec<usize>,       // Lengths of polynomials (replaces `Vec<u128>` for generality)
}

/// Generates multi-array indexes recursively.
pub fn generate_multi_array_indexes(
    symbols: &mut Vec<Symbol>,
    name: &str,
    lengths: &[usize],
    pol_id: usize,
    stage: usize,
    indexes: Vec<usize>,
) -> usize {
    if indexes.len() == lengths.len() {
        symbols.push(Symbol {
            name: name.to_string(),
            lengths: indexes.clone(),
            id: pol_id,
            stage,
            dim: 1,          // Default dimension
            air_group_id: 0, // Default value, update as needed
            pol_deg: None,
            values: None,
            symbol_type: "witness".to_string(), // Default type, update as needed
            pol_id: Some(pol_id),
            commit_id: None,
        });
        return pol_id + 1;
    }

    let mut current_pol_id = pol_id;
    for i in 0..lengths[indexes.len()] {
        let mut new_indexes = indexes.clone();
        new_indexes.push(i);
        current_pol_id = generate_multi_array_indexes(symbols, name, lengths, current_pol_id, stage, new_indexes);
    }

    current_pol_id
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

    /// Converts `ColsPil2` into a `HashMap<String, Vec<Value>>` where column names are based on symbols.
    pub fn to_hashmap(&self) -> HashMap<String, Vec<Value>> {
        let mut map = HashMap::new();

        // Ensure buffer is correctly grouped into columns
        for (i, symbol) in self.symbols.iter().enumerate() {
            let key = symbol.name.clone(); // Use actual symbol names instead of "col_{}"

            let values: Vec<Value> = self
                .buffer
                .chunks(self.n_cols) // Group buffer values into columns
                .map(|chunk| serde_json::to_value(chunk[i].clone()).unwrap()) // Extract the i-th element for this column
                .collect();

            map.insert(key, values);
        }

        map
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
