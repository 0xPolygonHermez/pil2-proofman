use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use num_bigint::BigUint;
use num_traits::Zero;
use serde_json::Value;

pub use pilout::pilout::Symbol;
pub use pilout::pilout::SymbolType;

/// Generates multi-array indexes recursively.
pub fn generate_multi_array_indexes(
    symbols: &mut Vec<Symbol>,
    name: &str,
    lengths: &[u32],
    pol_id: u32,
    stage: u32,
    indexes: Vec<u32>,
) -> u32 {
    if indexes.len() == lengths.len() {
        symbols.push(Symbol {
            name: name.to_string(),
            lengths: indexes.clone(),
            id: pol_id,
            stage: Some(stage),
            dim: 1,
            air_group_id: None,
            air_id: None,
            commit_id: None,
            debug_line: None,
            r#type: SymbolType::WitnessCol as i32,
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
    pub n: u32,
    pub n_cols: u32,
    pub buffer: Vec<BigUint>,
    pub field_mod: BigUint,
}

impl ColsPil2 {
    pub fn new(symbols: Vec<Symbol>, degree: u32, field_mod: BigUint) -> Self {
        let n_cols = symbols.len() as u32; // Convert `usize` to `u32`
        let buffer = vec![BigUint::zero(); degree as usize * n_cols as usize];

        Self {
            symbols,
            n: degree,
            n_cols, // Remains u32
            buffer,
            field_mod,
        }
    }

    /// Converts `ColsPil2` into a `HashMap<String, Vec<Value>>` where column names are based on symbols.
    pub fn to_hashmap(&self) -> HashMap<String, Vec<Value>> {
        let mut map = HashMap::new();

        // Ensure buffer is correctly grouped into columns
        for (i, symbol) in self.symbols.iter().enumerate() {
            let key = symbol.name.clone();

            let values: Vec<Value> = self
                .buffer
                .chunks(self.n_cols as usize)
                .map(|chunk| serde_json::to_value(chunk[i].clone()).unwrap())
                .collect();

            map.insert(key, values);
        }

        map
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
pub fn generate_fixed_cols(symbols: Vec<Symbol>, degree: u32, field_mod: BigUint) -> ColsPil2 {
    let mut fixed_symbols = Vec::new();

    for symbol in symbols.iter() {
        if symbol.stage.unwrap_or(0) != 0 {
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
                symbol.stage.unwrap_or(0),
                vec![],
            );
        }
    }

    ColsPil2::new(fixed_symbols, degree, field_mod)
}

/// Generates witness columns for the computation process.
pub fn generate_wtns_cols(symbols: Vec<Symbol>, degree: u32, field_mod: BigUint) -> ColsPil2 {
    let mut witness_symbols = Vec::new();

    for symbol in symbols.iter() {
        if symbol.stage.unwrap_or(1) != 1 {
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
                symbol.stage.unwrap_or(1),
                vec![],
            );
        }
    }

    ColsPil2::new(witness_symbols, degree, field_mod)
}
