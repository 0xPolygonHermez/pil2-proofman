#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub lengths: Vec<usize>,
    pub id: usize,
    pub stage: usize,
}

/// Generates multi-array indexes recursively.
///
/// # Arguments
/// - `symbols`: A mutable vector to store the generated `Symbol` structs.
/// - `name`: The name of the symbol.
/// - `lengths`: A slice containing the lengths of each dimension.
/// - `pol_id`: The starting ID for the symbols.
/// - `stage`: The stage associated with the symbol.
/// - `indexes`: The current indexes being processed (used in recursion, default to empty).
///
/// # Returns
/// The next available `pol_id` after processing all indexes.
pub fn generate_multi_array_indexes(
    symbols: &mut Vec<Symbol>,
    name: &str,
    lengths: &[usize],
    mut pol_id: usize,
    stage: usize,
    indexes: Vec<usize>,
) -> usize {
    if indexes.len() == lengths.len() {
        symbols.push(Symbol { name: name.to_string(), lengths: indexes, id: pol_id, stage });
        return pol_id + 1;
    }

    for i in 0..lengths[indexes.len()] {
        pol_id =
            generate_multi_array_indexes(symbols, name, lengths, pol_id, stage, [indexes.clone(), vec![i]].concat());
    }

    pol_id
}
