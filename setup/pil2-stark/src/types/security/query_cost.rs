//! Merkle query hash-cost model used by the FRI and STIR parameter calculators
//! to size grinding against per-query work.

/// Hashes to verify ONE Merkle authentication path over `n_leafs` leaves at the
/// given arity: `(arity - 1) * ceil(log_arity(n_leafs))`.
fn calculate_mtp_hashes(tree_arity: u64, n_leafs: f64) -> f64 {
    (tree_arity as f64 - 1.0) * (n_leafs.log2() / (tree_arity as f64).log2()).ceil()
}

/// Total Merkle hashes a verifier performs per query across a fold schedule: the
/// input codeword opening plus one opening at each intermediate folded layer.
pub(crate) fn calculate_query_num_hashes(tree_arity: u64, codeword_length: f64, folding_factors: &[u64]) -> f64 {
    if folding_factors.is_empty() {
        return 0.0;
    }
    let mut acc_folding_factor: f64 = 1.0;
    let mut total_hashes: f64 = 0.0;
    for &ff in &folding_factors[..folding_factors.len() - 1] {
        let n_leafs = codeword_length / acc_folding_factor;
        total_hashes += ff as f64 * calculate_mtp_hashes(tree_arity, n_leafs);
        acc_folding_factor *= ff as f64;
    }
    let n_leafs_input = codeword_length;
    total_hashes += folding_factors[0] as f64 * calculate_mtp_hashes(tree_arity, n_leafs_input);
    total_hashes
}
