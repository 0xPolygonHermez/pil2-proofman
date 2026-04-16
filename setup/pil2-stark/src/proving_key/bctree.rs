use proofman_starks_lib_c::compute_const_tree_c;

/// Build the constant polynomial Merkle tree and return the 4-element root.
///
/// Delegates to the C++ `build_const_tree_c` function in libstarks.a, which
/// reads `const_path`, parses tree parameters from `starkinfo_path`, extends
/// the polynomials, builds the Poseidon2 Merkle tree, and writes the root as
/// a JSON array to `verkey_path`.
pub fn compute_const_tree(const_path: &str, starkinfo_path: &str, verkey_path: &str) -> [u64; 4] {
    compute_const_tree_c(const_path, starkinfo_path, "", verkey_path)
}
