use std::fmt;

/// Polynomial Commitment Scheme.
///
/// A PCS is constructed from its *free* parameters (field, rate, folding
/// schedule, target security, regime kind) and deduces the rest — query
/// counts, grinding bits, gap widening — at construction time. Hence the
/// methods below take no regime: the instance already owns a fully solved
/// parameterization.
pub trait Pcs {
    /// Returns the name of the PCS.
    fn identifier(&self) -> &'static str;

    /// PCS-specific security levels, phase by phase.
    /// Entries are (descriptive label, bits of security).
    fn security_levels(&self) -> Vec<(String, u32)>;

    /// The minimum over all security levels.
    fn total_security_bits(&self) -> u32 {
        self.security_levels().into_iter().map(|(_, b)| b).min().unwrap_or(0)
    }

    /// Total Merkle openings.
    fn num_merkle_openings(&self) -> u64;

    /// Approximate verifier hashes spent.
    fn total_query_hashes(&self) -> f64;

    /// Estimated worst-case proof size in bits.
    fn proof_size_bits(&self) -> u64;

    /// Description of the parameters of the PCS.
    fn parameter_summary(&self) -> String;
}

/// How the coefficients c_i are chosen when combining `batch_size`
/// polynomials f_i into f_batch = Σ c_i · f_i.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Batching {
    /// c_i = γ^i for a random challenge γ.
    /// "Batching over parameterized curves", BCIKS20 Thm 6.2.
    /// Error depends on batch_size (ℓ in BCIKS20).
    Powers,
    /// c_0 = 1, c_i = r_i for independent random r_i.
    /// "Batching over affine spaces", BCIKS20 Thm 1.6.
    /// Error independent of batch_size.
    Affine,
    /// c_i = eq(r, i). Multilinear batching, BCHKS25 §4.1.
    /// Only usable with multilinear PCSs.
    Multilinear,
}

impl fmt::Display for Batching {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Batching::Powers => write!(f, "Powers"),
            Batching::Affine => write!(f, "Affine"),
            Batching::Multilinear => write!(f, "Multilinear"),
        }
    }
}

/// Bits of security from an error probability.
pub fn bits_of_security_from_error(epsilon: f64) -> u32 {
    debug_assert!(epsilon > 0.0 && epsilon.is_finite(), "invalid error {epsilon}");
    (-epsilon.log2()).floor().max(0.0) as u32
}

pub fn security_from_error(epsilon: f64) -> f64 {
    debug_assert!(epsilon > 0.0 && epsilon.is_finite(), "invalid error {epsilon}");
    -epsilon.log2()
}

/// ε ↦ ε · 2^-g. exp2 of an integer is exact, so no precision is lost.
pub fn apply_grinding(epsilon: f64, grinding_bits: u32) -> f64 {
    epsilon * f64::exp2(-(grinding_bits as f64))
}

/// Hashes to verify one Merkle path in a tree of `n_leafs` leaves.
pub fn merkle_path_hashes(tree_arity: u64, n_leafs: f64) -> f64 {
    (tree_arity as f64 - 1.0) * (n_leafs.log2() / (tree_arity as f64).log2()).ceil()
}

/// Approximate verifier hashes for one query opening in a tree over a domain
/// of `2^log_domain` points packed in cosets of size `2^k`: hash the coset
/// leaf plus verify one Merkle path.
pub fn coset_opening_hashes(log_domain: u32, k: u32, tree_arity: u64) -> f64 {
    let n_leafs = f64::exp2((log_domain - k) as f64);
    f64::exp2(k as f64) + merkle_path_hashes(tree_arity, n_leafs)
}

/// Size in bits of one Merkle opening in a tree of `n_leafs` leaves, each
/// holding a tuple of `tuple_size` elements of `element_size_bits`: the leaf
/// tuple itself plus its authentication path ((arity − 1) digests per level).
/// Worst case: shared path prefixes across openings are not deduplicated.
pub fn merkle_opening_size_bits(
    n_leafs: f64,
    tuple_size: u64,
    element_size_bits: u64,
    tree_arity: u64,
    hash_size_bits: u64,
) -> f64 {
    let depth = (n_leafs.log2() / (tree_arity as f64).log2()).ceil();
    (tuple_size * element_size_bits) as f64 + (tree_arity - 1) as f64 * depth * hash_size_bits as f64
}
