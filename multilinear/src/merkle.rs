//! Prover-side Merkle tree that keeps all levels so sibling paths can be
//! produced.

use fields::{linear_hash_seq, Field, Goldilocks, Hash};

pub const DIGEST_CELLS: usize = 4;

pub struct MerkleTree {
    pub arity: u64,
    /// `levels[0]` = leaf digests, each level a flat vec of 4-cell digests;
    /// the last level contains the single root digest.
    levels: Vec<Vec<Goldilocks>>,
}

impl MerkleTree {
    /// Build a tree over `leaves` (raw leaf data, hashed here with `H`).
    pub fn new<H: Hash<Goldilocks>>(leaves: &[Vec<Goldilocks>], arity: u64) -> Self {
        assert!(!leaves.is_empty());
        assert_eq!(arity as usize * DIGEST_CELLS, H::WIDTH, "arity * 4 must equal H::WIDTH");

        let mut level = Vec::with_capacity(leaves.len() * DIGEST_CELLS);
        for leaf in leaves {
            let st = linear_hash_seq::<Goldilocks, H>(leaf);
            level.extend_from_slice(&st.as_ref()[..DIGEST_CELLS]);
        }

        let mut levels = vec![level];
        while levels.last().unwrap().len() > DIGEST_CELLS {
            let cur = levels.last().unwrap();
            let n_dig = cur.len() / DIGEST_CELLS;
            let padded = n_dig.div_ceil(arity as usize) * arity as usize;
            let mut cur_padded = cur.clone();
            cur_padded.resize(padded * DIGEST_CELLS, Goldilocks::ZERO);

            let mut next = Vec::with_capacity((padded / arity as usize) * DIGEST_CELLS);
            for group in 0..(padded / arity as usize) {
                let mut st = H::State::default();
                let w = arity as usize * DIGEST_CELLS;
                st.as_mut()[..w].copy_from_slice(&cur_padded[group * w..(group + 1) * w]);
                H::hash(&mut st);
                next.extend_from_slice(&st.as_ref()[..DIGEST_CELLS]);
            }
            levels.push(next);
        }

        Self { arity, levels }
    }

    /// Build the tree by offloading the (Poseidon2) hashing to the C++
    /// AVX/AVX512/threaded kernel, then slicing its flat node buffer into the
    /// same per-level digest layout `new` produces. Byte-compatible with
    /// `new::<Poseidon2_16>` (guarded by `ffi_tree_matches_pure_rust`), so the
    /// pure-Rust verifier validates the resulting roots and paths unchanged.
    #[cfg(feature = "ffi-poseidon2")]
    pub fn from_ffi(leaves: &[Vec<Goldilocks>], arity: u64) -> Self {
        use fields::PrimeField64;
        assert!(!leaves.is_empty());
        debug_assert_eq!(arity, 4, "C++ Poseidon2 GL merkletree uses sponge width 16 == arity 4");
        let num_rows = leaves.len() as u64;
        let width = leaves[0].len();
        assert!(leaves.iter().all(|l| l.len() == width), "leaves must have uniform width");

        // Row-major, canonical-u64 input for the C++ kernel.
        let mut input = Vec::with_capacity(leaves.len() * width);
        for leaf in leaves {
            input.extend(leaf.iter().map(|v| v.as_canonical_u64()));
        }

        // Full node buffer (all levels including per-level padding).
        let num_nodes = tree_num_nodes(num_rows, arity);
        let mut nodes = vec![0u64; (num_nodes * DIGEST_CELLS as u64) as usize];
        proofman_starks_lib_c::poseidon2_merkletree_c(
            nodes.as_mut_ptr(),
            input.as_ptr(),
            width as u64,
            num_rows,
            arity,
            1,
        );

        // Slice the flat buffer into per-level (unpadded) digests — the layout
        // `path`/`root` expect. Mirrors `fields::partial_merkle_tree`'s walk.
        let flat: Vec<Goldilocks> = nodes.into_iter().map(Goldilocks::new).collect();
        let cells = |n: u64| (n * DIGEST_CELLS as u64) as usize;
        let mut levels: Vec<Vec<Goldilocks>> = Vec::new();
        let mut pending = num_rows;
        let mut start = 0usize;
        levels.push(flat[start..start + cells(pending)].to_vec());
        while pending > 1 {
            let extra = (arity - (pending % arity)) % arity;
            let next_n = pending.div_ceil(arity);
            start += cells(pending + extra);
            levels.push(flat[start..start + cells(next_n)].to_vec());
            pending = next_n;
        }

        Self { arity, levels }
    }

    pub fn root(&self) -> [Goldilocks; 4] {
        let top = self.levels.last().unwrap();
        [top[0], top[1], top[2], top[3]]
    }

    /// Sibling path for leaf `idx`: per level, the `arity − 1` sibling digests in position order.
    pub fn path(&self, mut idx: u64) -> Vec<Vec<Goldilocks>> {
        let arity = self.arity;
        let mut mp = Vec::with_capacity(self.levels.len().saturating_sub(1));
        for level in &self.levels[..self.levels.len() - 1] {
            let n_dig = (level.len() / DIGEST_CELLS) as u64;
            let group = idx / arity;
            let pos = idx % arity;
            let mut sibs = Vec::with_capacity((arity as usize - 1) * DIGEST_CELLS);
            for i in 0..arity {
                if i == pos {
                    continue;
                }
                let child = group * arity + i;
                if child < n_dig {
                    let off = (child * DIGEST_CELLS as u64) as usize;
                    sibs.extend_from_slice(&level[off..off + DIGEST_CELLS]);
                } else {
                    sibs.extend_from_slice(&[Goldilocks::ZERO; DIGEST_CELLS]);
                }
            }
            mp.push(sibs);
            idx = group;
        }
        mp
    }

    pub fn num_leaves(&self) -> usize {
        self.levels[0].len() / DIGEST_CELLS
    }
}

/// Total node count (all levels incl. per-level arity padding) for a tree of
/// `num_leaves` leaves — matches the C++ `getTreeNumElements/HASH_SIZE` and
/// `fields::partial_merkle_tree` formulas.
#[cfg(feature = "ffi-poseidon2")]
fn tree_num_nodes(num_leaves: u64, arity: u64) -> u64 {
    let mut num_nodes = num_leaves;
    let mut level = num_leaves;
    while level > 1 {
        num_nodes += (arity - (level % arity)) % arity;
        let next = level.div_ceil(arity);
        num_nodes += next;
        level = next;
    }
    num_nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::{partial_merkle_tree, verify_mt, Poseidon1_16, PrimeField64};
    use rand::{rng, RngExt};

    fn random_leaves(n: usize, len: usize) -> Vec<Vec<Goldilocks>> {
        let mut r = rng();
        (0..n).map(|_| (0..len).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect()).collect()
    }

    #[test]
    fn root_matches_partial_merkle_tree() {
        for n in [1usize, 3, 4, 16, 21] {
            let leaves = random_leaves(n, 7);
            let tree = MerkleTree::new::<Poseidon1_16>(&leaves, 4);

            // partial_merkle_tree consumes leaf *digests*
            let mut digests = Vec::with_capacity(n * 4);
            for leaf in &leaves {
                let st = linear_hash_seq::<Goldilocks, Poseidon1_16>(leaf);
                digests.extend_from_slice(&st.as_ref()[..4]);
            }
            let expected = partial_merkle_tree::<Goldilocks, Poseidon1_16>(&digests, n as u64, 4);
            assert_eq!(tree.root(), expected, "n = {n}");
        }
    }

    #[test]
    fn paths_verify_with_verify_mt() {
        let n = 21;
        let leaves = random_leaves(n, 6);
        let tree = MerkleTree::new::<Poseidon1_16>(&leaves, 4);
        let root = tree.root();
        for idx in [0usize, 1, 5, 15, 20] {
            let mp = tree.path(idx as u64);
            assert!(
                verify_mt::<Goldilocks, Poseidon1_16, Poseidon1_16>(&root, &[], &mp, idx as u64, &leaves[idx], 4, 0),
                "path for leaf {idx} must verify"
            );
            // Wrong leaf data must fail
            let mut bad = leaves[idx].clone();
            bad[0] += Goldilocks::ONE;
            assert!(!verify_mt::<Goldilocks, Poseidon1_16, Poseidon1_16>(&root, &[], &mp, idx as u64, &bad, 4, 0));
        }
    }
}
