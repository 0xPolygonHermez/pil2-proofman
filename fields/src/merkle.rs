//! Merkle-tree and linear-hash utilities, generic over the [`Hash`](crate::Hash)
//! trait so they work with any Poseidon family (Poseidon1 / Poseidon2) at any width.
//! Nothing here is Poseidon2-specific; the concrete hash is chosen by the caller via
//! the `H` / `LH` / `NH` type parameters.

use alloc::vec;
use alloc::vec::Vec;

use crate::PrimeField64;

/// Sponge-mode linear hash. The caller picks `H` (Poseidon1_12, Poseidon2_16, …);
/// the function uses `H::RATE` and `H::CAPACITY` to drive the absorb loop. Returns
/// the full state (`H::State` = `[F; H::WIDTH]`); the leaf digest is the first 4 cells.
pub fn linear_hash_seq<F: PrimeField64, H: crate::Hash<F>>(input: &[F]) -> H::State {
    assert!(H::WIDTH > 4);
    assert!(H::RATE > 0, "linear_hash_seq requires RATE > 0 (H::RATE = {})", H::RATE);
    let mut state: H::State = H::State::default();
    let size = input.len();
    let mut remaining = size;
    while remaining > 0 {
        let s = state.as_mut();
        if remaining != size {
            // Save current cells 0..4 (the digest / capacity carry).
            let mut carry = [F::ZERO; 4];
            carry.copy_from_slice(&s[..4]);
            s[H::RATE..H::RATE + 4].copy_from_slice(&carry);
        }
        let n = if remaining < H::RATE { remaining } else { H::RATE };
        for i in 0..(H::RATE - n) {
            s[n + i] = F::ZERO;
        }
        for i in 0..n {
            s[i] = input[size - remaining + i];
        }
        H::hash(&mut state);
        remaining -= n;
    }
    state
}

/// Walks a merkle path and recomputes the root in-place. `H` is the internal
/// merkle-node compression hash.
pub fn calculate_root_from_proof<F: PrimeField64, H: crate::Hash<F>>(
    value: &mut H::State,
    mp: &[Vec<F>],
    idx: &mut u64,
    offset: u64,
    arity: u64,
) {
    debug_assert_eq!(arity as usize * 4, H::WIDTH, "arity ({}) * 4 must equal H::WIDTH ({})", arity, H::WIDTH);
    if offset == mp.len() as u64 {
        return;
    }

    let curr_idx = *idx % arity;
    *idx /= arity;

    let mut inputs: H::State = H::State::default();
    {
        let in_slot = inputs.as_mut();
        let mut p = 0;
        for i in 0..arity {
            if i == curr_idx {
                continue;
            }
            for j in 0..4 {
                in_slot[(i * 4 + j) as usize] = mp[offset as usize][4 * p + j as usize];
            }
            p += 1;
        }
        for j in 0..4 {
            in_slot[(curr_idx * 4 + j) as usize] = value.as_ref()[j as usize];
        }
    }

    H::hash(&mut inputs);
    value.as_mut()[..4].copy_from_slice(&inputs.as_ref()[..4]);
    calculate_root_from_proof::<F, H>(value, mp, idx, offset + 1, arity);
}

/// Builds a partial merkle tree bottom-up using `H` as the internal-node hash.
pub fn partial_merkle_tree<F: PrimeField64, H: crate::Hash<F>>(input: &[F], num_elements: u64, arity: u64) -> [F; 4] {
    assert_eq!(
        arity as usize * H::CAPACITY,
        H::WIDTH,
        "arity ({}) * CAPACITY ({}) must equal H::WIDTH ({})",
        arity,
        H::CAPACITY,
        H::WIDTH
    );
    let mut num_nodes = num_elements;
    let mut nodes_level = num_elements;

    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }

    let cap = H::CAPACITY as u64;
    // Internal-node compression consumes `arity` child digests of `cap` cells each
    // (= the full hash state for our layouts: arity*cap == H::WIDTH). This must match
    // the layout used by `calculate_root_from_proof`, which fills `arity * 4` cells.
    let sponge_w = arity * cap;

    let mut cursor = vec![F::ZERO; (num_nodes * cap) as usize];
    cursor[..(num_elements * cap) as usize].copy_from_slice(&input[..(num_elements * cap) as usize]);

    let mut pending = num_elements;
    let mut next_n = pending.div_ceil(arity);
    let mut next_index = 0;

    while pending > 1 {
        let extra_zeros = (arity - (pending % arity)) % arity;

        if extra_zeros > 0 {
            let start = (next_index + pending * cap) as usize;
            let end = start + (extra_zeros * cap) as usize;
            cursor[start..end].fill(F::ZERO);
        }

        for i in 0..next_n {
            let mut pol_input: H::State = H::State::default();
            {
                let child_start = (next_index + i * sponge_w) as usize;
                let slot = pol_input.as_mut();
                slot[..(sponge_w as usize)].copy_from_slice(&cursor[child_start..child_start + sponge_w as usize]);
            }
            H::hash(&mut pol_input);
            let parent_start = (next_index + (pending + extra_zeros + i) * cap) as usize;
            cursor[parent_start..parent_start + H::CAPACITY].copy_from_slice(&pol_input.as_ref()[..H::CAPACITY]);
        }

        next_index += (pending + extra_zeros) * cap;
        pending = pending.div_ceil(arity);
        next_n = pending.div_ceil(arity);
    }

    let mut root = [F::ZERO; 4];
    root.copy_from_slice(&cursor[next_index as usize..next_index as usize + 4]);
    root
}

/// Verifies a merkle path. `LH` is the leaf linear hash; `NH` is the internal
/// merkle-node compression hash. Typical recurser configuration is
/// (`Poseidon1_12`, `Poseidon2_16`).
pub fn verify_mt<F, LH, NH>(
    root: &[F],
    last_level: &[F],
    mp: &[Vec<F>],
    idx: u64,
    v: &[F],
    arity: u64,
    last_level_verification: u64,
) -> bool
where
    F: PrimeField64,
    LH: crate::Hash<F>,
    NH: crate::Hash<F>,
{
    // Through the trait, not `linear_hash_seq` directly: a family whose leaf digest is not a sponge
    // absorb -- BLAKE3 -- overrides it, and calling the loop here would compute a digest its prover
    // never produced.
    let leaf_hash = LH::linear_hash(v);

    let mut value: NH::State = NH::State::default();
    value.as_mut()[..4].copy_from_slice(&leaf_hash);

    let mut query_idx = idx;
    calculate_root_from_proof::<F, NH>(&mut value, mp, &mut query_idx, 0, arity);

    if last_level_verification == 0 {
        for (i, r) in root.iter().enumerate().take(4) {
            if value.as_ref()[i] != *r {
                return false;
            }
        }
    } else {
        for i in 0..4 {
            if value.as_ref()[i] != last_level[query_idx as usize * 4 + i] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poseidon2_hash, Goldilocks, Poseidon2_16};

    #[test]
    fn partial_merkle_tree_single_parent_matches_full_compression() {
        // 4 leaf digests of 4 cells each = 16 input cells; arity 4 → exactly one parent
        // node whose hash is taken over ALL 16 child cells, so the root is the first 4
        // cells of that hash. Regression guard for the per-node input width: it must
        // read `arity * CAPACITY` (= 16) cells, not `RATE` (= 12).
        let input: [Goldilocks; 16] = core::array::from_fn(|i| Goldilocks::new(i as u64 + 1));
        let expected = poseidon2_hash::<Goldilocks, Poseidon2_16, 16>(&input);
        let root = partial_merkle_tree::<Goldilocks, Poseidon2_16>(&input, 4, 4);
        assert_eq!(root, [expected[0], expected[1], expected[2], expected[3]]);
    }
}
