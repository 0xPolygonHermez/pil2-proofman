use crate::PrimeField64;

/// Generic Poseidon-family hash trait. The state width is encoded in the
/// associated `State` type (= `[F; WIDTH]`), so callers don't have to repeat
/// the width as a separate const generic.
pub trait Hash<F: PrimeField64> {
    const WIDTH: usize;
    const RATE: usize;
    const CAPACITY: usize;
    type State: Default + Copy + AsRef<[F]> + AsMut<[F]>;

    /// Hash the state in-place. `state.as_ref().len()` is `WIDTH` by construction.
    fn hash(state: &mut Self::State);

    /// Digest a variable-length run of field elements: the LEAF hash of a Merkle tree.
    ///
    /// Separated from `hash` because the two are not the same operation for every family. A sponge
    /// absorbs `RATE` elements at a time and carries the capacity between permutations, which is
    /// what the default below does. A compression function like BLAKE3 has no rate to absorb at --
    /// it takes a whole block per call and carries a chaining value in its own state -- so it
    /// overrides this with the primitive its prover actually used. Getting this wrong does not fail
    /// loudly: the verifier simply computes a different leaf digest and every proof is rejected.
    fn linear_hash(input: &[F]) -> [F; 4]
    where
        Self: Sized,
    {
        let state = crate::merkle::linear_hash_seq::<F, Self>(input);
        let s = state.as_ref();
        [s[0], s[1], s[2], s[3]]
    }
}

// ── Poseidon2 family ─────────────────────────────────────────────────────────

use crate::{poseidon2_hash, Poseidon2_4, Poseidon2_8, Poseidon2_12, Poseidon2_16};

impl<F: PrimeField64> Hash<F> for Poseidon2_4 {
    const WIDTH: usize = 4;
    const RATE: usize = 0;
    const CAPACITY: usize = 4;
    type State = [F; 4];
    fn hash(state: &mut [F; 4]) {
        *state = poseidon2_hash::<F, Self, 4>(state);
    }
}

impl<F: PrimeField64> Hash<F> for Poseidon2_8 {
    const WIDTH: usize = 8;
    const RATE: usize = 4;
    const CAPACITY: usize = 4;
    type State = [F; 8];
    fn hash(state: &mut [F; 8]) {
        *state = poseidon2_hash::<F, Self, 8>(state);
    }
}

impl<F: PrimeField64> Hash<F> for Poseidon2_12 {
    const WIDTH: usize = 12;
    const RATE: usize = 8;
    const CAPACITY: usize = 4;
    type State = [F; 12];
    fn hash(state: &mut [F; 12]) {
        *state = poseidon2_hash::<F, Self, 12>(state);
    }
}

impl<F: PrimeField64> Hash<F> for Poseidon2_16 {
    const WIDTH: usize = 16;
    const RATE: usize = 12;
    const CAPACITY: usize = 4;
    type State = [F; 16];
    fn hash(state: &mut [F; 16]) {
        *state = poseidon2_hash::<F, Self, 16>(state);
    }
}

// ── Poseidon1 family ─────────────────────────────────────────────────────────

use crate::{poseidon1_hash, Poseidon1_8, Poseidon1_12, Poseidon1_16};

impl<F: PrimeField64> Hash<F> for Poseidon1_8 {
    const WIDTH: usize = 8;
    const RATE: usize = 4;
    const CAPACITY: usize = 4;
    type State = [F; 8];
    fn hash(state: &mut [F; 8]) {
        *state = poseidon1_hash::<F, Self, 8>(state);
    }
}

impl<F: PrimeField64> Hash<F> for Poseidon1_12 {
    const WIDTH: usize = 12;
    const RATE: usize = 8;
    const CAPACITY: usize = 4;
    type State = [F; 12];
    fn hash(state: &mut [F; 12]) {
        *state = poseidon1_hash::<F, Self, 12>(state);
    }
}

impl<F: PrimeField64> Hash<F> for Poseidon1_16 {
    const WIDTH: usize = 16;
    const RATE: usize = 12;
    const CAPACITY: usize = 4;
    type State = [F; 16];
    fn hash(state: &mut [F; 16]) {
        *state = poseidon1_hash::<F, Self, 16>(state);
    }
}

// ── BLAKE3 ───────────────────────────────────────────────────────────────────

use crate::Blake3Transcript;

/// BLAKE3 as this tree's hash, at the one width the Merkle geometry needs.
///
/// Eight field elements is a 64-byte BLAKE3 block, and at arity 2 a Merkle node is exactly two
/// four-element digests -- so `WIDTH = 8` is not a chosen sponge width but the block itself.
///
/// The C++ prover hands both jobs to ONE primitive: `Blake3Goldilocks::linearHash` is
/// `blake3core::hash_le64(input, size)` and `permuteTrunc` is the same call with `size = 8`. So the
/// node hash here is the leaf hash of eight elements, and `blake3_node_and_leaf_agree_at_width_8`
/// pins that identity rather than trusting it.
///
/// `RATE` and `CAPACITY` are carried for the trait's sake and are not used: `linear_hash` is
/// overridden, so nothing here ever runs the sponge absorb loop those two drive.
#[derive(Clone, Copy, Debug, Default)]
pub struct Blake3_8;

/// BLAKE3-256 of `input` as canonical little-endian u64 words, first four words of the digest.
/// Mirrors `blake3core::hash_le64` + `pack4`, canonicalisation included.
fn blake3_hash_le64<F: PrimeField64>(input: &[F]) -> [F; 4] {
    let mut t = Blake3Transcript::<F>::new();
    t.put(input);
    let s = t.get_state();
    [s[0], s[1], s[2], s[3]]
}

impl<F: PrimeField64> Hash<F> for Blake3_8 {
    const WIDTH: usize = 8;
    const RATE: usize = 4;
    const CAPACITY: usize = 4;
    type State = [F; 8];

    /// The node compression: eight elements in, the digest in cells 0..4. Cells 4..8 are cleared
    /// rather than left stale, because `calculate_root_from_proof` reuses the state across levels
    /// and a caller reading past the digest should see zeros, not the previous level's children.
    fn hash(state: &mut [F; 8]) {
        let dig = blake3_hash_le64::<F>(&state[..]);
        state[..4].copy_from_slice(&dig);
        state[4..].fill(F::ZERO);
    }

    fn linear_hash(input: &[F]) -> [F; 4] {
        blake3_hash_le64::<F>(input)
    }
}

#[cfg(test)]
mod blake3_tests {
    use super::*;
    use crate::{Field, Goldilocks};
    use alloc::vec::Vec;

    /// The C++ identity this whole impl rests on: `permuteTrunc(in[8])` and `linearHash(in, 8)` are
    /// the same `hash_le64` call. If the node hash ever stops being the leaf hash of eight
    /// elements, every Merkle path in a blake3 proof silently stops verifying.
    #[test]
    fn blake3_node_and_leaf_agree_at_width_8() {
        let input: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::from_u64(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1)));
        let leaf = <Blake3_8 as Hash<Goldilocks>>::linear_hash(&input);
        let mut state = input;
        <Blake3_8 as Hash<Goldilocks>>::hash(&mut state);
        assert_eq!(&state[..4], &leaf[..], "node hash must be the leaf hash of its eight children");
        assert_eq!(&state[4..], &[Goldilocks::ZERO; 4], "the tail must be cleared, not stale");
    }

    /// A leaf longer than one block has to keep hashing rather than truncate, and one shorter has to
    /// differ from the padded version -- the two failure modes a block-based digest invites.
    #[test]
    fn blake3_leaf_hash_depends_on_the_whole_input() {
        let long: Vec<Goldilocks> = (0..300u64).map(Goldilocks::from_u64).collect();
        let mut clipped = long.clone();
        clipped[299] = Goldilocks::from_u64(999);
        assert_ne!(
            <Blake3_8 as Hash<Goldilocks>>::linear_hash(&long),
            <Blake3_8 as Hash<Goldilocks>>::linear_hash(&clipped),
            "the last element of a 300-word leaf must reach the digest"
        );
        let short = [Goldilocks::from_u64(1), Goldilocks::from_u64(2)];
        let padded = [Goldilocks::from_u64(1), Goldilocks::from_u64(2), Goldilocks::ZERO, Goldilocks::ZERO];
        assert_ne!(
            <Blake3_8 as Hash<Goldilocks>>::linear_hash(&short),
            <Blake3_8 as Hash<Goldilocks>>::linear_hash(&padded),
            "length is part of the digest; zero-padding must not collide"
        );
    }
}
