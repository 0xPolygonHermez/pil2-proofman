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

    /// `Blake3_8` forced through [`crate::blake3_core`]. The backend is a target `cfg`, so this is
    /// the only way a native run reaches the guest's -- and at the level callers use, not bytes.
    #[derive(Clone, Copy, Debug, Default)]
    struct Blake3Core8;

    /// The transcript's `put` + `get_state` over the core, duplicated rather than shared so the
    /// two are compared; the test below ties it back to the real `Blake3_8::linear_hash`.
    fn core_hash_le64<F: PrimeField64>(input: &[F]) -> [F; 4] {
        let mut h = crate::blake3_core::Hasher::new();
        for x in input {
            h.update(&x.as_canonical_u64().to_le_bytes());
        }
        let mut buf = [0u8; 32];
        h.finalize_xof().fill(&mut buf);
        core::array::from_fn(|i| {
            F::from_u64(crate::blake3_transcript::canon(u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap())))
        })
    }

    impl<F: PrimeField64> Hash<F> for Blake3Core8 {
        const WIDTH: usize = 8;
        const RATE: usize = 4;
        const CAPACITY: usize = 4;
        type State = [F; 8];

        fn hash(state: &mut [F; 8]) {
            let dig = core_hash_le64::<F>(&state[..]);
            state[..4].copy_from_slice(&dig);
            state[4..].fill(F::ZERO);
        }

        fn linear_hash(input: &[F]) -> [F; 4] {
            core_hash_le64::<F>(input)
        }
    }

    fn elems(n: usize) -> Vec<Goldilocks> {
        // Above p on purpose: BLAKE3, unlike Poseidon, is not invariant under +p.
        (0..n as u64).map(|i| Goldilocks::from_u64(0x9E3779B97F4A7C15u64.wrapping_mul(i + 1))).collect()
    }

    /// Both Merkle jobs -- a trace row's leaf digest and a node's compression -- must match across
    /// backends at every width that changes branch, including past the 128-word chunk.
    #[test]
    fn the_two_backends_agree_on_leaf_and_node_hashes() {
        for n in [1usize, 4, 8, 9, 16, 127, 128, 129, 256, 300, 1000] {
            let input = elems(n);
            let core = <Blake3Core8 as Hash<Goldilocks>>::linear_hash(&input);
            let krate = <Blake3_8 as Hash<Goldilocks>>::linear_hash(&input);
            assert_eq!(core, krate, "leaf hash of {n} elements differs between the backends");
        }

        let input: [Goldilocks; 8] = core::array::from_fn(|i| elems(8)[i]);
        let mut a = input;
        let mut b = input;
        <Blake3_8 as Hash<Goldilocks>>::hash(&mut a);
        <Blake3Core8 as Hash<Goldilocks>>::hash(&mut b);
        assert_eq!(a, b, "node compression differs between the backends");
    }

    /// `calculate_root_from_proof` feeds each level's output in as the next level's input, so a
    /// divergence compounds rather than showing up once.
    #[test]
    fn the_two_backends_agree_on_a_merkle_root_and_path() {
        use crate::merkle::{calculate_root_from_proof, partial_merkle_tree};

        // 16 leaves of 4 cells: depth 4, so the path runs four compressions.
        let leaves = elems(16 * 4);
        let root_krate = partial_merkle_tree::<Goldilocks, Blake3_8>(&leaves, 16, 2);
        let root_core = partial_merkle_tree::<Goldilocks, Blake3Core8>(&leaves, 16, 2);
        assert_eq!(root_krate, root_core, "merkle roots differ between the backends");

        // Same starting digest and siblings through each backend.
        let mp: Vec<Vec<Goldilocks>> =
            (0..4).map(|lvl| elems(4).iter().map(|x| *x + Goldilocks::from_u64(lvl)).collect()).collect();
        let start: [Goldilocks; 8] = core::array::from_fn(|i| if i < 4 { leaves[i] } else { Goldilocks::ZERO });

        let mut v_krate = start;
        let mut i_krate = 5u64;
        calculate_root_from_proof::<Goldilocks, Blake3_8>(&mut v_krate, &mp, &mut i_krate, 0, 2);

        let mut v_core = start;
        let mut i_core = 5u64;
        calculate_root_from_proof::<Goldilocks, Blake3Core8>(&mut v_core, &mp, &mut i_core, 0, 2);

        assert_eq!(v_krate, v_core, "recomputed root from a merkle path differs between the backends");
    }

    /// The C++ identity this whole impl rests on: `permuteTrunc(in[8])` and `linearHash(in, 8)` are
    /// the same `hash_le64` call. If the node hash ever stops being the leaf hash of eight
    /// elements, every Merkle path in a blake3 proof silently stops verifying.
    #[test]
    fn blake3_node_and_leaf_agree_at_width_8() {
        let input: [Goldilocks; 8] =
            core::array::from_fn(|i| Goldilocks::from_u64(0x9E3779B97F4A7C15u64.wrapping_mul(i as u64 + 1)));
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

// ── SHA-256 ──────────────────────────────────────────────────────────────────

use sha2::{compress256, Digest, Sha256};

/// IVs for the two fixed-length constructions, mirroring `sha256_core.hpp`. Both are SHA-256
/// digests of ASCII domain strings; the test below recomputes them rather than trusting these.
const SHA_IV_NODE: [u32; 8] =
    [0x14FDA625, 0x32FCCD27, 0x853E32C5, 0xED19966D, 0x16699720, 0x63E7CCAD, 0x8AE17E84, 0xCA32C0F3];
const SHA_IV_GRIND: [u32; 8] =
    [0xA9AFF67F, 0xED176A33, 0xBF35D926, 0xE35A0AF2, 0x1A4F8C73, 0x7E6AB8B1, 0x3460F3DC, 0x8829058F];

/// Digest words read back as canonical Goldilocks, mirroring `sha256core::pack4`.
fn sha_pack4<F: PrimeField64>(h: &[u32; 8]) -> [F; 4] {
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        bytes[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    core::array::from_fn(|i| {
        F::from_u64(crate::blake3_transcript::canon(u64::from_le_bytes(
            bytes[8 * i..8 * i + 8].try_into().unwrap(),
        )))
    })
}

/// LEAVES: literal FIPS 180-4 over the canonical LE words.
fn sha256_leaf_hash<F: PrimeField64>(input: &[F]) -> [F; 4] {
    let mut hasher = Sha256::new();
    for x in input {
        hasher.update(x.as_canonical_u64().to_le_bytes());
    }
    let d = hasher.finalize();
    core::array::from_fn(|i| {
        F::from_u64(crate::blake3_transcript::canon(u64::from_le_bytes(
            d[8 * i..8 * i + 8].try_into().unwrap(),
        )))
    })
}

/// Compression chain from `iv`, no padding and no length. `input.len()` must be a positive
/// multiple of 8 -- a fixed width is what makes this unambiguous.
fn sha256_fixed_len_hash<F: PrimeField64>(iv: &[u32; 8], input: &[F]) -> [F; 4] {
    debug_assert!(!input.is_empty() && input.len() % 8 == 0, "fixed-length hash needs whole blocks");
    let mut state = *iv;
    for chunk in input.chunks_exact(8) {
        let mut block = [0u8; 64];
        for (i, x) in chunk.iter().enumerate() {
            block[8 * i..8 * i + 8].copy_from_slice(&x.as_canonical_u64().to_le_bytes());
        }
        compress256(&mut state, &[block.into()]);
    }
    sha_pack4(&state)
}

/// SHA-256 at the one width the Merkle geometry needs: eight elements is a 64-byte block, and at
/// arity 2 a node is two digests, so `WIDTH = 8` is the block, not a chosen sponge width.
///
/// UNLIKE `Blake3_8`, `hash` (node, `SHA_IV_NODE`) and `linear_hash` (leaf, FIPS) are DIFFERENT
/// functions, mirroring the C++ prover. `verify_mt` takes the two as separate generics.
///
/// `RATE`/`CAPACITY` are carried for the trait's sake: `linear_hash` is overridden, so the sponge
/// absorb loop they drive never runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sha256_8;

impl<F: PrimeField64> Hash<F> for Sha256_8 {
    const WIDTH: usize = 8;
    const RATE: usize = 4;
    const CAPACITY: usize = 4;
    type State = [F; 8];

    /// The NODE compression. Cells 4..8 are cleared because `calculate_root_from_proof` reuses
    /// the state across levels.
    fn hash(state: &mut [F; 8]) {
        let dig = sha256_fixed_len_hash::<F>(&SHA_IV_NODE, &state[..]);
        state[..4].copy_from_slice(&dig);
        state[4..].fill(F::ZERO);
    }

    fn linear_hash(input: &[F]) -> [F; 4] {
        sha256_leaf_hash::<F>(input)
    }
}

/// Grinding: a THIRD construction, one compression from `SHA_IV_GRIND`. A separate type because
/// `Sha256_8::hash` is the node hash; blake3 serves both from one type only because its node hash
/// and grinding permutation agree on cell 0.
#[derive(Clone, Copy, Debug, Default)]
#[allow(non_camel_case_types)] // matches the file's <Family>_<width> convention
pub struct Sha256Grind_8;

impl<F: PrimeField64> Hash<F> for Sha256Grind_8 {
    const WIDTH: usize = 8;
    const RATE: usize = 4;
    const CAPACITY: usize = 4;
    type State = [F; 8];

    /// `[c0, c1, c2, nonce, 0, 0, 0, 0]` in, digest in cells 0..4; the caller reads cell 0.
    fn hash(state: &mut [F; 8]) {
        let dig = sha256_fixed_len_hash::<F>(&SHA_IV_GRIND, &state[..]);
        state[..4].copy_from_slice(&dig);
        state[4..].fill(F::ZERO);
    }

    /// The sponge default would silently compute something the prover never produced.
    fn linear_hash(_input: &[F]) -> [F; 4] {
        panic!("Sha256Grind_8 is the grinding hash only; leaves go through Sha256_8")
    }
}

#[cfg(test)]
mod sha256_tests {
    use super::*;
    use crate::{Field, Goldilocks};

    // NODE: the arity-2 compression, from IV_NODE.
    const NODE_VECTORS: &[([u64; 8], [u64; 4])] = &[
        ([0x7E4328BC0F7DFB8A, 0xF4A61A7F8FA82E91, 0xE8155A62CD769FCC, 0xFFFFFFFF00000001, 0xFFFFFFFF02D42BB2, 0x236F92074A6D4F55, 0x67237FD0AF729E40, 0x52727028C64F128F], [0xABB97CBB37156A75, 0xE11CCA39909BA81E, 0x357D5589793F20C3, 0x1B2A13D22E88BEAB]),
        ([0xDF8AF97141D3F83E, 0x96247D0795E6E635, 0x1427003E960B43A0, 0xFFFFFFFF00000001, 0xFFFFFFFF01F43310, 0xE5A39CC78FEB3939, 0x4D5C06E32F50D754, 0x28E9D9A5A7540713], [0x37F7FB479B08215A, 0xB175F912619AE0CF, 0x3D4AAB28BCA9D655, 0xEDD2CE659AC05A3D]),
        ([0x03626B90D8D5EE5A, 0x7A9FA49FA8A30D21, 0x985BF1D1EFC92F1C, 0xFFFFFFFF00000001, 0xFFFFFFFFFF544927, 0x2C3FBD086064F6E5, 0x00059C1AAEEB8290, 0xAC50161C4A62E49F], [0x3EA5A6B98059202D, 0x2C0B74FFF6179B5B, 0x37C745C363E2C0A0, 0x2672607BD799AC05]),
    ];

    // GRIND: one compression from IV_GRIND.
    const GRIND_VECTORS: &[([u64; 8], [u64; 4])] = &[
        ([0x9408DD30E1CD2EE3, 0xDF608D5A20DA5C36, 0xE8AFAA98C4F680CD, 0xFFFFFFFF00000001, 0xFFFFFFFFAE7CD6EC, 0x86A98BFD9A1CCD4A, 0x8FC5C43ECC614D51, 0x4CCAC4C03602478C], [0x59E4D0DC02ED325C, 0x8675C5858E6D2375, 0x7348D77C4DC50B79, 0x978A1AA9503159C2]),
        ([0xEC5AD15E2E62AE10, 0x47C025FC226A0A1F, 0xF3B4140FB71BA9C2, 0xFFFFFFFF00000001, 0xFFFFFFFF6E44D8C4, 0x77368EADAB7C47C3, 0x0953E6036C58DB96, 0x02689F1F6E4384AD], [0xC77E8692AFBD624B, 0x6D789AB1B4E710C7, 0x0261B914684C6BD7, 0x3992BBF46C76CDF6]),
        ([0x44ACC58B7AF82D3D, 0xB01FBE9E23F9B808, 0xFEB87D86A940D2B7, 0xFFFFFFFF00000001, 0xFFFFFFFF2E0CDA9B, 0x67C3915DBCDBC23C, 0x82E207C80C5069DB, 0xB806797EA684C1CE], [0x7F6781596CA4BAF9, 0x80029D55EEE6D336, 0xC7828F231C5CF93A, 0x75347E7AB3E2A94E]),
    ];

    // LEAF: widths spanning all three padding cases.
    const LEAF_VECTORS: &[(&[u64], [u64; 4])] = &[
        (&[], [0x141CFC9842C4B0E3, 0x24B96F99C8F4FB9A, 0x4C939B64E441AE27, 0x55B852781B9995A4]),
        (&[0x6C576FAC43FD007C], [0xC019A9733D4DCB7B, 0x26D24CB5FB4BAD30, 0x27620D55795AC666, 0x6112DB0EE1CA430B]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F], [0x994BE62D5AEB6E83, 0x543E75E3BE0F7F03, 0xF1246EC1EC40E59A, 0x2077BA2D675E5F58]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F, 0x8DCEB534EFA548A2], [0xBA692C65FEBF02E2, 0x5E371E287C2BE5D3, 0x23D9F6B64E99409B, 0x42BF27E8CADEB127]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F, 0x8DCEB534EFA548A2, 0x10BF51ED74C7A3C9], [0x05707FC5896A4485, 0xC9A3EA123D513D62, 0x8BC63F1CF4B27CE8, 0xE2E74B2782F9E26E]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F, 0x8DCEB534EFA548A2, 0x10BF51ED74C7A3C9, 0xD6F84A5288BD02A4], [0x9E0EFF9FB502089E, 0xAB1EC3CC17E0150A, 0x59DDBB1F3C339BCE, 0xEB5F182D6CEEA009]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F, 0x8DCEB534EFA548A2, 0x10BF51ED74C7A3C9, 0xD6F84A5288BD02A4, 0xFFFFFFFF31314B9A, 0xBEEA83F6D126A876, 0x892769E4FD73A80D, 0xC89F7AF990AB7E98, 0x512A264047D22A07, 0xFFFFFFFFA2A0B6FF, 0xBFFAE0196A526891], [0xD305273B21657F56, 0xC2D2CA839EFBA933, 0x374487907E0F51CC, 0x2426C7C541ADCB53]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F, 0x8DCEB534EFA548A2, 0x10BF51ED74C7A3C9, 0xD6F84A5288BD02A4, 0xFFFFFFFF31314B9A, 0xBEEA83F6D126A876, 0x892769E4FD73A80D, 0xC89F7AF990AB7E98, 0x512A264047D22A07, 0xFFFFFFFFA2A0B6FF, 0xBFFAE0196A526891, 0x6CAE6A347228D1CC, 0x92780C3FF464962B, 0x9F64D9A9FE9C3BDE, 0xFFFFFFFF2ECDEDC1, 0xDDAA72D72463F040, 0xFAC87B1BC6687C8F, 0x8CBC97BCA0CB5772, 0x9F21B6C5D8E16E59], [0xDA014ECDAA69141D, 0xAA99A3BA201B75AE, 0xB369BFD896F75D36, 0x66CF7967195531DE]),
        (&[0x6C576FAC43FD007C, 0x826886B3864A1B1B, 0xA5FAE1992097AA0E, 0xFFFFFFFF00000001, 0xFFFFFFFF842420A6, 0x802181E6E230707F, 0x8DCEB534EFA548A2, 0x10BF51ED74C7A3C9, 0xD6F84A5288BD02A4, 0xFFFFFFFF31314B9A, 0xBEEA83F6D126A876, 0x892769E4FD73A80D, 0xC89F7AF990AB7E98, 0x512A264047D22A07, 0xFFFFFFFFA2A0B6FF, 0xBFFAE0196A526891, 0x6CAE6A347228D1CC, 0x92780C3FF464962B, 0x9F64D9A9FE9C3BDE, 0xFFFFFFFF2ECDEDC1, 0xDDAA72D72463F040, 0xFAC87B1BC6687C8F, 0x8CBC97BCA0CB5772, 0x9F21B6C5D8E16E59, 0xFFFFFFFFF5B2326F, 0xC9CCAF613E310133, 0x96F61E490C4C0446, 0x0E0F3D411BA0FB9D, 0xF6FA4AA3BDE79EE8, 0xFFFFFFFF8149B75A, 0x4880CF3F8E84D65A, 0xA3E94A246579D521, 0x521DB83816C2571C, 0x8C15A18A6413B53B, 0xFFFFFFFFDF345ACC, 0x3F8E2CB563183EE5, 0xCCD9859401272A90, 0x2F066AD41F376C9F, 0x0474C1B95F937A42, 0xFFFFFFFFFCA50B24, 0x398C09FFB8714D44, 0xA3447BC8C13AD243, 0xFC9F9514620CB416, 0x5BD74371D9C8132D, 0xFFFFFFFF9B336ED3, 0x49AD8A2CD3EC4A27, 0x2FA1B5E82D6CE32A, 0x8F415E0AA7AF45B1, 0x0362E25F80CE906C, 0xFFFFFFFF62E2BD91, 0x7568C75470A8DB7E, 0x49ABC9D1B24F9875, 0xA2A5B6F8442058E0, 0x00EA2F85378E40AF, 0xFFFFFFFF802C594F, 0xF94AA25107AA8F79, 0x5B1DDBAD5AFCC094, 0xEFF801675F82C753, 0xD61092EE4995B7E6, 0xFFFFFFFF908D46CB, 0x481750924ABF3B88, 0xA2D0962CFFB57037, 0x3C8B0BD890FA83FA, 0xD75B3C6BC36BBA41, 0xFFFFFFFF1F5E788C, 0x16BEC93F79F3DF5B], [0x7AF520279C9F0104, 0xFA89323BFA66EE47, 0x534562ECC0C4566F, 0xC2C5BA4E6EB1BC9D]),
    ];

    fn gl(v: &[u64]) -> alloc::vec::Vec<Goldilocks> {
        v.iter().map(|&x| Goldilocks::from_u64(x)).collect()
    }

    /// Recompute the IVs rather than trust the literals.
    #[test]
    fn the_domain_ivs_are_the_digests_of_their_strings() {
        for (want, s) in [
            (SHA_IV_NODE, &b"pil2-stark/sha256/merkle-node/v1"[..]),
            (SHA_IV_GRIND, &b"pil2-stark/sha256/grinding/v1"[..]),
        ] {
            let d = Sha256::digest(s);
            let got: [u32; 8] = core::array::from_fn(|i| u32::from_be_bytes(d[4 * i..4 * i + 4].try_into().unwrap()));
            assert_eq!(got, want);
        }
    }

    /// All three against the SAME independent reference that generates the C++ vectors, so Rust
    /// and C++ agree through a third party rather than by comparison.
    #[test]
    fn the_three_constructions_match_the_independent_reference() {
        for (input, want) in NODE_VECTORS {
            let mut st: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::from_u64(input[i]));
            <Sha256_8 as Hash<Goldilocks>>::hash(&mut st);
            for i in 0..4 {
                assert_eq!(st[i].as_canonical_u64(), want[i], "node word {i}");
            }
            assert!(st[4..].iter().all(|x| *x == Goldilocks::ZERO), "cells 4..8 must be cleared");
        }
        for (input, want) in GRIND_VECTORS {
            let mut st: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::from_u64(input[i]));
            <Sha256Grind_8 as Hash<Goldilocks>>::hash(&mut st);
            for i in 0..4 {
                assert_eq!(st[i].as_canonical_u64(), want[i], "grind word {i}");
            }
        }
        for (input, want) in LEAF_VECTORS {
            let got = <Sha256_8 as Hash<Goldilocks>>::linear_hash(&gl(input));
            for i in 0..4 {
                assert_eq!(got[i].as_canonical_u64(), want[i], "leaf word {i} at width {}", input.len());
            }
        }
    }

    /// The distinction the design rests on, and the one `Blake3_8` does NOT have: if these ever
    /// coincide the domain separation is gone and a leaf could stand in for a node.
    #[test]
    fn the_node_hash_differs_from_the_leaf_hash_at_the_node_width() {
        let input = gl(&NODE_VECTORS[0].0);
        let leaf = <Sha256_8 as Hash<Goldilocks>>::linear_hash(&input);
        let mut st: [Goldilocks; 8] = core::array::from_fn(|i| input[i]);
        <Sha256_8 as Hash<Goldilocks>>::hash(&mut st);
        assert_ne!(leaf, [st[0], st[1], st[2], st[3]]);
    }

    /// Grinding and the node hash differ only by their IV, so pin that they differ at all.
    #[test]
    fn grinding_differs_from_the_node_hash() {
        let input: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::from_u64(NODE_VECTORS[0].0[i]));
        let mut node = input;
        let mut grind = input;
        <Sha256_8 as Hash<Goldilocks>>::hash(&mut node);
        <Sha256Grind_8 as Hash<Goldilocks>>::hash(&mut grind);
        assert_ne!(node, grind);
    }
}
