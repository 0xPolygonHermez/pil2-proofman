use alloc::vec;
use alloc::vec::Vec;

use crate::{Hash, Poseidon1_16, Poseidon2_16, PrimeField64};

pub type TranscriptP1_16<F> = Transcript<F, Poseidon1_16>;
pub type TranscriptP2_16<F> = Transcript<F, Poseidon2_16>;

/// A hash whose permutation a rate/capacity SPONGE can be built on.
///
/// Not every `Hash` qualifies, and the one that does not is the reason this trait exists: BLAKE3's
/// transcript absorbs eight words per compression with the state in the chaining value, so wrapping
/// `Blake3_8` in this sponge silently produces different challenges from the prover's. `Blake3_8`
/// deliberately does NOT implement this, which turns that mistake into a compile error; blake3's
/// transcript is `Blake3Transcript`.
///
/// ```compile_fail
/// use proofman_fields::{Blake3_8, Goldilocks, Transcript};
/// // BLAKE3 is not a sponge: this must not compile.
/// let _t: Transcript<Goldilocks, Blake3_8> = Transcript::new();
/// ```
pub trait SpongeHash<F: PrimeField64>: Hash<F> {}

pub struct Transcript<F: PrimeField64, H: SpongeHash<F>> {
    state: H::State,
    pending: Vec<F>,
    out: H::State,
    pending_cursor: usize,
    out_cursor: usize,
    _marker: core::marker::PhantomData<H>,
}

impl<F: PrimeField64, H: SpongeHash<F>> Default for Transcript<F, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64, H: SpongeHash<F>> Transcript<F, H> {
    pub fn new() -> Self {
        Transcript {
            state: H::State::default(),
            pending: vec![F::ZERO; H::RATE],
            out: H::State::default(),
            pending_cursor: 0,
            out_cursor: 0,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn update_state(&mut self) {
        while self.pending_cursor < H::RATE {
            self.pending[self.pending_cursor] = F::ZERO;
            self.pending_cursor += 1;
        }

        let mut inputs = H::State::default();
        {
            let slot = inputs.as_mut();
            slot[..H::RATE].copy_from_slice(&self.pending);
            slot[H::RATE..H::WIDTH].copy_from_slice(&self.state.as_ref()[..H::CAPACITY]);
        }
        H::hash(&mut inputs);
        self.out_cursor = H::WIDTH;
        for i in 0..H::RATE {
            self.pending[i] = F::ZERO;
        }
        self.pending_cursor = 0;
        self.state = inputs;
        self.out = inputs;
    }

    pub fn add1(&mut self, input: F) {
        self.pending[self.pending_cursor] = input;
        self.pending_cursor += 1;
        self.out_cursor = 0;
        if self.pending_cursor == H::RATE {
            self.update_state();
        }
    }

    pub fn put(&mut self, inputs: &[F]) {
        for input in inputs.iter() {
            self.add1(*input);
        }
    }

    pub fn get_state(&mut self) -> Vec<F> {
        if self.pending_cursor > 0 {
            self.update_state();
        }
        self.state.as_ref().to_vec()
    }

    pub fn get_fields1(&mut self) -> F {
        if self.out_cursor == 0 {
            self.update_state();
        }
        let val = self.out.as_ref()[(H::WIDTH - self.out_cursor) % H::WIDTH];
        self.out_cursor -= 1;
        val
    }

    pub fn get_field(&mut self, value: &mut [F]) {
        for val in value.iter_mut().take(3) {
            *val = self.get_fields1();
        }
    }

    pub fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64> {
        let total_bits = n * n_bits;
        let n_fields = ((total_bits - 1) / 63) + 1;
        let mut fields = Vec::with_capacity(n_fields as usize);
        for _ in 0..n_fields {
            fields.push(self.get_fields1());
        }

        let mut cur_field = 0;
        let mut cur_bit = 0;

        let mut permutations = vec![0u64; n as usize];
        for i in 0..n {
            let mut a = 0u64;
            for j in 0..n_bits {
                let bit = (fields[cur_field].as_canonical_u64() >> cur_bit) & 1;
                if bit == 1 {
                    a += 1 << j;
                }
                cur_bit += 1;
                if cur_bit == 63 {
                    cur_bit = 0;
                    cur_field += 1;
                }
            }
            permutations[i as usize] = a;
        }

        permutations
    }
}

/// What a verifier needs of a transcript, so `stark_verify` can be generic over the CONSTRUCTION
/// and not over a hash it then wraps in a sponge.
///
/// The distinction is load-bearing: `Transcript<F, H>` is a rate-4 sponge, while BLAKE3's transcript
/// absorbs eight words per compression with the state in the chaining value. Parameterising the
/// verifier by `Blake3_8` and building a sponge from it produces different challenges from the
/// prover's -- see the test in `blake3_transcript.rs`.
pub trait TranscriptLike<F: PrimeField64> {
    fn new_transcript() -> Self;
    fn put(&mut self, inputs: &[F]);
    fn get_field(&mut self, value: &mut [F]);
    fn get_state(&mut self) -> Vec<F>;
    fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64>;
}

impl<F: PrimeField64, H: SpongeHash<F>> TranscriptLike<F> for Transcript<F, H> {
    fn new_transcript() -> Self {
        Self::new()
    }
    fn put(&mut self, inputs: &[F]) {
        Transcript::put(self, inputs)
    }
    fn get_field(&mut self, value: &mut [F]) {
        Transcript::get_field(self, value)
    }
    fn get_state(&mut self) -> Vec<F> {
        Transcript::get_state(self)
    }
    fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64> {
        Transcript::get_permutations(self, n, n_bits)
    }
}

// The sponge families. Blake3_8 is absent on purpose -- see `SpongeHash`.
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon1_8 {}
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon1_12 {}
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon1_16 {}
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon2_4 {}
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon2_8 {}
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon2_12 {}
impl<F: PrimeField64> SpongeHash<F> for crate::Poseidon2_16 {}
