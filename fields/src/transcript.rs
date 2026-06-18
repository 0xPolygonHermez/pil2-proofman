use alloc::vec;
use alloc::vec::Vec;

use crate::{Hash, Poseidon1_16, Poseidon2_16, PrimeField64};

pub type TranscriptP1_16<F> = Transcript<F, Poseidon1_16>;
pub type TranscriptP2_16<F> = Transcript<F, Poseidon2_16>;

pub struct Transcript<F: PrimeField64, H: Hash<F>> {
    state: H::State,
    pending: Vec<F>,
    out: H::State,
    pending_cursor: usize,
    out_cursor: usize,
    _marker: core::marker::PhantomData<H>,
}

impl<F: PrimeField64, H: Hash<F>> Default for Transcript<F, H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64, H: Hash<F>> Transcript<F, H> {
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
