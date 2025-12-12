extern crate alloc;
use alloc::{vec, vec::Vec};
use crate::{poseidon2_hash, Field, Goldilocks, PrimeField64};

#[derive(Default)]
pub struct Transcript {
    state: [Goldilocks; 12],
    pending: [Goldilocks; 8],
    out: [Goldilocks; 12],
    pending_cursor: u64,
    out_cursor: u64,
}

impl Transcript {
    pub fn new() -> Self {
        Transcript {
            state: [Goldilocks::ZERO; 12],
            pending: [Goldilocks::ZERO; 8],
            out: [Goldilocks::ZERO; 12],
            pending_cursor: 0,
            out_cursor: 0,
        }
    }

    pub fn update_state(&mut self) {
        while self.pending_cursor < 8 {
            self.pending[self.pending_cursor as usize] = Goldilocks::ZERO;
            self.pending_cursor += 1;
        }

        let mut inputs = vec![Goldilocks::ZERO; 12];
        inputs[..8].copy_from_slice(&self.pending);
        inputs[8..12].copy_from_slice(&self.state[..4]);
        let output = poseidon2_hash(&inputs);
        self.out_cursor = 12;
        for i in 0..8 {
            self.pending[i] = Goldilocks::ZERO;
        }
        self.pending_cursor = 0;
        self.state.copy_from_slice(&output[..12]);
        self.out.copy_from_slice(&output[..12]);
    }

    pub fn add1(&mut self, input: Goldilocks) {
        self.pending[self.pending_cursor as usize] = input;
        self.pending_cursor += 1;
        self.out_cursor = 0;
        if self.pending_cursor == 8 {
            self.update_state();
        }
    }

    pub fn put(&mut self, inputs: &mut [Goldilocks]) {
        for input in inputs.iter() {
            self.add1(*input);
        }
    }

    pub fn get_state(&mut self) -> Vec<Goldilocks> {
        if self.pending_cursor > 0 {
            self.update_state();
        }
        let mut state = Vec::with_capacity(4);
        for i in 0..4 {
            state.push(self.state[i]);
        }
        state
    }

    pub fn get_fields1(&mut self) -> Goldilocks {
        if self.out_cursor == 0 {
            self.update_state();
        }
        let val = self.out[(12 - self.out_cursor as usize) % 12];
        self.out_cursor -= 1;
        val
    }
    pub fn get_field(&mut self, value: &mut [Goldilocks]) {
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
                // pull out bit `cur_bit` of fields[cur_field]
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
