use crate::{poseidon2_hash, PrimeField64, Poseidon2Constants};

pub struct Transcript<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize> {
    state: [F; W],
    pending: Vec<F>,
    out: [F; W],
    pending_cursor: usize,
    out_cursor: usize,
    _marker: std::marker::PhantomData<C>,
}

impl<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize> Default for Transcript<F, C, W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize> Transcript<F, C, W> {
    pub fn new() -> Self {
        Transcript {
            state: [F::ZERO; W],
            pending: vec![F::ZERO; W - 4],
            out: [F::ZERO; W],
            pending_cursor: 0,
            out_cursor: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn update_state(&mut self) {
        while self.pending_cursor < (W - 4) {
            self.pending[self.pending_cursor] = F::ZERO;
            self.pending_cursor += 1;
        }

        let mut inputs: [F; W] = [F::ZERO; W];
        inputs[..W - 4].copy_from_slice(&self.pending);
        inputs[W - 4..W].copy_from_slice(&self.state[..(W - (W - 4))]);
        let output = poseidon2_hash::<F, C, W>(&inputs);
        self.out_cursor = W;
        for i in 0..W - 4 {
            self.pending[i] = F::ZERO;
        }
        self.pending_cursor = 0;
        self.state.copy_from_slice(&output[..W]);
        self.out.copy_from_slice(&output[..W]);
    }

    pub fn add1(&mut self, input: F) {
        self.pending[self.pending_cursor] = input;
        self.pending_cursor += 1;
        self.out_cursor = 0;
        if self.pending_cursor == W - 4 {
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
        let mut state = Vec::with_capacity(W);
        for i in 0..W {
            state.push(self.state[i]);
        }
        state
    }

    pub fn get_fields1(&mut self) -> F {
        if self.out_cursor == 0 {
            self.update_state();
        }
        let val = self.out[(W - self.out_cursor) % W];
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
