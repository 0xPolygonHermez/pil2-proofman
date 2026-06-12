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
