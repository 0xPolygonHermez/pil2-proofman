use crate::PrimeField64;
use crate::{DIAG_4, RC_4};

pub const HALF_ROUNDS: usize = 4;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const SPONGE_WIDTH: usize = 16;
pub const RATE: usize = SPONGE_WIDTH - 4;
pub const CAPACITY: usize = 4;

pub fn matmul_m4<F: PrimeField64>(input: &mut [F]) {
    let t0 = input[0] + input[1];
    let t1 = input[2] + input[3];
    let t2 = input[1] + input[1] + t1;
    let t3 = input[3] + input[3] + t0;
    let t1_2 = t1 + t1;
    let t0_2 = t0 + t0;
    let t4 = t1_2 + t1_2 + t3;
    let t5 = t0_2 + t0_2 + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;

    input[0] = t6;
    input[1] = t5;
    input[2] = t7;
    input[3] = t4;
}

pub fn matmul_external<F: PrimeField64>(input: &mut [F]) {
    for i in 0..SPONGE_WIDTH / 4 {
        matmul_m4(&mut input[i * 4..(i + 1) * 4]);
    }

    let mut stored = [F::ZERO; 4];
    for i in 0..4 {
        for j in 0..SPONGE_WIDTH / 4 {
            stored[i] += input[j * 4 + i];
        }
    }

    for (i, x) in input.iter_mut().enumerate() {
        *x += stored[i % 4];
    }
}

pub fn prodadd<F: PrimeField64>(input: &mut [F], d: &[u64], sum: F) {
    for i in 0..SPONGE_WIDTH {
        input[i] = input[i] * F::from_u64(d[i]) + sum;
    }
}
pub fn pow7add<F: PrimeField64>(input: &mut [F], c: &[F]) {
    for (i, x) in input.iter_mut().enumerate() {
        *x += c[i];
        *x = pow7(*x);
    }
}

pub fn pow7<F: PrimeField64>(input: F) -> F {
    let x2 = input * input;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * input
}

pub fn add<F: PrimeField64>(input: &mut [F]) -> F {
    let mut sum = F::ZERO;
    for x in input.iter() {
        sum += *x;
    }
    sum
}

pub fn poseidon2_hash<F: PrimeField64>(input: &[F]) -> Vec<F> {
    let mut state = input.to_vec();
    matmul_external(&mut state);
    for r in 0..HALF_ROUNDS {
        let c_slice: Vec<F> = RC_4[r * SPONGE_WIDTH..(r + 1) * SPONGE_WIDTH].iter().map(|&x| F::from_u64(x)).collect();
        pow7add(&mut state, &c_slice);
        matmul_external(&mut state);
    }

    for r in 0..N_PARTIAL_ROUNDS {
        state[0] += F::from_u64(RC_4[HALF_ROUNDS * SPONGE_WIDTH + r]);
        state[0] = pow7(state[0]);
        let sum = add(&mut state);
        prodadd(&mut state, &DIAG_4, sum);
    }

    for r in 0..HALF_ROUNDS {
        let c_slice: Vec<F> = RC_4[(HALF_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH)
            ..(HALF_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + (r + 1) * SPONGE_WIDTH)]
            .iter()
            .map(|&x| F::from_u64(x))
            .collect();
        pow7add(&mut state, &c_slice);
        matmul_external(&mut state);
    }
    state
}

pub fn linear_hash_seq<F: PrimeField64>(input: &[F]) -> Vec<F> {
    let mut state = vec![F::ZERO; SPONGE_WIDTH];
    let size = input.len();
    if size <= 4 {
        state[..size].copy_from_slice(&input[..size]);
        return state;
    }
    let mut remaining = size;
    while remaining > 0 {
        if remaining != size {
            for i in 0..4 {
                state[RATE + i] = state[i];
            }
        }
        let n = if remaining < RATE { remaining } else { RATE };
        for i in 0..(RATE - n) {
            state[n + i] = F::ZERO;
        }
        for i in 0..n {
            state[i] = input[size - remaining + i];
        }
        state = poseidon2_hash(&state);
        remaining -= n;
    }
    state
}

pub fn calculate_root_from_proof<F: PrimeField64>(
    value: &mut [F],
    mp: &[Vec<F>],
    idx: &mut u64,
    offset: u64,
    arity: u64,
) {
    if offset == mp.len() as u64 {
        return;
    }

    let curr_idx = *idx % arity;
    *idx /= arity;

    let mut inputs = vec![F::ZERO; SPONGE_WIDTH];
    let mut p = 0;
    for i in 0..arity {
        if i == curr_idx {
            continue;
        }
        for j in 0..4 {
            inputs[(i * 4 + j) as usize] = mp[offset as usize][4 * p + j as usize];
        }
        p += 1;
    }
    for j in 0..4 {
        inputs[(curr_idx * 4 + j) as usize] = value[j as usize];
    }

    let outputs = poseidon2_hash(&inputs);

    value[..4].copy_from_slice(&outputs[..4]);
    calculate_root_from_proof(value, mp, idx, offset + 1, arity);
}

pub fn partial_merkle_tree<F: PrimeField64>(input: &[F], num_elements: u64, arity: u64) -> [F; CAPACITY] {
    let mut num_nodes = num_elements;
    let mut nodes_level = num_elements;

    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }

    let mut cursor = vec![F::ZERO; (num_nodes * CAPACITY as u64) as usize];
    cursor[..(num_elements * CAPACITY as u64) as usize]
        .copy_from_slice(&input[..(num_elements * CAPACITY as u64) as usize]);

    let mut pending = num_elements;
    let mut next_n = pending.div_ceil(arity);
    let mut next_index = 0;

    while pending > 1 {
        let extra_zeros = (arity - (pending % arity)) % arity;

        if extra_zeros > 0 {
            let start = (next_index + pending * CAPACITY as u64) as usize;
            let end = start + (extra_zeros * CAPACITY as u64) as usize;
            cursor[start..end].fill(F::ZERO);
        }

        for i in 0..next_n {
            let mut pol_input = vec![F::ZERO; SPONGE_WIDTH];

            let child_start = (next_index + i * SPONGE_WIDTH as u64) as usize;
            pol_input[..SPONGE_WIDTH].copy_from_slice(&cursor[child_start..child_start + SPONGE_WIDTH]);

            // Compute hash
            let parent_start = (next_index + (pending + extra_zeros + i) * CAPACITY as u64) as usize;
            let parent_hash = poseidon2_hash(&pol_input);
            cursor[parent_start..parent_start + CAPACITY].copy_from_slice(&parent_hash[..CAPACITY]);
        }

        next_index += (pending + extra_zeros) * CAPACITY as u64;
        pending = pending.div_ceil(arity);
        next_n = pending.div_ceil(arity);
    }

    let mut root = [F::ZERO; CAPACITY];
    root.copy_from_slice(&cursor[next_index as usize..next_index as usize + CAPACITY]);
    root
}

pub fn verify_mt<F: PrimeField64>(
    root: &[F],
    last_level: &[F],
    mp: &[Vec<F>],
    idx: u64,
    v: &[F],
    arity: u64,
    last_level_verification: u64,
) -> bool {
    let mut value = linear_hash_seq(v);

    let mut query_idx = idx;
    calculate_root_from_proof(&mut value, mp, &mut query_idx, 0, arity);

    if last_level_verification == 0 {
        for i in 0..4 {
            if value[i] != root[i] {
                return false;
            }
        }
    } else {
        for i in 0..4 {
            if value[i] != last_level[query_idx as usize * 4 + i] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use crate::Goldilocks;

    #[allow(unused_imports)]
    use super::*;

    #[test]
    pub fn test_poseidon2() {
        let mut input = [
            Goldilocks::new(0),
            Goldilocks::new(1),
            Goldilocks::new(2),
            Goldilocks::new(3),
            Goldilocks::new(4),
            Goldilocks::new(5),
            Goldilocks::new(6),
            Goldilocks::new(7),
            Goldilocks::new(8),
            Goldilocks::new(9),
            Goldilocks::new(10),
            Goldilocks::new(11),
            Goldilocks::new(12),
            Goldilocks::new(13),
            Goldilocks::new(14),
            Goldilocks::new(15),
        ];
        let output = poseidon2_hash(&mut input);

        assert_eq!(output[0], Goldilocks::new(9639188652563994454));
        assert_eq!(output[1], Goldilocks::new(12273372933164734616));
        assert_eq!(output[2], Goldilocks::new(2905147255612444119));
        assert_eq!(output[3], Goldilocks::new(17581461329934617288));
        assert_eq!(output[4], Goldilocks::new(14390794100096760072));
        assert_eq!(output[5], Goldilocks::new(5468485695976078057));
        assert_eq!(output[6], Goldilocks::new(2832370985856357627));
        assert_eq!(output[7], Goldilocks::new(1116111836864400812));
        assert_eq!(output[8], Goldilocks::new(14997632823506024332));
        assert_eq!(output[9], Goldilocks::new(3976503894892102369));
        assert_eq!(output[10], Goldilocks::new(14874978986912301676));
        assert_eq!(output[11], Goldilocks::new(12458748982184310703));
        assert_eq!(output[12], Goldilocks::new(103345454961107931));
        assert_eq!(output[13], Goldilocks::new(3354965064850558444));
        assert_eq!(output[14], Goldilocks::new(14413825288474057217));
        assert_eq!(output[15], Goldilocks::new(4214638127285300968));
    }
}
