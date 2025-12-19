use crate::PrimeField64;
use crate::Poseidon2Constants;

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

pub fn matmul_external<F: PrimeField64, const W: usize>(input: &mut [F; W]) {
    for i in 0..W / 4 {
        matmul_m4(&mut input[i * 4..(i + 1) * 4]);
    }

    if W > 4 {
        let mut stored = [F::ZERO; 4];
        for i in 0..4 {
            for j in 0..W / 4 {
                stored[i] += input[j * 4 + i];
            }
        }

        for (i, x) in input.iter_mut().enumerate() {
            *x += stored[i % 4];
        }
    }
}

pub fn prodadd<F: PrimeField64, const W: usize>(input: &mut [F; W], d: &[u64], sum: F) {
    for i in 0..W {
        input[i] = input[i] * F::from_u64(d[i]) + sum;
    }
}
pub fn pow7add<F: PrimeField64, const W: usize>(input: &mut [F; W], c: &[F]) {
    for i in 0..W {
        input[i] += c[i];
        input[i] = pow7(input[i]);
    }
}

pub fn pow7<F: PrimeField64>(input: F) -> F {
    let x2 = input * input;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * input
}

pub fn add<F: PrimeField64, const W: usize>(input: &[F; W]) -> F {
    let mut sum = F::ZERO;
    for x in input.iter() {
        sum += *x;
    }
    sum
}

pub fn poseidon2_hash<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize>(input: &[F; W]) -> [F; W] {
    let mut state = *input;

    matmul_external::<F, W>(&mut state);

    for r in 0..C::HALF_ROUNDS {
        let mut c_slice = [F::ZERO; W];
        for (i, c) in c_slice.iter_mut().enumerate() {
            *c = F::from_u64(C::RC[r * W + i]);
        }
        pow7add::<F, W>(&mut state, &c_slice);
        matmul_external::<F, W>(&mut state);
    }

    for r in 0..C::N_PARTIAL_ROUNDS {
        state[0] += F::from_u64(C::RC[C::HALF_ROUNDS * W + r]);
        state[0] = pow7(state[0]);
        let sum = add::<F, W>(&state);
        prodadd::<F, W>(&mut state, C::DIAG, sum);
    }

    for r in 0..C::HALF_ROUNDS {
        let mut c_slice = [F::ZERO; W];
        for (i, c) in c_slice.iter_mut().enumerate() {
            *c = F::from_u64(C::RC[C::HALF_ROUNDS * W + C::N_PARTIAL_ROUNDS + r * W + i]);
        }
        pow7add::<F, W>(&mut state, &c_slice);
        matmul_external::<F, W>(&mut state);
    }

    state
}

pub fn linear_hash_seq<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize>(input: &[F]) -> [F; W] {
    assert!(W > 4);
    let mut state: [F; W] = [F::ZERO; W];
    let size = input.len();
    if size <= 4 {
        state[..size].copy_from_slice(&input[..size]);
        return state;
    }
    let mut remaining = size;
    while remaining > 0 {
        if remaining != size {
            for i in 0..4 {
                state[C::RATE + i] = state[i];
            }
        }
        let n = if remaining < C::RATE { remaining } else { C::RATE };
        for i in 0..(C::RATE - n) {
            state[n + i] = F::ZERO;
        }
        for i in 0..n {
            state[i] = input[size - remaining + i];
        }
        state = poseidon2_hash::<F, C, W>(&state);
        remaining -= n;
    }
    state
}

pub fn calculate_root_from_proof<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize>(
    value: &mut [F; W],
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

    let mut inputs: [F; W] = [F::ZERO; W];
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

    let outputs = poseidon2_hash::<F, C, W>(&inputs);

    value[..4].copy_from_slice(&outputs[..4]);
    calculate_root_from_proof::<F, C, W>(value, mp, idx, offset + 1, arity);
}

pub fn partial_merkle_tree<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize>(
    input: &[F],
    num_elements: u64,
    arity: u64,
) -> [F; 4] {
    let mut num_nodes = num_elements;
    let mut nodes_level = num_elements;

    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }

    let mut cursor = vec![F::ZERO; (num_nodes * C::CAPACITY as u64) as usize];
    cursor[..(num_elements * C::CAPACITY as u64) as usize]
        .copy_from_slice(&input[..(num_elements * C::CAPACITY as u64) as usize]);

    let mut pending = num_elements;
    let mut next_n = pending.div_ceil(arity);
    let mut next_index = 0;

    while pending > 1 {
        let extra_zeros = (arity - (pending % arity)) % arity;

        if extra_zeros > 0 {
            let start = (next_index + pending * C::CAPACITY as u64) as usize;
            let end = start + (extra_zeros * C::CAPACITY as u64) as usize;
            cursor[start..end].fill(F::ZERO);
        }

        for i in 0..next_n {
            let mut pol_input: [F; W] = [F::ZERO; W];

            let child_start = (next_index + i * C::SPONGE_WIDTH as u64) as usize;
            pol_input[..C::SPONGE_WIDTH].copy_from_slice(&cursor[child_start..child_start + C::SPONGE_WIDTH]);

            // Compute hash
            let parent_start = (next_index + (pending + extra_zeros + i) * C::CAPACITY as u64) as usize;
            let parent_hash = poseidon2_hash::<F, C, W>(&pol_input);
            cursor[parent_start..parent_start + C::CAPACITY].copy_from_slice(&parent_hash[..C::CAPACITY]);
        }

        next_index += (pending + extra_zeros) * C::CAPACITY as u64;
        pending = pending.div_ceil(arity);
        next_n = pending.div_ceil(arity);
    }

    let mut root = [F::ZERO; 4];
    root.copy_from_slice(&cursor[next_index as usize..next_index as usize + 4]);
    root
}

pub fn verify_mt<F: PrimeField64, C: Poseidon2Constants<W>, const W: usize>(
    root: &[F],
    last_level: &[F],
    mp: &[Vec<F>],
    idx: u64,
    v: &[F],
    arity: u64,
    last_level_verification: u64,
) -> bool {
    let mut value = linear_hash_seq::<F, C, W>(v);

    let mut query_idx = idx;
    calculate_root_from_proof::<F, C, W>(&mut value, mp, &mut query_idx, 0, arity);

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
    use crate::{Goldilocks, Poseidon16, Poseidon12, Poseidon4, Poseidon8};

    #[allow(unused_imports)]
    use super::*;

    #[test]
    pub fn test_poseidon2_16() {
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
        let output = poseidon2_hash::<Goldilocks, Poseidon16, 16>(&mut input);

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

    #[test]
    pub fn test_poseidon2_4() {
        let mut input = [Goldilocks::new(0), Goldilocks::new(1), Goldilocks::new(2), Goldilocks::new(3)];
        let output = poseidon2_hash::<Goldilocks, Poseidon4, 4>(&mut input);

        assert_eq!(output[0], Goldilocks::new(8466914293353944746));
        assert_eq!(output[1], Goldilocks::new(9589318970755021278));
        assert_eq!(output[2], Goldilocks::new(5769801005587200741));
        assert_eq!(output[3], Goldilocks::new(17288820341814263849));
    }

    #[test]
    pub fn test_poseidon2_8() {
        let mut input = [
            Goldilocks::new(0),
            Goldilocks::new(1),
            Goldilocks::new(2),
            Goldilocks::new(3),
            Goldilocks::new(4),
            Goldilocks::new(5),
            Goldilocks::new(6),
            Goldilocks::new(7),
        ];
        let output = poseidon2_hash::<Goldilocks, Poseidon8, 8>(&mut input);

        assert_eq!(output[0], Goldilocks::new(14266028122062624699));
        assert_eq!(output[1], Goldilocks::new(5353147180106052723));
        assert_eq!(output[2], Goldilocks::new(15203350112844181434));
        assert_eq!(output[3], Goldilocks::new(17630919042639565165));
        assert_eq!(output[4], Goldilocks::new(16601551015858213987));
        assert_eq!(output[5], Goldilocks::new(10184091939013874068));
        assert_eq!(output[6], Goldilocks::new(16774100645754596496));
        assert_eq!(output[7], Goldilocks::new(12047415603622314780));
    }

    #[test]
    pub fn test_poseidon2_12() {
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
        ];
        let output = poseidon2_hash::<Goldilocks, Poseidon12, 12>(&mut input);

        assert_eq!(output[0], Goldilocks::new(138186169299091649));
        assert_eq!(output[1], Goldilocks::new(2237493815125627916));
        assert_eq!(output[2], Goldilocks::new(7098449130000758157));
        assert_eq!(output[3], Goldilocks::new(16681569560651424230));
        assert_eq!(output[4], Goldilocks::new(2885694034573886267));
        assert_eq!(output[5], Goldilocks::new(1987263728465303211));
        assert_eq!(output[6], Goldilocks::new(4895658260063552408));
        assert_eq!(output[7], Goldilocks::new(16782691522897809445));
        assert_eq!(output[8], Goldilocks::new(6250362358359317026));
        assert_eq!(output[9], Goldilocks::new(8723968546836371205));
        assert_eq!(output[10], Goldilocks::new(17025428646788054631));
        assert_eq!(output[11], Goldilocks::new(7660698892044183277));
    }
}
