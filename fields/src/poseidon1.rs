use crate::pow7;
use crate::Poseidon1Constants;
use crate::PrimeField64;

fn matmul<F: PrimeField64, const W: usize>(mat: &[u64], state: &mut [F; W]) {
    let old = *state;
    for i in 0..W {
        let mut sum = old[0] * F::from_u64(mat[i]);
        for j in 1..W {
            sum += old[j] * F::from_u64(mat[j * W + i]);
        }
        state[i] = sum;
    }
}

pub fn poseidon1_hash<F: PrimeField64, P: Poseidon1Constants<W>, const W: usize>(input: &[F; W]) -> [F; W] {
    let mut state = *input;

    // Initial ARC: state += C[0..W]
    for (i, s) in state.iter_mut().enumerate() {
        *s += F::from_u64(P::C[i]);
    }

    // First HALF_FULL_ROUNDS-1 full rounds with M matrix.
    for r in 0..(P::HALF_FULL_ROUNDS - 1) {
        for (i, s) in state.iter_mut().enumerate() {
            *s = pow7(*s) + F::from_u64(P::C[(r + 1) * W + i]);
        }
        matmul::<F, W>(P::M, &mut state);
    }

    // Transition full round with P matrix.
    for (i, s) in state.iter_mut().enumerate() {
        *s = pow7(*s) + F::from_u64(P::C[P::HALF_FULL_ROUNDS * W + i]);
    }
    matmul::<F, W>(P::P, &mut state);

    // 22 partial rounds with sparse S matrices.
    let partial_c_base = (P::HALF_FULL_ROUNDS + 1) * W;
    let stride = 2 * W - 1;
    for r in 0..P::N_PARTIAL_ROUNDS {
        // state[0] = pow7(state[0]) + C[partial_c_base + r]
        state[0] = pow7(state[0]) + F::from_u64(P::C[partial_c_base + r]);

        let s_base = stride * r;

        // s0 = sum_j state[j] * S[s_base + j]
        let mut s0 = state[0] * F::from_u64(P::S[s_base]);
        for (j, s) in state.iter().enumerate().skip(1) {
            s0 += *s * F::from_u64(P::S[s_base + j]);
        }

        // state[t] += state[0] * S[s_base + (W - 1) + t] for t in 1..W
        let s0_active = state[0];
        for (t, s) in state.iter_mut().enumerate().skip(1) {
            *s += s0_active * F::from_u64(P::S[s_base + (W - 1) + t]);
        }

        state[0] = s0;
    }

    // Last HALF_FULL_ROUNDS-1 full rounds with M matrix.
    let post_partial_base = (P::HALF_FULL_ROUNDS + 1) * W + P::N_PARTIAL_ROUNDS;
    for r in 0..(P::HALF_FULL_ROUNDS - 1) {
        for (i, s) in state.iter_mut().enumerate() {
            *s = pow7(*s) + F::from_u64(P::C[post_partial_base + r * W + i]);
        }
        matmul::<F, W>(P::M, &mut state);
    }

    // Final round: pow7 + M (no ARC).
    for s in state.iter_mut() {
        *s = pow7(*s);
    }
    matmul::<F, W>(P::M, &mut state);

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poseidon1_constants::Poseidon1_12;
    use crate::Goldilocks;

    // Ground-truth values produced by the C++ reference Poseidon1_seq
    // (setup/circom/poseidon_goldilocks.cpp).

    #[test]
    fn poseidon1_12_zero_input() {
        let input = [Goldilocks::new(0); 12];
        let out = poseidon1_hash::<Goldilocks, Poseidon1_12, 12>(&input);
        let expected = [
            Goldilocks::new(4330397376401421145),
            Goldilocks::new(14124799381142128323),
            Goldilocks::new(8742572140681234676),
            Goldilocks::new(14345658006221440202),
            Goldilocks::new(15524073338516903644),
            Goldilocks::new(5091405722150716653),
            Goldilocks::new(15002163819607624508),
            Goldilocks::new(2047012902665707362),
            Goldilocks::new(16106391063450633726),
            Goldilocks::new(4680844749859802542),
            Goldilocks::new(15019775476387350140),
            Goldilocks::new(1698615465718385111),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn poseidon1_12_sequence_input() {
        let input: [Goldilocks; 12] = core::array::from_fn(|i| Goldilocks::new(i as u64));
        let out = poseidon1_hash::<Goldilocks, Poseidon1_12, 12>(&input);
        let expected = [
            Goldilocks::new(15442313428170673822),
            Goldilocks::new(6009603122036124231),
            Goldilocks::new(15276919505380083749),
            Goldilocks::new(7005999589691109842),
            Goldilocks::new(4703821519083557360),
            Goldilocks::new(14636568497518936639),
            Goldilocks::new(7976624690322644239),
            Goldilocks::new(1802209762296193110),
            Goldilocks::new(17313479547752415775),
            Goldilocks::new(16435059422334172133),
            Goldilocks::new(14537566946116046030),
            Goldilocks::new(6632157367509271963),
        ];
        assert_eq!(out, expected);
    }
}
