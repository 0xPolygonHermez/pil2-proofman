use crate::pow7;
use crate::Poseidon1Constants;
use crate::PrimeField64;

#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
extern "C" {
    fn syscall_poseidon1(state: *mut u64);
}

#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
#[inline]
fn poseidon1_hash_syscall(state: &mut [u64; 16]) {
    unsafe {
        syscall_poseidon1(state.as_mut_ptr());
    }
}

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
    cfg_if::cfg_if! {
        if #[cfg(all(target_os = "zkvm", target_vendor = "zisk"))] {
            if W == 16 {
                let mut state_u64 = [0u64; 16];
                for i in 0..16 {
                    state_u64[i] = input[i].as_canonical_u64();
                }
                poseidon1_hash_syscall(&mut state_u64);
                let mut result = [F::ZERO; W];
                for i in 0..16 {
                    result[i] = F::from_u64(state_u64[i]);
                }
                return result;
            }
        }
    }

    // Native implementation
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
    use crate::poseidon1_constants::{Poseidon1_12, Poseidon1_16, Poseidon1_8};
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

    #[test]
    fn poseidon1_16_zero_input() {
        let input = [Goldilocks::new(0); 16];
        let out = poseidon1_hash::<Goldilocks, Poseidon1_16, 16>(&input);
        let expected = [
            Goldilocks::new(543339775275048841),
            Goldilocks::new(13406197515273506800),
            Goldilocks::new(16355280640120539189),
            Goldilocks::new(15188646379150690726),
            Goldilocks::new(9354230821846213963),
            Goldilocks::new(2346697939566408112),
            Goldilocks::new(7619017200564581325),
            Goldilocks::new(228656875195661331),
            Goldilocks::new(17072924943878933846),
            Goldilocks::new(9274179898046949852),
            Goldilocks::new(17957639320403343698),
            Goldilocks::new(2237659060219097400),
            Goldilocks::new(10040284204272520954),
            Goldilocks::new(14389846334735599737),
            Goldilocks::new(17768004018560868840),
            Goldilocks::new(14237542884911959017),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn poseidon1_16_sequence_input() {
        let input: [Goldilocks; 16] = core::array::from_fn(|i| Goldilocks::new(i as u64));
        let out = poseidon1_hash::<Goldilocks, Poseidon1_16, 16>(&input);
        let expected = [
            Goldilocks::new(9350316517402464675),
            Goldilocks::new(12030202759022745826),
            Goldilocks::new(4859973758198429733),
            Goldilocks::new(15185438940901174775),
            Goldilocks::new(367739838966239011),
            Goldilocks::new(4276588024047887050),
            Goldilocks::new(1856543552381299387),
            Goldilocks::new(9084938887562314446),
            Goldilocks::new(6457218870141715263),
            Goldilocks::new(9574990127189291069),
            Goldilocks::new(13211544215836788163),
            Goldilocks::new(12635059628010643534),
            Goldilocks::new(8076414907562360476),
            Goldilocks::new(16536794806098064096),
            Goldilocks::new(6270191904161927611),
            Goldilocks::new(7308253070792633232),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn poseidon1_8_zero_input() {
        let input = [Goldilocks::new(0); 8];
        let out = poseidon1_hash::<Goldilocks, Poseidon1_8, 8>(&input);
        let expected = [
            Goldilocks::new(10843407380721191157),
            Goldilocks::new(12480894873209202472),
            Goldilocks::new(3310578452386834554),
            Goldilocks::new(243575549172213111),
            Goldilocks::new(10828976750644631960),
            Goldilocks::new(3180618067839798747),
            Goldilocks::new(14106729840943200108),
            Goldilocks::new(11601868679023094360),
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn poseidon1_8_sequence_input() {
        let input: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::new(i as u64));
        let out = poseidon1_hash::<Goldilocks, Poseidon1_8, 8>(&input);
        let expected = [
            Goldilocks::new(2431226948502761687),
            Goldilocks::new(9427563026145807618),
            Goldilocks::new(6827549936272051660),
            Goldilocks::new(16907684411084503785),
            Goldilocks::new(10131745626715172913),
            Goldilocks::new(17448305483431576765),
            Goldilocks::new(9066501914269485014),
            Goldilocks::new(12095238468458521303),
        ];
        assert_eq!(out, expected);
    }
}
