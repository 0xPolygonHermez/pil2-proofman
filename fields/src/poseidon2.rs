use crate::PrimeField64;
use crate::Poseidon2Constants;

#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
extern "C" {
    fn syscall_poseidon2(state: *mut u64);
}

#[cfg(all(target_os = "zkvm", target_vendor = "zisk"))]
#[inline]
fn poseidon2_hash_syscall(state: &mut [u64; 16]) {
    unsafe {
        syscall_poseidon2(state.as_mut_ptr());
    }
}

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
    cfg_if::cfg_if! {
        if #[cfg(all(target_os = "zkvm", target_vendor = "zisk"))] {
            if W == 16 {
                let mut state_u64 = [0u64; 16];
                for i in 0..16 {
                    state_u64[i] = input[i].as_canonical_u64();
                }
                poseidon2_hash_syscall(&mut state_u64);
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

#[cfg(test)]
mod tests {
    use crate::{Goldilocks, Poseidon2_16, Poseidon2_12, Poseidon2_4, Poseidon2_8};

    #[allow(unused_imports)]
    use super::*;

    #[test]
    pub fn test_poseidon2_16() {
        let input = [
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
        let output = poseidon2_hash::<Goldilocks, Poseidon2_16, 16>(&input);

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
        let input = [Goldilocks::new(0), Goldilocks::new(1), Goldilocks::new(2), Goldilocks::new(3)];
        let output = poseidon2_hash::<Goldilocks, Poseidon2_4, 4>(&input);

        assert_eq!(output[0], Goldilocks::new(8466914293353944746));
        assert_eq!(output[1], Goldilocks::new(9589318970755021278));
        assert_eq!(output[2], Goldilocks::new(5769801005587200741));
        assert_eq!(output[3], Goldilocks::new(17288820341814263849));
    }

    #[test]
    pub fn test_poseidon2_8() {
        let input = [
            Goldilocks::new(0),
            Goldilocks::new(1),
            Goldilocks::new(2),
            Goldilocks::new(3),
            Goldilocks::new(4),
            Goldilocks::new(5),
            Goldilocks::new(6),
            Goldilocks::new(7),
        ];
        let output = poseidon2_hash::<Goldilocks, Poseidon2_8, 8>(&input);

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
        let input = [
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
        let output = poseidon2_hash::<Goldilocks, Poseidon2_12, 12>(&input);

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
