use crate::{Field, Goldilocks};

pub const HALF_ROUNDS: usize = 4;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const SPONGE_WIDTH: usize = 12;

pub const D: [Goldilocks; 12] = [
    Goldilocks::new(0xc3b6c08e23ba9300),
    Goldilocks::new(0xd84b5de94a324fb6),
    Goldilocks::new(0x0d0c371c5b35b84f),
    Goldilocks::new(0x7964f570e7188037),
    Goldilocks::new(0x5daf18bbd996604b),
    Goldilocks::new(0x6743bc47b9595257),
    Goldilocks::new(0x5528b9362c59bb70),
    Goldilocks::new(0xac45e25b7127b68b),
    Goldilocks::new(0xa2077d7dfbb606b5),
    Goldilocks::new(0xf3faac6faee378ae),
    Goldilocks::new(0x0c6388b51545e883),
    Goldilocks::new(0xd27dbb6944917b60),
];

pub const C: [Goldilocks; 118] = [
    Goldilocks::new(0x13dcf33aba214f46),
    Goldilocks::new(0x30b3b654a1da6d83),
    Goldilocks::new(0x1fc634ada6159b56),
    Goldilocks::new(0x937459964dc03466),
    Goldilocks::new(0xedd2ef2ca7949924),
    Goldilocks::new(0xede9affde0e22f68),
    Goldilocks::new(0x8515b9d6bac9282d),
    Goldilocks::new(0x6b5c07b4e9e900d8),
    Goldilocks::new(0x1ec66368838c8a08),
    Goldilocks::new(0x9042367d80d1fbab),
    Goldilocks::new(0x400283564a3c3799),
    Goldilocks::new(0x4a00be0466bca75e),
    Goldilocks::new(0x7913beee58e3817f),
    Goldilocks::new(0xf545e88532237d90),
    Goldilocks::new(0x22f8cb8736042005),
    Goldilocks::new(0x6f04990e247a2623),
    Goldilocks::new(0xfe22e87ba37c38cd),
    Goldilocks::new(0xd20e32c85ffe2815),
    Goldilocks::new(0x117227674048fe73),
    Goldilocks::new(0x4e9fb7ea98a6b145),
    Goldilocks::new(0xe0866c232b8af08b),
    Goldilocks::new(0x00bbc77916884964),
    Goldilocks::new(0x7031c0fb990d7116),
    Goldilocks::new(0x240a9e87cf35108f),
    Goldilocks::new(0x2e6363a5a12244b3),
    Goldilocks::new(0x5e1c3787d1b5011c),
    Goldilocks::new(0x4132660e2a196e8b),
    Goldilocks::new(0x3a013b648d3d4327),
    Goldilocks::new(0xf79839f49888ea43),
    Goldilocks::new(0xfe85658ebafe1439),
    Goldilocks::new(0xb6889825a14240bd),
    Goldilocks::new(0x578453605541382b),
    Goldilocks::new(0x4508cda8f6b63ce9),
    Goldilocks::new(0x9c3ef35848684c91),
    Goldilocks::new(0x0812bde23c87178c),
    Goldilocks::new(0xfe49638f7f722c14),
    Goldilocks::new(0x8e3f688ce885cbf5),
    Goldilocks::new(0xb8e110acf746a87d),
    Goldilocks::new(0xb4b2e8973a6dabef),
    Goldilocks::new(0x9e714c5da3d462ec),
    Goldilocks::new(0x6438f9033d3d0c15),
    Goldilocks::new(0x24312f7cf1a27199),
    Goldilocks::new(0x23f843bb47acbf71),
    Goldilocks::new(0x9183f11a34be9f01),
    Goldilocks::new(0x839062fbb9d45dbf),
    Goldilocks::new(0x24b56e7e6c2e43fa),
    Goldilocks::new(0xe1683da61c962a72),
    Goldilocks::new(0xa95c63971a19bfa7),
    Goldilocks::new(0x4adf842aa75d4316),
    Goldilocks::new(0xf8fbb871aa4ab4eb),
    Goldilocks::new(0x68e85b6eb2dd6aeb),
    Goldilocks::new(0x07a0b06b2d270380),
    Goldilocks::new(0xd94e0228bd282de4),
    Goldilocks::new(0x8bdd91d3250c5278),
    Goldilocks::new(0x209c68b88bba778f),
    Goldilocks::new(0xb5e18cdab77f3877),
    Goldilocks::new(0xb296a3e808da93fa),
    Goldilocks::new(0x8370ecbda11a327e),
    Goldilocks::new(0x3f9075283775dad8),
    Goldilocks::new(0xb78095bb23c6aa84),
    Goldilocks::new(0x3f36b9fe72ad4e5f),
    Goldilocks::new(0x69bc96780b10b553),
    Goldilocks::new(0x3f1d341f2eb7b881),
    Goldilocks::new(0x4e939e9815838818),
    Goldilocks::new(0xda366b3ae2a31604),
    Goldilocks::new(0xbc89db1e7287d509),
    Goldilocks::new(0x6102f411f9ef5659),
    Goldilocks::new(0x58725c5e7ac1f0ab),
    Goldilocks::new(0x0df5856c798883e7),
    Goldilocks::new(0xf7bb62a8da4c961b),
    Goldilocks::new(0xc68be7c94882a24d),
    Goldilocks::new(0xaf996d5d5cdaedd9),
    Goldilocks::new(0x9717f025e7daf6a5),
    Goldilocks::new(0x6436679e6e7216f4),
    Goldilocks::new(0x8a223d99047af267),
    Goldilocks::new(0xbb512e35a133ba9a),
    Goldilocks::new(0xfbbf44097671aa03),
    Goldilocks::new(0xf04058ebf6811e61),
    Goldilocks::new(0x5cca84703fac7ffb),
    Goldilocks::new(0x9b55c7945de6469f),
    Goldilocks::new(0x8e05bf09808e934f),
    Goldilocks::new(0x2ea900de876307d7),
    Goldilocks::new(0x7748fff2b38dfb89),
    Goldilocks::new(0x6b99a676dd3b5d81),
    Goldilocks::new(0xac4bb7c627cf7c13),
    Goldilocks::new(0xadb6ebe5e9e2f5ba),
    Goldilocks::new(0x2d33378cafa24ae3),
    Goldilocks::new(0x1e5b73807543f8c2),
    Goldilocks::new(0x09208814bfebb10f),
    Goldilocks::new(0x782e64b6bb5b93dd),
    Goldilocks::new(0xadd5a48eac90b50f),
    Goldilocks::new(0xadd4c54c736ea4b1),
    Goldilocks::new(0xd58dbb86ed817fd8),
    Goldilocks::new(0x6d5ed1a533f34ddd),
    Goldilocks::new(0x28686aa3e36b7cb9),
    Goldilocks::new(0x591abd3476689f36),
    Goldilocks::new(0x047d766678f13875),
    Goldilocks::new(0xa2a11112625f5b49),
    Goldilocks::new(0x21fd10a3f8304958),
    Goldilocks::new(0xf9b40711443b0280),
    Goldilocks::new(0xd2697eb8b2bde88e),
    Goldilocks::new(0x3493790b51731b3f),
    Goldilocks::new(0x11caf9dd73764023),
    Goldilocks::new(0x7acfb8f72878164e),
    Goldilocks::new(0x744ec4db23cefc26),
    Goldilocks::new(0x1e00e58f422c6340),
    Goldilocks::new(0x21dd28d906a62dda),
    Goldilocks::new(0xf32a46ab5f465b5f),
    Goldilocks::new(0xbfce13201f3f7e6b),
    Goldilocks::new(0xf30d2e7adb5304e2),
    Goldilocks::new(0xecdf4ee4abad48e9),
    Goldilocks::new(0xf94e82182d395019),
    Goldilocks::new(0x4ee52e3744d887c5),
    Goldilocks::new(0xa1341c7cac0083b2),
    Goldilocks::new(0x2302fb26c30c834a),
    Goldilocks::new(0xaea3c587273bf7d3),
    Goldilocks::new(0xf798e24961823ec7),
    Goldilocks::new(0x962deba3e9a2cd94),
];

pub fn matmul_m4(input: &mut [Goldilocks]) {
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

pub fn matmul_external(input: &mut [Goldilocks]) {
    matmul_m4(&mut input[0..4]);
    matmul_m4(&mut input[4..8]);
    matmul_m4(&mut input[8..12]);

    let stored = [
        input[0] + input[4] + input[8],
        input[1] + input[5] + input[9],
        input[2] + input[6] + input[10],
        input[3] + input[7] + input[11],
    ];

    for (i, x) in input.iter_mut().enumerate() {
        *x += stored[i % 4];
    }
}

pub fn prodadd(input: &mut [Goldilocks], sum: Goldilocks) {
    for i in 0..SPONGE_WIDTH {
        input[i] = input[i] * D[i] + sum;
    }
}
pub fn pow7add(input: &mut [Goldilocks], c: &[Goldilocks]) {
    for (i, x) in input.iter_mut().enumerate() {
        *x += c[i];
        *x = pow7(*x);
    }
}

pub fn pow7(input: Goldilocks) -> Goldilocks {
    let x2 = input * input;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    x6 * input
}

pub fn add(input: &mut [Goldilocks]) -> Goldilocks {
    let mut sum = Goldilocks::ZERO;
    for x in input.iter() {
        sum += *x;
    }
    sum
}

pub fn poseidon2_hash(input: &mut [Goldilocks]) -> Vec<Goldilocks> {
    let mut state = input.to_vec();
    matmul_external(&mut state);
    for r in 0..HALF_ROUNDS {
        pow7add(&mut state, &C[r * SPONGE_WIDTH..(r + 1) * SPONGE_WIDTH]);
        matmul_external(&mut state);
    }

    for r in 0..N_PARTIAL_ROUNDS {
        state[0] += C[HALF_ROUNDS * SPONGE_WIDTH + r];
        state[0] = pow7(state[0]);
        let sum = add(&mut state);
        prodadd(&mut state, sum);
    }

    for r in 0..HALF_ROUNDS {
        pow7add(
            &mut state,
            &C[(HALF_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH)
                ..(HALF_ROUNDS * SPONGE_WIDTH + N_PARTIAL_ROUNDS + (r + 1) * SPONGE_WIDTH)],
        );
        matmul_external(&mut state);
    }
    state
}

pub fn linear_hash_seq(input: &[Goldilocks]) -> Vec<Goldilocks> {
    let mut state = vec![Goldilocks::ZERO; SPONGE_WIDTH];
    let size = input.len();
    if size <= 4 {
        state[..size].copy_from_slice(&input[..size]);
        return state;
    }
    let mut remaining = size;
    while remaining > 0 {
        if remaining != size {
            for i in 0..4 {
                state[8 + i] = state[i];
            }
        }
        let n = if remaining < 8 { remaining } else { 8 };
        for i in 0..(8 - n) {
            state[n + i] = Goldilocks::ZERO;
        }
        for i in 0..n {
            state[i] = input[size - remaining + i];
        }
        state = poseidon2_hash(&mut state);
        remaining -= n;
    }
    state
}

pub fn calculate_root_from_proof(value: &mut [Goldilocks], mp: &[Vec<Goldilocks>], idx: u64, offset: u64, arity: u64) {
    if offset == mp.len() as u64 {
        return;
    }

    let curr_idx = idx % arity;
    let next_idx = idx / arity;

    let mut inputs = vec![Goldilocks::ZERO; SPONGE_WIDTH];
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

    let outputs = poseidon2_hash(&mut inputs);

    value[..4].copy_from_slice(&outputs[..4]);
    calculate_root_from_proof(value, mp, next_idx, offset + 1, arity);
}

pub fn verify_mt(root: &[Goldilocks], mp: &[Vec<Goldilocks>], idx: u64, v: &[Goldilocks], arity: u64) -> bool {
    let mut value = linear_hash_seq(v);

    calculate_root_from_proof(&mut value, mp, idx, 0, arity);
    for i in 0..4 {
        if value[i] != root[i] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
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
        ];
        let output = poseidon2_hash(&mut input);

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
