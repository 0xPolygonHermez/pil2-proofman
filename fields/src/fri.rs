use crate::{CubicExtensionField, Field, Goldilocks};

const OMEGAS_INV: [u64; 33] = [
    0x1,
    0xffffffff00000000,
    0xfffeffff00000001,
    0xfffffeff00000101,
    0xffefffff00100001,
    0xfbffffff04000001,
    0xdfffffff20000001,
    0x3fffbfffc0,
    0x7f4949dce07bf05d,
    0x4bd6bb172e15d48c,
    0x38bc97652b54c741,
    0x553a9b711648c890,
    0x55da9bb68958caa,
    0xa0a62f8f0bb8e2b6,
    0x276fd7ae450aee4b,
    0x7b687b64f5de658f,
    0x7de5776cbda187e9,
    0xd2199b156a6f3b06,
    0xd01c8acd8ea0e8c0,
    0x4f38b2439950a4cf,
    0x5987c395dd5dfdcf,
    0x46cf3d56125452b1,
    0x909c4b1a44a69ccb,
    0xc188678a32a54199,
    0xf3650f9ddfcaffa8,
    0xe8ef0e3e40a92655,
    0x7c8abec072bb46a6,
    0xe0bfc17d5c5a7a04,
    0x4c6b8a5a0b79f23a,
    0x6b4d20533ce584fe,
    0xe5cceae468a70ec2,
    0x8958579f296dac7a,
    0x16d265893b5b7e85,
];
const DOMAIN_SIZE_INVERSE: [u64; 33] = [
    0x0000000000000001, // 1^{-1}
    0x7fffffff80000001, // 2^{-1}
    0xbfffffff40000001, // (1 << 2)^{-1}
    0xdfffffff20000001, // (1 << 3)^{-1}
    0xefffffff10000001,
    0xf7ffffff08000001,
    0xfbffffff04000001,
    0xfdffffff02000001,
    0xfeffffff01000001,
    0xff7fffff00800001,
    0xffbfffff00400001,
    0xffdfffff00200001,
    0xffefffff00100001,
    0xfff7ffff00080001,
    0xfffbffff00040001,
    0xfffdffff00020001,
    0xfffeffff00010001,
    0xffff7fff00008001,
    0xffffbfff00004001,
    0xffffdfff00002001,
    0xffffefff00001001,
    0xfffff7ff00000801,
    0xfffffbff00000401,
    0xfffffdff00000201,
    0xfffffeff00000101,
    0xffffff7f00000081,
    0xffffffbf00000041,
    0xffffffdf00000021,
    0xffffffef00000011,
    0xfffffff700000009,
    0xfffffffb00000005,
    0xfffffffd00000003,
    0xfffffffe00000002, // (1 << 32)^{-1}
];

fn bit_reverse(mut x: u32, bits: usize) -> u32 {
    // layers of bit swaps
    x = ((x >> 1) & 0x5555_5555) | ((x & 0x5555_5555) << 1);
    x = ((x >> 2) & 0x3333_3333) | ((x & 0x3333_3333) << 2);
    x = ((x >> 4) & 0x0F0F_0F0F) | ((x & 0x0F0F_0F0F) << 4);
    x = ((x >> 8) & 0x00FF_00FF) | ((x & 0x00FF_00FF) << 8);
    x = x.rotate_left(16);
    // drop upper bits
    x >> (32 - bits)
}

pub fn intt_tiny(data: &mut [Goldilocks], n_bits: usize, n_cols: usize) {
    let n = 1 << n_bits;
    // allocate scratch for bit-reversed copy
    let mut vals = vec![Goldilocks::ZERO; n * n_cols];
    // bit-reverse permutation
    for i in 0..n {
        let r = bit_reverse(i as u32, n_bits) as usize;
        for c in 0..n_cols {
            vals[r * n_cols + c] = data[i * n_cols + c];
        }
    }

    // Cooley-Tukey inverse NTT
    for stage in 0..n_bits {
        let m = 1 << (stage + 1);
        let half_m = m >> 1;
        // compute twiddles for this stage
        // use OMEGAS_INV[stage+1] so that stage 0 uses the N/2 root, stage 1 uses N/4, etc.
        let omega_inv = OMEGAS_INV[stage + 1];
        let mut twiddles = Vec::with_capacity(half_m);
        twiddles.push(Goldilocks::ONE);
        for j in 1..half_m {
            twiddles.push(twiddles[j - 1] * Goldilocks::new(omega_inv));
        }

        for k in (0..n).step_by(m) {
            for (j, w) in twiddles.iter().enumerate().take(half_m) {
                for c in 0..n_cols {
                    let idx1 = (k + j) * n_cols + c;
                    let idx2 = (k + j + half_m) * n_cols + c;
                    let u = vals[idx1];
                    let t = vals[idx2] * *w;
                    // inverse butterfly: even = u + t, odd = u - t
                    vals[idx1] = u + t;
                    vals[idx2] = u - t;
                }
            }
        }
    }

    // scale by n^{-1}
    let inv_n = Goldilocks::new(DOMAIN_SIZE_INVERSE[n_bits]);
    for v in vals.iter_mut() {
        *v *= inv_n;
    }

    // write back to original buffer
    data.copy_from_slice(&vals);
}

pub fn verify_fold(
    n_bits_ext: u64,
    current_bits: u64,
    prev_bits: u64,
    challenge: CubicExtensionField<Goldilocks>,
    idx: u64,
    vals: &[Goldilocks],
) -> Vec<Goldilocks> {
    let mut shift = Goldilocks::new(Goldilocks::SHIFT);

    for _ in 0..(n_bits_ext - prev_bits) {
        shift = shift * shift;
    }

    let n_x = 1 << (prev_bits - current_bits);

    assert_eq!(vals.len(), n_x * 3, "vals length {} does not match expected nX {}", vals.len(), n_x);

    let w = Goldilocks::new(Goldilocks::W[prev_bits as usize]);
    let sinv = (shift * w.exp_u64(idx)).inverse();

    let mut ppar_c = vec![Goldilocks::ZERO; n_x * 3];
    ppar_c[..(n_x * 3)].copy_from_slice(&vals[..(n_x * 3)]);

    intt_tiny(&mut ppar_c, (prev_bits - current_bits) as usize, 3);

    let aux = challenge * sinv;
    eval_pol(n_x, &ppar_c, aux)
}

/// Evaluate the degree-`degree` polynomial `p` at point `x[0]`
/// and write the result into `res` at index `res_idx`.
pub fn eval_pol(degree: usize, p: &[Goldilocks], x: CubicExtensionField<Goldilocks>) -> Vec<Goldilocks> {
    if degree == 0 {
        return vec![Goldilocks::ZERO; 3];
    }

    let mut res =
        CubicExtensionField { value: [p[(degree - 1) * 3], p[(degree - 1) * 3 + 1], p[(degree - 1) * 3 + 2]] };

    for i in (0..degree - 1).rev() {
        let p_i = CubicExtensionField { value: [p[3 * i], p[3 * i + 1], p[3 * i + 2]] };
        res = res * x + p_i;
    }

    res.value.to_vec()
}
