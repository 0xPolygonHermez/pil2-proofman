//! Mutated proofs for the tests of the Solidity verifier (pilfflonk/docs/verifier.md#tests): the
//! arithmetic of `Fr` over `num-bigint` that they need, and [`fixup`], which recomputes `invZh` and
//! `inv` of a mutated proof for its own transcript (pilfflonk/docs/protocol.md#inverses), as an
//! attacker would, so that the proof gets past those checks to the deeper ones, `checkQPieces` and
//! the pairing. The Solidity tests of `pilfflonk-setup` (`setup/pilfflonk/tests/solidity.rs`) and
//! the differential fuzzer (`fuzz.rs`, which the CLI's end to end runs) share it.
//!
//! Include it with `#[path = ".../pilfflonk/tests/data/mutations.rs"] mod mutations;`.

use std::sync::OnceLock;

use num_bigint::BigUint;
use proofman_pilfflonk::{verifier_challenges, FrBytes, Proof, Vkey, BN254_Q, BN254_R};

/// BN254's scalar field modulus `r`.
pub fn r() -> BigUint {
    static R: OnceLock<BigUint> = OnceLock::new();
    R.get_or_init(|| BigUint::parse_bytes(BN254_R.as_bytes(), 10).unwrap()).clone()
}

/// BN254's base field modulus `q`.
pub fn q() -> BigUint {
    static Q: OnceLock<BigUint> = OnceLock::new();
    Q.get_or_init(|| BigUint::parse_bytes(BN254_Q.as_bytes(), 10).unwrap()).clone()
}

pub fn big(value: &FrBytes) -> BigUint {
    BigUint::from_bytes_le(&value.to_le_bytes())
}

pub fn plus_one(value: &FrBytes) -> FrBytes {
    FrBytes::from_decimal(&((big(value) + 1u32) % r()).to_string()).unwrap()
}

/// The 32 big-endian bytes of `value < 2^256`.
pub fn be_word(value: &BigUint) -> [u8; 32] {
    let bytes = value.to_bytes_be();
    let mut word = [0u8; 32];
    word[32 - bytes.len()..].copy_from_slice(&bytes);
    word
}

/// `s mod r` as a scalar.
pub fn fr(s: &BigUint) -> FrBytes {
    FrBytes::from_decimal(&(s % r()).to_str_radix(10)).unwrap()
}

pub fn fr_inv(a: &BigUint) -> BigUint {
    a.modpow(&(r() - 2u32), &r())
}

/// `a − b mod r`.
pub fn fr_sub(a: &BigUint, b: &BigUint) -> BigUint {
    (a + r() - (b % r())) % r()
}

/// `5^((r−1)/n)`, a primitive `n`-th root of unity (`shplonk.js`, `rootOfUnity`).
pub fn root_of_unity(n: u64) -> BigUint {
    BigUint::from(5u32).modpow(&((r() - 1u32) / n), &r())
}

/// `ξ = xiSeed^powerW`.
pub fn xi_of(vkey: &Vkey, xi_seed: &BigUint) -> BigUint {
    xi_seed.modpow(&BigUint::from(vkey.power_w), &r())
}

/// The roots `T` of an `f` of `k` polynomials and these offsets, offset-major (`shplonk.js`,
/// `computeRoots`): `x_j = xiSeed^(powerW/k)·ω_{kN}^s·w_k^j`.
pub fn roots(vkey: &Vkey, k: u64, offsets: &[i64], xi_seed: &BigUint) -> Vec<BigUint> {
    let r = r();
    let seed = xi_seed.modpow(&BigUint::from(vkey.power_w / k), &r);
    let (omega_kn, w_k) = (root_of_unity(k << vkey.power), root_of_unity(k));
    let mut t = Vec::new();
    for &s in offsets {
        let power = omega_kn.modpow(&BigUint::from(s.unsigned_abs()), &r);
        let mut x = &seed * if s < 0 { fr_inv(&power) } else { power } % &r;
        for _ in 0..k {
            t.push(x.clone());
            x = x * &w_k % &r;
        }
    }
    t
}

/// `proof` with `invZh` and `inv` recomputed for its own transcript
/// (pilfflonk/docs/protocol.md#inverses; `shplonk.js`, `computeZerofiers` and
/// `computeInverseDenominators`, as an attacker would), so that a mutated proof, with the auxiliary
/// inverses of its `ξ` (`Calldata::encode`), gets past those checks to the deeper ones:
/// `checkQPieces` and the pairing.
pub fn fixup(vkey: &Vkey, mut proof: Proof, publics: &[FrBytes]) -> Proof {
    let ch = verifier_challenges(vkey, &proof, publics).expect("a proof whose points the transcript absorbs");
    let (xi_seed, y) = (big(&ch.xi_seed), big(&ch.y));
    let r = r();
    let xi = xi_of(vkey, &xi_seed);
    proof.inv_zh = fr(&fr_inv(&fr_sub(&xi.modpow(&BigUint::from(1u64 << vkey.power), &r), &BigUint::from(1u32))));
    let mut product = BigUint::from(1u32);
    for (i, f) in vkey.layout.0.iter().enumerate() {
        let t = roots(vkey, f.k, &f.offsets, &xi_seed);
        if i > 0 {
            product = t.iter().fold(product, |z, x| z * fr_sub(&y, x) % &r);
        }
        for (m, x) in t.iter().enumerate() {
            let den =
                t.iter().enumerate().filter(|&(l, _)| l != m).fold(fr_sub(&y, x), |d, (_, xl)| d * fr_sub(x, xl) % &r);
            product = product * den % &r;
        }
    }
    proof.inv = fr(&fr_inv(&product));
    proof
}

/// The position of the piece `Q<i>` among the proof's evaluations, if `Q` is split: after the
/// evMap's, in the order of the layout (pilfflonk/docs/formats.md#proof).
pub fn piece_position(vkey: &Vkey, piece: u64) -> Option<usize> {
    let q_stage = vkey.layout.0.last()?.stage;
    let names: Vec<&str> = vkey
        .layout
        .0
        .iter()
        .filter(|f| f.stage == q_stage)
        .flat_map(|f| f.pols.iter().map(|p| p.name.as_str()))
        .collect();
    if names.len() < 2 {
        return None;
    }
    let name = format!("Q{piece}");
    names.iter().position(|n| *n == name).map(|at| vkey.ev_map.len() + at)
}

/// Adds to `proof`'s split `Q` a multiple of `ξ^(M·N)·Q_1 − Q_0` that leaves `Σ_i ξ^(i·M·N)·Q_i(ξ)`
/// as it is: `Q_0 += ξ^(M·N)·d`, `Q_1 −= d` (pilfflonk/docs/protocol.md#q-pieces). `checkQPieces`
/// passes, and the pairing refuses it.
pub fn rebalance_pieces(vkey: &Vkey, proof: &mut Proof, xi_seed: &BigUint, d: u64) {
    let (Some(q0), Some(q1)) = (piece_position(vkey, 0), piece_position(vkey, 1)) else { return };
    let shift = xi_of(vkey, xi_seed).modpow(&BigUint::from(vkey.max_q_degree << vkey.power), &r());
    proof.evaluations[q0] = fr(&(big(&proof.evaluations[q0]) + shift * d));
    proof.evaluations[q1] = fr(&fr_sub(&big(&proof.evaluations[q1]), &BigUint::from(d)));
}
