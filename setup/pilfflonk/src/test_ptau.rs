//! For tests only (feature `test-ptau`): powers-of-tau files written here, instead of downloading
//! one or running snarkjs (pilfflonk/docs/README.md#tests). Each holds only what pilfflonk reads,
//! sections 1 to 3, as the C++ test ptau does
//! (`pil2-stark/test/pilfflonk/pilfflonk_test_ptau.hpp`).
//!
//! - [`tau_one_ptau`]: `τ = 1`. Every power `[τ^i]₁` is the generator `G` of G1, and `[τ]₂` the
//!   generator of G2, so the commitment of a fixed `f = Σ_j p_j(X^k)·X^j` is `f(1)·G = (Σ_j
//!   p_j(1))·G`, and `p_j(1)` is the first row of column `j` (the domain starts at `ω^0 = 1`): a
//!   test knows the points in advance, with no curve arithmetic. But it is no SRS to prove with:
//!   the blinding `(X^N − 1)·b(X)` vanishes at 1, so every commitment is the same whatever the
//!   blinding, and anyone can open anything (`forgeProof` of the JS tests does).
//! - [`fixed_tau_ptau`]: a full-width `τ`, [`TEST_TAU`] by default, the `τ` of the C++ helper
//!   (`testTau()`), whose points this port computes with its own arithmetic over `num-bigint` (the
//!   C++ one is ffiasm's) and which the tests pin against the C++'s. A prover's proof of it
//!   verifies only if it is sound, as with a real ptau, and the blinding changes the commitments.

use std::path::Path;

use num_bigint::BigUint;
use proofman_pilfflonk::{FqBytes, G1Affine, G2Affine, BN254_Q, BN254_R};

/// BN254's base field modulus `q`, little-endian, in hex: byte by byte as it is stored.
const Q_LE: &str = "47fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430";

/// The generator of G1, `(1, 2)`, in Montgomery form (`c·2^256 mod q`), little-endian: as a
/// snarkjs ptau stores it. The same constants as `provers/starks-lib-c/tests/pilfflonk_srs.rs`;
/// the C++ reader checks that `[1]₁` is the generator, so a wrong digit fails every test.
const G1_MONTGOMERY_LE: [&str; 2] = [
    "9d0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    "3a1b1e8b1b87baa67b168eeb51d6f114588cf2f0de46ddcc5ebe0f3483ef141c",
];

/// The generator of G2, `x.c0, x.c1, y.c0, y.c1`, in the same form.
const G2_MONTGOMERY_LE: [&str; 4] = [
    "2620bc02d1b5838e72017b493519ebdcdf1a81974726b8fb3b5096af41385719",
    "40614ca87d73b4afc4d802585add4360862fa052fc50e9096b7bea3a83f0fe14",
    "f6e96b889dfa9d61789b9ef597d27ffefe7d1b23621a9eff06429eaeeb7efd28",
    "ee5618c7565b0964bb3c7d3222f957dc76103533be35f9558264fd93e6a0a40d",
];

/// The bytes 64 hex digits spell, in the order they are written.
fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).filter_map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok()).collect()
}

/// A binfile as snarkjs and rapidsnark write them: type, version, then each section as (id, size,
/// bytes).
fn binfile(file_type: &[u8; 4], sections: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = file_type.to_vec();
    out.extend(1u32.to_le_bytes());
    out.extend((sections.len() as u32).to_le_bytes());
    for (id, contents) in sections {
        out.extend(id.to_le_bytes());
        out.extend((contents.len() as u64).to_le_bytes());
        out.extend(contents);
    }
    out
}

/// A ptau with `τ = 1` and `n_g1` powers `[τ^i]₁`: `n_g1` copies of G1's generator and two of
/// G2's. Its power is the smallest a snarkjs ptau of that many powers has, `2^(power+1) − 1 ≥
/// n_g1`, as the C++ test ptau's (the reader relies on the sizes of the sections only).
pub fn tau_one_ptau(n_g1: usize) -> Vec<u8> {
    let power = (1..usize::BITS - 1).find(|&p| (1usize << (p + 1)) > n_g1).unwrap_or(usize::BITS - 1);
    let mut header = 32u32.to_le_bytes().to_vec();
    header.extend(bytes(Q_LE));
    header.extend(power.to_le_bytes());
    header.extend(power.to_le_bytes());
    let g1: Vec<u8> = G1_MONTGOMERY_LE.iter().flat_map(|c| bytes(c)).collect();
    let g2: Vec<u8> = G2_MONTGOMERY_LE.iter().flat_map(|c| bytes(c)).collect();
    binfile(b"ptau", &[(1, header), (2, g1.repeat(n_g1)), (3, g2.repeat(2))])
}

/// Writes [`tau_one_ptau`] at `path`.
pub fn write_tau_one_ptau(path: &Path, n_g1: usize) -> std::io::Result<()> {
    std::fs::write(path, tau_one_ptau(n_g1))
}

// ---------------------------------------------------------------------------------------------
// A full-width τ
// ---------------------------------------------------------------------------------------------

/// The `τ` of the C++ test ptau (`testTau()` of `pil2-stark/test/pilfflonk/pilfflonk_test_ptau.cpp`),
/// a 253-bit scalar, in hexadecimal.
pub const TEST_TAU: &str = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809";

/// The generator of G2 in canonical form, `x.c0, x.c1, y.c0, y.c1` (the `X_2` of a `τ = 1` vkey).
const G2_GENERATOR: [&str; 4] = [
    "10857046999023057135944570762232829481370756359578518086990519993285655852781",
    "11559732032986387107991004021392285783925812861821192530917403151452391805634",
    "8495653923123431417604973247489272438418190587263600148770280649306958101930",
    "4082367875863433681332203403145435568316851327593401208105741076214120093531",
];

fn decimal(s: &str) -> BigUint {
    BigUint::parse_bytes(s.as_bytes(), 10).unwrap_or_default()
}

/// The arithmetic of a field the curve points of this module have coordinates in.
trait Field: Clone + PartialEq {
    fn zero(q: &BigUint) -> Self;
    fn one(q: &BigUint) -> Self;
    fn add(&self, o: &Self, q: &BigUint) -> Self;
    fn sub(&self, o: &Self, q: &BigUint) -> Self;
    fn mul(&self, o: &Self, q: &BigUint) -> Self;
    fn inv(&self, q: &BigUint) -> Self;
    fn is_zero(&self) -> bool;
}

/// Fq: a value below q.
#[derive(Clone, PartialEq)]
struct Fq(BigUint);

impl Field for Fq {
    fn zero(_: &BigUint) -> Self {
        Fq(BigUint::ZERO)
    }
    fn one(_: &BigUint) -> Self {
        Fq(BigUint::from(1u32))
    }
    fn add(&self, o: &Self, q: &BigUint) -> Self {
        Fq((&self.0 + &o.0) % q)
    }
    fn sub(&self, o: &Self, q: &BigUint) -> Self {
        Fq((&self.0 + q - &o.0) % q)
    }
    fn mul(&self, o: &Self, q: &BigUint) -> Self {
        Fq((&self.0 * &o.0) % q)
    }
    fn inv(&self, q: &BigUint) -> Self {
        Fq(self.0.modpow(&(q - 2u32), q))
    }
    fn is_zero(&self) -> bool {
        self.0 == BigUint::ZERO
    }
}

/// Fq2 = Fq[u]/(u² + 1): `c0 + c1·u`.
#[derive(Clone, PartialEq)]
struct Fq2(Fq, Fq);

impl Field for Fq2 {
    fn zero(q: &BigUint) -> Self {
        Fq2(Fq::zero(q), Fq::zero(q))
    }
    fn one(q: &BigUint) -> Self {
        Fq2(Fq::one(q), Fq::zero(q))
    }
    fn add(&self, o: &Self, q: &BigUint) -> Self {
        Fq2(self.0.add(&o.0, q), self.1.add(&o.1, q))
    }
    fn sub(&self, o: &Self, q: &BigUint) -> Self {
        Fq2(self.0.sub(&o.0, q), self.1.sub(&o.1, q))
    }
    fn mul(&self, o: &Self, q: &BigUint) -> Self {
        // (a0 + a1·u)(b0 + b1·u) = a0·b0 − a1·b1 + (a0·b1 + a1·b0)·u
        Fq2(self.0.mul(&o.0, q).sub(&self.1.mul(&o.1, q), q), self.0.mul(&o.1, q).add(&self.1.mul(&o.0, q), q))
    }
    fn inv(&self, q: &BigUint) -> Self {
        // 1/(a0 + a1·u) = (a0 − a1·u)/(a0² + a1²)
        let norm = self.0.mul(&self.0, q).add(&self.1.mul(&self.1, q), q).inv(q);
        Fq2(self.0.mul(&norm, q), Fq::zero(q).sub(&self.1, q).mul(&norm, q))
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
}

/// `k·P` on a curve `y² = x³ + b` for `P = (x, y)` affine and not the point at infinity, by
/// double-and-add in Jacobian coordinates (`dbl-2009-l` and `madd-2007-bl`, which do not use b).
/// `None` for the point at infinity.
fn scalar_mul<F: Field>(p: &(F, F), k: &BigUint, q: &BigUint) -> Option<(F, F)> {
    let two = |a: &F| a.add(a, q);
    let mut acc: Option<(F, F, F)> = None;
    for bit in (0..k.bits()).rev() {
        if let Some((x, y, z)) = acc.take() {
            // Doubling.
            let a = x.mul(&x, q);
            let b = y.mul(&y, q);
            let c = b.mul(&b, q);
            let xb = x.add(&b, q);
            let d = two(&xb.mul(&xb, q).sub(&a, q).sub(&c, q));
            let e = two(&a).add(&a, q);
            let f = e.mul(&e, q);
            let x3 = f.sub(&two(&d), q);
            let y3 = e.mul(&d.sub(&x3, q), q).sub(&two(&two(&two(&c))), q);
            let z3 = two(&y.mul(&z, q));
            acc = if z3.is_zero() { None } else { Some((x3, y3, z3)) };
        }
        if k.bit(bit) {
            acc = match acc {
                None => Some((p.0.clone(), p.1.clone(), F::one(q))),
                Some((x1, y1, z1)) => {
                    // Mixed addition of the affine P.
                    let z1z1 = z1.mul(&z1, q);
                    let u2 = p.0.mul(&z1z1, q);
                    let s2 = p.1.mul(&z1, q).mul(&z1z1, q);
                    let h = u2.sub(&x1, q);
                    let r = two(&s2.sub(&y1, q));
                    if h.is_zero() {
                        // P itself (doubling) or −P (infinity); neither happens for a scalar below the
                        // order of the group, but they are handled.
                        if r.is_zero() {
                            let (x, y) = scalar_mul(p, &BigUint::from(2u32), q)?;
                            Some((x, y, F::one(q)))
                        } else {
                            None
                        }
                    } else {
                        let hh = h.mul(&h, q);
                        let i = two(&two(&hh));
                        let j = h.mul(&i, q);
                        let v = x1.mul(&i, q);
                        let x3 = r.mul(&r, q).sub(&j, q).sub(&two(&v), q);
                        let y3 = r.mul(&v.sub(&x3, q), q).sub(&two(&y1.mul(&j, q)), q);
                        let z1h = z1.add(&h, q);
                        let z3 = z1h.mul(&z1h, q).sub(&z1z1, q).sub(&hh, q);
                        Some((x3, y3, z3))
                    }
                }
            };
        }
    }
    let (x, y, z) = acc?;
    let zi = z.inv(q);
    let zi2 = zi.mul(&zi, q);
    Some((x.mul(&zi2, q), y.mul(&zi2, q).mul(&zi, q)))
}

/// A coordinate as a ptau stores it: Montgomery form, `c·2^256 mod q`, 32 bytes little-endian.
fn montgomery_le(c: &Fq, q: &BigUint) -> Vec<u8> {
    let m = (&c.0 << 256u32) % q;
    let mut bytes = m.to_bytes_le();
    bytes.resize(32, 0);
    bytes
}

/// `[s]₁` for each scalar, in parallel: a scalar multiplication each.
fn g1_multiples(scalars: &[BigUint], q: &BigUint) -> Vec<Vec<u8>> {
    let g = (Fq(BigUint::from(1u32)), Fq(BigUint::from(2u32)));
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(scalars.len().max(1));
    let chunk = scalars.len().div_ceil(threads).max(1);
    let point = |s: &BigUint| match scalar_mul(&g, s, q) {
        Some((x, y)) => [montgomery_le(&x, q), montgomery_le(&y, q)].concat(),
        // [0]₁: (0, 0), which the SRS reader refuses but for [1]₁ is never needed.
        None => vec![0; 64],
    };
    std::thread::scope(|scope| {
        let handles: Vec<_> =
            scalars.chunks(chunk).map(|part| scope.spawn(move || part.iter().map(point).collect::<Vec<_>>())).collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

/// A ptau with the full-width `τ = tau` (below `r`) and `n_g1` powers `[τ^i]₁`, and `[1]₂`, `[τ]₂`:
/// the bytes the C++ helper's `writeTestPtau` writes for the same `τ` and `nG1`, with the same power
/// in the header as [`tau_one_ptau`] (see [the module](self)). Scalar multiplications over
/// `num-bigint`: a few seconds for a few hundred powers in a debug build.
pub fn fixed_tau_ptau(n_g1: usize, tau: &BigUint) -> Vec<u8> {
    let q = decimal(BN254_Q);
    let r = decimal(BN254_R);
    let tau = tau % &r;
    let power = (1..usize::BITS - 1).find(|&p| (1usize << (p + 1)) > n_g1).unwrap_or(usize::BITS - 1);
    let mut header = 32u32.to_le_bytes().to_vec();
    header.extend(bytes(Q_LE));
    header.extend(power.to_le_bytes());
    header.extend(power.to_le_bytes());

    let mut powers = Vec::with_capacity(n_g1);
    let mut t = BigUint::from(1u32);
    for _ in 0..n_g1 {
        powers.push(t.clone());
        t = (&t * &tau) % &r;
    }
    let g1: Vec<u8> = g1_multiples(&powers, &q).concat();

    let [xc0, xc1, yc0, yc1] = G2_GENERATOR.map(|c| Fq(decimal(c)));
    let generator = (Fq2(xc0, xc1), Fq2(yc0, yc1));
    let g2_point = |p: &(Fq2, Fq2)| {
        [montgomery_le(&p.0 .0, &q), montgomery_le(&p.0 .1, &q), montgomery_le(&p.1 .0, &q), montgomery_le(&p.1 .1, &q)]
            .concat()
    };
    let mut g2 = g2_point(&generator);
    match scalar_mul(&generator, &tau, &q) {
        Some(p) => g2.extend(g2_point(&p)),
        None => g2.extend([0u8; 128]),
    }
    binfile(b"ptau", &[(1, header), (2, g1), (3, g2)])
}

/// The τ of [`TEST_TAU`].
pub fn test_tau() -> BigUint {
    BigUint::parse_bytes(TEST_TAU.as_bytes(), 16).unwrap_or_default()
}

/// Writes [`fixed_tau_ptau`] at `path`.
pub fn write_fixed_tau_ptau(path: &Path, n_g1: usize, tau: &BigUint) -> std::io::Result<()> {
    std::fs::write(path, fixed_tau_ptau(n_g1, tau))
}

/// `s·G`, `G = (1, 2)` the generator of G1, with this module's arithmetic: for a test that builds a
/// proof by hand, knowing `τ`. The point at infinity, `(0, 0)`, for `s ≡ 0 mod r`.
pub fn g1_times(s: &BigUint) -> G1Affine {
    let q = decimal(BN254_Q);
    let g = (Fq(BigUint::from(1u32)), Fq(BigUint::from(2u32)));
    let fq = |c: &Fq| FqBytes::from_decimal(&c.0.to_str_radix(10)).unwrap_or_default();
    match scalar_mul(&g, &(s % decimal(BN254_R)), &q) {
        Some((x, y)) => G1Affine { x: fq(&x), y: fq(&y) },
        None => G1Affine::INFINITY,
    }
}

/// `s·[1]₂`, as [`g1_times`]: the `X_2 = [τ]₂` of a vkey of `τ = s`. All zeros for `s ≡ 0 mod r`.
pub fn g2_times(s: &BigUint) -> G2Affine {
    let q = decimal(BN254_Q);
    let [xc0, xc1, yc0, yc1] = G2_GENERATOR.map(|c| Fq(decimal(c)));
    let fq = |c: &Fq| FqBytes::from_decimal(&c.0.to_str_radix(10)).unwrap_or_default();
    match scalar_mul(&(Fq2(xc0, xc1), Fq2(yc0, yc1)), &(s % decimal(BN254_R)), &q) {
        Some((x, y)) => G2Affine { x: [fq(&x.0), fq(&x.1)], y: [fq(&y.0), fq(&y.1)] },
        None => G2Affine::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The points of the C++ helper's ptau for `τ = testTau()`, as it writes them (Montgomery,
    /// little-endian), for `[τ]₁` (section 2, point 1) and `[τ]₂` (section 3, point 1): computed
    /// with ffiasm by `writeTestPtau`, independently of this module's arithmetic.
    const CPP_TAU_G1: &str = concat!(
        "c34dd15691ad86dc7badd0794ebc7370cc8dfff25c2fe30acec55179628c961e",
        "0ce9ab226fcc5b0cac334a33ad6dc137b77064180ac0a4d563583e8ef853590d",
    );
    const CPP_TAU_G2: &str = concat!(
        "fd63e7690f03cebf929ce48003ca1db9211dbfbfdbfad4f1b49c4cd3f7112305",
        "802430f19151b130ade5273d82eeed930a9768a68e3f66aff566f57a7e8b321f",
        "765a9da48dd3d3cc75f7f473c9968b08aa5ccc3f2ad3f4180d49cb7130718518",
        "93285c64523a78de08bc85be8cf58553178eaaaed29c0a2b53c300f30d3cf406",
    );

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Section `id` of a binfile of this module.
    fn section(file: &[u8], id: u32) -> &[u8] {
        let mut at = 12;
        while at < file.len() {
            let sid = u32::from_le_bytes(file[at..at + 4].try_into().unwrap());
            let size = u64::from_le_bytes(file[at + 4..at + 12].try_into().unwrap()) as usize;
            if sid == id {
                return &file[at + 12..at + 12 + size];
            }
            at += 12 + size;
        }
        panic!("no section {id}");
    }

    #[test]
    fn the_multiples_are_the_ptaus_points() {
        let tau = test_tau();
        let file = fixed_tau_ptau(2, &tau);
        let q = decimal(BN254_Q);
        // The ptau's [τ]₁ and [τ]₂, out of Montgomery form (c·R⁻¹ mod q, R = 2^256).
        let r_inv = (BigUint::from(1u32) << 256u32).modpow(&(&q - 2u32), &q);
        let canonical = |le: &[u8]| (BigUint::from_bytes_le(le) * &r_inv % &q).to_str_radix(10);
        let (g1, g2) = (section(&file, 2), section(&file, 3));
        let p = g1_times(&tau);
        assert_eq!((p.x.to_decimal(), p.y.to_decimal()), (canonical(&g1[64..96]), canonical(&g1[96..128])));
        let x2 = g2_times(&tau);
        let coordinates = [x2.x[0], x2.x[1], x2.y[0], x2.y[1]].map(|c| c.to_decimal());
        let expected: Vec<String> = (0..4).map(|c| canonical(&g2[128 + 32 * c..160 + 32 * c])).collect();
        assert_eq!(coordinates.to_vec(), expected);
        assert!(g1_times(&BigUint::ZERO).is_infinity());
        assert_eq!(g2_times(&decimal(BN254_R)), G2Affine::default());
    }

    #[test]
    fn tau_one_is_the_ptau_of_tau_one() {
        for n in [1, 2, 3, 8, 17] {
            assert_eq!(fixed_tau_ptau(n, &BigUint::from(1u32)), tau_one_ptau(n), "n = {n}");
        }
    }

    #[test]
    fn the_test_tau_gives_the_cpp_helpers_points() {
        let file = fixed_tau_ptau(4, &test_tau());
        let g1 = section(&file, 2);
        let g2 = section(&file, 3);
        assert_eq!((g1.len(), g2.len()), (4 * 64, 2 * 128));
        assert_eq!(hex(&g1[..64]), G1_MONTGOMERY_LE.concat(), "[1]₁");
        assert_eq!(hex(&g2[..128]), G2_MONTGOMERY_LE.concat(), "[1]₂");
        assert_eq!(hex(&g1[64..128]), CPP_TAU_G1, "[τ]₁");
        assert_eq!(hex(&g2[128..]), CPP_TAU_G2, "[τ]₂");
    }
}
