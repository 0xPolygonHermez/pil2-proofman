//! The boundary zerofiers `Zi_b(ξ)` the verifier passes to `q_verify`.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use proofman_fields::{CubicExtensionField, Field, Goldilocks};

use crate::verifier::Boundary;

type Ext = CubicExtensionField<Goldilocks>;

/// Why the zerofiers cannot be evaluated; `Display` gives C++ `validateBoundaries`'s messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZiError {
    NotEveryRowFirst,
    Unknown {
        index: usize,
        name: String,
    },
    CycleTooLong {
        index: usize,
        cycle_bits: u64,
        n_bits: u32,
    },
    NoOffsets {
        index: usize,
    },
    OffsetOutsideCycle {
        index: usize,
        offset: u64,
    },
    OffsetsNotIncreasing {
        index: usize,
    },
    /// `Zh(ξ) = 0`, which also covers every `V_O(ξ) = 0` since each `V_O` divides `Zh`.
    ZhIsZero,
}

impl fmt::Display for ZiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZiError::NotEveryRowFirst => write!(f, "boundary 0 must be everyRow"),
            ZiError::Unknown { index, name } => write!(f, "boundary {index}: unknown boundary {name}"),
            ZiError::CycleTooLong { index, cycle_bits, n_bits } => {
                write!(f, "boundary {index}: cycle 2^{cycle_bits} exceeds N=2^{n_bits}")
            }
            ZiError::NoOffsets { index } => write!(f, "boundary {index}: a domain without offsets"),
            ZiError::OffsetOutsideCycle { index, offset } => {
                write!(f, "boundary {index}: offset {offset} is outside the cycle")
            }
            ZiError::OffsetsNotIncreasing { index } => {
                write!(f, "boundary {index}: offsets must be strictly increasing")
            }
            ZiError::ZhIsZero => write!(f, "Zh(xi) is zero"),
        }
    }
}

/// Boundary `index` as a domain: a cycle within N, sorted distinct offsets inside it.
fn validate_domain(index: usize, b: &Boundary, n_bits: u32) -> Result<(), ZiError> {
    if b.name != "domain" {
        return Err(ZiError::Unknown { index, name: b.name.clone() });
    }
    if b.cycle_bits > n_bits as u64 {
        return Err(ZiError::CycleTooLong { index, cycle_bits: b.cycle_bits, n_bits });
    }
    if b.offsets.is_empty() {
        return Err(ZiError::NoOffsets { index });
    }
    for (k, &offset) in b.offsets.iter().enumerate() {
        if offset >= 1 << b.cycle_bits {
            return Err(ZiError::OffsetOutsideCycle { index, offset });
        }
        // a repeated offset squares a factor of V_O, and Zh/V_O is no longer a polynomial
        if k > 0 && offset <= b.offsets[k - 1] {
            return Err(ZiError::OffsetsNotIncreasing { index });
        }
    }
    Ok(())
}

/// C++ `validateBoundaries`: everyRow first, then domains.
pub fn validate_boundaries(boundaries: &[Boundary], n_bits: u32) -> Result<(), ZiError> {
    if boundaries.first().is_none_or(|b| b.name != "everyRow") {
        return Err(ZiError::NotEveryRowFirst);
    }
    boundaries.iter().enumerate().skip(1).try_for_each(|(i, b)| validate_domain(i, b, n_bits))
}

/// `ξ^(2^j)` for `j = 0..=n_bits`.
fn pow2_chain(xi: Ext, n_bits: u32) -> Vec<Ext> {
    let mut pows = Vec::with_capacity(n_bits as usize + 1);
    pows.push(xi);
    for j in 0..n_bits as usize {
        pows.push(pows[j].square());
    }
    pows
}

/// `Zi_b(ξ)` of a validated boundary, given the chain and `Zh(ξ) ≠ 0`.
fn zi_from_chain(pows: &[Ext], n_bits: u32, zh: Ext, b: &Boundary) -> Ext {
    if b.name == "everyRow" {
        return zh.inverse();
    }
    let c = b.cycle_bits as usize;
    let xi_m = pows[n_bits as usize - c];
    let w = Goldilocks::new(Goldilocks::W[c]);
    let v = b.offsets.iter().fold(Ext::one(), |v, &o| v * (xi_m - w.exp_u64(o)));
    if b.complement {
        v
    } else {
        zh * v.inverse()
    }
}

/// `ξ^N` and `Zh(ξ)`, rejecting `Zh(ξ) = 0`.
fn chain_and_zh(xi: Ext, n_bits: u32) -> Result<(Vec<Ext>, Ext), ZiError> {
    let pows = pow2_chain(xi, n_bits);
    let zh = pows[n_bits as usize] - Goldilocks::ONE;
    if zh.is_zero() {
        return Err(ZiError::ZhIsZero);
    }
    Ok((pows, zh))
}

/// `Zi_b(ξ)`: `Zh(ξ)⁻¹` for everyRow, `Zh(ξ)/V_O(ξ)` for a domain, `V_O(ξ)` for its complement, with
/// `V_O(X) = ∏_{o∈O} (X^m − w(c)^o)` and `m = N/2^c`.
pub fn domain_zi(xi: Ext, n_bits: u32, b: &Boundary) -> Result<Ext, ZiError> {
    if b.name != "everyRow" {
        validate_domain(0, b, n_bits)?;
    }
    let (pows, zh) = chain_and_zh(xi, n_bits)?;
    Ok(zi_from_chain(&pows, n_bits, zh, b))
}

/// What `q_verify` and the quotient check read at ξ.
pub struct Zerofiers {
    /// `ξ^N`.
    pub x_n: Ext,
    /// `Zi_b(ξ)` per boundary.
    pub zi: Vec<Ext>,
}

/// Every boundary's `Zi_b(ξ)`, from one `ξ^(2^j)` chain.
pub fn zi_vector(xi: Ext, n_bits: u32, boundaries: &[Boundary]) -> Result<Zerofiers, ZiError> {
    validate_boundaries(boundaries, n_bits)?;
    let (pows, zh) = chain_and_zh(xi, n_bits)?;
    let zi = boundaries.iter().map(|b| zi_from_chain(&pows, n_bits, zh, b)).collect();
    Ok(Zerofiers { x_n: pows[n_bits as usize], zi })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use proofman_fields::PrimeField64;
    use serde::Deserialize;

    const N_BITS: u32 = 6;

    fn ext(v: [u64; 3]) -> Ext {
        CubicExtensionField { value: v.map(Goldilocks::new) }
    }

    fn canon(e: Ext) -> [u64; 3] {
        e.value.map(|x| x.as_canonical_u64())
    }

    fn every_row() -> Boundary {
        Boundary { name: "everyRow".to_string(), cycle_bits: 0, offsets: vec![], complement: false }
    }

    fn domain(cycle_bits: u64, offsets: &[u64], complement: bool) -> Boundary {
        Boundary { name: "domain".to_string(), cycle_bits, offsets: offsets.to_vec(), complement }
    }

    /// The C++ gtest's domains: FIRST, LAST, !FIRST, LIMITS_16, ODD, !ODD.
    fn task4_boundaries() -> Vec<Boundary> {
        vec![
            every_row(),
            domain(6, &[0], false),
            domain(6, &[63], false),
            domain(6, &[0], true),
            domain(4, &[0, 15], false),
            domain(1, &[1], false),
            domain(1, &[1], true),
        ]
    }

    fn in_domain(b: &Boundary, row: u64) -> bool {
        if b.name == "everyRow" {
            return true;
        }
        b.offsets.contains(&(row & ((1 << b.cycle_bits) - 1))) != b.complement
    }

    /// The definition: `Zh(ξ) / ∏_{h ∈ H_b} (ξ − g^h)`, `H_b` the rows `b` holds on; `Zh(ξ)⁻¹` for everyRow.
    fn brute_force(xi: Ext, b: &Boundary) -> Ext {
        let n = 1u64 << N_BITS;
        let zh = xi.pow(n) - Goldilocks::ONE;
        if b.name == "everyRow" {
            return zh.inverse();
        }
        let g = Goldilocks::new(Goldilocks::W[N_BITS as usize]);
        let mut den = Ext::one();
        for h in (0..n).filter(|&h| in_domain(b, h)) {
            den *= xi - g.exp_u64(h);
        }
        zh * den.inverse()
    }

    #[test]
    fn domain_matches_brute_force() {
        let xi = ext([0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9, 0x94d049bb133111eb]);
        let boundaries = task4_boundaries();
        let all = zi_vector(xi, N_BITS, &boundaries).expect("ξ is off H");
        assert_eq!(canon(all.x_n), canon(xi.pow(1 << N_BITS)));
        for (i, b) in boundaries.iter().enumerate() {
            let want = canon(brute_force(xi, b));
            assert_eq!(canon(domain_zi(xi, N_BITS, b).unwrap()), want, "boundary {i}");
            assert_eq!(canon(all.zi[i]), want, "boundary {i} in zi_vector");
        }
    }

    #[derive(Deserialize)]
    struct Point {
        xi: [u64; 3],
        #[serde(default)]
        zi: Vec<[u64; 3]>,
        #[serde(default)]
        reject: bool,
    }

    #[derive(Deserialize)]
    struct Vectors {
        #[serde(rename = "nBits")]
        n_bits: u32,
        boundaries: Vec<Boundary>,
        points: Vec<Point>,
    }

    /// `ProverHelpers(starkInfo, z)`'s values, written by pil2-stark's `domainZerofierTest`.
    #[test]
    fn domain_matches_cpp_point() {
        let v: Vectors =
            serde_json::from_str(include_str!("../tests/data/domain_zi_vectors.json")).expect("vector file");
        assert_eq!(v.boundaries.len(), task4_boundaries().len());
        assert!(v.points.iter().any(|p| p.xi[1] != 0 || p.xi[2] != 0), "no point off the base field");
        assert!(v.points.iter().any(|p| p.reject), "no rejected point");
        for p in &v.points {
            let got = zi_vector(ext(p.xi), v.n_bits, &v.boundaries);
            if p.reject {
                assert_eq!(got.err(), Some(ZiError::ZhIsZero), "ξ = {:?} must be rejected", p.xi);
                continue;
            }
            let got: Vec<[u64; 3]> = got.expect("ξ is off H").zi.into_iter().map(canon).collect();
            assert_eq!(got, p.zi, "ξ = {:?}", p.xi);
        }
    }

    #[test]
    fn domain_rejects_a_point_on_a_domain() {
        // g^3 is a root of ODD's V_O
        let g3 = ext([Goldilocks::new(Goldilocks::W[N_BITS as usize]).exp_u64(3).as_canonical_u64(), 0, 0]);
        assert_eq!(zi_vector(g3, N_BITS, &task4_boundaries()).err(), Some(ZiError::ZhIsZero));
        assert_eq!(domain_zi(g3, N_BITS, &domain(1, &[1], false)), Err(ZiError::ZhIsZero));
    }

    /// The messages of the C++ gtest `DomainZerofier.ValidateBoundaries`.
    #[test]
    fn domain_rejects_malformed_boundaries() {
        let xi = ext([7, 11, 13]);
        let msg = |b: &[Boundary]| zi_vector(xi, N_BITS, b).err().map(|e| e.to_string());
        let er = every_row();
        assert_eq!(msg(&[]).unwrap(), "boundary 0 must be everyRow");
        assert_eq!(msg(&[domain(1, &[1], false)]).unwrap(), "boundary 0 must be everyRow");
        assert_eq!(
            msg(&[er.clone(), domain(4, &[3, 3], false)]).unwrap(),
            "boundary 1: offsets must be strictly increasing"
        );
        assert_eq!(
            msg(&[er.clone(), domain(4, &[15, 0], false)]).unwrap(),
            "boundary 1: offsets must be strictly increasing"
        );
        assert_eq!(msg(&[er.clone(), domain(4, &[16], false)]).unwrap(), "boundary 1: offset 16 is outside the cycle");
        assert_eq!(msg(&[er.clone(), domain(7, &[0], false)]).unwrap(), "boundary 1: cycle 2^7 exceeds N=2^6");
        assert_eq!(msg(&[er.clone(), domain(4, &[], false)]).unwrap(), "boundary 1: a domain without offsets");
        let frame = Boundary { name: "everyFrame".to_string(), ..every_row() };
        assert_eq!(msg(&[er.clone(), frame.clone()]).unwrap(), "boundary 1: unknown boundary everyFrame");
        assert_eq!(validate_boundaries(&task4_boundaries(), N_BITS), Ok(()));
        // domain_zi refuses rather than panics
        assert!(domain_zi(xi, N_BITS, &frame).is_err());
        assert!(domain_zi(xi, N_BITS, &domain(7, &[0], false)).is_err());
    }
}
