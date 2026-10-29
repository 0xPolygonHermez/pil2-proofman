//! The boundary zerofiers `Zi_b(ξ)` the verifier passes to `q_verify`.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use proofman_fields::{CubicExtensionField, Field, Goldilocks};

use crate::verifier::Boundary;

type Ext = CubicExtensionField<Goldilocks>;

/// Why the zerofiers cannot be evaluated; `Invalid` carries C++ `validateBoundaries`'s message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZiError {
    Invalid(String),
    /// `Zh(ξ) = 0`, which also covers every `V_O(ξ) = 0` since each `V_O` divides `Zh`.
    ZhIsZero,
}

impl fmt::Display for ZiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZiError::Invalid(m) => f.write_str(m),
            ZiError::ZhIsZero => write!(f, "Zh(xi) is zero"),
        }
    }
}

/// Boundary `index` as a domain: a cycle within N, sorted distinct offsets inside it.
fn validate_domain(index: usize, b: &Boundary, n_bits: u32) -> Result<(), ZiError> {
    let err = |m: String| Err(ZiError::Invalid(format!("boundary {index}: {m}")));
    if b.name != "domain" {
        return err(format!("unknown boundary {}", b.name));
    }
    if b.cycle_bits > n_bits as u64 {
        return err(format!("cycle 2^{} exceeds N=2^{n_bits}", b.cycle_bits));
    }
    if b.offsets.is_empty() {
        return err(String::from("a domain without offsets"));
    }
    for (k, &offset) in b.offsets.iter().enumerate() {
        if offset >= 1 << b.cycle_bits {
            return err(format!("offset {offset} is outside the cycle"));
        }
        // a repeated offset squares a factor of V_O, and Zh/V_O is no longer a polynomial
        if k > 0 && offset <= b.offsets[k - 1] {
            return err(String::from("offsets must be strictly increasing"));
        }
    }
    Ok(())
}

/// C++ `validateBoundaries`: everyRow first, then domains.
pub fn validate_boundaries(boundaries: &[Boundary], n_bits: u32) -> Result<(), ZiError> {
    if boundaries.first().is_none_or(|b| b.name != "everyRow") {
        return Err(ZiError::Invalid(String::from("boundary 0 must be everyRow")));
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

/// What `q_verify` and the quotient check read at ξ.
pub struct Zerofiers {
    /// `ξ^N`.
    pub x_n: Ext,
    /// `Zi_b(ξ)` per boundary.
    pub zi: Vec<Ext>,
}

/// `Zi_b(ξ)`: `Zh⁻¹` (everyRow), `Zh/V_O` (domain), `V_O` (complement); `V_O(X) = ∏ (X^(N/2^c) − w(c)^o)`.
pub fn zi_vector(xi: Ext, n_bits: u32, boundaries: &[Boundary]) -> Result<Zerofiers, ZiError> {
    validate_boundaries(boundaries, n_bits)?;
    let pows = pow2_chain(xi, n_bits);
    let zh = pows[n_bits as usize] - Goldilocks::ONE;
    if zh.is_zero() {
        return Err(ZiError::ZhIsZero);
    }
    let zi = boundaries.iter().map(|b| zi_from_chain(&pows, n_bits, zh, b)).collect();
    Ok(Zerofiers { x_n: pows[n_bits as usize], zi })
}

/// `(ξ^N, Zi)`, or `None`, logged, when they cannot be evaluated.
pub(crate) fn zerofiers(xi: Ext, n_bits: u64, boundaries: &[Boundary]) -> Option<(Ext, Vec<Ext>)> {
    match zi_vector(xi, n_bits as u32, boundaries) {
        Ok(z) => Some((z.x_n, z.zi)),
        Err(_e) => {
            v_error!("Zerofiers: {}", _e);
            None
        }
    }
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
    fn vector_boundaries() -> Vec<Boundary> {
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
        assert_eq!(v.boundaries.len(), vector_boundaries().len());
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
    fn a_domain_without_its_rows_is_refused() {
        let ok: Vec<Boundary> = serde_json::from_str(r#"[{"name":"everyRow"},{"name":"firstRow"}]"#).unwrap();
        assert_eq!(ok[1].name, "firstRow");
        for bad in [
            r#"{"name":"domain","offsets":[0],"complement":false}"#,
            r#"{"name":"domain","cycleBits":1,"offsets":[0]}"#,
        ] {
            assert!(serde_json::from_str::<Boundary>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn domain_rejects_a_point_on_a_domain() {
        // g^3 is a root of ODD's V_O
        let g3 = ext([Goldilocks::new(Goldilocks::W[N_BITS as usize]).exp_u64(3).as_canonical_u64(), 0, 0]);
        assert_eq!(zi_vector(g3, N_BITS, &vector_boundaries()).err(), Some(ZiError::ZhIsZero));
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
        assert_eq!(msg(&[er.clone(), frame]).unwrap(), "boundary 1: unknown boundary everyFrame");
        assert_eq!(validate_boundaries(&vector_boundaries(), N_BITS), Ok(()));
    }
}
