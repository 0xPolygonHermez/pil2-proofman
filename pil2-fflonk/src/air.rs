//! Reconstructing the quotient evaluation the opening scheme omits.
//!
//! The AIR's constraints hold on the trace domain exactly when the combined
//! constraint expression vanishes there, which is to say when it is divisible
//! by the domain's vanishing polynomial:
//!
//! ```text
//! cExp(X) = Q(X) · Z_H(X),    Z_H(X) = X^N - 1
//! ```
//!
//! So `Q(xi) = cExp(xi) / Z_H(xi)`. The prover sends the inverse rather than
//! the quotient, and the verifier recovers the value it needs with one
//! multiplication instead of a field inversion.
//!
//! That inverse is a prover-supplied hint, so it is checked here rather than
//! trusted: a prover free to choose `invZh` could scale the quotient to
//! whatever value made its opening consistent, which is the whole constraint
//! system. Checking costs one multiplication and one exponentiation.
//!
//! This is the one input [`crate::linearisation`] cannot derive on its own,
//! and it is why that module takes reconstructed evaluations as a parameter.

use anyhow::{Result, bail};
use num_bigint::BigUint;
use num_traits::One;

use crate::fr;
use crate::proof::{INV_ZH_KEY, ShPlonkProof};

/// `Z_H(x) = x^N - 1` for a domain of `2^n_bits` rows.
pub fn zh_at(x: &BigUint, n_bits: u32) -> BigUint {
    fr::sub(&x.modpow(&(BigUint::one() << n_bits), &crate::transcript::fr_modulus()), &BigUint::one())
}

/// Check the prover's claimed `1 / Z_H(xi)`.
///
/// A zero `Z_H(xi)` would mean the challenge landed on the trace domain, where
/// the quotient is not defined. That is negligibly unlikely with an honest
/// transcript, so meeting it means something is wrong rather than unlucky.
pub fn check_inv_zh(inv_zh: &BigUint, xi: &BigUint, n_bits: u32) -> Result<()> {
    let zh = zh_at(xi, n_bits);
    if zh == BigUint::from(0u32) {
        bail!("xi is a root of Z_H: the challenge landed on the trace domain");
    }
    if fr::mul(inv_zh, &zh) != BigUint::one() {
        bail!("the proof's invZh is not the inverse of Z_H(xi)");
    }
    Ok(())
}

/// `Q(xi) = cExp(xi) · invZh`, after checking the hint.
pub fn quotient_at(c_exp: &BigUint, inv_zh: &BigUint, xi: &BigUint, n_bits: u32) -> Result<BigUint> {
    check_inv_zh(inv_zh, xi, n_bits)?;
    Ok(fr::mul(c_exp, inv_zh))
}

/// Read `invZh` from a proof.
pub fn inv_zh(proof: &ShPlonkProof) -> Result<BigUint> {
    let raw = match proof.evaluations.get(INV_ZH_KEY) {
        Some(v) => v,
        None => bail!("proof is missing the {INV_ZH_KEY} evaluation"),
    };
    fr::from_decimal(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;
    use crate::verifier_code::{Inputs, VerifierCode};

    const INFO: &str = include_str!("../tests/fixtures/pilfflonk.verifierinfo.json");

    fn n_bits() -> u32 {
        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        info["pilPower"].as_u64().unwrap() as u32
    }

    /// The whole reconstruction, end to end, against the quotient the prover
    /// actually committed to. This is the last input the opening scheme needed
    /// that it could not derive for itself.
    #[test]
    fn reconstructs_the_provers_quotient_evaluation() {
        let r = reference::load();
        let info: serde_json::Value = serde_json::from_str(INFO).unwrap();
        let code = VerifierCode::from_json(&info).unwrap();

        let evals = code.evaluations(&r.setup.pols_map, &r.proof).unwrap();
        let c_exp = code
            .evaluate(&Inputs { evals: &evals, challenges: &r.air_challenges, publics: &r.publics, x: &r.xi })
            .unwrap();

        let got = quotient_at(&c_exp, &inv_zh(&r.proof).unwrap(), &r.xi, n_bits()).unwrap();
        assert_eq!(got, r.quotient_evaluation);
    }

    /// The prover's invZh really is the inverse of the vanishing polynomial at
    /// xi, for the domain the key declares.
    #[test]
    fn the_proofs_inv_zh_is_the_true_inverse() {
        let r = reference::load();
        assert_eq!(n_bits(), 8, "the reference AIR has 256 rows");
        check_inv_zh(&inv_zh(&r.proof).unwrap(), &r.xi, n_bits()).unwrap();
    }

    /// A forged inverse is rejected. Without this check a prover could pick
    /// invZh to make any constraint value produce whatever quotient its
    /// opening claimed.
    #[test]
    fn rejects_a_forged_inverse() {
        let r = reference::load();
        let honest = inv_zh(&r.proof).unwrap();

        for forged in [BigUint::from(1u32), fr::add(&honest, &BigUint::one()), BigUint::from(0u32)] {
            assert!(check_inv_zh(&forged, &r.xi, n_bits()).is_err());
            assert!(quotient_at(&BigUint::from(7u32), &forged, &r.xi, n_bits()).is_err());
        }
    }

    /// The domain size is part of the statement: the same proof read against a
    /// different N gives a different Z_H, and the hint no longer checks out.
    #[test]
    fn the_domain_size_is_bound_by_the_check() {
        let r = reference::load();
        let honest = inv_zh(&r.proof).unwrap();
        assert!(check_inv_zh(&honest, &r.xi, n_bits() + 1).is_err());
    }

    #[test]
    fn zh_vanishes_on_the_trace_domain() {
        let r = reference::load();
        let w = fr::from_decimal(&r.setup.omegas["w"]).unwrap();

        // w generates the trace domain, so every power of it is a root.
        for k in [0u64, 1, 2, 37, 255] {
            assert_eq!(zh_at(&fr::pow(&w, k), 8), BigUint::from(0u32), "w^{k}");
        }
        // And the challenge is not one of them.
        assert_ne!(zh_at(&r.xi, 8), BigUint::from(0u32));
    }

    /// A challenge on the domain is refused rather than divided by zero.
    #[test]
    fn a_challenge_on_the_domain_is_rejected() {
        let r = reference::load();
        let w = fr::from_decimal(&r.setup.omegas["w"]).unwrap();
        let err = check_inv_zh(&BigUint::one(), &w, 8).unwrap_err().to_string();
        assert!(err.contains("root of Z_H"), "{err}");
    }

    #[test]
    fn reports_a_proof_without_the_hint() {
        let r = reference::load();
        let mut proof = r.proof.clone();
        proof.evaluations.remove(INV_ZH_KEY);
        assert!(inv_zh(&proof).is_err());
    }
}
