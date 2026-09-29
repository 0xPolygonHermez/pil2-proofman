//! Polynomials over `Fr` in coefficient form, lowest degree first, and the evaluation of a column
//! off `H` by barycentric interpolation. Quadratic algorithms throughout: the oracle is for
//! fixtures of a few hundred rows, and has to be obviously right rather than fast.

use crate::error::{invalid, PilfflonkResult};

use super::fr::{batch_inverse, Fr};

/// `p` without its zero coefficients of highest degree. The zero polynomial is empty.
pub fn trim(mut p: Vec<Fr>) -> Vec<Fr> {
    while p.last().is_some_and(Fr::is_zero) {
        p.pop();
    }
    p
}

pub fn add(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    let zero = Fr::zero();
    let n = a.len().max(b.len());
    trim((0..n).map(|i| a.get(i).unwrap_or(&zero) + b.get(i).unwrap_or(&zero)).collect())
}

pub fn sub(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    let zero = Fr::zero();
    let n = a.len().max(b.len());
    trim((0..n).map(|i| a.get(i).unwrap_or(&zero) - b.get(i).unwrap_or(&zero)).collect())
}

pub fn neg(a: &[Fr]) -> Vec<Fr> {
    a.iter().map(|c| -c).collect()
}

pub fn scale(a: &[Fr], factor: &Fr) -> Vec<Fr> {
    trim(a.iter().map(|c| c * factor).collect())
}

pub fn mul(a: &[Fr], b: &[Fr]) -> Vec<Fr> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Fr::zero(); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate().filter(|(_, x)| !x.is_zero()) {
        for (j, y) in b.iter().enumerate() {
            out[i + j] = &out[i + j] + &(x * y);
        }
    }
    trim(out)
}

/// `p(x)`, by Horner.
pub fn evaluate(p: &[Fr], x: &Fr) -> Fr {
    p.iter().rev().fold(Fr::zero(), |acc, c| &(&acc * x) + c)
}

/// `p(factor·X)`: the polynomial of a column read at an offset, `factor = ω^offset`.
pub fn compose_scaled(p: &[Fr], factor: &Fr) -> Vec<Fr> {
    let mut power = Fr::one();
    let mut out = Vec::with_capacity(p.len());
    for c in p {
        out.push(c * &power);
        power = &power * factor;
    }
    trim(out)
}

/// `(a / b, a mod b)` by long division.
pub fn div_rem(a: &[Fr], b: &[Fr]) -> PilfflonkResult<(Vec<Fr>, Vec<Fr>)> {
    let b = trim(b.to_vec());
    let Some(lead) = b.last() else {
        return invalid!("division by the zero polynomial");
    };
    let lead_inv = lead.inv()?;
    let mut rem = trim(a.to_vec());
    if rem.len() < b.len() {
        return Ok((Vec::new(), rem));
    }
    let mut quot = vec![Fr::zero(); rem.len() - b.len() + 1];
    for k in (0..quot.len()).rev() {
        let c = &rem[k + b.len() - 1] * &lead_inv;
        if !c.is_zero() {
            // Most zerofiers are sparse: X^N - 1 has two terms.
            for (j, bj) in b.iter().enumerate().filter(|(_, bj)| !bj.is_zero()) {
                rem[k + j] = &rem[k + j] - &(&c * bj);
            }
        }
        quot[k] = c;
    }
    rem.truncate(b.len() - 1);
    Ok((trim(quot), trim(rem)))
}

/// `Π_j (X - roots_j)`.
pub fn from_roots(roots: &[Fr]) -> Vec<Fr> {
    let mut p = vec![Fr::one()];
    for root in roots {
        // p·(X - root)
        let mut next = vec![Fr::zero(); p.len() + 1];
        for (i, c) in p.iter().enumerate() {
            next[i + 1] = &next[i + 1] + c;
            next[i] = &next[i] - &(c * root);
        }
        p = next;
    }
    p
}

/// The coefficients of the polynomial of degree `< N` that takes `values[j]` at `ω^j`, `N` the
/// number of values: the inverse DFT, `c_k = (1/N)·Σ_j values[j]·ω^(-jk)`, computed as it reads.
pub fn interpolate(values: &[Fr], omega: &Fr) -> PilfflonkResult<Vec<Fr>> {
    let n = values.len();
    let n_inv = Fr::from_u64(n as u64).inv()?;
    let omega_inv = omega.inv()?;
    let mut out = Vec::with_capacity(n);
    let mut w_k = Fr::one(); // ω^(-k)
    for _ in 0..n {
        let mut acc = Fr::zero();
        let mut w_jk = Fr::one(); // ω^(-jk)
        for v in values {
            acc = &acc + &(v * &w_jk);
            w_jk = &w_jk * &w_k;
        }
        out.push(&acc * &n_inv);
        w_k = &w_k * &omega_inv;
    }
    Ok(trim(out))
}

/// Evaluates columns over `H = {ω^j}` at one point `x`:
/// `p(x) = (x^N - 1)/N · Σ_j values[j]·ω^j/(x - ω^j)`, and `values[k]` if `x = ω^k`.
pub struct Barycentric {
    at_row: Option<usize>,
    /// `(x^N - 1)/N · ω^j/(x - ω^j)`, the Lagrange basis at `x`.
    weights: Vec<Fr>,
}

impl Barycentric {
    pub fn new(x: &Fr, omega: &Fr, n: usize) -> PilfflonkResult<Self> {
        let mut roots = Vec::with_capacity(n);
        let mut w = Fr::one();
        for _ in 0..n {
            roots.push(w.clone());
            w = &w * omega;
        }
        if let Some(k) = roots.iter().position(|root| root == x) {
            return Ok(Self { at_row: Some(k), weights: Vec::new() });
        }
        let differences: Vec<Fr> = roots.iter().map(|root| x - root).collect();
        let inverses = batch_inverse(&differences)?;
        let factor = &(&x.pow_u64(n as u64) - &Fr::one()) * &Fr::from_u64(n as u64).inv()?;
        let weights = roots.iter().zip(&inverses).map(|(root, inv)| &factor * &(root * inv)).collect();
        Ok(Self { at_row: None, weights })
    }

    pub fn evaluate(&self, values: &[Fr]) -> PilfflonkResult<Fr> {
        if let Some(k) = self.at_row {
            return match values.get(k) {
                Some(v) => Ok(v.clone()),
                None => invalid!("a column of {} values has no row {k}", values.len()),
            };
        }
        if values.len() != self.weights.len() {
            return invalid!("a column of {} values over an H of {} points", values.len(), self.weights.len());
        }
        Ok(values.iter().zip(&self.weights).fold(Fr::zero(), |acc, (v, w)| &acc + &(v * w)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::fr::omega;
    use super::*;

    fn fr(v: u64) -> Fr {
        Fr::from_u64(v)
    }

    fn some_polynomial(n: usize) -> Vec<Fr> {
        (0..n as u64).map(|i| &fr(i * i + 3) * &fr(7).pow_u64(i)).collect()
    }

    #[test]
    fn interpolation_inverts_evaluation_on_h() {
        let w = omega(4).unwrap();
        let p = some_polynomial(16);
        let values: Vec<Fr> = (0..16).map(|j| evaluate(&p, &w.pow_u64(j))).collect();
        assert_eq!(interpolate(&values, &w).unwrap(), p);
    }

    #[test]
    fn barycentric_evaluation_is_horners_off_and_on_h() {
        let w = omega(4).unwrap();
        let p = some_polynomial(16);
        let values: Vec<Fr> = (0..16).map(|j| evaluate(&p, &w.pow_u64(j))).collect();
        for x in [fr(2), fr(123456789), -&fr(5)] {
            let b = Barycentric::new(&x, &w, 16).unwrap();
            assert_eq!(b.evaluate(&values).unwrap(), evaluate(&p, &x), "at {x}");
        }
        let at_row_3 = Barycentric::new(&w.pow_u64(3), &w, 16).unwrap();
        assert_eq!(at_row_3.evaluate(&values).unwrap(), values[3]);
        assert!(Barycentric::new(&fr(2), &w, 16).unwrap().evaluate(&values[1..]).is_err());
    }

    #[test]
    fn long_division_and_products() {
        let a = some_polynomial(9);
        let b = some_polynomial(4);
        let product = mul(&a, &b);
        assert_eq!(product.len(), 12);
        let (q, r) = div_rem(&product, &b).unwrap();
        assert_eq!((q, r), (a.clone(), Vec::new()));
        let plus_one = add(&product, &[fr(1)]);
        let (q, r) = div_rem(&plus_one, &b).unwrap();
        assert_eq!((q, r), (a.clone(), vec![fr(1)]));
        assert!(div_rem(&a, &[]).is_err());
        assert_eq!(sub(&add(&a, &b), &b), a);
        assert_eq!(add(&a, &neg(&a)), Vec::<Fr>::new());
        assert_eq!(scale(&a, &Fr::zero()), Vec::<Fr>::new());
    }

    #[test]
    fn the_roots_of_unity_give_x_n_minus_1() {
        let w = omega(3).unwrap();
        let roots: Vec<Fr> = (0..8).map(|j| w.pow_u64(j)).collect();
        let mut expected = vec![Fr::zero(); 9];
        expected[0] = -&Fr::one();
        expected[8] = Fr::one();
        assert_eq!(from_roots(&roots), expected);
    }

    #[test]
    fn composing_with_a_scaled_x_is_evaluating_at_the_scaled_point() {
        let p = some_polynomial(6);
        let factor = fr(11);
        let x = fr(13);
        assert_eq!(evaluate(&compose_scaled(&p, &factor), &x), evaluate(&p, &(&factor * &x)));
    }
}
