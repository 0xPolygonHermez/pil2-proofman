//! The LogUp-GKR fractional sumcheck (Papini–Haböck, eprint 2023/1284).
//!
//! Given input fractions `(num(b), den(b))` indexed by the hypercube
//! `b ∈ {0,1}^K`, prove that
//!
//! `Σ_b num(b)/den(b) = p_out/q_out`
//!
//! via a layered binary tree of fraction additions
//!
//! `(p₁, q₁) ⊕ (p₂, q₂) = (p₁·q₂ + p₂·q₁, q₁·q₂)`.
//!
//! Depth-`d` nodes merge adjacent pairs of depth-`d+1` nodes, so the merged
//! variable is always `X₁` (the LSB, per the crate convention):
//! `p_d(y) = p_{d+1}(0,y)·q_{d+1}(1,y) + p_{d+1}(1,y)·q_{d+1}(0,y)` and
//! `q_d(y) = q_{d+1}(0,y)·q_{d+1}(1,y)`.
//!
//! The GKR walk starts from the output pair and, per layer, reduces claims
//! `p̃_d(r), q̃_d(r)` to claims on the next layer: batch with a challenge `λ`
//! and sumcheck
//!
//! `p̃_d(r) + λ·q̃_d(r) = Σ_{x∈{0,1}^d} eq(r,x)·[p(0,x)·q(1,x) + p(1,x)·q(0,x) + λ·q(0,x)·q(1,x)]`
//!
//! (tables at depth `d+1`). The eq factor is handled analytically: round
//! messages carry only the degree-2 factor `h` (evaluations at `1, 2`; `h(0)`
//! is recovered from the Gruen inter-round check `(1−rᵢ)·h(0) + rᵢ·h(1) =
//! claim`, exactly as the zerocheck). After the `d` rounds bind `ρ`, the
//! prover sends the four split values `p(0,ρ), p(1,ρ), q(0,ρ), q(1,ρ)`; a
//! fresh challenge `μ` interpolates them into single claims at `(μ, ρ)`.
//!
//! The walk terminates in MLE evaluation claims on the *input* tables at a
//! random point, which the caller discharges (for LogUp: an eq-weighted
//! sumcheck over the bus-term expressions, then the PCS opening).

use crate::eq::eq_evals;
use crate::error::MlError;
use crate::evaluator::{eval_instrs, eval_operand_cone, operand_eval, LeafSource, RowSource, Val};
use crate::hypercube::{fold_mle, Ext};
use crate::ir::{AirIr, BusIr};
use crate::par::map_range;
use crate::sumcheck::{interpolate_at, SumcheckOracle};
use crate::transcript::MlTranscript;
use crate::zerocheck::ZerocheckOracle;
use fields::{Field, Goldilocks};
use serde::{Deserialize, Serialize};

/// The layered fraction-addition circuit, fully materialized.
///
/// Numerators of the input layer live in the base field (bus numerators are
/// `±sel` / `mul` values); denominators and every internal layer are extension
/// field (denominators contain the bus challenge `γ`).
pub struct FractionTree {
    /// Input-layer numerators, length `2^K`.
    pub input_p: Vec<Goldilocks>,
    /// Input-layer denominators, length `2^K`.
    pub input_q: Vec<Ext>,
    /// `layers[d] = (p, q)` node values at depth `d` (`2^d` entries each),
    /// for `d = 0` (output) `..= K−1`. Empty when `K = 0`.
    layers: Vec<(Vec<Ext>, Vec<Ext>)>,
}

impl FractionTree {
    /// Build the tree. Rejects a zero input denominator (the fraction sum
    /// would be undefined); internal denominators are products of the input
    /// ones, hence automatically nonzero.
    pub fn new(input_p: Vec<Goldilocks>, input_q: Vec<Ext>) -> Result<Self, MlError> {
        if input_p.len() != input_q.len() || input_p.is_empty() || !input_p.len().is_power_of_two() {
            return Err(MlError::Malformed("fraction tree input must be a nonempty power of two".into()));
        }
        if input_q.iter().any(|q| q.is_zero()) {
            return Err(MlError::Malformed("fraction tree: zero denominator in the input layer".into()));
        }

        let k = input_p.len().trailing_zeros() as usize;
        let mut layers: Vec<(Vec<Ext>, Vec<Ext>)> = Vec::with_capacity(k);
        if k > 0 {
            let half = input_p.len() / 2;
            let p = map_range(half, |i| input_q[2 * i + 1] * input_p[2 * i] + input_q[2 * i] * input_p[2 * i + 1]);
            let q = map_range(half, |i| input_q[2 * i] * input_q[2 * i + 1]);
            layers.push((p, q));
            while layers.last().unwrap().0.len() > 1 {
                let (lp, lq) = layers.last().unwrap();
                let half = lp.len() / 2;
                let p = map_range(half, |i| lp[2 * i] * lq[2 * i + 1] + lp[2 * i + 1] * lq[2 * i]);
                let q = map_range(half, |i| lq[2 * i] * lq[2 * i + 1]);
                layers.push((p, q));
            }
            layers.reverse();
        }
        Ok(Self { input_p, input_q, layers })
    }

    /// Number of input variables `K` (input length `2^K`).
    pub fn n_vars(&self) -> usize {
        self.input_p.len().trailing_zeros() as usize
    }

    /// The output fraction `(p_out, q_out)`, with `Σ_b num(b)/den(b) = p_out/q_out`.
    pub fn output(&self) -> (Ext, Ext) {
        match self.layers.first() {
            Some((p, q)) => (p[0], q[0]),
            None => (Ext::from_base(self.input_p[0]), self.input_q[0]),
        }
    }
}

/// Prover messages of the GKR walk over a [`FractionTree`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalSumcheckProof {
    /// The output fraction.
    pub p_out: Ext,
    pub q_out: Ext,
    /// `layer_round_polys[j][round] = [h(1), h(2)]` — Gruen round messages of
    /// the depth-`j` → depth-`j+1` sumcheck (`j` rounds at layer `j`).
    pub layer_round_polys: Vec<Vec<Vec<Ext>>>,
    /// `layer_claims[j] = [p(0,ρ), p(1,ρ), q(0,ρ), q(1,ρ)]` of the
    /// depth-`j+1` tables at the bound point `ρ`.
    pub layer_claims: Vec<[Ext; 4]>,
}

/// The claims the walk terminates in: MLE evaluations of the input
/// numerator/denominator tables at `point`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputClaims {
    pub point: Vec<Ext>,
    pub p: Ext,
    pub q: Ext,
}

/// The per-layer sumcheck oracle over the split (even/odd = `X₁ = 0/1`)
/// tables of one tree layer, with the eq factor handled analytically
/// (suffix tables + Gruen), as in `ZerocheckOracle`.
struct LayerOracle {
    p: PTables,
    q0: Vec<Ext>,
    q1: Vec<Ext>,
    lambda: Ext,
    eq_suffix: Vec<Ext>,
}

/// Numerator split tables: base field for the input layer until the first
/// `bind` (folding by an extension challenge promotes them).
enum PTables {
    Base(Vec<Goldilocks>, Vec<Goldilocks>),
    Ext(Vec<Ext>, Vec<Ext>),
}

impl LayerOracle {
    /// Oracle for the claim at `r` (`r.len() = depth − 1`) on the depth-`r.len()`
    /// layer, over the depth-`r.len()+1` tables of `tree`.
    fn new(tree: &FractionTree, r: &[Ext], lambda: Ext) -> Self {
        let depth = r.len() + 1;
        debug_assert!(depth <= tree.n_vars());
        let half = 1usize << r.len();

        let (p, q0, q1) = if depth == tree.n_vars() {
            let p0 = map_range(half, |i| tree.input_p[2 * i]);
            let p1 = map_range(half, |i| tree.input_p[2 * i + 1]);
            let q0 = map_range(half, |i| tree.input_q[2 * i]);
            let q1 = map_range(half, |i| tree.input_q[2 * i + 1]);
            (PTables::Base(p0, p1), q0, q1)
        } else {
            let (lp, lq) = &tree.layers[depth];
            let p0 = map_range(half, |i| lp[2 * i]);
            let p1 = map_range(half, |i| lp[2 * i + 1]);
            let q0 = map_range(half, |i| lq[2 * i]);
            let q1 = map_range(half, |i| lq[2 * i + 1]);
            (PTables::Ext(p0, p1), q0, q1)
        };

        let eq_suffix = if r.is_empty() { Vec::new() } else { eq_evals(&r[1..]) };
        Self { p, q0, q1, lambda, eq_suffix }
    }

    /// Evaluations `[h(0), h(1), h(2)]` of the degree-2 non-eq factor of the
    /// current round polynomial.
    fn round_evals(&self) -> [Ext; 3] {
        let half = self.eq_suffix.len();
        let mut h = [Ext::ZERO; 3];
        // value at X = 2 by linearity: v(2) = 2·v(1) − v(0)
        match &self.p {
            PTables::Base(p0, p1) => {
                for i in 0..half {
                    let w = self.eq_suffix[i];
                    let (a0, a1) = (p0[2 * i], p0[2 * i + 1]);
                    let (b0, b1) = (p1[2 * i], p1[2 * i + 1]);
                    let (c0, c1) = (self.q0[2 * i], self.q0[2 * i + 1]);
                    let (d0, d1) = (self.q1[2 * i], self.q1[2 * i + 1]);
                    let (a2, b2) = (a1 + a1 - a0, b1 + b1 - b0);
                    let (c2, d2) = (c1 + c1 - c0, d1 + d1 - d0);
                    h[0] += w * (d0 * a0 + c0 * b0 + self.lambda * (c0 * d0));
                    h[1] += w * (d1 * a1 + c1 * b1 + self.lambda * (c1 * d1));
                    h[2] += w * (d2 * a2 + c2 * b2 + self.lambda * (c2 * d2));
                }
            }
            PTables::Ext(p0, p1) => {
                for i in 0..half {
                    let w = self.eq_suffix[i];
                    let (a0, a1) = (p0[2 * i], p0[2 * i + 1]);
                    let (b0, b1) = (p1[2 * i], p1[2 * i + 1]);
                    let (c0, c1) = (self.q0[2 * i], self.q0[2 * i + 1]);
                    let (d0, d1) = (self.q1[2 * i], self.q1[2 * i + 1]);
                    let (a2, b2) = (a1 + a1 - a0, b1 + b1 - b0);
                    let (c2, d2) = (c1 + c1 - c0, d1 + d1 - d0);
                    h[0] += w * (a0 * d0 + b0 * c0 + self.lambda * (c0 * d0));
                    h[1] += w * (a1 * d1 + b1 * c1 + self.lambda * (c1 * d1));
                    h[2] += w * (a2 * d2 + b2 * c2 + self.lambda * (c2 * d2));
                }
            }
        }
        h
    }

    fn bind(&mut self, r: Ext) {
        match &mut self.p {
            // First bind: folding base tables by an extension challenge yields
            // extension tables, so subsequent rounds run in `Ext`.
            PTables::Base(p0, p1) => {
                let promote = |t: &[Goldilocks]| -> Vec<Ext> {
                    let half = t.len() / 2;
                    (0..half).map(|i| Ext::from_base(t[2 * i]) + r * (t[2 * i + 1] - t[2 * i])).collect()
                };
                let (e0, e1) = (promote(p0), promote(p1));
                self.p = PTables::Ext(e0, e1);
            }
            PTables::Ext(p0, p1) => {
                fold_mle(p0, r);
                fold_mle(p1, r);
            }
        }
        fold_mle(&mut self.q0, r);
        fold_mle(&mut self.q1, r);
        let half = self.eq_suffix.len() / 2;
        for i in 0..half {
            self.eq_suffix[i] = self.eq_suffix[2 * i] + self.eq_suffix[2 * i + 1];
        }
        self.eq_suffix.truncate(half);
    }

    /// The split values `[p(0,ρ), p(1,ρ), q(0,ρ), q(1,ρ)]` once every round
    /// variable has been bound (tables of length 1).
    fn final_values(&self) -> [Ext; 4] {
        debug_assert_eq!(self.q0.len(), 1);
        match &self.p {
            PTables::Base(p0, p1) => [Ext::from_base(p0[0]), Ext::from_base(p1[0]), self.q0[0], self.q1[0]],
            PTables::Ext(p0, p1) => [p0[0], p1[0], self.q0[0], self.q1[0]],
        }
    }
}

/// Run the GKR walk over `tree`, producing the proof and the input-layer
/// evaluation claims. Transcript order per layer `j = 0..K`: sample `λ`;
/// `j` Gruen rounds (absorb `[h(1), h(2)]`, sample the round challenge);
/// absorb the four split values; sample `μ`. The output pair is absorbed
/// up front so every challenge binds it.
pub fn prove_fractional_sum(
    tree: &FractionTree,
    transcript: &mut MlTranscript,
) -> (FractionalSumcheckProof, InputClaims) {
    let k = tree.n_vars();
    let (p_out, q_out) = tree.output();
    transcript.absorb_ext(&p_out);
    transcript.absorb_ext(&q_out);

    let mut layer_round_polys = Vec::with_capacity(k);
    let mut layer_claims = Vec::with_capacity(k);
    let mut point: Vec<Ext> = Vec::new();
    let (mut claim_p, mut claim_q) = (p_out, q_out);

    for j in 0..k {
        let lambda = transcript.challenge();
        let mut oracle = LayerOracle::new(tree, &point, lambda);

        let mut rounds = Vec::with_capacity(j);
        let mut rho = Vec::with_capacity(j);
        for _ in 0..j {
            let evals = oracle.round_evals();
            let sent = evals[1..].to_vec();
            transcript.absorb_exts(&sent);
            let ch = transcript.challenge();
            oracle.bind(ch);
            rounds.push(sent);
            rho.push(ch);
        }

        let vals = oracle.final_values();
        transcript.absorb_exts(&vals);
        let mu = transcript.challenge();
        let [p0, p1, q0, q1] = vals;
        claim_p = p0 + mu * (p1 - p0);
        claim_q = q0 + mu * (q1 - q0);

        // The next claim point: the split variable is X₁, so μ is prepended.
        let mut next = Vec::with_capacity(j + 1);
        next.push(mu);
        next.extend(rho);
        point = next;

        layer_round_polys.push(rounds);
        layer_claims.push(vals);
    }

    (
        FractionalSumcheckProof { p_out, q_out, layer_round_polys, layer_claims },
        InputClaims { point, p: claim_p, q: claim_q },
    )
}

/// Verify the GKR walk of a fractional sumcheck over `2^n_vars` input
/// fractions, returning the input-layer evaluation claims the caller must
/// discharge. Rejects a zero output denominator.
pub fn verify_fractional_sum(
    proof: &FractionalSumcheckProof,
    n_vars: usize,
    transcript: &mut MlTranscript,
) -> Result<InputClaims, MlError> {
    if proof.layer_round_polys.len() != n_vars || proof.layer_claims.len() != n_vars {
        return Err(MlError::Malformed(format!("fractional sumcheck: expected {n_vars} layers")));
    }
    if proof.q_out.is_zero() {
        return Err(MlError::Malformed("fractional sumcheck: zero output denominator".into()));
    }

    transcript.absorb_ext(&proof.p_out);
    transcript.absorb_ext(&proof.q_out);

    let mut point: Vec<Ext> = Vec::new();
    let (mut claim_p, mut claim_q) = (proof.p_out, proof.q_out);

    for j in 0..n_vars {
        let lambda = transcript.challenge();
        let mut claim = claim_p + lambda * claim_q;

        let rounds = &proof.layer_round_polys[j];
        if rounds.len() != j {
            return Err(MlError::Malformed(format!("fractional sumcheck layer {j}: expected {j} rounds")));
        }
        let mut rho = Vec::with_capacity(j);
        for (i, sent) in rounds.iter().enumerate() {
            if sent.len() != 2 {
                return Err(MlError::Malformed(format!(
                    "fractional sumcheck layer {j} round {i}: expected 2 evaluations"
                )));
            }
            transcript.absorb_exts(sent);
            let ch = transcript.challenge();
            // Gruen: sent[0] = h(1); recover h(0) = (claim − rᵢ·h(1)) / (1 − rᵢ).
            let ri = point[i];
            let h0 = (claim - ri * sent[0]) * (Ext::ONE - ri).inverse();
            claim = interpolate_at(&[h0, sent[0], sent[1]], ch);
            rho.push(ch);
        }

        // Final check of the layer sumcheck: the eq factor is absorbed by the
        // Gruen rounds, so the claim equals the batched merge on the split values.
        let [p0, p1, q0, q1] = proof.layer_claims[j];
        if claim != p0 * q1 + p1 * q0 + lambda * (q0 * q1) {
            return Err(MlError::FinalCheck(format!(
                "fractional sumcheck layer {j}: claim inconsistent with split values"
            )));
        }

        transcript.absorb_exts(&proof.layer_claims[j]);
        let mu = transcript.challenge();
        claim_p = p0 + mu * (p1 - p0);
        claim_q = q0 + mu * (q1 - q0);

        let mut next = Vec::with_capacity(j + 1);
        next.push(mu);
        next.extend(rho);
        point = next;
    }

    Ok(InputClaims { point, p: claim_p, q: claim_q })
}

// --- LogUp bus phase: the GKR walk applied to an AIR's `BusIr` ---

/// Prover messages of the bus phase, carried in the proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusProof {
    /// The GKR walk over the fraction tree.
    pub fractional: FractionalSumcheckProof,
    /// Round messages `[h(1), …, h(d)]` of the input-layer reduction sumcheck
    /// (`n_bits` Gruen rounds of degree `d = max_term_degree`).
    pub reduction_round_polys: Vec<Vec<Ext>>,
}

/// Row/term layout of the fraction-tree input: `index = t + row·M` with
/// `M = terms.len().next_power_of_two()` — term bits in the low variables.
/// Padded terms hold the neutral fraction `0/1`.
fn bus_layout(bus: &BusIr) -> (usize, usize) {
    let cap = bus.terms.len().next_power_of_two();
    (cap, cap.trailing_zeros() as usize)
}

/// Round-polynomial degree of the input-layer reduction.
fn bus_round_degree(bus: &BusIr) -> usize {
    (bus.max_term_degree as usize).max(1)
}

/// Evaluate every bus term over the trace, producing the fraction-tree input
/// tables. Numerators must stay in the base field (they are `±sel`/`mul`
/// expressions); denominators contain `γ` and are extension-valued.
#[allow(clippy::too_many_arguments)]
fn build_bus_input(
    ir: &AirIr,
    bus: &BusIr,
    witness: &[Vec<Vec<Goldilocks>>],
    consts: &[Vec<Goldilocks>],
    customs: &[Vec<Vec<Goldilocks>>],
    publics: &[Goldilocks],
    challenges: &[Ext],
    air_values: &[Ext],
    airgroup_values: &[Ext],
    proof_values: &[Ext],
) -> Result<(Vec<Goldilocks>, Vec<Ext>), MlError> {
    let n_rows = 1usize << ir.n_bits;
    let (cap, _) = bus_layout(bus);
    let mut p = vec![Goldilocks::ZERO; cap * n_rows];
    let mut q = vec![Ext::ONE; cap * n_rows];

    let mut temps: Vec<Val> = Vec::new();
    for row in 0..n_rows {
        let src = RowSource {
            witness,
            consts,
            customs,
            publics,
            challenges,
            air_values,
            airgroup_values,
            proof_values,
            row,
            n_rows,
        };
        eval_instrs(ir, &src, &mut temps);
        for (t, term) in bus.terms.iter().enumerate() {
            // Numerators are `±sel`/`mul` expressions: base-field *values*,
            // though the evaluator may carry them as extension elements (e.g.
            // when a selector reads an air value, which `RowSource` always
            // returns as `Ext`). Accept any base-embedded value.
            let num = match operand_eval(ir, &src, &temps, &term.num) {
                Val::B(x) => x,
                Val::E(e) if e.value[1].is_zero() && e.value[2].is_zero() => e.value[0],
                Val::E(_) => {
                    return Err(MlError::Unsupported("bus term numerator must be a base-field value".into()));
                }
            };
            p[t + row * cap] = num;
            q[t + row * cap] = operand_eval(ir, &src, &temps, &term.den).to_ext();
        }
    }
    Ok((p, q))
}

/// Leaf source for the scalar ("direct") bus terms: scalars only — a column
/// reference in a scalar term is a setup bug.
struct ScalarSource<'a> {
    publics: &'a [Goldilocks],
    challenges: &'a [Ext],
    air_values: &'a [Ext],
    airgroup_values: &'a [Ext],
    proof_values: &'a [Ext],
}

impl LeafSource for ScalarSource<'_> {
    fn witness(&self, _stage: u8, _col: u32, _row_offset: i32) -> Val {
        unreachable!("scalar bus terms must not reference witness columns")
    }
    fn constant(&self, _col: u32, _row_offset: i32) -> Val {
        unreachable!("scalar bus terms must not reference fixed columns")
    }
    fn custom(&self, _commit: u8, _col: u32, _row_offset: i32) -> Val {
        unreachable!("scalar bus terms must not reference custom commits")
    }
    fn public(&self, idx: u32) -> Val {
        Val::B(self.publics[idx as usize])
    }
    fn challenge(&self, idx: u32) -> Val {
        Val::E(self.challenges[idx as usize])
    }
    fn air_value(&self, idx: u32) -> Val {
        Val::E(self.air_values[idx as usize])
    }
    fn airgroup_value(&self, idx: u32) -> Val {
        Val::E(self.airgroup_values[idx as usize])
    }
    fn proof_value(&self, idx: u32) -> Val {
        Val::E(self.proof_values[idx as usize])
    }
}

/// The scalar bus terms as one fraction `(num, den)`. Verifier-evaluable:
/// scalar terms read publics, challenges and air/airgroup values only (never
/// the result airgroup value itself).
pub fn eval_scalar_fraction(
    ir: &AirIr,
    bus: &BusIr,
    publics: &[Goldilocks],
    challenges: &[Ext],
    air_values: &[Ext],
    airgroup_values: &[Ext],
    proof_values: &[Ext],
) -> Result<(Ext, Ext), MlError> {
    let src = ScalarSource { publics, challenges, air_values, airgroup_values, proof_values };
    let mut temps: Vec<Val> = Vec::new();
    let (mut num, mut den) = (Ext::ZERO, Ext::ONE);
    for term in &bus.scalar_terms {
        let n_t = eval_operand_cone(ir, &src, &mut temps, &term.num).to_ext();
        let d_t = eval_operand_cone(ir, &src, &mut temps, &term.den).to_ext();
        if d_t.is_zero() {
            return Err(MlError::Malformed("scalar bus term has zero denominator".into()));
        }
        num = num * d_t + n_t * den;
        den *= d_t;
    }
    Ok((num, den))
}

/// Prover-side bus context, built (transcript-free) before the PIOP phases so
/// the result airgroup value is fixed prior to any absorption that reads it.
pub struct BusProver {
    tree: FractionTree,
    /// The instance's net bus contribution: `p_out/q_out` plus the scalar
    /// fraction — the value of the result airgroup value (`gsum_result`).
    pub result: Ext,
}

impl BusProver {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ir: &AirIr,
        bus: &BusIr,
        witness: &[Vec<Vec<Goldilocks>>],
        consts: &[Vec<Goldilocks>],
        customs: &[Vec<Vec<Goldilocks>>],
        publics: &[Goldilocks],
        challenges: &[Ext],
        air_values: &[Ext],
        airgroup_values: &[Ext],
        proof_values: &[Ext],
    ) -> Result<Self, MlError> {
        let (p, q) = build_bus_input(
            ir,
            bus,
            witness,
            consts,
            customs,
            publics,
            challenges,
            air_values,
            airgroup_values,
            proof_values,
        )?;
        let tree = FractionTree::new(p, q)?;
        let (s_num, s_den) =
            eval_scalar_fraction(ir, bus, publics, challenges, air_values, airgroup_values, proof_values)?;
        let (p_out, q_out) = tree.output();
        let result = p_out * q_out.inverse() + s_num * s_den.inverse();
        Ok(Self { tree, result })
    }

    /// Run the GKR walk and the input-layer reduction sumcheck, returning the
    /// proof and the reduction point `v` at which the witness columns must be
    /// opened (through the `BusRot` kernels).
    #[allow(clippy::too_many_arguments)]
    pub fn prove(
        &self,
        ir: &AirIr,
        bus: &BusIr,
        witness: &[Vec<Vec<Goldilocks>>],
        consts: &[Vec<Goldilocks>],
        customs: &[Vec<Vec<Goldilocks>>],
        publics: &[Goldilocks],
        challenges: &[Ext],
        air_values: &[Ext],
        airgroup_values: &[Ext],
        proof_values: &[Ext],
        transcript: &mut MlTranscript,
    ) -> (BusProof, Vec<Ext>) {
        let n = ir.n_bits as usize;
        let (_, nu) = bus_layout(bus);
        debug_assert_eq!(self.tree.n_vars(), nu + n);

        // GKR walk: output pair → input-table claims at `u = (u_term ‖ u_row)`.
        let (fractional, input_claims) = prove_fractional_sum(&self.tree, transcript);
        let (u_term, u_row) = input_claims.point.split_at(nu);

        // Input-layer reduction: batch p/q with μ and prove
        //   p̂(u) + μ·q̂(u) − μ·pad = Σ_x eq(u_row, x)·Σ_t c_t·(Num_t(x) + μ·Den_t(x))
        // with c_t = eq(bits(t), u_term) and pad the padded terms' constant
        // contribution (num 0, den 1).
        let c = eq_evals(u_term);
        let mu = transcript.challenge();
        let mut roots = Vec::with_capacity(2 * bus.terms.len());
        for (t, term) in bus.terms.iter().enumerate() {
            roots.push((term.num, c[t]));
            roots.push((term.den, mu * c[t]));
        }

        let mut oracle = ZerocheckOracle::with_roots(
            ir,
            witness,
            consts,
            customs,
            publics,
            challenges,
            air_values,
            airgroup_values,
            proof_values,
            u_row,
            roots,
            bus_round_degree(bus),
            0,
        );

        let mut reduction_round_polys = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let evals = oracle.round_evals();
            let sent = evals[1..].to_vec();
            transcript.absorb_exts(&sent);
            let ch = transcript.challenge();
            oracle.bind(ch);
            reduction_round_polys.push(sent);
            v.push(ch);
        }

        (BusProof { fractional, reduction_round_polys }, v)
    }
}

/// Verifier-side output of the bus phase transcript replay. The final claim
/// is checked against the claimed openings (at the `BusRot` kernels) once the
/// claims matrix has been absorbed.
pub struct BusVerification {
    /// The reduction point `v`.
    pub point: Vec<Ext>,
    /// Final claim of the reduction sumcheck: must equal
    /// `Σ_t c_t·(Num_t + μ·Den_t)` on the claimed openings at `v`.
    pub final_claim: Ext,
    /// The p/q batching challenge.
    pub mu: Ext,
    /// `c_t = eq(bits(t), u_term)` for `t < terms.len()`.
    pub term_weights: Vec<Ext>,
    /// The output fraction (enters the result-airgroup-value check).
    pub p_out: Ext,
    pub q_out: Ext,
}

/// Replay the bus phase: verify the GKR walk and the reduction rounds.
pub fn verify_bus_phase(
    bus: &BusIr,
    proof: &BusProof,
    n_bits: usize,
    transcript: &mut MlTranscript,
) -> Result<BusVerification, MlError> {
    let (cap, nu) = bus_layout(bus);
    let k = nu + n_bits;

    let input_claims = verify_fractional_sum(&proof.fractional, k, transcript)?;
    let (u_term, u_row) = input_claims.point.split_at(nu);

    let c = eq_evals(u_term);
    let mu = transcript.challenge();
    let pad: Ext = c[bus.terms.len()..cap].iter().copied().sum();
    let mut claim = input_claims.p + mu * input_claims.q - mu * pad;

    let d = bus_round_degree(bus);
    if proof.reduction_round_polys.len() != n_bits {
        return Err(MlError::Malformed(format!("bus reduction: expected {n_bits} round polynomials")));
    }
    let mut v = Vec::with_capacity(n_bits);
    for (round, sent) in proof.reduction_round_polys.iter().enumerate() {
        if sent.len() != d {
            return Err(MlError::Malformed(format!("bus reduction round {round}: expected {d} evaluations")));
        }
        transcript.absorb_exts(sent);
        let ch = transcript.challenge();
        // Gruen: sent[0] = h(1); recover h(0) = (claim − rₖ·h(1)) / (1 − rₖ).
        let rk = u_row[round];
        let g0 = (claim - rk * sent[0]) * (Ext::ONE - rk).inverse();
        let mut evals = Vec::with_capacity(d + 1);
        evals.push(g0);
        evals.extend_from_slice(sent);
        claim = interpolate_at(&evals, ch);
        v.push(ch);
    }

    Ok(BusVerification {
        point: v,
        final_claim: claim,
        mu,
        term_weights: c[..bus.terms.len()].to_vec(),
        p_out: proof.fractional.p_out,
        q_out: proof.fractional.q_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypercube::{mle_eval, mle_eval_base};
    use fields::{Field, PrimeField64};
    use rand::{rng, RngExt};

    fn random_ext() -> Ext {
        let mut r = rng();
        Ext::from_array(&[
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
            Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64),
        ])
    }

    fn random_inputs(k: usize) -> (Vec<Goldilocks>, Vec<Ext>) {
        let mut r = rng();
        let p = (0..(1usize << k)).map(|_| Goldilocks::new(r.random::<u64>() % Goldilocks::ORDER_U64)).collect();
        let q = (0..(1usize << k))
            .map(|_| loop {
                let v = random_ext();
                if !v.is_zero() {
                    return v;
                }
            })
            .collect();
        (p, q)
    }

    /// `p_out/q_out` must equal the brute-force fraction sum via inversions.
    #[test]
    fn fraction_tree_matches_bruteforce() {
        for k in [1usize, 2, 5, 8] {
            let (p, q) = random_inputs(k);
            let tree = FractionTree::new(p.clone(), q.clone()).expect("build");
            let (p_out, q_out) = tree.output();
            let brute: Ext = p.iter().zip(q.iter()).map(|(&pi, &qi)| qi.inverse() * pi).sum();
            assert_eq!(p_out * q_out.inverse(), brute, "k={k}");
        }
    }

    /// Padding with the neutral fraction (0, 1) must not change the sum.
    #[test]
    fn neutral_padding_preserves_sum() {
        let k = 4;
        let (mut p, mut q) = random_inputs(k);
        let tree = FractionTree::new(p.clone(), q.clone()).expect("build");
        let (p_out, q_out) = tree.output();

        p.resize(1 << (k + 1), Goldilocks::ZERO);
        q.resize(1 << (k + 1), Ext::ONE);
        let padded = FractionTree::new(p, q).expect("build padded");
        let (pp, qq) = padded.output();

        // Equal as fractions: p_out/q_out == pp/qq.
        assert_eq!(p_out * qq, pp * q_out);
    }

    #[test]
    fn zero_denominator_rejected() {
        let (p, mut q) = random_inputs(3);
        q[5] = Ext::ZERO;
        assert!(FractionTree::new(p, q).is_err());
    }

    /// Full prove→verify walk: same challenges on both sides, and the final
    /// claims must equal the true input-table MLEs at the returned point.
    #[test]
    fn gkr_roundtrip_matches_input_mles() {
        for k in [1usize, 2, 3, 6, 9] {
            let (p, q) = random_inputs(k);
            let tree = FractionTree::new(p.clone(), q.clone()).expect("build");

            let mut tp = MlTranscript::new();
            let (proof, prover_claims) = prove_fractional_sum(&tree, &mut tp);

            let mut tv = MlTranscript::new();
            let verifier_claims = verify_fractional_sum(&proof, k, &mut tv).expect("verify");
            assert_eq!(prover_claims, verifier_claims, "k={k}");

            assert_eq!(verifier_claims.point.len(), k);
            assert_eq!(verifier_claims.p, mle_eval_base(&p, &verifier_claims.point), "numerator claim k={k}");
            assert_eq!(verifier_claims.q, mle_eval(&q, &verifier_claims.point), "denominator claim k={k}");
        }
    }

    /// Tampering with any message class must either fail in-walk or surface
    /// as final claims inconsistent with the true input MLEs.
    #[test]
    fn tampered_walk_rejected() {
        let k = 5;
        let (p, q) = random_inputs(k);
        let tree = FractionTree::new(p.clone(), q.clone()).expect("build");
        let (proof, _) = prove_fractional_sum(&tree, &mut MlTranscript::new());

        let rejected = |proof: &FractionalSumcheckProof| -> bool {
            match verify_fractional_sum(proof, k, &mut MlTranscript::new()) {
                Err(_) => true,
                Ok(claims) => claims.p != mle_eval_base(&p, &claims.point) || claims.q != mle_eval(&q, &claims.point),
            }
        };

        let mut t1 = proof.clone();
        t1.p_out += Ext::ONE;
        assert!(rejected(&t1), "tampered p_out");

        let mut t2 = proof.clone();
        t2.q_out += Ext::ONE;
        assert!(rejected(&t2), "tampered q_out");

        let mut t3 = proof.clone();
        t3.layer_round_polys[k - 1][0][0] += Ext::ONE;
        assert!(rejected(&t3), "tampered round polynomial");

        let mut t4 = proof.clone();
        t4.layer_claims[2][1] += Ext::ONE;
        assert!(rejected(&t4), "tampered split value");

        let mut t5 = proof.clone();
        t5.layer_claims[k - 1][3] += Ext::ONE;
        assert!(rejected(&t5), "tampered input-layer split value");
    }
}
