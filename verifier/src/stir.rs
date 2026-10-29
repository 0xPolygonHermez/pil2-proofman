// STIR verification (Arnon, Chiesa, Fenzi, Yogev — ePrint 2024/390, Construction 5.2): the
// native-Rust twin of the C++ verifier in `pil2-stark/src/starkpil/stir/stir.hpp` plus the STARK
// glue of `stark_verify.hpp`.
//
// Split like the C++ side: `stir_verify` is the STIR argument alone — it replays the transcript
// from right after the challenges that defined f_0, re-derives every query index, and hands each
// round-1 claim about f_0 to a caller-supplied check. `stark_verify_stir` is the whole STARK
// verifier: the shared prefix (publics, stage roots, evals, stage-tree openings, the quotient
// check) is kept textually in sync with `stark_verify` in `verifier.rs`, and the f_0 claims are
// recorded during `stir_verify` and compared against the recomputed DEEP polynomial afterwards —
// the STIR counterpart of FRI's s1_vals consistency check.
//
// Two deliberate mirrors of the C++ verifier:
//  - Âns_i is taken from the proof (`ansCoeffs`) and checked to interpolate G_i at one random
//    point ρ_i through the shake polynomial `shakeCoeffs` (step 2(f) in stir.hpp) — never
//    interpolated here. Both hints are absorbed, zero padding included, before ρ_i is squeezed.
//  - r_out is squeezed once; a proof whose r_out lands in L_i is rejected (the prover aborts
//    there), the same single-squeeze rule as the recursion circuit.

use alloc::vec;
use alloc::vec::Vec;

#[allow(unused)]
use num_traits::Float;

use proofman_fields::{
    partial_merkle_tree, verify_fold, verify_mt, CubicExtensionField, Field, Goldilocks, Hash, PrimeField64,
    TranscriptLike,
};

use crate::layout::{merkle_proof_levels, sibling_words};
use crate::multiproof::{expand_stir_checked, Proven};
use crate::replay::{prefix_transcript, stir_transcript, StirTranscript};
use crate::domain::zi_vector;
use crate::verifier::{deep_denominators_inv, Boundary, DeepQueryConsts, FriEvalGroup};

type E3 = CubicExtensionField<Goldilocks>;

fn e3_zero() -> E3 {
    CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] }
}

fn e3_one() -> E3 {
    CubicExtensionField { value: [Goldilocks::ONE, Goldilocks::ZERO, Goldilocks::ZERO] }
}

/// Embed a base-field element into F = Goldilocks³.
fn embed(x: Goldilocks) -> E3 {
    CubicExtensionField { value: [x, Goldilocks::ZERO, Goldilocks::ZERO] }
}

/// Parameters of one STIR execution plus the commitment geometry it shares with the STARK —
/// the counterpart of the C++ `StirParams`.
#[derive(Clone, Debug)]
pub struct StirParams {
    /// kᵢ in bits: iteration i folds by 2^{kᵢ} (length M).
    pub folding_factors: Vec<u64>,
    /// log₂ dᵢ, the degree bound of fᵢ (length M+1; the last entry bounds the final polynomial).
    pub log_degrees: Vec<u64>,
    /// log₂|Lᵢ|, the evaluation domain of fᵢ (length M+1).
    pub log_domain_sizes: Vec<u64>,
    /// tᵢ: shift queries into fᵢ (length M).
    pub num_queries: Vec<u64>,
    /// Grinding bits on iteration i+1's query message (length M).
    pub grinding_bits_queries: Vec<u64>,
    pub arity: u64,
    pub last_level_verification: u64,
    /// Absorb hash(p) instead of p into the transcript.
    pub hash_commits: bool,
}

impl StirParams {
    /// M, the number of iterations.
    pub fn m(&self) -> usize {
        self.folding_factors.len()
    }

    /// The schedule invariants of Construction 5.2 — the checks of the C++ `Parameters::validate`,
    /// returning false instead of asserting (the schedule is codegen-time data, but the verifier
    /// must not misbehave on a corrupted build).
    pub fn validate(&self) -> bool {
        let m = self.m();
        if m < 1
            || self.log_degrees.len() != m + 1
            || self.log_domain_sizes.len() != m + 1
            || self.num_queries.len() != m
            || self.grinding_bits_queries.len() != m
        {
            v_error!("STIR parameters have inconsistent lengths");
            return false;
        }
        for i in 0..m {
            let k = self.folding_factors[i];
            if k < 1
                || self.log_degrees[i + 1] + k != self.log_degrees[i]
                || self.log_domain_sizes[i + 1] + 1 != self.log_domain_sizes[i]
                || self.log_degrees[i] >= self.log_domain_sizes[i]
                || self.log_degrees[i + 1] > self.log_domain_sizes[i] - k
            {
                v_error!("STIR schedule is invalid at iteration {}", i);
                return false;
            }
            // Remark 5.3: Lᵢ^{kᵢ} ∩ Lᵢ₊₁ = ∅, so Fill is never needed. The cosets
            // shift^{kᵢ}·⟨ω⟩ and shift·⟨ω⟩ meet iff shift^{kᵢ−1} lies in the larger subgroup.
            let shift = Goldilocks::new(Goldilocks::SHIFT);
            let ratio = shift.exp_u64((1u64 << k) - 1);
            let log_max = core::cmp::max(self.log_domain_sizes[i] - k, self.log_domain_sizes[i + 1]);
            if ratio.exp_power_of_2(log_max as usize) == Goldilocks::ONE {
                v_error!("STIR domains are not disjoint at iteration {} (Remark 5.3)", i);
                return false;
            }
            if self.grinding_bits_queries[i] >= 64 {
                v_error!("STIR grinding bits out of range at iteration {}", i);
                return false;
            }
        }
        self.log_degrees[m] <= self.log_domain_sizes[m]
    }
}

/// The constants a generated STIR verifier bakes in — the counterpart of `VerifierInfo` for a
/// circuit whose low-degree test is STIR. The shared STARK fields mean the same as there.
pub struct StirVerifierInfo {
    pub n_stages: u32,
    pub n_constants: u64,
    pub n_evals: u64,
    pub n_bits: u64,
    pub n_bits_ext: u64,
    pub n_challenges: u64,
    pub n_challenges_total: u64,
    pub num_vals: Vec<u64>,
    pub opening_points: Vec<i64>,
    pub boundaries: Vec<Boundary>,
    pub q_deg: u64,
    pub q_index: u64,
    /// The DEEP polynomial's evaluation table, the same one FRI's `VerifierInfo` carries.
    pub fri_ev_groups: Vec<FriEvalGroup>,
    pub stir: StirParams,
}

/// The flat proof's size in bytes, publics excluded.
pub fn expected_stir_proof_size_bytes(info: &StirVerifierInfo) -> usize {
    info.proof_layout().flat_proof_words() * 8
}

/// The parsed STIR section of a flat proof buffer, in `proof2pointer` order.
pub struct StirSection {
    /// Roots of T_0..T_{M−1}. Unlike FRI, the first one is a commitment of its own (to f_0).
    pub roots: Vec<[Goldilocks; 4]>,
    /// `cosets[i][q]`: the opened coset of T_i at query q, 2^{kᵢ} extension elements flat.
    pub cosets: Vec<Vec<Vec<Goldilocks>>>,
    /// `siblings[i][q]`: the Merkle path of that opening, levels flat (`sibling_words` each).
    pub siblings: Vec<Vec<Vec<Goldilocks>>>,
    /// `last_levels[i]`: the published bottom level of T_i (empty when lastLevelVerification = 0).
    pub last_levels: Vec<Vec<Goldilocks>>,
    /// β_{i,1}, i = 1..M−1 (s = 1).
    pub betas: Vec<E3>,
    /// p, as d_M coefficients.
    pub final_pol: Vec<E3>,
    /// One grinding nonce per query message.
    pub nonces: Vec<u64>,
    /// ans_coeffs[i−1]: the coefficients of Âns_i, i = 1..M−1, zero-padded to s + t_{i−1}.
    pub ans_coeffs: Vec<Vec<E3>>,
    /// shake_coeffs[i−1]: the coefficients of the shake polynomial Sh_i, same padding.
    pub shake_coeffs: Vec<Vec<E3>>,
}

/// Walk the STIR section out of `proof` starting at `*p`, which the caller has length-checked
/// against `stir_section_size_words`.
pub fn parse_stir_section(proof: &[u64], p: &mut usize, params: &StirParams) -> StirSection {
    parse_stir_section_impl(proof, p, params, true)
}

/// [`parse_stir_section`]; without `with_siblings` the Merkle paths are stepped over, `siblings` empty.
fn parse_stir_section_impl(proof: &[u64], p: &mut usize, params: &StirParams, with_siblings: bool) -> StirSection {
    let m = params.m();
    let n_sibs_per_level = ((params.arity - 1) * 4) as usize;
    let num_nodes_level = if params.last_level_verification == 0 {
        0
    } else {
        params.arity.pow(params.last_level_verification as u32) as usize
    };

    let mut roots = Vec::with_capacity(m);
    for _ in 0..m {
        let mut root = [Goldilocks::ZERO; 4];
        for r in &mut root {
            *r = Goldilocks::new(proof[*p]);
            *p += 1;
        }
        roots.push(root);
    }

    let mut cosets: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(m);
    let mut siblings: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(m);
    let mut last_levels: Vec<Vec<Goldilocks>> = Vec::with_capacity(m);
    for i in 0..m {
        let k = 1usize << params.folding_factors[i];
        let log_leaves = params.log_domain_sizes[i] - params.folding_factors[i];
        let n_sibs = merkle_proof_levels(log_leaves, params.arity, params.last_level_verification);
        let t = params.num_queries[i] as usize;

        let mut tree_cosets = Vec::with_capacity(t);
        for _ in 0..t {
            let mut vals = Vec::with_capacity(k * 3);
            for _ in 0..k * 3 {
                vals.push(Goldilocks::new(proof[*p]));
                *p += 1;
            }
            tree_cosets.push(vals);
        }
        cosets.push(tree_cosets);

        let n = n_sibs * n_sibs_per_level;
        if with_siblings {
            let mut tree_siblings = Vec::with_capacity(t);
            for _ in 0..t {
                tree_siblings.push(proof[*p..][..n].iter().map(|&x| Goldilocks::new(x)).collect::<Vec<_>>());
                *p += n;
            }
            siblings.push(tree_siblings);
        } else {
            *p += t * n;
        }

        let mut level = Vec::with_capacity(num_nodes_level * 4);
        for _ in 0..num_nodes_level * 4 {
            level.push(Goldilocks::new(proof[*p]));
            *p += 1;
        }
        last_levels.push(level);
    }

    let mut betas = Vec::with_capacity(m - 1);
    for _ in 1..m {
        betas.push(CubicExtensionField {
            value: [Goldilocks::new(proof[*p]), Goldilocks::new(proof[*p + 1]), Goldilocks::new(proof[*p + 2])],
        });
        *p += 3;
    }

    let d_final = 1usize << params.log_degrees[m];
    let mut final_pol = Vec::with_capacity(d_final);
    for _ in 0..d_final {
        final_pol.push(CubicExtensionField {
            value: [Goldilocks::new(proof[*p]), Goldilocks::new(proof[*p + 1]), Goldilocks::new(proof[*p + 2])],
        });
        *p += 3;
    }

    let mut nonces = Vec::with_capacity(m);
    for _ in 0..m {
        nonces.push(proof[*p]);
        *p += 1;
    }

    let read_hints = |proof: &[u64], p: &mut usize| -> Vec<Vec<E3>> {
        let mut hints = Vec::with_capacity(m - 1);
        for i in 1..m {
            let n_g_max = 1 + params.num_queries[i - 1] as usize;
            let mut coeffs = Vec::with_capacity(n_g_max);
            for _ in 0..n_g_max {
                coeffs.push(CubicExtensionField {
                    value: [Goldilocks::new(proof[*p]), Goldilocks::new(proof[*p + 1]), Goldilocks::new(proof[*p + 2])],
                });
                *p += 3;
            }
            hints.push(coeffs);
        }
        hints
    };
    let ans_coeffs = read_hints(proof, p);
    let shake_coeffs = read_hints(proof, p);

    StirSection { roots, cosets, siblings, last_levels, betas, final_pol, nonces, ans_coeffs, shake_coeffs }
}

// ---------------------------------------------------------------------------------------------
// The polynomial arithmetic (`stir_math.hpp`)
// ---------------------------------------------------------------------------------------------

/// Horner at a base-field point: each step is three base products, not an extension product.
fn eval_pol_base(coeffs: &[E3], x: Goldilocks) -> E3 {
    let mut out = e3_zero();
    for c in coeffs.iter().rev() {
        out = out * x + *c;
    }
    out
}

/// The coefficients of ∏_{a∈points}(X − a), lowest degree first.
fn vanishing_poly(points: &[E3]) -> Vec<E3> {
    let mut v = vec![e3_one()];
    for a in points {
        // v ← v·(X − a): descending so v[m] is still the old value when it moves up.
        v.push(e3_zero());
        for m in (1..v.len()).rev() {
            v[m] = v[m - 1] - v[m] * *a;
        }
        v[0] = e3_zero() - v[0] * *a;
    }
    v
}

/// Horner evaluation of Σⱼ coeffs[j]·Xʲ at x ∈ F.
fn eval_pol_e3(coeffs: &[E3], x: E3) -> E3 {
    let mut out = e3_zero();
    for c in coeffs.iter().rev() {
        out = out * x + *c;
    }
    out
}

/// Σ_{j=0}^{e} yʲ, in closed form (1 − y^{e+1}) / (1 − y) unless y = 1.
fn geometric_sum(y: E3, e: u64) -> E3 {
    let one = e3_one();
    if y == one {
        return embed(Goldilocks::new(e + 1));
    }
    (one - y.pow(e + 1)) * (one - y).inverse()
}

/// The coefficients of Âns, the unique polynomial of degree < |S| with Âns(a) = Ans(a) for all
/// a ∈ S (Newton interpolation, O(|S|²)). The points are distinct: `QuotientContext::add` dedups.
#[cfg(test)]
fn interpolate(points: &[E3], values: &[E3]) -> Vec<E3> {
    let n = points.len();
    let mut coeffs = vec![e3_zero(); n];
    if n == 0 {
        return coeffs;
    }

    // Newton divided differences: c[j] = Ans[a_0, …, a_j].
    let mut c: Vec<E3> = values.to_vec();
    for level in 1..n {
        for j in (level..n).rev() {
            let num = c[j] - c[j - 1];
            let den = points[j] - points[j - level];
            c[j] = num * den.inverse();
        }
    }

    // Newton form → monomial coefficients, Horner-style from the top:
    //   Âns = c_0 + (X − a_0)(c_1 + (X − a_1)(c_2 + …))
    // Starting from P = c_{n−1}, repeat P ← P·(X − a_j) + c_j for j = n−2 down to 0.
    coeffs[0] = c[n - 1];
    for (deg, j) in (0..n - 1).rev().enumerate() {
        let a = points[j];
        // P·(X − a): descending m so coeffs[m] is still the old value when it is moved up.
        for m in (0..=deg).rev() {
            let lo = coeffs[m];
            coeffs[m + 1] += lo;
            coeffs[m] = e3_zero() - lo * a;
        }
        coeffs[0] += c[j];
    }
    coeffs
}

/// (G, Ans, r_comb) of one iteration: everything step 2(e) of Construction 5.2 needs in order to
/// turn a value of the committed gᵢ into the corresponding value of the virtual
///   fᵢ = DegCor(dᵢ, r_comb, Quotient(gᵢ, Gᵢ, Ansᵢ, Fillᵢ), dᵢ − |Gᵢ|).
struct QuotientContext {
    points: Vec<E3>,
    values: Vec<E3>,
    ans_coeffs: Vec<E3>,
    /// ∏_{a∈G}(X − a) expanded, set with Âns: `apply` then evaluates it per point in O(|G|) base
    /// products instead of multiplying |G| extension factors.
    vanishing: Vec<E3>,
    r_comb: E3,
}

impl QuotientContext {
    fn new() -> Self {
        QuotientContext {
            points: Vec::new(),
            values: Vec::new(),
            ans_coeffs: Vec::new(),
            vanishing: Vec::new(),
            r_comb: e3_zero(),
        }
    }

    fn size(&self) -> usize {
        self.points.len()
    }

    fn reset(&mut self, r_comb: E3) {
        self.points.clear();
        self.values.clear();
        self.ans_coeffs.clear();
        self.vanishing.clear();
        self.r_comb = r_comb;
    }

    /// G is a set: the shift queries are sampled with replacement, so a repeated point is
    /// dropped. Prover and verifier must apply the same rule.
    fn add(&mut self, pt: E3, val: E3) {
        if self.points.contains(&pt) {
            return;
        }
        self.points.push(pt);
        self.values.push(val);
    }

    #[cfg(test)]
    fn build(&mut self) {
        self.ans_coeffs = interpolate(&self.points, &self.values);
        self.vanishing = vanishing_poly(&self.points);
    }

    /// Âns from the prover's hint: its first |G| coefficients. The caller checks the padding
    /// beyond |G| and, through `check_shake`, that the hint interpolates G.
    fn set_ans(&mut self, coeffs: &[E3]) {
        self.ans_coeffs = coeffs[..self.size()].to_vec();
        self.vanishing = vanishing_poly(&self.points);
    }

    /// The shake check (stir_math.hpp `checkShake`): with the prover's Âns and Sh, at the random ρ
    ///   Sh(ρ) = Σ_{(a,y)∈G} (Âns(ρ) − y) / (ρ − a).
    /// Sh = Σ (Âns − y)/(X − a) is a polynomial exactly when Âns(a) = y on all of G; otherwise the
    /// right-hand side is a proper rational function no polynomial agrees with on more than
    /// |G| + deg Sh points. Fails rather than dividing by zero when ρ ∈ G.
    fn check_shake(&self, shake: &[E3], rho: E3) -> bool {
        let ans_rho = eval_pol_e3(&self.ans_coeffs, rho);
        let lhs = eval_pol_e3(shake, rho);
        let mut rhs = e3_zero();
        for (a, y) in self.points.iter().zip(&self.values) {
            let den = rho - *a;
            if den == e3_zero() {
                return false;
            }
            rhs += (ans_rho - *y) * den.inverse();
        }
        lhs == rhs
    }

    ///   fᵢ(x) = ( (gᵢ(x) − Âns(x)) / ∏_{a∈G}(x − a) ) · Σ_{j=0}^{|G|} (r_comb·x)ʲ.
    /// The denominator is non-zero because G ∩ L = ∅ for every domain this runs on: the
    /// out-of-domain samples are rejected if they land in Lᵢ and the shift queries live in
    /// Lᵢ₋₁^{kᵢ₋₁}, disjoint from Lᵢ by Remark 5.3 (checked in `validate`).
    fn apply(&self, gx: E3, x: Goldilocks) -> E3 {
        let ans = eval_pol_base(&self.ans_coeffs, x);
        let denom = eval_pol_base(&self.vanishing, x);
        // geometric_sum's own division folded into this one: a single inversion.
        let y = self.r_comb * x;
        let one = e3_one();
        if y == one {
            return (gx - ans) * denom.inverse() * geometric_sum(y, self.size() as u64);
        }
        (gx - ans) * (one - y.pow(self.size() as u64 + 1)) * (denom * (one - y)).inverse()
    }
}

/// The g-th point of Lᵢ = shift·⟨ω⟩ with |Lᵢ| = 2^{log_size} (shift = 7 for every Lᵢ).
fn domain_point(shift: Goldilocks, log_size: u64, g: u64) -> Goldilocks {
    shift * Goldilocks::new(Goldilocks::W[log_size as usize]).exp_u64(g)
}

/// Whether the extension element x lies in shift·⟨ω⟩ (only base-field elements can):
/// x ∈ shift·⟨ω_n⟩ ⇔ (x / shift)^{2ⁿ} = 1.
fn domain_contains(shift: Goldilocks, log_size: u64, x: E3) -> bool {
    if x.value[1] != Goldilocks::ZERO || x.value[2] != Goldilocks::ZERO {
        return false;
    }
    let y = x.value[0] * shift.inverse();
    y.exp_power_of_2(log_size as usize) == Goldilocks::ONE
}

/// The proof-of-work check of a query message: permute [c₀, c₁, c₂, nonce, 0, …] with the
/// fixed-width grinding hash and require the first output limb below 2^{64−bits}.
fn check_grinding<GrindingHash: Hash<Goldilocks>>(challenge: &E3, nonce: u64, bits: u64) -> bool {
    if bits == 0 {
        return true;
    }
    let mut pow_state = <GrindingHash as Hash<Goldilocks>>::State::default();
    {
        let state = pow_state.as_mut();
        state[0] = challenge.value[0];
        state[1] = challenge.value[1];
        state[2] = challenge.value[2];
        state[3] = Goldilocks::new(nonce);
    }
    <GrindingHash as Hash<Goldilocks>>::hash(&mut pow_state);
    pow_state.as_ref()[0].as_canonical_u64() < 1u64 << (64 - bits)
}

// ---------------------------------------------------------------------------------------------
// The STIR argument (`STIR::verify`)
// ---------------------------------------------------------------------------------------------

/// Whether `nonce` grinds iteration i's query message (the prover's `sampleShiftQueries`).
fn check_query_grinding<GrindingHash: Hash<Goldilocks>>(params: &StirParams, i: usize, query: &E3, nonce: u64) -> bool {
    if !check_grinding::<GrindingHash>(query, nonce, params.grinding_bits_queries[i - 1]) {
        v_error!("Invalid grinding in STIR iteration {}", i);
        return false;
    }
    true
}

/// Read query q of iteration i out of T_{i−1} and recompute
///   Ansᵢ(r_shift) = Fold(fᵢ₋₁, kᵢ₋₁, r_foldᵢ₋₁)(r_shift).
///
/// The leaf holds the kᵢ₋₁ preimages of r_shift. For i = 1 (`prev_ctx` = None) those are values
/// of f_0 directly and the claim at the query's own member is handed to `check_f0`; for i ≥ 2
/// they are values of the committed gᵢ₋₁, and the virtual fᵢ₋₁ is obtained pointwise with
/// ctx[i−1] — this is where a prover that committed a gᵢ₋₁ disagreeing with Ansᵢ₋₁ produces a
/// non-low-degree fᵢ₋₁ and is caught downstream.
#[allow(clippy::too_many_arguments)]
fn fold_at_query<LeafHash, CompressionHash>(
    section: &StirSection,
    params: &StirParams,
    prev_ctx: Option<&QuotientContext>,
    i: usize,
    q: usize,
    raw: u64,
    r_fold: E3,
    check_f0: &mut dyn FnMut(usize, u64, E3) -> bool,
    proven: Option<&[u64]>,
) -> Option<E3>
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
{
    let tree = i - 1;
    let log_k = params.folding_factors[tree];
    let log_l_prev = params.log_domain_sizes[tree];
    let n_leaves = 1u64 << (log_l_prev - log_k);
    let leaf = raw % n_leaves;
    let vals = &section.cosets[tree][q];

    // `proven`: the multiproof check already bound T_{i−1}'s openings to its root; this pins its
    // leaf indices to ours.
    let opened = match proven {
        Some(ix) => ix.get(q) == Some(&leaf),
        None => verify_mt::<Goldilocks, LeafHash, CompressionHash>(
            &section.roots[tree],
            &section.last_levels[tree],
            &section.siblings[tree][q],
            leaf,
            vals,
            params.arity,
            params.last_level_verification,
        ),
    };
    if !opened {
        v_error!("Merkle verification of T_{} failed at query {}", tree, q);
        return None;
    }

    let shift = Goldilocks::new(Goldilocks::SHIFT);
    let folded = if let Some(ctx) = prev_ctx {
        // Member j of the leaf is fᵢ₋₁'s domain index leaf + j·nLeaves — `commit`'s coset layout.
        let k = 1usize << log_k;
        let mut fvals = vec![Goldilocks::ZERO; k * 3];
        for j in 0..k {
            let x = domain_point(shift, log_l_prev, leaf + j as u64 * n_leaves);
            let gx = CubicExtensionField { value: [vals[j * 3], vals[j * 3 + 1], vals[j * 3 + 2]] };
            let fx = ctx.apply(gx, x);
            fvals[j * 3..j * 3 + 3].copy_from_slice(&fx.value);
        }
        fold_coset(log_l_prev, log_k, r_fold, leaf, &fvals)
    } else {
        // f_0 is not the prover's to choose: hand the value at the query's own member (not the
        // coset representative — see stir.hpp on why) to the caller's cross-check.
        let member = (raw / n_leaves) as usize;
        let claim = CubicExtensionField { value: [vals[member * 3], vals[member * 3 + 1], vals[member * 3 + 2]] };
        if !check_f0(q, raw, claim) {
            v_error!("f_0 cross-check failed at query {}", q);
            return None;
        }
        fold_coset(log_l_prev, log_k, r_fold, leaf, vals)
    };
    Some(folded)
}

/// Fold(f, 2^{log_k}, r) at the coset of leaf `m` of L = 7·⟨ω⟩, |L| = 2^{log_l} — the C++
/// `foldCoset`. `verify_fold` with prev_bits = n_bits_ext is exactly it: the shift-squaring
/// loop degenerates and it INTTs the coset and Horner-evaluates at r·(7·ω^m)⁻¹.
fn fold_coset(log_l: u64, log_k: u64, r: E3, m: u64, coset: &[Goldilocks]) -> E3 {
    let v = verify_fold(log_l, log_l - log_k, log_l, r, m, coset);
    CubicExtensionField { value: [v[0], v[1], v[2]] }
}

/// Verify the STIR argument alone: `transcript` must be in the state the prover received it in,
/// right after the challenges that defined f_0. `check_f0` is called once per round-1 query (in
/// query order) with a uniform index of L_0 and the value T_0's leaf claims for that point;
/// returning false rejects. The STARK verifier records the claims and checks them against the
/// recomputed DEEP polynomial afterwards; a self-contained test can compare directly.
pub fn stir_verify<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
    transcript: &mut TranscriptT,
    section: &StirSection,
    params: &StirParams,
    check_f0: &mut dyn FnMut(usize, u64, E3) -> bool,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptT: TranscriptLike<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    if !params.validate() {
        return false;
    }
    let challenges = stir_transcript(
        transcript,
        params,
        &section.roots,
        &section.betas,
        &section.ans_coeffs,
        &section.shake_coeffs,
        &section.final_pol,
        &section.nonces,
    );
    stir_verify_impl::<LeafHash, CompressionHash, GrindingHash>(section, params, &challenges, check_f0, None)
}

/// [`stir_verify`] given the STIR rounds' challenges, with `proven[i]` the leaves of T_i a multiproof
/// expansion already checked.
fn stir_verify_impl<LeafHash, CompressionHash, GrindingHash>(
    section: &StirSection,
    params: &StirParams,
    challenges: &StirTranscript,
    check_f0: &mut dyn FnMut(usize, u64, E3) -> bool,
    proven: Option<&[Vec<u64>]>,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    if !params.validate() {
        return false;
    }
    if proven.is_some_and(|p| p.len() != params.m()) {
        return false;
    }
    let m = params.m();
    let shift = Goldilocks::new(Goldilocks::SHIFT);

    // With a published bottom level, the root must be the reduction of that level: check it once
    // per tree, before anything is read out of it.
    if params.last_level_verification > 0 {
        for i in 0..m {
            let n_leaves = 1u64 << (params.log_domain_sizes[i] - params.folding_factors[i]);
            let mut num_nodes_level = n_leaves;
            while num_nodes_level > params.arity.pow(params.last_level_verification as u32) {
                num_nodes_level = num_nodes_level.div_ceil(params.arity);
            }
            let computed_root = partial_merkle_tree::<Goldilocks, CompressionHash>(
                &section.last_levels[i],
                num_nodes_level,
                params.arity,
            );
            if computed_root != section.roots[i] {
                v_error!("Root of T_{} does not match its published last level", i);
                return false;
            }
        }
    }

    // ctx[i] is iteration i's (Gᵢ, Ansᵢ, r_combᵢ): what turns an opened value of the committed
    // gᵢ into the corresponding value of the virtual fᵢ. ctx[0] is unused — f_0 is committed
    // directly, not as a quotient.
    let mut ctx: Vec<QuotientContext> = (0..m).map(|_| QuotientContext::new()).collect();
    // A replay of another schedule would have the zip below skip rounds.
    if challenges.rounds.len() + 1 != m {
        return false;
    }
    let mut r_fold = challenges.r_fold0;

    // ---- Main loop: iterations i = 1, …, M−1 --------------------------------------------------
    for (i, round) in (1..m).zip(&challenges.rounds) {
        // 2(b)  r_out ← F \ Lᵢ (s = 1): one squeeze; the prover aborts if it lands in Lᵢ.
        let r_out = round.r_out;
        if domain_contains(shift, params.log_domain_sizes[i], r_out) {
            v_error!("r_out lands in L_{}", i);
            return false;
        }

        // 2(c)  the prover's β; 2(d)  r_foldᵢ, r_combᵢ, r_shift.
        let r_comb = round.r_comb;
        if !check_query_grinding::<GrindingHash>(params, i, &round.query, section.nonces[i - 1]) {
            return false;
        }
        let raw = &round.queries;

        // 2(e)  build (Gᵢ, Ansᵢ): the out-of-domain claim, plus the fold values the verifier
        //       recomputes itself from T_{i−1}. No equality is checked here — the binding is what
        //       the quotient does to the next iteration's opened values.
        let (done, rest) = ctx.split_at_mut(i);
        let cur = &mut rest[0];
        cur.reset(r_comb);
        cur.add(r_out, section.betas[i - 1]);
        let log_k = params.folding_factors[i - 1];
        let log_lprev_k = params.log_domain_sizes[i - 1] - log_k; // log |Lᵢ₋₁^{kᵢ₋₁}|
        let shift_k = shift.exp_power_of_2(log_k as usize);
        let prev_ctx = if i == 1 { None } else { Some(&done[i - 1]) };
        for (q, &raw_q) in raw.iter().enumerate() {
            let tree_proven = proven.map(|p| p[i - 1].as_slice());
            let Some(v) = fold_at_query::<LeafHash, CompressionHash>(
                section,
                params,
                prev_ctx,
                i,
                q,
                raw_q,
                r_fold,
                check_f0,
                tree_proven,
            ) else {
                return false;
            };
            let pt = embed(domain_point(shift_k, log_lprev_k, raw_q % (1u64 << log_lprev_k)));
            cur.add(pt, v);
        }
        if cur.size() as u64 >= 1u64 << params.log_degrees[i] {
            v_error!("|G| is not below d_i in STIR iteration {}", i);
            return false;
        }

        // 2(f)  Âns_i and Sh_i from the proof: zero-padded beyond |G_i| (a longer Âns_i is not the
        //       interpolant), absorbed padding included, then ρ_i is squeezed and Âns_i is checked to
        //       interpolate G_i at ρ_i alone.
        let n_g = cur.size();
        let ans = &section.ans_coeffs[i - 1];
        let sh = &section.shake_coeffs[i - 1];
        if ans.len() < n_g || sh.len() != ans.len() {
            v_error!("malformed Âns/shake hints in STIR iteration {}", i);
            return false;
        }
        if ans[n_g..].iter().chain(sh[n_g..].iter()).any(|c| *c != e3_zero()) {
            v_error!("Âns/shake hint of STIR iteration {} is not zero-padded beyond |G|", i);
            return false;
        }
        cur.set_ans(ans);
        if !cur.check_shake(sh, round.rho) {
            v_error!("Âns does not interpolate G in STIR iteration {}", i);
            return false;
        }

        r_fold = round.r_fold;
    }

    // ---- Final step ---------------------------------------------------------------------------
    // p in the clear, then the only explicit equality check of the whole protocol:
    //   Fold(f_{M−1}, k_{M−1}, r_fold_{M−1})(r_shift) = p(r_shift).
    // p arrives as d_M coefficients, so its degree bound is structural — no INTT check needed.
    if !check_query_grinding::<GrindingHash>(params, m, &challenges.final_query, section.nonces[m - 1]) {
        return false;
    }
    let raw = &challenges.final_queries;
    let log_k = params.folding_factors[m - 1];
    let log_lprev_k = params.log_domain_sizes[m - 1] - log_k;
    let shift_k = shift.exp_power_of_2(log_k as usize);
    let prev_ctx = if m == 1 { None } else { Some(&ctx[m - 1]) };
    for (q, &raw_q) in raw.iter().enumerate() {
        let tree_proven = proven.map(|p| p[m - 1].as_slice());
        let Some(v) = fold_at_query::<LeafHash, CompressionHash>(
            section,
            params,
            prev_ctx,
            m,
            q,
            raw_q,
            r_fold,
            check_f0,
            tree_proven,
        ) else {
            return false;
        };
        let x = embed(domain_point(shift_k, log_lprev_k, raw_q % (1u64 << log_lprev_k)));
        let px = eval_pol_e3(&section.final_pol, x);
        if px != v {
            v_error!("STIR final consistency check failed at query {}", q);
            return false;
        }
    }

    true
}

// ---------------------------------------------------------------------------------------------
// The whole STARK verifier for a STIR proof (`stark_verify.hpp`'s STIR path)
// ---------------------------------------------------------------------------------------------

/// The STIR counterpart of `stark_verify`, for proofs of circuits whose low-degree test is STIR.
/// Same contract: `proof` is [n_publics, publics…, flat proof], `vk` the 4-limb constant root,
/// `q_verify` the generated quotient evaluator; f_0 is recomputed from `fri_ev_groups`.
#[allow(clippy::type_complexity)]
pub fn stark_verify_stir<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
    proof: &[u64],
    vk: &[u64],
    verifier_info: &StirVerifierInfo,
    q_verify: QVerify,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptT: TranscriptLike<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    stark_verify_stir_impl::<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
        proof,
        vk,
        verifier_info,
        q_verify,
        None,
    )
}

/// [`stark_verify_stir`] of a multiproof (`wire`, as [`crate::prune_proof`] writes it): the expansion
/// checks every tree's openings against its root once, sharing the nodes the queries' paths
/// have in common, so the verifier does not hash them a second time.
pub fn verify_multiproof_stir<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
    wire: &[u64],
    vk: &[u64],
    verifier_info: &StirVerifierInfo,
    q_verify: QVerify,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptT: TranscriptLike<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    let Some((flat, proven)) =
        expand_stir_checked::<LeafHash, CompressionHash, TranscriptT>(wire, vk, &verifier_info.proof_layout())
    else {
        return false;
    };
    stark_verify_stir_impl::<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
        &flat,
        vk,
        verifier_info,
        q_verify,
        Some(&proven),
    )
}

type QVerify = fn(
    &[CubicExtensionField<Goldilocks>],
    &[CubicExtensionField<Goldilocks>],
    &[Goldilocks],
    &[CubicExtensionField<Goldilocks>],
) -> CubicExtensionField<Goldilocks>;

/// `proven`: what the multiproof check already did — each tree's leaves checked against its root,
/// and the transcript replayed from the same words this parses.
fn stark_verify_stir_impl<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
    proof: &[u64],
    vk: &[u64],
    verifier_info: &StirVerifierInfo,
    q_verify: QVerify,
    proven: Option<&Proven>,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptT: TranscriptLike<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    let leaves = proven.map(|p| p.leaves.as_slice());
    if proof.is_empty() || vk.len() < 4 {
        return false;
    }
    if !verifier_info.stir.validate() || verifier_info.stir.log_domain_sizes[0] != verifier_info.n_bits_ext {
        v_error!("STIR schedule does not match the STARK geometry");
        return false;
    }
    let params = &verifier_info.stir;

    // ---- Proof parsing: the shared prefix, kept in sync with `stark_verify`, at t₀ queries ----
    let n_siblings = merkle_proof_levels(verifier_info.n_bits_ext, params.arity, params.last_level_verification) as u64;
    let n_siblings_per_level = sibling_words(params.arity) as u64;

    let root_c = [Goldilocks::new(vk[0]), Goldilocks::new(vk[1]), Goldilocks::new(vk[2]), Goldilocks::new(vk[3])];

    let mut p: usize = 0;

    // `try_from`, not `as`: on a 32-bit target a count of 2^32 + k would truncate to k.
    let Ok(n_publics) = usize::try_from(proof[p]) else {
        return false;
    };
    p += 1;

    let Some(expected_total) =
        1usize.checked_add(n_publics).and_then(|s| s.checked_add(expected_stir_proof_size_bytes(verifier_info) / 8))
    else {
        return false;
    };
    if proof.len() != expected_total {
        return false;
    }

    // Pin the publics to one encoding: verification only ever compares field
    // elements, so `x` and `x + p` pass identically, but a caller reading the raw
    // words back as outputs would see two different values.
    let mut publics = Vec::with_capacity(n_publics);
    for i in 0..n_publics {
        let word = proof[p];
        if word >= Goldilocks::ORDER_U64 {
            v_error!("Public {i} is not a canonical Goldilocks element: {word} >= {}", Goldilocks::ORDER_U64);
            return false;
        }
        publics.push(Goldilocks::new(word));
        p += 1;
    }

    let mut roots = Vec::with_capacity(verifier_info.n_stages as usize + 1);
    for _ in 0..verifier_info.n_stages + 1 {
        let mut root = [Goldilocks::ZERO; 4];
        for r in &mut root {
            *r = Goldilocks::new(proof[p]);
            p += 1;
        }
        roots.push(root);
    }

    let mut evals = Vec::with_capacity(verifier_info.n_evals as usize);
    for _ in 0..verifier_info.n_evals {
        let eval = CubicExtensionField {
            value: [Goldilocks::new(proof[p]), Goldilocks::new(proof[p + 1]), Goldilocks::new(proof[p + 2])],
        };
        p += 3;
        evals.push(eval);
    }

    let n_queries = params.num_queries[0] as usize;
    let n_stages_plus_2 = verifier_info.n_stages as usize + 2;
    let n_sibs = n_siblings as usize;
    let n_sibs_per_lvl = n_siblings_per_level as usize;

    let mut s0_vals: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(n_queries);
    let mut s0_siblings: Vec<Vec<Vec<Goldilocks>>> = Vec::with_capacity(n_queries);
    let mut s0_last_levels: Vec<Vec<Goldilocks>> = Vec::with_capacity(n_stages_plus_2);

    for _q in 0..n_queries {
        let mut query_vals = Vec::with_capacity(n_stages_plus_2);
        let mut vals = Vec::with_capacity(verifier_info.n_constants as usize);
        for _ in 0..verifier_info.n_constants {
            vals.push(Goldilocks::new(proof[p]));
            p += 1;
        }
        query_vals.push(vals);
        s0_vals.push(query_vals);
    }

    // With `proven` the paths are never read: step over them.
    let sib_words = n_sibs * n_sibs_per_lvl;
    for _q in 0..n_queries {
        if proven.is_none() {
            let siblings = proof[p..][..sib_words].iter().map(|&x| Goldilocks::new(x)).collect();
            s0_siblings.push(vec![siblings]);
        }
        p += sib_words;
    }

    let num_nodes_level = params.arity.pow(params.last_level_verification as u32) * 4;
    let num_nodes_lvl = num_nodes_level as usize;

    if params.last_level_verification > 0 {
        let mut last_level_nodes = Vec::with_capacity(num_nodes_lvl);
        for _ in 0..num_nodes_level {
            last_level_nodes.push(Goldilocks::new(proof[p]));
            p += 1;
        }
        s0_last_levels.push(last_level_nodes);
    }

    for i in 0..verifier_info.n_stages + 1 {
        let num_vals_i = verifier_info.num_vals[i as usize] as usize;

        for query_vals in s0_vals.iter_mut() {
            let mut vals = Vec::with_capacity(num_vals_i);
            for _ in 0..num_vals_i {
                vals.push(Goldilocks::new(proof[p]));
                p += 1;
            }
            query_vals.push(vals);
        }

        for q in 0..n_queries {
            if let Some(query_siblings) = s0_siblings.get_mut(q) {
                query_siblings.push(proof[p..][..sib_words].iter().map(|&x| Goldilocks::new(x)).collect());
            }
            p += sib_words;
        }

        if params.last_level_verification > 0 {
            let mut last_level_nodes = Vec::with_capacity(num_nodes_lvl);
            for _ in 0..num_nodes_level {
                last_level_nodes.push(Goldilocks::new(proof[p]));
                p += 1;
            }
            s0_last_levels.push(last_level_nodes);
        }
    }

    if params.last_level_verification == 0 {
        // One (empty) entry per tree, so `check_query` can index them unconditionally —
        // `verify_mt` only reads a last level when lastLevelVerification > 0.
        s0_last_levels.resize(n_stages_plus_2, Vec::new());
    }

    // ---- The STIR section replaces FRI's roots/steps/final-pol/nonce tail ---------------------
    let section = parse_stir_section_impl(proof, &mut p, params, proven.is_none());

    // ---- Transcript replay: the shared prefix -------------------------------------------------
    // 2-stage AIRs with 6 stage challenges only: else a root goes unabsorbed or xi/vf1/vf2 read zero.
    if verifier_info.n_stages != 2 || verifier_info.n_challenges != 6 || verifier_info.n_challenges_total < 6 {
        v_error!(
            "verifier only supports 2-stage AIRs with 6 stage challenges, got n_stages = {}, n_challenges = {}",
            verifier_info.n_stages,
            verifier_info.n_challenges
        );
        return false;
    }
    let mut challenges = vec![e3_zero(); verifier_info.n_challenges_total as usize];
    let replayed;
    let (prefix, stir_challenges) = if let Some(p) = proven {
        (p.prefix, &p.stir)
    } else {
        let (mut transcript, prefix) = prefix_transcript::<TranscriptT>(
            &root_c,
            &publics,
            &[roots[0], roots[1], roots[2]],
            &evals,
            params.hash_commits,
        );
        replayed = stir_transcript(
            &mut transcript,
            params,
            &section.roots,
            &section.betas,
            &section.ans_coeffs,
            &section.shake_coeffs,
            &section.final_pol,
            &section.nonces,
        );
        (prefix, &replayed)
    };
    challenges[..prefix.len()].copy_from_slice(&prefix);

    // ---- The STIR rounds own the transcript from here. The round-1 shift queries come back as
    // uniform indices of L_0 and become the shared query set of the whole STARK; T_0's claims
    // about f_0 at those rows are recorded for the DEEP cross-check below.
    let mut queries_l0 = vec![0u64; n_queries];
    let mut f0_claims = vec![e3_zero(); n_queries];
    {
        let mut record_f0 = |q: usize, raw: u64, claim: E3| -> bool {
            queries_l0[q] = raw;
            f0_claims[q] = claim;
            true
        };
        let n_stage_trees = verifier_info.n_stages as usize + 2;
        if leaves.is_some_and(|l| l.len() != n_stage_trees + params.m()) {
            return false;
        }
        if !stir_verify_impl::<LeafHash, CompressionHash, GrindingHash>(
            &section,
            params,
            stir_challenges,
            &mut record_f0,
            leaves.map(|l| &l[n_stage_trees..]),
        ) {
            v_error!("STIR verification failed");
            return false;
        }
    }

    // ---- Shared machinery at the round-1 query set: kept in sync with `stark_verify` ----------
    let xi_challenge = challenges[verifier_info.n_challenges as usize - 3];

    let Some(xdivxsub) = deep_denominators_inv(
        &queries_l0,
        &verifier_info.opening_points,
        xi_challenge,
        verifier_info.n_bits,
        verifier_info.n_bits_ext,
    ) else {
        v_error!("DEEP denominator x - xi * w^o is zero");
        return false;
    };

    let (x_n, zi) = match zi_vector(xi_challenge, verifier_info.n_bits as u32, &verifier_info.boundaries) {
        Ok(z) => (z.x_n, z.zi),
        Err(e) => {
            v_error!("Zerofiers: {}", e);
            return false;
        }
    };

    v_debug!("Verifying proof");

    let deep_consts =
        DeepQueryConsts::new(&verifier_info.fri_ev_groups, verifier_info.n_challenges, &challenges, &evals);

    // The multiproof check opened the stage trees at its own round-1 indices: pin them to ours.
    if let Some(l) = leaves {
        if l[..verifier_info.n_stages as usize + 2].iter().any(|ix| *ix != queries_l0) {
            v_error!("multiproof expansion opened the stage trees at other rows");
            return false;
        }
    }

    let check_query = |q: usize| -> bool {
        // 1) Fixed MT
        if proven.is_none()
            && !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
                &root_c,
                &s0_last_levels[0],
                &s0_siblings[q][0],
                queries_l0[q],
                &s0_vals[q][0],
                params.arity,
                params.last_level_verification,
            )
        {
            v_error!("Fixed MT verification failed for query {}", q);
            return false;
        }

        // 2) stage MTs
        for (s, root) in roots.iter().enumerate().take(verifier_info.n_stages as usize + 1) {
            if proven.is_none()
                && !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
                    root,
                    &s0_last_levels[s + 1],
                    &s0_siblings[q][s + 1],
                    queries_l0[q],
                    &s0_vals[q][s + 1],
                    params.arity,
                    params.last_level_verification,
                )
            {
                v_error!("Stage MT verification failed for query {}", q);
                return false;
            }
        }

        // 3) DEEP consistency: the recomputed DEEP polynomial at this row must equal what T_0's
        //    leaf claimed for f_0 there — the STIR counterpart of FRI's s1_vals comparison.
        let deep = deep_consts.eval(&verifier_info.fri_ev_groups, &s0_vals[q], &xdivxsub[q]);
        if deep != f0_claims[q] {
            v_error!("DEEP/f_0 consistency check failed for query {}", q);
            return false;
        }

        true
    };

    let all_valid = (0..n_queries).all(check_query);

    if !all_valid {
        return false;
    }

    // The STIR trees checked their published last levels inside `stir_verify`; the stage and
    // constant trees are checked here, as in `stark_verify`.
    if params.last_level_verification > 0 {
        let mut num_nodes_level = 1u64 << verifier_info.n_bits_ext;
        while num_nodes_level > params.arity.pow(params.last_level_verification as u32) {
            num_nodes_level = num_nodes_level.div_ceil(params.arity);
        }

        for s in 0..verifier_info.n_stages + 1 {
            let computed_root = partial_merkle_tree::<Goldilocks, CompressionHash>(
                &s0_last_levels[s as usize + 1],
                num_nodes_level,
                params.arity,
            );
            for i in 0..4 {
                if computed_root[i] != roots[s as usize][i] {
                    v_error!("Stage {} Merkle tree root recomputation failed", s + 1);
                    return false;
                }
            }
        }

        let computed_root_c =
            partial_merkle_tree::<Goldilocks, CompressionHash>(&s0_last_levels[0], num_nodes_level, params.arity);

        for i in 0..4 {
            if computed_root_c[i] != root_c[i] {
                v_error!("Stage fixed Merkle tree root recomputation failed");
                return false;
            }
        }
    }

    v_debug!("Verifying Quotient polynomial");
    let mut x_acc = CubicExtensionField { value: [Goldilocks::ONE, Goldilocks::ZERO, Goldilocks::ZERO] };
    let mut q = CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] };
    for i in 0..verifier_info.q_deg {
        q += x_acc * evals[(verifier_info.q_index + i) as usize];
        x_acc *= x_n;
    }

    let q_val = q_verify(&challenges, &evals, &publics, &zi);
    if q_val != q {
        v_error!("Quotient polynomial verification failed");
        return false;
    }
    v_debug!("Quotient polynomial verification passed");
    v_debug!("Proof verification succeeded");

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(a: u64, b: u64, c: u64) -> E3 {
        CubicExtensionField { value: [Goldilocks::new(a), Goldilocks::new(b), Goldilocks::new(c)] }
    }

    #[test]
    fn geometric_sum_matches_the_naive_sum() {
        let y = e(12345, 678, 9);
        for exp in 0..7u64 {
            let mut naive = e3_zero();
            let mut pow = e3_one();
            for _ in 0..=exp {
                naive += pow;
                pow *= y;
            }
            assert_eq!(geometric_sum(y, exp), naive);
        }
        assert_eq!(geometric_sum(e3_one(), 4), embed(Goldilocks::new(5)));
    }

    #[test]
    fn interpolation_reproduces_the_values() {
        let points = [e(1, 2, 3), e(4, 5, 6), e(7, 0, 1), e(2, 9, 4)];
        let values = [e(10, 20, 30), e(1, 1, 1), e(0, 3, 7), e(5, 5, 5)];
        let coeffs = interpolate(&points, &values);
        assert_eq!(coeffs.len(), 4);
        for (p, v) in points.iter().zip(values.iter()) {
            assert_eq!(eval_pol_e3(&coeffs, *p), *v);
        }
    }

    #[test]
    fn quotient_context_dedups_and_anchors_ans() {
        let mut ctx = QuotientContext::new();
        ctx.reset(e(3, 1, 4));
        ctx.add(e(1, 0, 0), e(2, 0, 0));
        ctx.add(e(5, 0, 0), e(7, 0, 0));
        ctx.add(e(5, 0, 0), e(9, 9, 9)); // repeated point: dropped, G is a set
        assert_eq!(ctx.size(), 2);
        ctx.build();

        // Âns interpolates the kept values.
        assert_eq!(eval_pol_e3(&ctx.ans_coeffs, e(1, 0, 0)), e(2, 0, 0));
        assert_eq!(eval_pol_e3(&ctx.ans_coeffs, e(5, 0, 0)), e(7, 0, 0));

        // apply is the paper's formula: ((g(x) − Âns(x)) / ∏(x − a)) · Σ_{j=0}^{|G|} (r_comb·x)ʲ.
        let x = e(11, 0, 0);
        let gx = e(100, 50, 25);
        let ans = eval_pol_e3(&ctx.ans_coeffs, x);
        let denom = (x - e(1, 0, 0)) * (x - e(5, 0, 0));
        let expected = (gx - ans) * denom.inverse() * geometric_sum(ctx.r_comb * x, 2);
        assert_eq!(ctx.apply(gx, Goldilocks::new(11)), expected);
    }

    #[test]
    fn vanishing_poly_is_the_product_of_its_roots() {
        let points = [e(1, 2, 3), e(5, 0, 7), e(9, 4, 0), e(13, 1, 1)];
        let v = vanishing_poly(&points);
        assert_eq!(v.len(), points.len() + 1);
        for x in [Goldilocks::new(11), Goldilocks::new(1 << 40), Goldilocks::new(3)] {
            let direct = points.iter().fold(e3_one(), |acc, a| acc * (embed(x) - *a));
            assert_eq!(eval_pol_base(&v, x), direct);
        }
    }

    /// The shake polynomial of a small G, built by hand as Σ (Âns − y)/(X − a): the check must
    /// accept it and reject any other Âns, at a point that is not in G.
    #[test]
    fn shake_check_accepts_the_interpolant_and_rejects_others() {
        let mut ctx = QuotientContext::new();
        ctx.reset(e(3, 1, 4));
        ctx.add(e(1, 0, 0), e(2, 0, 0));
        ctx.add(e(5, 0, 0), e(7, 0, 0));
        ctx.add(e(9, 2, 0), e(4, 4, 1));
        ctx.build();
        let ans = ctx.ans_coeffs.clone();
        // (Âns − y)/(X − a) by synthetic division, summed over G.
        let n = ans.len();
        let mut shake = vec![e3_zero(); n - 1];
        for (a, y) in ctx.points.iter().zip(&ctx.values) {
            let mut acc = ans[n - 1];
            for j in (0..n - 1).rev() {
                shake[j] += acc;
                acc = acc * *a + ans[j];
            }
            assert_eq!(acc, *y, "Âns must interpolate G");
        }
        let rho = e(123, 456, 789);
        assert!(ctx.check_shake(&shake, rho));
        // Zero padding beyond deg Sh is harmless…
        let mut padded = shake.clone();
        padded.push(e3_zero());
        assert!(ctx.check_shake(&padded, rho));
        // …a different Âns is not.
        let mut other = ctx.ans_coeffs.clone();
        other[0] += e(1, 0, 0);
        ctx.set_ans(&other);
        assert!(!ctx.check_shake(&shake, rho));
    }

    #[test]
    fn schedule_validation_mirrors_the_paper_invariants() {
        let params = StirParams {
            folding_factors: vec![3, 3, 2],
            log_degrees: vec![12, 9, 6, 4],
            log_domain_sizes: vec![14, 13, 12, 11],
            num_queries: vec![12, 8, 6],
            grinding_bits_queries: vec![2, 2, 2],
            arity: 4,
            last_level_verification: 0,
            hash_commits: false,
        };
        assert!(params.validate());

        let mut broken = StirParams { log_degrees: vec![12, 9, 6, 5], ..params };
        assert!(!broken.validate(), "d_{{i+1}}·k_i = d_i must be enforced");
        broken.log_degrees = vec![12, 9, 6, 4];
        broken.log_domain_sizes = vec![14, 13, 12, 10];
        assert!(!broken.validate(), "|L_{{i+1}}| = |L_i|/2 must be enforced");
    }
}
