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
//  - Âns is recomputed by Newton interpolation from (G_i, Ans_i); the `ansCoeffs` hints on the
//    wire exist for the recursion circuit only and are skipped here.
//  - r_out is re-squeezed while it lands in L_i, exactly as the prover drew it. (The recursion
//    circuit instead constrains a single squeeze — a proof whose r_out needed a re-squeeze
//    verifies here but is not recursable; completeness loss 2^-128 per sample.)

use alloc::vec;
use alloc::vec::Vec;

#[allow(unused)]
use num_traits::Float;

use proofman_fields::{
    partial_merkle_tree, verify_fold, verify_mt, CubicExtensionField, Field, Goldilocks, Hash, PrimeField64, Transcript,
};

use crate::verifier::Boundary;

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
    pub stir: StirParams,
}

/// Merkle path levels one query publishes: the tree's depth less the levels the published last
/// level replaces (the GL case of the C++ `merkleProofLevels`).
fn merkle_proof_levels(n_bits: u64, arity: u64, last_level_verification: u64) -> usize {
    if n_bits == 0 {
        return 0;
    }
    let levels = ((n_bits as f64) / (arity as f64).log2()).ceil() as u64;
    levels.saturating_sub(last_level_verification) as usize
}

/// The u64 words the flat proof buffer spends on the STIR section — the GL case of the C++
/// `StarkInfo::stirProofSectionSize`, which is exactly what `Proofs::proof2pointer` writes.
pub fn stir_section_size_words(params: &StirParams) -> usize {
    let m = params.m();
    let n_sibs_per_level = ((params.arity - 1) * 4) as usize;
    let num_nodes_level = if params.last_level_verification == 0 {
        0
    } else {
        params.arity.pow(params.last_level_verification as u32) as usize
    };

    let mut size = m * 4; // roots of T_0..T_{M−1}
    for i in 0..m {
        let k = 1usize << params.folding_factors[i];
        let log_leaves = params.log_domain_sizes[i] - params.folding_factors[i];
        let n_sibs = merkle_proof_levels(log_leaves, params.arity, params.last_level_verification);
        size += params.num_queries[i] as usize * k * 3; // opened cosets
        size += params.num_queries[i] as usize * n_sibs * n_sibs_per_level; // their Merkle paths
        size += num_nodes_level * 4; // published last level
    }
    size += (m - 1) * 3; // β (s = 1)
    size += (1usize << params.log_degrees[m]) * 3; // p, in coefficients
    size += m; // one nonce per query message
    for i in 1..m {
        size += (1 + params.num_queries[i - 1] as usize) * 3; // Âns hints, zero-padded
    }
    size
}

pub fn expected_stir_proof_size_bytes(info: &StirVerifierInfo) -> usize {
    // The shared prefix, sized like `expected_proof_size_bytes` with n_queries = t₀ — every
    // stage-level consumer of a STIR proof keys on the round-1 query count.
    let log_arity = (info.stir.arity as f64).log2();
    let n_siblings =
        (((info.n_bits_ext as f64 / log_arity).ceil()) as u64).saturating_sub(info.stir.last_level_verification);
    let n_siblings_per_level = (info.stir.arity - 1) * 4;
    let n_queries = info.stir.num_queries[0];
    let num_nodes_level = info.stir.arity.pow(info.stir.last_level_verification as u32) * 4;
    let last_level_extra = if info.stir.last_level_verification > 0 { num_nodes_level } else { 0 };

    let mut p: u64 = 0;

    // roots: (n_stages + 1) groups of 4
    p += 4 * (info.n_stages as u64 + 1);

    // evals: n_evals cubic extension elements (3 each)
    p += 3 * info.n_evals;

    // s0 vals: n_queries * n_constants
    p += n_queries * info.n_constants;

    // s0 siblings: n_queries * n_siblings * n_siblings_per_level
    p += n_queries * n_siblings * n_siblings_per_level;

    // s0 last level
    p += last_level_extra;

    // stage queries: n_stages + 1 iterations
    for i in 0..(info.n_stages as u64 + 1) {
        let num_vals_i = info.num_vals[i as usize];
        p += n_queries * num_vals_i;
        p += n_queries * n_siblings * n_siblings_per_level;
        p += last_level_extra;
    }

    // The STIR section replaces FRI's roots/steps/final-pol/nonce tail entirely.
    (p as usize + stir_section_size_words(&info.stir)) * 8
}

/// The parsed STIR section of a flat proof buffer, in `proof2pointer` order.
pub struct StirSection {
    /// Roots of T_0..T_{M−1}. Unlike FRI, the first one is a commitment of its own (to f_0).
    pub roots: Vec<[Goldilocks; 4]>,
    /// cosets[i][q]: the opened coset of T_i at query q, 2^{kᵢ} extension elements flat.
    pub cosets: Vec<Vec<Vec<Goldilocks>>>,
    /// siblings[i][q]: the Merkle path of that opening.
    pub siblings: Vec<Vec<Vec<Vec<Goldilocks>>>>,
    /// last_levels[i]: the published bottom level of T_i (empty when lastLevelVerification = 0).
    pub last_levels: Vec<Vec<Goldilocks>>,
    /// β_{i,1}, i = 1..M−1 (s = 1).
    pub betas: Vec<E3>,
    /// p, as d_M coefficients.
    pub final_pol: Vec<E3>,
    /// One grinding nonce per query message.
    pub nonces: Vec<u64>,
}

/// Walk the STIR section out of `proof` starting at `*p`, which the caller has length-checked
/// against `stir_section_size_words`. The Âns coefficient hints at the end are recursion-circuit
/// material and are skipped: this verifier recomputes Âns itself.
pub fn parse_stir_section(proof: &[u64], p: &mut usize, params: &StirParams) -> StirSection {
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
    let mut siblings: Vec<Vec<Vec<Vec<Goldilocks>>>> = Vec::with_capacity(m);
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

        let mut tree_siblings = Vec::with_capacity(t);
        for _ in 0..t {
            let mut path = Vec::with_capacity(n_sibs);
            for _ in 0..n_sibs {
                let mut level = Vec::with_capacity(n_sibs_per_level);
                for _ in 0..n_sibs_per_level {
                    level.push(Goldilocks::new(proof[*p]));
                    *p += 1;
                }
                path.push(level);
            }
            tree_siblings.push(path);
        }
        siblings.push(tree_siblings);

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

    // Âns hints: on the wire for the recursion circuit, unused here.
    for i in 1..m {
        *p += (1 + params.num_queries[i - 1] as usize) * 3;
    }

    StirSection { roots, cosets, siblings, last_levels, betas, final_pol, nonces }
}

// ---------------------------------------------------------------------------------------------
// The polynomial arithmetic (`stir_math.hpp`)
// ---------------------------------------------------------------------------------------------

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
    r_comb: E3,
}

impl QuotientContext {
    fn new() -> Self {
        QuotientContext { points: Vec::new(), values: Vec::new(), ans_coeffs: Vec::new(), r_comb: e3_zero() }
    }

    fn size(&self) -> usize {
        self.points.len()
    }

    fn reset(&mut self, r_comb: E3) {
        self.points.clear();
        self.values.clear();
        self.ans_coeffs.clear();
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

    fn build(&mut self) {
        self.ans_coeffs = interpolate(&self.points, &self.values);
    }

    ///   fᵢ(x) = ( (gᵢ(x) − Âns(x)) / ∏_{a∈G}(x − a) ) · Σ_{j=0}^{|G|} (r_comb·x)ʲ.
    /// The denominator is non-zero because G ∩ L = ∅ for every domain this runs on: the
    /// out-of-domain samples are re-squeezed out of Lᵢ and the shift queries live in
    /// Lᵢ₋₁^{kᵢ₋₁}, disjoint from Lᵢ by Remark 5.3 (checked in `validate`).
    fn apply(&self, gx: E3, x: E3) -> E3 {
        let ans = eval_pol_e3(&self.ans_coeffs, x);
        let mut denom = e3_one();
        for a in &self.points {
            denom *= x - *a;
        }
        let geo = geometric_sum(self.r_comb * x, self.size() as u64);
        (gx - ans) * denom.inverse() * geo
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

/// Re-derive iteration i's shift queries as uniform indices of Lᵢ₋₁ — the prover's
/// `sampleShiftQueries`, with the grinding checked instead of searched.
fn derive_shift_queries<TranscriptHash, GrindingHash>(
    transcript: &mut Transcript<Goldilocks, TranscriptHash>,
    params: &StirParams,
    i: usize,
    nonce: u64,
) -> Option<Vec<u64>>
where
    TranscriptHash: Hash<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    let mut c = e3_zero();
    transcript.get_field(&mut c.value);
    if !check_grinding::<GrindingHash>(&c, nonce, params.grinding_bits_queries[i - 1]) {
        v_error!("Invalid grinding in STIR iteration {}", i);
        return None;
    }
    let mut transcript_queries: Transcript<Goldilocks, TranscriptHash> = Transcript::new();
    transcript_queries.put(&c.value);
    transcript_queries.put(&[Goldilocks::new(nonce)]);
    Some(transcript_queries.get_permutations(params.num_queries[i - 1], params.log_domain_sizes[i - 1]))
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

    if !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
        &section.roots[tree],
        &section.last_levels[tree],
        &section.siblings[tree][q],
        leaf,
        vals,
        params.arity,
        params.last_level_verification,
    ) {
        v_error!("Merkle verification of T_{} failed at query {}", tree, q);
        return None;
    }

    let shift = Goldilocks::new(Goldilocks::SHIFT);
    let folded = if let Some(ctx) = prev_ctx {
        // Member j of the leaf is fᵢ₋₁'s domain index leaf + j·nLeaves — `commit`'s coset layout.
        let k = 1usize << log_k;
        let mut fvals = vec![Goldilocks::ZERO; k * 3];
        for j in 0..k {
            let x = embed(domain_point(shift, log_l_prev, leaf + j as u64 * n_leaves));
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
pub fn stir_verify<LeafHash, CompressionHash, TranscriptHash, GrindingHash>(
    transcript: &mut Transcript<Goldilocks, TranscriptHash>,
    section: &StirSection,
    params: &StirParams,
    check_f0: &mut dyn FnMut(usize, u64, E3) -> bool,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptHash: Hash<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    if !params.validate() {
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

    // ---- Initial commitment and the first folding challenge ----------------------------------
    transcript.put(&section.roots[0]);
    let mut r_fold = e3_zero();
    transcript.get_field(&mut r_fold.value);

    // ---- Main loop: iterations i = 1, …, M−1 --------------------------------------------------
    for i in 1..m {
        transcript.put(&section.roots[i]);

        // 2(b)  r_out ← F \ Lᵢ (s = 1), drawn exactly as the prover drew it.
        let mut r_out = e3_zero();
        loop {
            transcript.get_field(&mut r_out.value);
            if !domain_contains(shift, params.log_domain_sizes[i], r_out) {
                break;
            }
        }

        // 2(c)  the prover's β.
        transcript.put(&section.betas[i - 1].value);

        // 2(d)  r_foldᵢ, r_combᵢ, r_shift.
        let mut r_fold_next = e3_zero();
        transcript.get_field(&mut r_fold_next.value);
        let mut r_comb = e3_zero();
        transcript.get_field(&mut r_comb.value);
        let Some(raw) =
            derive_shift_queries::<TranscriptHash, GrindingHash>(transcript, params, i, section.nonces[i - 1])
        else {
            return false;
        };

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
            let Some(v) =
                fold_at_query::<LeafHash, CompressionHash>(section, params, prev_ctx, i, q, raw_q, r_fold, check_f0)
            else {
                return false;
            };
            let pt = embed(domain_point(shift_k, log_lprev_k, raw_q % (1u64 << log_lprev_k)));
            cur.add(pt, v);
        }
        if cur.size() as u64 >= 1u64 << params.log_degrees[i] {
            v_error!("|G| is not below d_i in STIR iteration {}", i);
            return false;
        }
        cur.build();

        r_fold = r_fold_next;
    }

    // ---- Final step ---------------------------------------------------------------------------
    // p in the clear, then the only explicit equality check of the whole protocol:
    //   Fold(f_{M−1}, k_{M−1}, r_fold_{M−1})(r_shift) = p(r_shift).
    // p arrives as d_M coefficients, so its degree bound is structural — no INTT check needed.
    if !params.hash_commits {
        for coeff in &section.final_pol {
            transcript.put(&coeff.value);
        }
    } else {
        let mut transcript_final_pol: Transcript<Goldilocks, TranscriptHash> = Transcript::new();
        for coeff in &section.final_pol {
            transcript_final_pol.put(&coeff.value);
        }
        let hash = transcript_final_pol.get_state();
        transcript.put(&hash[0..4]);
    }

    let Some(raw) = derive_shift_queries::<TranscriptHash, GrindingHash>(transcript, params, m, section.nonces[m - 1])
    else {
        return false;
    };
    let log_k = params.folding_factors[m - 1];
    let log_lprev_k = params.log_domain_sizes[m - 1] - log_k;
    let shift_k = shift.exp_power_of_2(log_k as usize);
    let prev_ctx = if m == 1 { None } else { Some(&ctx[m - 1]) };
    for (q, &raw_q) in raw.iter().enumerate() {
        let Some(v) =
            fold_at_query::<LeafHash, CompressionHash>(section, params, prev_ctx, m, q, raw_q, r_fold, check_f0)
        else {
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
/// and the two function pointers are the generated straight-line evaluators (they are
/// LDT-agnostic — identical to the ones a FRI verifier of the same circuit would bake in).
#[allow(clippy::type_complexity)]
pub fn stark_verify_stir<LeafHash, CompressionHash, TranscriptHash, GrindingHash>(
    proof: &[u64],
    vk: &[u64],
    verifier_info: &StirVerifierInfo,
    q_verify: fn(
        &[CubicExtensionField<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
        &[Goldilocks],
        &[CubicExtensionField<Goldilocks>],
    ) -> CubicExtensionField<Goldilocks>,
    queries_verify: fn(
        &[CubicExtensionField<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
        &[Vec<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
    ) -> CubicExtensionField<Goldilocks>,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptHash: Hash<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    if proof.is_empty() || vk.len() < 4 {
        return false;
    }
    if !verifier_info.stir.validate() || verifier_info.stir.log_domain_sizes[0] != verifier_info.n_bits_ext {
        v_error!("STIR schedule does not match the STARK geometry");
        return false;
    }
    let params = &verifier_info.stir;

    // ---- Proof parsing: the shared prefix, kept in sync with `stark_verify`, at t₀ queries ----
    let n_siblings: u64 = (((verifier_info.n_bits_ext as f64 / (params.arity as f64).log2()).ceil()) as u64)
        .saturating_sub(params.last_level_verification);
    let n_siblings_per_level = (params.arity - 1) * 4;

    let root_c = [Goldilocks::new(vk[0]), Goldilocks::new(vk[1]), Goldilocks::new(vk[2]), Goldilocks::new(vk[3])];

    let mut p: usize = 0;

    let n_publics = proof[p];
    p += 1;

    let Some(expected_total) = 1usize
        .checked_add(n_publics as usize)
        .and_then(|s| s.checked_add(expected_stir_proof_size_bytes(verifier_info) / 8))
    else {
        return false;
    };
    if proof.len() != expected_total {
        return false;
    }

    // Pin the publics to one encoding: verification only ever compares field
    // elements, so `x` and `x + p` pass identically, but a caller reading the raw
    // words back as outputs would see two different values.
    let mut publics = Vec::with_capacity(n_publics as usize);
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
    let mut s0_siblings: Vec<Vec<Vec<Vec<Goldilocks>>>> = Vec::with_capacity(n_queries);
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

    for _q in 0..n_queries {
        let mut query_siblings = Vec::with_capacity(n_stages_plus_2);
        let mut siblings = Vec::with_capacity(n_sibs);
        for _ in 0..n_sibs {
            let mut sibling = Vec::with_capacity(n_sibs_per_lvl);
            for _ in 0..n_sibs_per_lvl {
                sibling.push(Goldilocks::new(proof[p]));
                p += 1;
            }
            siblings.push(sibling);
        }
        query_siblings.push(siblings);
        s0_siblings.push(query_siblings);
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

        for query_siblings in s0_siblings.iter_mut() {
            let mut siblings = Vec::with_capacity(n_sibs);
            for _ in 0..n_sibs {
                let mut sibling = Vec::with_capacity(n_sibs_per_lvl);
                for _ in 0..n_sibs_per_lvl {
                    sibling.push(Goldilocks::new(proof[p]));
                    p += 1;
                }
                siblings.push(sibling);
            }
            query_siblings.push(siblings);
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
    let section = parse_stir_section(proof, &mut p, params);

    // ---- Transcript replay: the shared prefix, kept in sync with `stark_verify` ---------------
    let mut challenges = vec![e3_zero(); verifier_info.n_challenges_total as usize];

    let mut transcript: Transcript<Goldilocks, TranscriptHash> = Transcript::<Goldilocks, TranscriptHash>::new();
    transcript.put(&root_c);
    if n_publics > 0 {
        if !params.hash_commits {
            transcript.put(&publics);
        } else {
            let mut transcript_publics: Transcript<Goldilocks, TranscriptHash> =
                Transcript::<Goldilocks, TranscriptHash>::new();
            transcript_publics.put(&publics);
            let hash = transcript_publics.get_state();
            transcript.put(&hash[0..4]);
        }
    }
    transcript.put(&roots[0]);
    transcript.get_field(&mut challenges[0].value);
    transcript.get_field(&mut challenges[1].value);

    transcript.put(&roots[1]);
    transcript.get_field(&mut challenges[2].value);
    transcript.put(&roots[2]);

    transcript.get_field(&mut challenges[3].value);

    if !params.hash_commits {
        for i in 0..verifier_info.n_evals {
            transcript.put(&evals[i as usize].value);
        }
    } else {
        let mut transcript_evals: Transcript<Goldilocks, TranscriptHash> =
            Transcript::<Goldilocks, TranscriptHash>::new();
        for i in 0..verifier_info.n_evals {
            transcript_evals.put(&evals[i as usize].value);
        }
        let hash = transcript_evals.get_state();
        transcript.put(&hash[0..4]);
    }

    transcript.get_field(&mut challenges[4].value);
    transcript.get_field(&mut challenges[5].value);

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
        if !stir_verify::<LeafHash, CompressionHash, TranscriptHash, GrindingHash>(
            &mut transcript,
            &section,
            params,
            &mut record_f0,
        ) {
            v_error!("STIR verification failed");
            return false;
        }
    }

    // ---- Shared machinery at the round-1 query set: kept in sync with `stark_verify` ----------
    let xi_challenge = challenges[verifier_info.n_challenges as usize - 3];

    let w_ext = Goldilocks::new(Goldilocks::W[verifier_info.n_bits_ext as usize]);
    let w_bits = Goldilocks::new(Goldilocks::W[verifier_info.n_bits as usize]);
    let n_opening_points = verifier_info.opening_points.len();

    let mut xdivxsub: Vec<Vec<CubicExtensionField<Goldilocks>>> = Vec::with_capacity(n_queries);
    for &query in queries_l0.iter() {
        let mut query_xdivxsub = Vec::with_capacity(n_opening_points);
        let x = CubicExtensionField {
            value: [Goldilocks::new(Goldilocks::SHIFT) * w_ext.exp_u64(query), Goldilocks::ZERO, Goldilocks::ZERO],
        };
        for o in 0..n_opening_points {
            let mut wi = Goldilocks::ONE;
            let abs_opening = verifier_info.opening_points[o].unsigned_abs();
            for _ in 0..abs_opening {
                wi *= w_bits;
            }

            if verifier_info.opening_points[o] < 0 {
                wi = wi.inverse();
            }

            query_xdivxsub.push((x - (xi_challenge * wi)).inverse());
        }
        xdivxsub.push(query_xdivxsub);
    }

    let x_n = xi_challenge.pow(1 << verifier_info.n_bits);

    let z_n = x_n - Goldilocks::ONE;
    let z_n_inv = z_n.inverse();
    let mut zi = Vec::with_capacity(verifier_info.boundaries.len() + 1);
    zi.push(z_n_inv);
    for boundary in &verifier_info.boundaries {
        if boundary.name == "everyRow" {
            continue;
        }

        // Handling for boundaries other than "everyRow" is intentionally deferred.
        // If support for additional boundary types is required, implement logic here.
    }

    v_debug!("Verifying proof");

    let check_query = |q: usize| -> bool {
        // 1) Fixed MT
        if !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
            &root_c,
            &s0_last_levels[0],
            &s0_siblings[q][0],
            queries_l0[q],
            &s0_vals[q][0],
            params.arity,
            params.last_level_verification,
        ) {
            v_error!("Fixed MT verification failed for query {}", q);
            return false;
        }

        // 2) stage MTs
        for (s, root) in roots.iter().enumerate().take(verifier_info.n_stages as usize + 1) {
            if !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
                root,
                &s0_last_levels[s + 1],
                &s0_siblings[q][s + 1],
                queries_l0[q],
                &s0_vals[q][s + 1],
                params.arity,
                params.last_level_verification,
            ) {
                v_error!("Stage MT verification failed for query {}", q);
                return false;
            }
        }

        // 3) DEEP consistency: the recomputed DEEP polynomial at this row must equal what T_0's
        //    leaf claimed for f_0 there — the STIR counterpart of FRI's s1_vals comparison.
        let deep = queries_verify(&challenges, &evals, &s0_vals[q], &xdivxsub[q]);
        if deep != f0_claims[q] {
            v_error!("DEEP/f_0 consistency check failed for query {}", q);
            return false;
        }

        true
    };

    #[cfg(feature = "parallel")]
    let all_valid = {
        use rayon::prelude::*;
        (0..n_queries).into_par_iter().all(check_query)
    };
    #[cfg(not(feature = "parallel"))]
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
                naive = naive + pow;
                pow = pow * y;
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
        let x = e(11, 3, 2);
        let gx = e(100, 50, 25);
        let ans = eval_pol_e3(&ctx.ans_coeffs, x);
        let denom = (x - e(1, 0, 0)) * (x - e(5, 0, 0));
        let expected = (gx - ans) * denom.inverse() * geometric_sum(ctx.r_comb * x, 2);
        assert_eq!(ctx.apply(gx, x), expected);
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
