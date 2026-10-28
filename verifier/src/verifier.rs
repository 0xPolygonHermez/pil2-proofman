use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[allow(unused)]
use num_traits::Float;

use crate::layout::{merkle_proof_levels, sibling_words};
use crate::replay::{fri_transcript, prefix_transcript};
use proofman_fields::{
    cubic_mul_acc, intt_tiny, DotAcc, partial_merkle_tree, verify_fold, verify_mt, CubicExtensionField, Field,
    Goldilocks, Hash, PrimeField64, TranscriptLike,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Boundary {
    pub name: String,
    pub offset_min: Option<u64>,
    pub offset_max: Option<u64>,
}

pub struct VerifierInfo {
    pub n_stages: u32,
    pub n_constants: u64,
    pub n_evals: u64,
    pub n_bits: u64,
    pub n_bits_ext: u64,
    pub arity: u64,
    pub n_fri_queries: u64,
    pub n_fri_steps: u64,
    pub n_challenges: u64,
    pub n_challenges_total: u64,
    pub fri_steps: Vec<u64>,
    pub hash_commits: bool,
    pub num_vals: Vec<u64>,
    pub opening_points: Vec<i64>,
    pub boundaries: Vec<Boundary>,
    pub q_deg: u64,
    pub q_index: u64,
    pub last_level_verification: u64,
    pub pow_bits: u64,
    pub fri_ev_groups: Vec<FriEvalGroup>,
}

/// The flat proof's size in bytes, publics excluded.
pub fn expected_proof_size_bytes(info: &VerifierInfo) -> usize {
    info.proof_layout().flat_proof_words() * 8
}

/// One opened column that the FRI polynomial reads at a query point.
#[derive(Debug, Clone)]
pub struct FriEvalRef {
    /// 0 for constants, `stage` for a witness column, `n_stages + 1 + commit_id` for a custom one.
    pub bucket: u16,
    pub offset: u16,
    pub dim: u8,
    /// vf2's exponent: the polynomial's first appearance in the evaluation map.
    pub col: u16,
}

impl FriEvalRef {
    pub const fn new(bucket: u16, offset: u16, dim: u8, col: u16) -> Self {
        Self { bucket, offset, dim, col }
    }
}

/// The columns opened at one opening point, in evaluation-map order.
#[derive(Debug, Clone)]
pub struct FriEvalGroup {
    pub opening: u16,
    pub refs: Vec<FriEvalRef>,
}

/// Evaluates the FRI polynomial at one query point, mirroring the setup's
/// `fri_poly.rs`: terms weighted by vf2^col, scaled by `xDivXSubXi`, Horner on `vf1` across openings.
pub fn fri_query_verify(
    info: &VerifierInfo,
    challenges: &[CubicExtensionField<Goldilocks>],
    evals: &[CubicExtensionField<Goldilocks>],
    vals: &[Vec<Goldilocks>],
    xdivxsub: &[CubicExtensionField<Goldilocks>],
) -> CubicExtensionField<Goldilocks> {
    deep_query_eval(&info.fri_ev_groups, info.n_challenges, challenges, evals, vals, xdivxsub)
}

/// `1 / (x_q - xi * w^o)` for every query `q` and opening point `o`, one row per query. The
/// `xi * w^o` terms do not depend on the query, and the whole table shares one inversion
/// (Montgomery's trick). `None` if any denominator is zero.
pub(crate) fn deep_denominators_inv(
    queries: &[u64],
    opening_points: &[i64],
    xi: CubicExtensionField<Goldilocks>,
    n_bits: u64,
    n_bits_ext: u64,
) -> Option<Vec<Vec<CubicExtensionField<Goldilocks>>>> {
    let w_ext = Goldilocks::new(Goldilocks::W[n_bits_ext as usize]);
    let w_bits = Goldilocks::new(Goldilocks::W[n_bits as usize]);
    let xi_wi: Vec<CubicExtensionField<Goldilocks>> = opening_points
        .iter()
        // w has order 2^n_bits, so its exponents work mod 2^n_bits: a negative o needs no inversion.
        .map(|&o| xi * w_bits.exp_u64(o as u64 & ((1u64 << n_bits) - 1)))
        .collect();

    let n_open = opening_points.len();
    if n_open == 0 {
        return Some(vec![Vec::new(); queries.len()]);
    }
    let mut den = Vec::with_capacity(queries.len() * n_open);
    for &q in queries {
        let x = Goldilocks::new(Goldilocks::SHIFT) * w_ext.exp_u64(q);
        den.extend(xi_wi.iter().map(|&s| CubicExtensionField { value: [x, Goldilocks::ZERO, Goldilocks::ZERO] } - s));
    }

    // prefix[i] = den[0] * ... * den[i-1]
    let mut prefix = Vec::with_capacity(den.len());
    let mut acc = CubicExtensionField::<Goldilocks>::one();
    for &d in &den {
        prefix.push(acc);
        acc *= d;
    }
    if acc.is_zero() {
        return None;
    }
    let mut inv = acc.inverse();
    for i in (0..den.len()).rev() {
        let d = den[i];
        den[i] = inv * prefix[i];
        inv *= d;
    }
    Some(den.chunks(n_open).map(|c| c.to_vec()).collect())
}

/// The DEEP polynomial at one query point from its `FriEvalGroup` table: FRI's first-layer value
/// and STIR's f_0 alike (both are the setup's `deepExpId`).
pub fn deep_query_eval(
    fri_ev_groups: &[FriEvalGroup],
    n_challenges: u64,
    challenges: &[CubicExtensionField<Goldilocks>],
    evals: &[CubicExtensionField<Goldilocks>],
    vals: &[Vec<Goldilocks>],
    xdivxsub: &[CubicExtensionField<Goldilocks>],
) -> CubicExtensionField<Goldilocks> {
    let consts = DeepQueryConsts::new(fri_ev_groups, n_challenges, challenges, evals);
    consts.eval(fri_ev_groups, vals, xdivxsub)
}

/// The query-independent half of [`deep_query_eval`], built once per proof.
pub(crate) struct DeepQueryConsts {
    vf1: CubicExtensionField<Goldilocks>,
    vf2_pow: Vec<CubicExtensionField<Goldilocks>>,
    /// Per group, `sum(eval * vf2^col)`: the evals are the same at every query.
    group_evals: Vec<CubicExtensionField<Goldilocks>>,
}

impl DeepQueryConsts {
    pub(crate) fn new(
        fri_ev_groups: &[FriEvalGroup],
        n_challenges: u64,
        challenges: &[CubicExtensionField<Goldilocks>],
        evals: &[CubicExtensionField<Goldilocks>],
    ) -> Self {
        let vf1 = challenges[n_challenges as usize - 2];
        let vf2 = challenges[n_challenges as usize - 1];

        let n_cols = fri_ev_groups.iter().flat_map(|g| g.refs.iter()).map(|e| e.col as usize + 1).max().unwrap_or(0);
        let mut vf2_pow = Vec::with_capacity(n_cols);
        let mut p = CubicExtensionField::<Goldilocks>::one();
        for _ in 0..n_cols {
            vf2_pow.push(p);
            p *= vf2;
        }

        // ev_map is sorted by opening point, so the groups walk `evals` in order.
        let mut eval_idx = 0;
        let group_evals = fri_ev_groups
            .iter()
            .map(|group| {
                let mut acc = CubicExtensionField::<Goldilocks>::zero();
                for e in &group.refs {
                    acc += evals[eval_idx] * vf2_pow[e.col as usize];
                    eval_idx += 1;
                }
                acc
            })
            .collect();

        Self { vf1, vf2_pow, group_evals }
    }

    /// `sum((col - eval) * vf2^col)` split as `sum(col * vf2^col) - group_evals`, so a base-field
    /// column costs a base-by-extension product instead of a full extension one.
    pub(crate) fn eval(
        &self,
        fri_ev_groups: &[FriEvalGroup],
        vals: &[Vec<Goldilocks>],
        xdivxsub: &[CubicExtensionField<Goldilocks>],
    ) -> CubicExtensionField<Goldilocks> {
        let mut fri = CubicExtensionField::<Goldilocks>::zero();
        for (g, group) in fri_ev_groups.iter().enumerate() {
            // One reduction per coordinate per group, not per term.
            let mut acc = [DotAcc::default(); 3];
            for e in &group.refs {
                let leaf = &vals[e.bucket as usize];
                let off = e.offset as usize;
                let p = self.vf2_pow[e.col as usize].value;
                if e.dim == 1 {
                    for (acc, p) in acc.iter_mut().zip(p) {
                        acc.add(leaf[off], p);
                    }
                } else {
                    cubic_mul_acc(&mut acc, &[leaf[off], leaf[off + 1], leaf[off + 2]], &p);
                }
            }
            let acc = CubicExtensionField { value: [acc[0].finish(), acc[1].finish(), acc[2].finish()] };
            let scaled = (acc - self.group_evals[g]) * xdivxsub[group.opening as usize];
            fri = if g == 0 { scaled } else { fri * self.vf1 + scaled };
        }
        fri
    }
}

#[allow(clippy::type_complexity)]
pub fn stark_verify<LeafHash, CompressionHash, TranscriptT, GrindingHash>(
    proof: &[u64],
    vk: &[u64],
    verifier_info: &VerifierInfo,
    q_verify: fn(
        &[CubicExtensionField<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
        &[Goldilocks],
        &[CubicExtensionField<Goldilocks>],
    ) -> CubicExtensionField<Goldilocks>,
) -> bool
where
    LeafHash: Hash<Goldilocks>,
    CompressionHash: Hash<Goldilocks>,
    TranscriptT: TranscriptLike<Goldilocks>,
    GrindingHash: Hash<Goldilocks>,
{
    if proof.is_empty() || vk.len() < 4 {
        return false;
    }
    // The size check reads the step list, the parser the step count.
    if verifier_info.fri_steps.len() as u64 != verifier_info.n_fri_steps || verifier_info.fri_steps.is_empty() {
        v_error!(
            "verifier_info is inconsistent: {} FRI steps listed, {} declared",
            verifier_info.fri_steps.len(),
            verifier_info.n_fri_steps
        );
        return false;
    }

    let n_siblings =
        merkle_proof_levels(verifier_info.n_bits_ext, verifier_info.arity, verifier_info.last_level_verification)
            as u64;
    let n_siblings_per_level = sibling_words(verifier_info.arity) as u64;

    let root_c = [Goldilocks::new(vk[0]), Goldilocks::new(vk[1]), Goldilocks::new(vk[2]), Goldilocks::new(vk[3])];

    let mut p = 0;

    // `try_from`, not `as`: on a 32-bit target a count of 2^32 + k would truncate to k.
    let Ok(n_publics) = usize::try_from(proof[p as usize]) else {
        return false;
    };
    p += 1;

    let Some(expected_total) =
        1usize.checked_add(n_publics).and_then(|s| s.checked_add(expected_proof_size_bytes(verifier_info) / 8))
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
        let word = proof[p as usize];
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
            *r = Goldilocks::new(proof[p as usize]);
            p += 1;
        }
        roots.push(root);
    }

    let mut evals = Vec::with_capacity(verifier_info.n_evals as usize);
    for _ in 0..verifier_info.n_evals {
        let eval = CubicExtensionField {
            value: [
                Goldilocks::new(proof[p as usize]),
                Goldilocks::new(proof[p as usize + 1]),
                Goldilocks::new(proof[p as usize + 2]),
            ],
        };
        p += 3;
        evals.push(eval);
    }

    let n_queries = verifier_info.n_fri_queries as usize;
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
            vals.push(Goldilocks::new(proof[p as usize]));
            p += 1;
        }
        query_vals.push(vals);
        s0_vals.push(query_vals);
    }

    for _q in 0..n_queries {
        let mut query_siblings = Vec::with_capacity(n_stages_plus_2);
        let siblings: Vec<Goldilocks> =
            proof[p as usize..][..n_sibs * n_sibs_per_lvl].iter().map(|&x| Goldilocks::new(x)).collect();
        p += (n_sibs * n_sibs_per_lvl) as u64;
        query_siblings.push(siblings);
        s0_siblings.push(query_siblings);
    }

    let num_nodes_level = verifier_info.arity.pow(verifier_info.last_level_verification as u32) * 4;
    let num_nodes_lvl = num_nodes_level as usize;

    if verifier_info.last_level_verification > 0 {
        let mut last_level_nodes = Vec::with_capacity(num_nodes_lvl);
        for _ in 0..num_nodes_level {
            last_level_nodes.push(Goldilocks::new(proof[p as usize]));
            p += 1;
        }
        s0_last_levels.push(last_level_nodes);
    }

    for i in 0..verifier_info.n_stages + 1 {
        let num_vals_i = verifier_info.num_vals[i as usize] as usize;

        for query_vals in s0_vals.iter_mut() {
            let mut vals = Vec::with_capacity(num_vals_i);
            for _ in 0..num_vals_i {
                vals.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            query_vals.push(vals);
        }

        for query_siblings in s0_siblings.iter_mut() {
            let siblings: Vec<Goldilocks> =
                proof[p as usize..][..n_sibs * n_sibs_per_lvl].iter().map(|&x| Goldilocks::new(x)).collect();
            p += (n_sibs * n_sibs_per_lvl) as u64;
            query_siblings.push(siblings);
        }

        if verifier_info.last_level_verification > 0 {
            let mut last_level_nodes = Vec::with_capacity(num_nodes_lvl);
            for _ in 0..num_nodes_level {
                last_level_nodes.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            s0_last_levels.push(last_level_nodes);
        }
    }

    if verifier_info.last_level_verification == 0 {
        // `check_query` indexes a last level per tree even when there is none.
        s0_last_levels.resize(n_stages_plus_2, Vec::new());
    }

    let n_fri_steps_minus_1 = (verifier_info.n_fri_steps - 1) as usize;
    let mut roots_fri = Vec::with_capacity(n_fri_steps_minus_1);
    for _ in 1..verifier_info.n_fri_steps {
        let mut root = [Goldilocks::ZERO; 4];
        for r in &mut root {
            *r = Goldilocks::new(proof[p as usize]);
            p += 1;
        }
        roots_fri.push(root);
    }

    let mut siblings_fri: Vec<Vec<Vec<Goldilocks>>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_fri_steps_minus_1)).collect();
    let mut vals_fri: Vec<Vec<Vec<Goldilocks>>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_fri_steps_minus_1)).collect();
    let mut last_levels_fri: Vec<Vec<Goldilocks>> = Vec::with_capacity(n_fri_steps_minus_1);

    let n_siblings_per_level_fri = ((verifier_info.arity - 1) * 4) as usize;

    for i in 1..verifier_info.n_fri_steps {
        let vals_size =
            ((1 << (verifier_info.fri_steps[(i - 1) as usize] - verifier_info.fri_steps[i as usize])) * 3) as usize;

        for val_fri in vals_fri.iter_mut().take(n_queries) {
            let mut vals = Vec::with_capacity(vals_size);
            for _ in 0..vals_size {
                vals.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            val_fri.push(vals);
        }

        let n_siblings_fri = merkle_proof_levels(
            verifier_info.fri_steps[i as usize],
            verifier_info.arity,
            verifier_info.last_level_verification,
        );

        for query_siblings in siblings_fri.iter_mut() {
            let n = n_siblings_fri * n_siblings_per_level_fri;
            let siblings: Vec<Goldilocks> = proof[p as usize..][..n].iter().map(|&x| Goldilocks::new(x)).collect();
            p += n as u64;
            query_siblings.push(siblings);
        }

        if verifier_info.last_level_verification > 0 {
            let mut last_level_nodes = Vec::with_capacity(num_nodes_lvl);
            for _ in 0..num_nodes_level {
                last_level_nodes.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            last_levels_fri.push(last_level_nodes);
        }
    }

    if verifier_info.last_level_verification == 0 {
        last_levels_fri.resize(n_fri_steps_minus_1, Vec::new());
    }

    let final_pol_capacity = 1usize << verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize];
    let mut final_pol = Vec::with_capacity(final_pol_capacity);
    for _ in 0..final_pol_capacity {
        let pol = CubicExtensionField {
            value: [
                Goldilocks::new(proof[p as usize]),
                Goldilocks::new(proof[p as usize + 1]),
                Goldilocks::new(proof[p as usize + 2]),
            ],
        };
        p += 3;
        final_pol.push(pol);
    }

    let nonce = Goldilocks::new(proof[p as usize]);

    // Everything below is hardcoded for a 2-stage AIR (root1, root2, rootQ) and challenge
    // slots 0..=5: a different stage count would leave the Q root unabsorbed (forgeable) or
    // index `roots` out of bounds. Reject rather than mis-verify.
    if verifier_info.n_stages != 2 {
        v_error!("verifier only supports 2-stage AIRs, got n_stages = {}", verifier_info.n_stages);
        return false;
    }

    let mut challenges = vec![
        CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] };
        verifier_info.n_challenges_total as usize
    ];

    let mut zi = Vec::with_capacity(verifier_info.boundaries.len() + 1);

    let (mut transcript, prefix) = prefix_transcript::<TranscriptT>(
        &root_c,
        &publics,
        &[roots[0], roots[1], roots[2]],
        &evals,
        verifier_info.hash_commits,
    );
    let fri = fri_transcript(
        &mut transcript,
        &roots_fri,
        &final_pol,
        nonce,
        &verifier_info.fri_steps,
        verifier_info.n_fri_queries,
        verifier_info.hash_commits,
    );

    // Every declared slot is written (6 stage challenges, the folds, the query challenge): an unwritten
    // one stays zero, which the prover can predict.
    let n_written = prefix.len() + fri.folds.len() + 1;
    if verifier_info.n_challenges as usize != prefix.len() || n_written != challenges.len() {
        v_error!(
            "verifier_info is inconsistent: the transcript writes {} challenges but {} are declared",
            n_written,
            challenges.len()
        );
        return false;
    }
    let last_challenge_index = challenges.len() - 1;
    challenges[..prefix.len()].copy_from_slice(&prefix);
    challenges[prefix.len()..last_challenge_index].copy_from_slice(&fri.folds);
    challenges[last_challenge_index] = fri.query;
    // Proof-of-work grinding hash. Uses its own fixed-width hash (Poseidon2_8 for
    // Poseidon2, Poseidon1_8 for Poseidon1), independent of the transcript/Merkle
    // widths. The input `[c0, c1, c2, nonce]` is written into a zero-padded state
    // of the grinding hash's width and permuted in place; the check reads cell 0.
    let mut pow_state = <GrindingHash as Hash<Goldilocks>>::State::default();
    {
        let state = pow_state.as_mut();
        state[0] = fri.query.value[0];
        state[1] = fri.query.value[1];
        state[2] = fri.query.value[2];
        state[3] = nonce;
    }
    <GrindingHash as Hash<Goldilocks>>::hash(&mut pow_state);
    // At 0 bits every nonce passes (and the shift below would overflow).
    if verifier_info.pow_bits > 0 && pow_state.as_ref()[0].as_canonical_u64() >= 1 << (64 - verifier_info.pow_bits) {
        v_error!("Proof of work verification failed");
        return false;
    }
    let fri_queries = fri.queries;

    let xi_challenge = challenges[verifier_info.n_challenges as usize - 3];

    let xdivxsub = match deep_denominators_inv(
        &fri_queries[..n_queries.min(fri_queries.len())],
        &verifier_info.opening_points,
        xi_challenge,
        verifier_info.n_bits,
        verifier_info.n_bits_ext,
    ) {
        Some(t) => t,
        None => {
            v_error!("DEEP denominator x - xi * w^o is zero");
            return false;
        }
    };

    let x_n = xi_challenge.pow(1 << verifier_info.n_bits);

    let z_n = x_n - Goldilocks::ONE;
    let z_n_inv = z_n.inverse();
    zi.push(z_n_inv);
    for boundary in &verifier_info.boundaries {
        if boundary.name == "everyRow" {
            continue;
        }

        // Handling for boundaries other than "everyRow" is intentionally deferred.
        // If support for additional boundary types is required, implement logic here.
    }

    let mut final_pol_vals: Vec<Goldilocks> = Vec::with_capacity(final_pol.len() * 3);
    for pol in &final_pol {
        final_pol_vals.extend_from_slice(&pol.value);
    }

    v_debug!("Verifying proof");

    let deep_consts =
        DeepQueryConsts::new(&verifier_info.fri_ev_groups, verifier_info.n_challenges, &challenges, &evals);

    let check_query = |q: usize| -> bool {
        // 1) Fixed MT
        if !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
            &root_c,
            &s0_last_levels[0],
            &s0_siblings[q][0],
            fri_queries[q],
            &s0_vals[q][0],
            verifier_info.arity,
            verifier_info.last_level_verification,
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
                fri_queries[q],
                &s0_vals[q][s + 1],
                verifier_info.arity,
                verifier_info.last_level_verification,
            ) {
                v_error!("Stage MT verification failed for query {}", q);
                return false;
            }
        }

        // 3) FRI Query
        let idx = fri_queries[q] % (1 << verifier_info.fri_steps[0]);
        let query_fri = deep_consts.eval(&verifier_info.fri_ev_groups, &s0_vals[q], &xdivxsub[q]);

        let valid_query = if verifier_info.n_fri_steps > 1 {
            let group_idx = (idx / (1 << verifier_info.fri_steps[1])) as usize;
            query_fri[0] == vals_fri[q][0][group_idx * 3]
                && query_fri[1] == vals_fri[q][0][group_idx * 3 + 1]
                && query_fri[2] == vals_fri[q][0][group_idx * 3 + 2]
        } else {
            query_fri == final_pol[idx as usize]
        };
        if !valid_query {
            v_error!("FRI query verification failed for query {}", q);
            return false;
        }

        // 4) FRI folding && MT
        for s in 0..verifier_info.n_fri_steps - 1 {
            let idx = fri_queries[q] % (1 << verifier_info.fri_steps[s as usize + 1]);
            if !verify_mt::<Goldilocks, LeafHash, CompressionHash>(
                &roots_fri[s as usize],
                &last_levels_fri[s as usize],
                &siblings_fri[q][s as usize],
                idx,
                &vals_fri[q][s as usize],
                verifier_info.arity,
                verifier_info.last_level_verification,
            ) {
                v_error!("FRI step MT verification failed for query {}", q);
                return false;
            }

            let value = verify_fold(
                verifier_info.n_bits_ext,
                verifier_info.fri_steps[s as usize + 1],
                verifier_info.fri_steps[s as usize],
                challenges[verifier_info.n_challenges as usize + s as usize], // r^fold_s
                idx,
                &vals_fri[q][s as usize],
            );

            if s as usize + 1 < verifier_info.n_fri_steps as usize - 1 {
                let group_idx = (idx / (1 << verifier_info.fri_steps[s as usize + 2])) as usize;
                for (i, val) in value.iter().enumerate().take(3usize) {
                    if vals_fri[q][s as usize + 1][group_idx * 3 + i] != *val {
                        v_error!("FRI foldings verification failed at step {} for query {}", s as usize + 1, q,);
                        return false;
                    }
                }
            } else {
                for (i, val) in value.iter().enumerate().take(3usize) {
                    if final_pol[idx as usize][i] != *val {
                        v_error!("Final polynomial verification failed at index {} for query {}", idx, q,);
                        return false;
                    }
                }
            }
        }

        true
    };

    let all_valid = (0..n_queries).all(check_query);

    if !all_valid {
        return false;
    }

    if verifier_info.last_level_verification > 0 {
        let mut num_nodes_level = 1 << verifier_info.n_bits_ext;
        while num_nodes_level > verifier_info.arity.pow(verifier_info.last_level_verification as u32) {
            num_nodes_level = num_nodes_level.div_ceil(verifier_info.arity);
        }

        for s in 0..verifier_info.n_stages + 1 {
            let computed_root = partial_merkle_tree::<Goldilocks, CompressionHash>(
                &s0_last_levels[s as usize + 1],
                num_nodes_level,
                verifier_info.arity,
            );
            for i in 0..4 {
                if computed_root[i] != roots[s as usize][i] {
                    v_error!("Stage {} Merkle tree root recomputation failed", s + 1);
                    return false;
                }
            }
        }

        let computed_root_c = partial_merkle_tree::<Goldilocks, CompressionHash>(
            &s0_last_levels[0],
            num_nodes_level,
            verifier_info.arity,
        );

        for i in 0..4 {
            if computed_root_c[i] != root_c[i] {
                v_error!("Stage fixed Merkle tree root recomputation failed");
                return false;
            }
        }

        for s in 0..(verifier_info.n_fri_steps - 1) {
            let mut num_nodes_level = 1 << verifier_info.fri_steps[s as usize + 1];
            while num_nodes_level > verifier_info.arity.pow(verifier_info.last_level_verification as u32) {
                num_nodes_level = num_nodes_level.div_ceil(verifier_info.arity);
            }
            let computed_root = partial_merkle_tree::<Goldilocks, CompressionHash>(
                &last_levels_fri[s as usize],
                num_nodes_level,
                verifier_info.arity,
            );
            for i in 0..4 {
                if computed_root[i] != roots_fri[s as usize][i] {
                    v_error!("Stage {} FRI Merkle tree root recomputation failed", s + 1);
                    return false;
                }
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

    v_debug!("Verifying final polynomial");
    let terminal_bits = verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize];
    let final_pol_size = 1usize << terminal_bits;
    intt_tiny(&mut final_pol_vals, terminal_bits as usize, 3);
    // The terminal layer carries 2^terminal_bits evaluations at the committed rate, so the degree
    // bound is 2^(terminal_bits - blowup). A wrapping_sub here put `init` past the end whenever the
    // terminal layer was smaller than the blowup, which skipped this check -- and it IS the check
    // that gives FRI its degree bound. Every shipped schedule clears it, poseidon vadcop_final by
    // one bit, so refuse rather than wrap.
    let Some(shift) = terminal_bits.checked_sub(verifier_info.n_bits_ext - verifier_info.n_bits) else {
        v_error!(
            "FRI terminal layer 2^{} is smaller than the blowup 2^{}; the final polynomial has no \
             degree bound left to check",
            terminal_bits,
            verifier_info.n_bits_ext - verifier_info.n_bits
        );
        return false;
    };
    let init = 1usize << shift;
    for i in init..final_pol_size {
        for j in 0..3usize {
            if final_pol_vals[i * 3 + j] != Goldilocks::ZERO {
                v_error!("Final polynomial has non-zero value at index {}: {:?}", i, final_pol_vals[i * 3 + j]);
                return false;
            }
        }
    }
    v_debug!("Final polynomial verification passed");
    v_debug!("Proof verification succeeded");

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gl(x: u64) -> CubicExtensionField<Goldilocks> {
        CubicExtensionField { value: [Goldilocks::new(x), Goldilocks::ZERO, Goldilocks::ZERO] }
    }

    /// A `VerifierInfo` carrying only what `fri_query_verify` reads.
    fn info(groups: Vec<FriEvalGroup>) -> VerifierInfo {
        VerifierInfo {
            n_stages: 2,
            n_constants: 0,
            n_evals: 0,
            n_bits: 0,
            n_bits_ext: 0,
            arity: 2,
            n_fri_queries: 0,
            n_fri_steps: 1,
            n_challenges: 6,
            n_challenges_total: 12,
            fri_steps: vec![],
            hash_commits: true,
            num_vals: vec![],
            opening_points: vec![],
            boundaries: vec![],
            q_deg: 0,
            q_index: 0,
            last_level_verification: 0,
            pow_bits: 0,
            fri_ev_groups: groups,
        }
    }

    fn r(bucket: u16, offset: u16, dim: u8, col: u16) -> FriEvalRef {
        FriEvalRef { bucket, offset, dim, col }
    }

    // challenges[4] = vf1, challenges[5] = vf2 for n_challenges = 6.
    fn challenges(vf1: u64, vf2: u64) -> Vec<CubicExtensionField<Goldilocks>> {
        vec![gl(0), gl(0), gl(0), gl(0), gl(vf1), gl(vf2)]
    }

    /// A single opened column contributes `(col - eval) * xDivXSubXi`.
    #[test]
    fn scales_a_single_opened_column_by_its_opening_weight() {
        let i = info(vec![FriEvalGroup { opening: 0, refs: vec![r(1, 2, 1, 0)] }]);
        let vals = vec![vec![], vec![Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::new(7)]];

        let got = fri_query_verify(&i, &challenges(10, 2), &[gl(3)], &vals, &[gl(5)]);

        assert_eq!(got, gl((7 - 3) * 5));
    }

    /// Each column is weighted by vf2 to the power of its global index.
    #[test]
    fn weights_columns_by_vf2_to_their_index() {
        let i = info(vec![FriEvalGroup { opening: 0, refs: vec![r(1, 0, 1, 0), r(1, 1, 1, 1)] }]);
        let vals = vec![vec![], vec![Goldilocks::new(7), Goldilocks::new(9)]];

        let got = fri_query_verify(&i, &challenges(10, 2), &[gl(3), gl(4)], &vals, &[gl(1)]);

        // (7-3)*2^0 + (9-4)*2^1
        assert_eq!(got, gl(14));
    }

    /// A column keeps its vf2 power at every opening.
    #[test]
    fn a_column_keeps_its_vf2_power_across_openings() {
        let i = info(vec![
            FriEvalGroup { opening: 0, refs: vec![r(1, 0, 1, 0), r(1, 1, 1, 1)] },
            FriEvalGroup { opening: 1, refs: vec![r(1, 1, 1, 1)] },
        ]);
        let vals = vec![vec![], vec![Goldilocks::new(7), Goldilocks::new(9)]];

        let got = fri_query_verify(&i, &challenges(10, 2), &[gl(3), gl(4), gl(6)], &vals, &[gl(1), gl(1)]);

        // group0 = 4 + 5*2 = 14, group1 = (9-6)*2 = 6, fri = 14*10 + 6
        assert_eq!(got, gl(146));
    }

    /// Distinct opening points combine in a Horner chain on vf1.
    #[test]
    fn combines_opening_points_on_vf1() {
        let i = info(vec![
            FriEvalGroup { opening: 0, refs: vec![r(1, 0, 1, 0)] },
            FriEvalGroup { opening: 1, refs: vec![r(1, 1, 1, 1)] },
        ]);
        let vals = vec![vec![], vec![Goldilocks::new(7), Goldilocks::new(9)]];

        let got = fri_query_verify(&i, &challenges(10, 2), &[gl(3), gl(4)], &vals, &[gl(1), gl(1)]);

        // group0 = 4, group1 = 5*2 (col 1), fri = 4*10 + 10
        assert_eq!(got, gl(50));
    }

    /// An extension-field column reads three consecutive slots of its leaf.
    #[test]
    fn reads_three_slots_for_an_extension_column() {
        let i = info(vec![FriEvalGroup { opening: 0, refs: vec![r(1, 0, 3, 0)] }]);
        let vals = vec![vec![], vec![Goldilocks::new(1), Goldilocks::new(2), Goldilocks::new(3)]];
        let eval = CubicExtensionField { value: [Goldilocks::new(1), Goldilocks::new(1), Goldilocks::new(1)] };

        let got = fri_query_verify(&i, &challenges(10, 2), &[eval], &vals, &[gl(1)]);

        assert_eq!(got, CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::new(1), Goldilocks::new(2)] });
    }

    /// Empty opening points are skipped, so the index is recorded, not positional.
    #[test]
    fn uses_the_recorded_opening_index_not_the_group_position() {
        let i = info(vec![FriEvalGroup { opening: 2, refs: vec![r(1, 0, 1, 0)] }]);
        let vals = vec![vec![], vec![Goldilocks::new(7)]];

        let got = fri_query_verify(&i, &challenges(10, 2), &[gl(3)], &vals, &[gl(0), gl(0), gl(5)]);

        assert_eq!(got, gl((7 - 3) * 5));
    }
}
