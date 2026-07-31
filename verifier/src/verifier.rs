use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

#[allow(unused)]
use num_traits::Float;

use fields::{
    intt_tiny, partial_merkle_tree, verify_fold, verify_mt, CubicExtensionField, Field, Goldilocks, Hash, PrimeField64,
    Transcript,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
}

pub fn expected_proof_size_bytes(info: &VerifierInfo) -> usize {
    let log_arity = (info.arity as f64).log2();
    let n_siblings = ((info.n_bits_ext as f64 / log_arity).ceil()) as u64 - info.last_level_verification;
    let n_siblings_per_level = (info.arity - 1) * 4;
    let n_queries = info.n_fri_queries;
    let num_nodes_level = info.arity.pow(info.last_level_verification as u32) * 4;
    let last_level_extra = if info.last_level_verification > 0 { num_nodes_level } else { 0 };

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

    // fri roots: (n_fri_steps - 1) groups of 4
    p += 4 * (info.n_fri_steps - 1);

    // fri data: (n_fri_steps - 1) iterations
    for i in 1..info.n_fri_steps {
        let vals_size = (1u64 << (info.fri_steps[(i - 1) as usize] - info.fri_steps[i as usize])) * 3;
        p += n_queries * vals_size;

        let n_siblings_fri =
            ((info.fri_steps[i as usize] as f64 / log_arity).ceil()) as u64 - info.last_level_verification;
        p += n_queries * n_siblings_fri * n_siblings_per_level;

        p += last_level_extra;
    }

    // final polynomial
    let final_pol_capacity = 1u64 << info.fri_steps[(info.n_fri_steps - 1) as usize];
    p += 3 * final_pol_capacity;

    // nonce
    p += 1;

    (p as usize) * 8
}

#[allow(clippy::type_complexity)]
pub fn stark_verify<LeafHash, CompressionHash, TranscriptHash, GrindingHash>(
    proof: &[u64],
    vk: &[u64],
    verifier_info: &VerifierInfo,
    q_verify: fn(
        &[CubicExtensionField<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
        &[Goldilocks],
        &[CubicExtensionField<Goldilocks>],
    ) -> CubicExtensionField<Goldilocks>,
    queries_fri_verify: fn(
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

    let n_siblings: u64 = ((verifier_info.n_bits_ext as f64 / (verifier_info.arity as f64).log2()).ceil()) as u64
        - verifier_info.last_level_verification;
    let n_siblings_per_level = (verifier_info.arity - 1) * 4;

    let root_c = [Goldilocks::new(vk[0]), Goldilocks::new(vk[1]), Goldilocks::new(vk[2]), Goldilocks::new(vk[3])];

    let mut p = 0;

    let n_publics = proof[p as usize];
    p += 1;

    let Some(expected_total) = 1usize
        .checked_add(n_publics as usize)
        .and_then(|s| s.checked_add(expected_proof_size_bytes(verifier_info) / 8))
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
    let mut s0_siblings: Vec<Vec<Vec<Vec<Goldilocks>>>> = Vec::with_capacity(n_queries);
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
        let mut siblings = Vec::with_capacity(n_sibs);
        for _ in 0..n_sibs {
            let mut sibling = Vec::with_capacity(n_sibs_per_lvl);
            for _ in 0..n_sibs_per_lvl {
                sibling.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            siblings.push(sibling);
        }
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
            let mut siblings = Vec::with_capacity(n_sibs);
            for _ in 0..n_sibs {
                let mut sibling = Vec::with_capacity(n_sibs_per_lvl);
                for _ in 0..n_sibs_per_lvl {
                    sibling.push(Goldilocks::new(proof[p as usize]));
                    p += 1;
                }
                siblings.push(sibling);
            }
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

    let mut siblings_fri: Vec<Vec<Vec<Vec<Goldilocks>>>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_fri_steps_minus_1)).collect();
    let mut vals_fri: Vec<Vec<Vec<Goldilocks>>> =
        (0..n_queries).map(|_| Vec::with_capacity(n_fri_steps_minus_1)).collect();
    let mut last_levels_fri: Vec<Vec<Goldilocks>> = Vec::with_capacity(n_fri_steps_minus_1);

    let log_arity = (verifier_info.arity as f64).log2();
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

        let n_siblings_fri = ((verifier_info.fri_steps[i as usize] as f64 / log_arity).ceil()) as usize
            - verifier_info.last_level_verification as usize;

        for query_siblings in siblings_fri.iter_mut() {
            let mut siblings = Vec::with_capacity(n_siblings_fri);
            for _ in 0..n_siblings_fri {
                let mut sibling = Vec::with_capacity(n_siblings_per_level_fri);
                for _ in 0..n_siblings_per_level_fri {
                    sibling.push(Goldilocks::new(proof[p as usize]));
                    p += 1;
                }
                siblings.push(sibling);
            }
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

    let mut challenges = vec![
        CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] };
        verifier_info.n_challenges_total as usize
    ];

    let mut xdivxsub: Vec<Vec<CubicExtensionField<Goldilocks>>> = Vec::with_capacity(n_queries);
    let mut zi = Vec::with_capacity(verifier_info.boundaries.len() + 1);

    let mut transcript: Transcript<Goldilocks, TranscriptHash> = Transcript::<Goldilocks, TranscriptHash>::new();
    transcript.put(&root_c);
    if n_publics > 0 {
        if !verifier_info.hash_commits {
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

    if !verifier_info.hash_commits {
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

    let mut c = 6;
    for i in 0..verifier_info.n_fri_steps {
        if i > 0 {
            transcript.get_field(&mut challenges[c].value);
        }
        c += 1;
        if i < verifier_info.n_fri_steps - 1 {
            transcript.put(&roots_fri[i as usize]);
        } else {
            let final_pol_size = 1 << verifier_info.fri_steps[i as usize];
            if !verifier_info.hash_commits {
                for j in 0..final_pol_size {
                    transcript.put(&final_pol[j as usize].value);
                }
            } else {
                let mut transcript_final_pol: Transcript<Goldilocks, TranscriptHash> =
                    Transcript::<Goldilocks, TranscriptHash>::new();
                for j in 0..final_pol_size {
                    transcript_final_pol.put(&final_pol[j as usize].value);
                }
                let hash = transcript_final_pol.get_state();
                transcript.put(&hash[0..4]);
            }
        }
    }

    transcript.get_field(&mut challenges[c].value);
    let last_challenge_index = challenges.len() - 1;
    // Proof-of-work grinding hash. Uses its own fixed-width hash (Poseidon2_8 for
    // Poseidon2, Poseidon1_8 for Poseidon1), independent of the transcript/Merkle
    // widths. The input `[c0, c1, c2, nonce]` is written into a zero-padded state
    // of the grinding hash's width and permuted in place; the check reads cell 0.
    let mut pow_state = <GrindingHash as Hash<Goldilocks>>::State::default();
    {
        let state = pow_state.as_mut();
        state[0] = challenges[last_challenge_index].value[0];
        state[1] = challenges[last_challenge_index].value[1];
        state[2] = challenges[last_challenge_index].value[2];
        state[3] = nonce;
    }
    <GrindingHash as Hash<Goldilocks>>::hash(&mut pow_state);
    if pow_state.as_ref()[0].as_canonical_u64() >= 1 << (64 - verifier_info.pow_bits) {
        v_error!("Proof of work verification failed");
        return false;
    }
    let mut transcript_permutation: Transcript<Goldilocks, TranscriptHash> =
        Transcript::<Goldilocks, TranscriptHash>::new();
    transcript_permutation.put(&challenges[last_challenge_index].value);
    transcript_permutation.put(&[nonce]);
    let fri_queries = transcript_permutation.get_permutations(verifier_info.n_fri_queries, verifier_info.fri_steps[0]);

    let xi_challenge = challenges[verifier_info.n_challenges as usize - 3];

    let w_ext = Goldilocks::new(Goldilocks::W[verifier_info.n_bits_ext as usize]);
    let w_bits = Goldilocks::new(Goldilocks::W[verifier_info.n_bits as usize]);
    let n_opening_points = verifier_info.opening_points.len();

    for &fri_query in fri_queries.iter().take(n_queries) {
        let mut query_xdivxsub = Vec::with_capacity(n_opening_points);
        let x = CubicExtensionField {
            value: [Goldilocks::new(Goldilocks::SHIFT) * w_ext.exp_u64(fri_query), Goldilocks::ZERO, Goldilocks::ZERO],
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
        let query_fri = queries_fri_verify(&challenges, &evals, &s0_vals[q], &xdivxsub[q]);

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
                challenges[verifier_info.n_challenges as usize + s as usize + 1],
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

    #[cfg(feature = "parallel")]
    let all_valid = (0..n_queries).into_par_iter().all(check_query);
    #[cfg(not(feature = "parallel"))]
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
    let final_pol_size = 1 << verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize];
    intt_tiny(&mut final_pol_vals, verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize] as usize, 3);
    let init = 1
        << (verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize]
            .wrapping_sub(verifier_info.n_bits_ext - verifier_info.n_bits));
    for i in init..final_pol_size as usize {
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
