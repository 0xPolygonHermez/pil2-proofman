use fields::{intt_tiny, verify_fold, verify_mt, CubicExtensionField, Field, Goldilocks, Transcript};

#[derive(Debug, Clone)]
pub struct Boundary {
    pub name: String,
    pub offset_min: Option<u64>,
    pub offset_max: Option<u64>,
}

pub struct VerifierInfo {
    pub root_c: [Goldilocks; 4],
    pub n_publics: u64,
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
}

#[inline]
pub fn log2(mut n: u64) -> u32 {
    assert!(n != 0, "log2(0) is undefined");
    let mut res: u32 = 0;
    while n != 1 {
        n >>= 1; // divide by 2
        res += 1;
    }
    res
}

#[allow(clippy::type_complexity)]
pub fn stark_verify(
    proof: &[u64],
    verifier_info: &VerifierInfo,
    q_verify: fn(
        &[CubicExtensionField<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
        &[Goldilocks],
        &[CubicExtensionField<Goldilocks>],
        CubicExtensionField<Goldilocks>,
    ) -> CubicExtensionField<Goldilocks>,
    queries_fri_verify: fn(
        &[CubicExtensionField<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
        &[Vec<Goldilocks>],
        &[CubicExtensionField<Goldilocks>],
    ) -> CubicExtensionField<Goldilocks>,
) -> bool {
    let mut leaves: u64 = 1 << verifier_info.n_bits_ext;
    let mut n_siblings: u64 = 0;

    while leaves > 1 {
        leaves = leaves.div_ceil(verifier_info.arity);
        n_siblings += 1;
    }

    let n_siblings_per_level = (verifier_info.arity - 1) * 4;

    let mut p = 1;

    let mut publics = Vec::new();
    for _ in 0..verifier_info.n_publics {
        publics.push(Goldilocks::new(proof[p as usize]));
        p += 1;
    }

    let mut roots = Vec::new();
    for _ in 0..verifier_info.n_stages + 1 {
        let mut root = [Goldilocks::ZERO; 4];
        for r in &mut root {
            *r = Goldilocks::new(proof[p as usize]);
            p += 1;
        }
        roots.push(root);
    }

    let mut evals = Vec::new();
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

    let mut s0_vals = Vec::new();
    let mut s0_siblings = Vec::new();

    for q in 0..verifier_info.n_fri_queries {
        s0_vals.push(Vec::new());
        let mut vals = Vec::new();
        for _ in 0..verifier_info.n_constants {
            vals.push(Goldilocks::new(proof[p as usize]));
            p += 1;
        }
        s0_vals[q as usize].push(vals);
    }

    for q in 0..verifier_info.n_fri_queries {
        s0_siblings.push(Vec::new());
        let mut siblings = Vec::new();
        for _ in 0..n_siblings {
            let mut sibling = Vec::new();
            for _ in 0..n_siblings_per_level {
                sibling.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            siblings.push(sibling);
        }
        s0_siblings[q as usize].push(siblings);
    }

    for i in 0..verifier_info.n_stages + 1 {
        s0_vals.push(Vec::new());
        s0_siblings.push(Vec::new());
        for q in 0..verifier_info.n_fri_queries {
            let mut vals = Vec::new();
            for _ in 0..verifier_info.num_vals[i as usize] {
                vals.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            s0_vals[q as usize].push(vals);
        }

        for q in 0..verifier_info.n_fri_queries {
            let mut siblings = Vec::new();
            for _ in 0..n_siblings {
                let mut sibling = Vec::new();
                for _ in 0..n_siblings_per_level {
                    sibling.push(Goldilocks::new(proof[p as usize]));
                    p += 1;
                }
                siblings.push(sibling);
            }
            s0_siblings[q as usize].push(siblings);
        }
    }

    let mut roots_fri = Vec::new();
    for _ in 1..verifier_info.n_fri_steps {
        let mut root = [Goldilocks::ZERO; 4];
        for r in &mut root {
            *r = Goldilocks::new(proof[p as usize]);
            p += 1;
        }
        roots_fri.push(root);
    }

    let mut siblings_fri = vec![Vec::new(); verifier_info.n_fri_queries as usize];
    let mut vals_fri = vec![Vec::new(); verifier_info.n_fri_queries as usize];
    for i in 1..verifier_info.n_fri_steps {
        for val_fri in vals_fri.iter_mut().take(verifier_info.n_fri_queries as usize) {
            let mut vals = Vec::new();
            for _ in 0..(1 << (verifier_info.fri_steps[(i - 1) as usize] - verifier_info.fri_steps[i as usize])) * 3 {
                vals.push(Goldilocks::new(proof[p as usize]));
                p += 1;
            }
            val_fri.push(vals);
        }

        for q in 0..verifier_info.n_fri_queries {
            let mut leaves: u64 = 1 << verifier_info.fri_steps[i as usize];
            let mut n_siblings_fri: u64 = 0;

            while leaves > 1 {
                leaves = leaves.div_ceil(verifier_info.arity);
                n_siblings_fri += 1;
            }
            let n_siblings_per_level_fri = (verifier_info.arity - 1) * 4;
            let mut siblings = Vec::new();
            for _ in 0..n_siblings_fri {
                let mut sibling = Vec::new();
                for _ in 0..n_siblings_per_level_fri {
                    sibling.push(Goldilocks::new(proof[p as usize]));
                    p += 1;
                }
                siblings.push(sibling);
            }
            siblings_fri[q as usize].push(siblings);
        }
    }

    let mut final_pol = Vec::new();
    for _ in 0..(1 << verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize]) {
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

    assert!(p == proof.len() as u64, "Proof length mismatch: expected {}, got {}", proof.len(), p);

    let mut challenges = vec![
        CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] };
        verifier_info.n_challenges_total as usize
    ];

    let mut xdivxsub = Vec::new();
    let mut zi = Vec::new();

    let mut transcript = Transcript::new();
    transcript.put(&mut verifier_info.root_c.clone());
    if verifier_info.n_publics > 0 {
        if !verifier_info.hash_commits {
            transcript.put(&mut publics);
        } else {
            let mut transcript_publics = Transcript::new();
            transcript_publics.put(&mut publics);
            let mut hash = transcript_publics.get_state();
            transcript.put(&mut hash);
        }
    }
    transcript.put(&mut roots[0]);
    transcript.get_field(&mut challenges[0].value);
    transcript.get_field(&mut challenges[1].value);

    transcript.put(&mut roots[1]);
    transcript.get_field(&mut challenges[2].value);
    transcript.put(&mut roots[2]);

    transcript.get_field(&mut challenges[3].value);

    if !verifier_info.hash_commits {
        for i in 0..verifier_info.n_evals {
            transcript.put(&mut evals[i as usize].value);
        }
    } else {
        let mut transcript_evals = Transcript::new();
        for i in 0..verifier_info.n_evals {
            transcript_evals.put(&mut evals[i as usize].value);
        }
        let mut hash = transcript_evals.get_state();
        transcript.put(&mut hash);
    }

    transcript.get_field(&mut challenges[4].value);
    transcript.get_field(&mut challenges[5].value);

    let mut c = 6;
    for i in 0..verifier_info.n_fri_steps {
        transcript.get_field(&mut challenges[c].value);
        c += 1;
        if i < verifier_info.n_fri_steps - 1 {
            transcript.put(&mut roots_fri[i as usize]);
        } else {
            let final_pol_size = 1 << verifier_info.fri_steps[i as usize];
            if !verifier_info.hash_commits {
                for j in 0..final_pol_size {
                    transcript.put(&mut final_pol[j as usize].value);
                }
            } else {
                let mut transcript_final_pol = Transcript::new();
                for j in 0..final_pol_size {
                    transcript_final_pol.put(&mut final_pol[j as usize].value);
                }
                let mut hash = transcript_final_pol.get_state();
                transcript.put(&mut hash);
            }
        }
    }

    transcript.get_field(&mut challenges[c].value);
    let mut transcript_permutation = Transcript::new();
    let last_challenge_index = challenges.len() - 1;
    transcript_permutation.put(&mut challenges[last_challenge_index].value);
    let fri_queries = transcript_permutation.get_permutations(verifier_info.n_fri_queries, verifier_info.fri_steps[0]);

    let xi_challenge = challenges[verifier_info.n_challenges as usize - 3];

    for q in 0..verifier_info.n_fri_queries as usize {
        xdivxsub.push(Vec::new());
        let w = Goldilocks::new(Goldilocks::W[verifier_info.n_bits_ext as usize]);
        let x = CubicExtensionField {
            value: [Goldilocks::new(Goldilocks::SHIFT) * w.exp_u64(fri_queries[q]), Goldilocks::ZERO, Goldilocks::ZERO],
        };
        for o in 0..verifier_info.opening_points.len() {
            let mut wi = Goldilocks::ONE;
            let abs_opening = verifier_info.opening_points[o].unsigned_abs();
            for _ in 0..abs_opening {
                wi *= Goldilocks::new(Goldilocks::W[verifier_info.n_bits as usize]);
            }

            if verifier_info.opening_points[o] < 0 {
                wi = wi.inverse();
            }

            let val = (x - (xi_challenge * wi)).inverse() * x;

            xdivxsub[q].push(val);
        }
    }

    let x_n = xi_challenge.pow(1 << verifier_info.n_bits);

    let z_n = x_n - Goldilocks::ONE;
    let z_n_inv = z_n.inverse();
    zi.push(z_n_inv);
    for boundary in &verifier_info.boundaries {
        if boundary.name == "everyRow" {
            continue;
        }

        // TODO
    }
    let mut final_pol_vals: Vec<Goldilocks> = final_pol
        .iter() // borrow each CubicExtensionField
        .flat_map(|pol| pol.value.iter().cloned())
        .collect();

    tracing::debug!("Verifying proof");
    for q in 0..verifier_info.n_fri_queries as usize {
        // 1) Fixed MT
        if !verify_mt(&verifier_info.root_c, &s0_siblings[q][0], fri_queries[q], &s0_vals[q][0], verifier_info.arity) {
            tracing::error!("Fixed MT verification failed for query {}", q);
            return false;
        }

        // 2) stage MTs
        for (s, root) in roots.iter().enumerate().take(verifier_info.n_stages as usize + 1) {
            if !verify_mt(root, &s0_siblings[q][s + 1], fri_queries[q], &s0_vals[q][s + 1], verifier_info.arity) {
                tracing::error!("Stage MT verification failed for query {}", q);
                return false;
            }
        }

        // 3) FRI step MTs
        for s in 0..(verifier_info.n_fri_steps - 1) {
            let idx = fri_queries[q] % (1 << verifier_info.fri_steps[s as usize + 1]);
            if !verify_mt(
                &roots_fri[s as usize],
                &siblings_fri[q][s as usize],
                idx,
                &vals_fri[q][s as usize],
                verifier_info.arity,
            ) {
                tracing::error!("FRI step MT verification failed for query {}", q);
                return false;
            }
        }

        // 4) FRI Queries
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
            tracing::error!("FRI query verification failed for query {}", q);
            return false;
        }

        // 5) FRI foldings
        for s in 1..verifier_info.n_fri_steps as usize {
            let idx = fri_queries[q] % (1 << verifier_info.fri_steps[s]);
            let value = verify_fold(
                verifier_info.n_bits_ext,
                verifier_info.fri_steps[s],
                verifier_info.fri_steps[s - 1],
                challenges[verifier_info.n_challenges as usize + s],
                fri_queries[q] % (1 << verifier_info.fri_steps[s]),
                &vals_fri[q][s - 1],
            );
            if s < verifier_info.n_fri_steps as usize - 1 {
                let group_idx = (idx / (1 << verifier_info.fri_steps[s + 1])) as usize;
                for (i, val) in value.iter().enumerate().take(3usize) {
                    if vals_fri[q][s][group_idx * 3 + i] != *val {
                        tracing::error!("FRI foldings verification failed at step {} for query {}", s, q,);
                        return false;
                    }
                }
            } else {
                for (i, val) in value.iter().enumerate().take(3usize) {
                    if final_pol[idx as usize][i] != *val {
                        tracing::error!("Final polynomial verification failed at index {} for query {}", idx, q,);
                        return false;
                    }
                }
            }
        }
    }

    tracing::debug!("Verifying Quotient polynomial");
    let mut x_acc = CubicExtensionField { value: [Goldilocks::ONE, Goldilocks::ZERO, Goldilocks::ZERO] };
    let mut q = CubicExtensionField { value: [Goldilocks::ZERO, Goldilocks::ZERO, Goldilocks::ZERO] };
    for i in 0..verifier_info.q_deg {
        q += x_acc * evals[(verifier_info.q_index + i) as usize];
        x_acc *= x_n;
    }

    let q_val = q_verify(&challenges, &evals, &publics, &zi, xi_challenge);
    if q_val != q {
        tracing::error!("Quotient polynomial verification failed");
        return false;
    }
    tracing::debug!("Quotient polynomial verification passed");

    tracing::debug!("Verifying final polynomial");
    let final_pol_size = 1 << verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize];
    intt_tiny(&mut final_pol_vals, verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize] as usize, 3);
    let init = 1
        << (verifier_info.fri_steps[(verifier_info.n_fri_steps - 1) as usize]
            .wrapping_sub(verifier_info.n_bits_ext - verifier_info.n_bits));
    for i in init..final_pol_size as usize {
        for j in 0..3usize {
            if final_pol_vals[i * 3 + j] != Goldilocks::ZERO {
                tracing::error!("Final polynomial has non-zero value at index {}: {:?}", i, final_pol_vals[i * 3 + j]);
                return false;
            }
        }
    }
    tracing::debug!("Final polynomial verification passed");
    tracing::debug!("Proof verification succeeded");

    true
}
