// The Fiat-Shamir transcript as pure functions of the proof: every challenge and query index,
// nothing checked. The verifiers check what these return; the multiproof codec takes the indices.

use alloc::vec::Vec;

use proofman_fields::{CubicExtensionField, Field, Goldilocks, TranscriptLike};

use crate::stir::StirParams;

type E3 = CubicExtensionField<Goldilocks>;

fn squeeze<T: TranscriptLike<Goldilocks>>(t: &mut T) -> E3 {
    let mut c = CubicExtensionField { value: [Goldilocks::ZERO; 3] };
    t.get_field(&mut c.value);
    c
}

/// Absorb `vals`, skipping an empty put: BLAKE3's rewinds its XOF even then.
fn put<T: TranscriptLike<Goldilocks>>(t: &mut T, vals: &[Goldilocks]) {
    if !vals.is_empty() {
        t.put(vals);
    }
}

/// Absorb `vals`, or with `hash_commits` the first four words of their own transcript's state.
fn put_committed<T: TranscriptLike<Goldilocks>>(t: &mut T, vals: &[Goldilocks], hash_commits: bool) {
    if !hash_commits {
        put(t, vals);
    } else {
        let mut inner = T::new_transcript();
        inner.put(vals);
        t.put(&inner.get_state()[0..4]);
    }
}

fn flatten(e3s: &[E3]) -> Vec<Goldilocks> {
    e3s.iter().flat_map(|e| e.value).collect()
}

/// The query challenge, and the indices it and `nonce` draw.
fn draw_queries<T: TranscriptLike<Goldilocks>>(t: &mut T, nonce: Goldilocks, n: u64, n_bits: u64) -> (E3, Vec<u64>) {
    let c = squeeze(t);
    let mut tq = T::new_transcript();
    tq.put(&c.value);
    tq.put(&[nonce]);
    (c, tq.get_permutations(n, n_bits))
}

/// The shared prefix of a 2-stage AIR: verkey, publics, root1/root2/rootQ, evals; challenges 0 … 5.
pub(crate) fn prefix_transcript<T: TranscriptLike<Goldilocks>>(
    root_c: &[Goldilocks; 4],
    publics: &[Goldilocks],
    roots: &[[Goldilocks; 4]; 3],
    evals: &[E3],
    hash_commits: bool,
) -> (T, [E3; 6]) {
    let mut t = T::new_transcript();
    t.put(root_c);
    if !publics.is_empty() {
        put_committed(&mut t, publics, hash_commits);
    }
    t.put(&roots[0]);
    let c0 = squeeze(&mut t);
    let c1 = squeeze(&mut t);
    t.put(&roots[1]);
    let c2 = squeeze(&mut t);
    t.put(&roots[2]);
    let c3 = squeeze(&mut t);
    put_committed(&mut t, &flatten(evals), hash_commits);
    let c4 = squeeze(&mut t);
    let c5 = squeeze(&mut t);
    (t, [c0, c1, c2, c3, c4, c5])
}

pub(crate) struct FriTranscript {
    /// r^fold_i, drawn after f_i's root.
    pub folds: Vec<E3>,
    pub query: E3,
    /// Indices of f₀'s domain.
    pub queries: Vec<u64>,
}

/// FRI after the prefix: the step roots, the final polynomial, then the queries.
pub(crate) fn fri_transcript<T: TranscriptLike<Goldilocks>>(
    t: &mut T,
    roots: &[[Goldilocks; 4]],
    final_pol: &[E3],
    nonce: Goldilocks,
    steps: &[u64],
    n_queries: u64,
    hash_commits: bool,
) -> FriTranscript {
    let mut folds = Vec::with_capacity(roots.len());
    for root in roots {
        t.put(root);
        folds.push(squeeze(t));
    }
    put_committed(t, &flatten(final_pol), hash_commits);
    let (query, queries) = draw_queries(t, nonce, n_queries, steps[0]);
    FriTranscript { folds, query, queries }
}

pub(crate) struct StirRound {
    pub r_out: E3,
    pub r_fold: E3,
    pub r_comb: E3,
    /// Grinds round i and draws its indices of L_{i−1}.
    pub query: E3,
    pub queries: Vec<u64>,
    pub rho: E3,
}

pub(crate) struct StirTranscript {
    pub r_fold0: E3,
    /// Iterations 1 … M−1.
    pub rounds: Vec<StirRound>,
    pub final_query: E3,
    pub final_queries: Vec<u64>,
}

/// STIR after the prefix, in `stir.hpp`'s order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stir_transcript<T: TranscriptLike<Goldilocks>>(
    t: &mut T,
    params: &StirParams,
    roots: &[[Goldilocks; 4]],
    betas: &[E3],
    ans: &[Vec<E3>],
    shake: &[Vec<E3>],
    final_pol: &[E3],
    nonces: &[u64],
) -> StirTranscript {
    let m = params.m();
    t.put(&roots[0]);
    let r_fold0 = squeeze(t);
    let mut rounds = Vec::with_capacity(m - 1);
    for i in 1..m {
        t.put(&roots[i]);
        let r_out = squeeze(t);
        t.put(&betas[i - 1].value);
        let r_fold = squeeze(t);
        let r_comb = squeeze(t);
        let (query, queries) =
            draw_queries(t, Goldilocks::new(nonces[i - 1]), params.num_queries[i - 1], params.log_domain_sizes[i - 1]);
        put(t, &flatten(&ans[i - 1]));
        put(t, &flatten(&shake[i - 1]));
        let rho = squeeze(t);
        rounds.push(StirRound { r_out, r_fold, r_comb, query, queries, rho });
    }
    put_committed(t, &flatten(final_pol), params.hash_commits);
    let (final_query, final_queries) =
        draw_queries(t, Goldilocks::new(nonces[m - 1]), params.num_queries[m - 1], params.log_domain_sizes[m - 1]);
    StirTranscript { r_fold0, rounds, final_query, final_queries }
}
