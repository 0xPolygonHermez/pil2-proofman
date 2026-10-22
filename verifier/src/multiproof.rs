// Merkle multiproofs: `prune` drops every sibling another query's path already yields, `expand`
// rebuilds the flat proof word for word; the verifier still checks all of it. Wire form: the flat
// proof minus its sibling blocks, `MULTIPROOF_TAG` after the publics, then each tree's kept
// siblings by level, parent and child. Every other word is canonical: one wire form per proof.

use alloc::vec;
use alloc::vec::Vec;

use proofman_fields::{CubicExtensionField, Field, Goldilocks, Hash, PrimeField64, TranscriptLike};

use crate::layout::{sibling_words, LowDegreeLayout, ProofLayout, Segment};
use crate::replay::{fri_transcript, prefix_transcript, stir_transcript};

type E3 = CubicExtensionField<Goldilocks>;

/// The body's first word: not a field element, so no flat proof starts with it; low bits = version.
pub const MULTIPROOF_TAG: u64 = 0xFFFF_FFFF_FFFF_0001;

/// Whether `wire` is tagged as a multiproof.
pub fn is_multiproof(wire: &[u64]) -> bool {
    let tag_at = wire.first().and_then(|&n| usize::try_from(n).ok()?.checked_add(1));
    tag_at.and_then(|i| wire.get(i)) == Some(&MULTIPROOF_TAG)
}

/// The kept words of a proof, with each tree's leaf offset and each data segment's offset.
struct Split {
    dense: Vec<u64>,
    leaves: Vec<usize>,
    data: Vec<usize>,
}

impl Split {
    fn gl(&self, at: usize, n: usize) -> Vec<Goldilocks> {
        self.dense[at..at + n].iter().map(|&w| Goldilocks::new(w)).collect()
    }
    fn roots(&self, at: usize, n: usize) -> Vec<[Goldilocks; 4]> {
        self.gl(at, 4 * n).chunks(4).map(|r| [r[0], r[1], r[2], r[3]]).collect()
    }
    fn e3s(&self, at: usize, n: usize) -> Vec<E3> {
        self.gl(at, 3 * n).chunks(3).map(|e| CubicExtensionField { value: [e[0], e[1], e[2]] }).collect()
    }
}

/// The trees of a proof, in order: (leaf width, queries, path levels).
fn trees(layout: &ProofLayout) -> Vec<(usize, usize, usize)> {
    layout
        .segments(0)
        .into_iter()
        .filter_map(|s| match s {
            Segment::Tree { width, n_queries, n_levels, .. } => Some((width, n_queries, n_levels)),
            Segment::Data(_) => None,
        })
        .collect()
}

fn supported(layout: &ProofLayout) -> bool {
    let ldt_ok = match &layout.ldt {
        LowDegreeLayout::Fri { steps, .. } => !steps.is_empty() && steps.windows(2).all(|w| w[0] > w[1]),
        // Its tree geometry is the layout's own.
        LowDegreeLayout::Stir(stir) => {
            stir.validate()
                && stir.arity == layout.arity
                && stir.last_level_verification == layout.last_level_verification
                && stir.hash_commits == layout.hash_commits
        }
    };
    // Three stage roots, as the verifiers; and no empty leaves.
    ldt_ok
        && layout.n_stages == 2
        && layout.num_vals.len() > layout.n_stages as usize
        && (2..=MAX_ARITY as u64).contains(&layout.arity)
        && trees(layout).iter().all(|&(width, _, _)| width > 0)
}

/// Collect the kept words of `proof` (flat, or a pruned one's prefix), and a flat one's sibling blocks.
fn split<'a>(layout: &ProofLayout, proof: &'a [u64], flat: bool) -> Option<(Split, Vec<&'a [u64]>)> {
    if !supported(layout) {
        return None;
    }
    // Attacker-chosen: bound it before any arithmetic.
    let n_publics = usize::try_from(*proof.first()?).ok()?;
    if n_publics >= proof.len() {
        return None;
    }
    let sib_w = sibling_words(layout.arity);
    let mut split = Split { dense: Vec::new(), leaves: Vec::new(), data: Vec::new() };
    let mut sibling_blocks = Vec::new();
    let mut p = 0usize;
    for seg in layout.segments(n_publics) {
        match seg {
            Segment::Data(n) => {
                split.data.push(split.dense.len());
                split.dense.extend_from_slice(proof.get(p..p + n)?);
                p += n;
            }
            Segment::Tree { width, n_queries, n_levels, .. } => {
                let n = n_queries * width;
                split.leaves.push(split.dense.len());
                split.dense.extend_from_slice(proof.get(p..p + n)?);
                p += n;
                if flat {
                    let s = n_queries * n_levels * sib_w;
                    sibling_blocks.push(proof.get(p..p + s)?);
                    p += s;
                }
            }
        }
    }
    if flat && p != proof.len() {
        return None;
    }
    Some((split, sibling_blocks))
}

/// Per tree, the leaf each query opens.
fn tree_indices<T: TranscriptLike<Goldilocks>>(
    layout: &ProofLayout,
    split: &Split,
    vk: &[u64],
) -> Option<Vec<Vec<u64>>> {
    let root_c: [Goldilocks; 4] = core::array::from_fn(|i| Goldilocks::new(vk[i]));
    let roots = split.roots(split.data[1], 3);
    let (mut t, _) = prefix_transcript::<T>(
        &root_c,
        &split.gl(1, split.dense[0] as usize),
        &[roots[0], roots[1], roots[2]],
        &split.e3s(split.data[2], layout.n_evals as usize),
        layout.hash_commits,
    );

    // Data segments: publics, roots, evals, stage last levels, ldt roots, ldt last levels, tail.
    let n_stage_trees = layout.n_stages as usize + 2;
    let ldt_roots = split.data[3 + n_stage_trees];
    let tail = *split.data.last()?;
    let (round1, ldt_trees): (Vec<u64>, Vec<Vec<u64>>) = match &layout.ldt {
        LowDegreeLayout::Fri { steps, n_queries } => {
            let n = steps.len();
            let n_final = 1usize << steps[n - 1];
            let fri = fri_transcript(
                &mut t,
                &split.roots(ldt_roots, n - 1),
                &split.e3s(tail, n_final),
                Goldilocks::new(split.dense[tail + 3 * n_final]),
                steps,
                *n_queries,
                layout.hash_commits,
            );
            let folded = steps[1..].iter().map(|&bits| fri.queries.iter().map(|q| q % (1 << bits)).collect()).collect();
            (fri.queries, folded)
        }
        LowDegreeLayout::Stir(stir) => {
            // The tail: β_1 … β_{M−1}, p, the nonces, then Âns and Sh.
            let m = stir.m();
            let n_final = 1usize << stir.log_degrees[m];
            let final_pol = tail + 3 * (m - 1);
            let nonces = final_pol + 3 * n_final;
            let mut at = nonces + m;
            let mut hints = || {
                (1..m)
                    .map(|i| {
                        let n = 1 + stir.num_queries[i - 1] as usize;
                        let h = split.e3s(at, n);
                        at += 3 * n;
                        h
                    })
                    .collect::<Vec<_>>()
            };
            let ans = hints();
            let shake = hints();
            let st = stir_transcript(
                &mut t,
                stir,
                &split.roots(ldt_roots, m),
                &split.e3s(tail, m - 1),
                &ans,
                &shake,
                &split.e3s(final_pol, n_final),
                &split.dense[nonces..nonces + m],
            );
            // Round i's queries index L_{i−1}; T_{i−1}'s leaves are their cosets.
            let raw: Vec<Vec<u64>> =
                st.rounds.into_iter().map(|r| r.queries).chain(core::iter::once(st.final_queries)).collect();
            let folded = raw
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let n_leaves = 1u64 << (stir.log_domain_sizes[i] - stir.folding_factors[i]);
                    q.iter().map(|r| r % n_leaves).collect()
                })
                .collect();
            (raw[0].clone(), folded)
        }
    };
    let mut out = vec![round1; n_stage_trees];
    out.extend(ldt_trees);
    Some(out)
}

/// Widest node the codec walks: a node hash takes `arity` four-word digests in a state of at most 16.
const MAX_ARITY: usize = 4;

/// The slot of child `j` in the sibling list of a node at child position `curr`.
fn slot(j: usize, curr: usize) -> usize {
    if j < curr {
        j
    } else {
        j - 1
    }
}

/// A level's known nodes, sorted and distinct, and for each query the index of its node in them.
fn leaves(idx: &[u64]) -> (Vec<u64>, Vec<usize>) {
    let mut nodes = idx.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    let node_of = idx.iter().map(|i| nodes.binary_search(i).unwrap()).collect();
    (nodes, node_of)
}

/// The known nodes of a level split into runs that share a parent: `(first node, end)`.
fn parent_runs(nodes: &[u64], arity: u64) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    for i in 1..=nodes.len() {
        if i == nodes.len() || nodes[i] / arity != nodes[start] / arity {
            runs.push((start, i));
            start = i;
        }
    }
    runs
}

fn prune_tree(idx: &[u64], n_levels: usize, arity: u64, sibs: &[u64], out: &mut Vec<u64>) {
    let (a, w) = (arity as usize, sibling_words(arity));
    let (mut nodes, node_of) = leaves(idx);
    // A query whose path runs through each node, to copy its siblings from.
    let mut rep = vec![usize::MAX; nodes.len()];
    for (q, &n) in node_of.iter().enumerate().rev() {
        rep[n] = q;
    }
    for level in 0..n_levels {
        let runs = parent_runs(&nodes, arity);
        let mut next_nodes = Vec::with_capacity(runs.len());
        let mut next_rep = Vec::with_capacity(runs.len());
        for &(first, end) in &runs {
            let mut known = [false; MAX_ARITY];
            for &n in &nodes[first..end] {
                known[(n % arity) as usize] = true;
            }
            let (q, curr) = (rep[first], (nodes[first] % arity) as usize);
            for j in (0..a).filter(|&j| !known[j]) {
                let at = (q * n_levels + level) * w + slot(j, curr) * 4;
                out.extend_from_slice(&sibs[at..at + 4]);
            }
            next_nodes.push(nodes[first] / arity);
            next_rep.push(q);
        }
        nodes = next_nodes;
        rep = next_rep;
    }
}

/// One tree's flat sibling block, from its leaf digests and the kept siblings at `kept[*kp..]`.
fn expand_tree<NH: Hash<Goldilocks>>(
    idx: &[u64],
    n_levels: usize,
    arity: u64,
    leaf_digests: &[[Goldilocks; 4]],
    kept: &[u64],
    kp: &mut usize,
) -> Option<Vec<u64>> {
    let (a, w) = (arity as usize, sibling_words(arity));
    let mut out = vec![0u64; idx.len() * n_levels * w];
    let (mut nodes, mut node_of) = leaves(idx);
    // Duplicate indices keep the first query's leaf; a different second one fails its own path.
    let mut digest = vec![[Goldilocks::ZERO; 4]; nodes.len()];
    for (q, &n) in node_of.iter().enumerate().rev() {
        digest[n] = leaf_digests[q];
    }
    for level in 0..n_levels {
        let runs = parent_runs(&nodes, arity);
        let mut next_nodes = Vec::with_capacity(runs.len());
        let mut next_digest = Vec::with_capacity(runs.len());
        let mut children = Vec::with_capacity(runs.len());
        let mut run_of = vec![0usize; nodes.len()];
        for (r, &(first, end)) in runs.iter().enumerate() {
            let mut ch: [Option<[Goldilocks; 4]>; MAX_ARITY] = [None; MAX_ARITY];
            for n in first..end {
                ch[(nodes[n] % arity) as usize] = Some(digest[n]);
                run_of[n] = r;
            }
            let mut full = [[Goldilocks::ZERO; 4]; MAX_ARITY];
            let mut state = NH::State::default();
            for j in 0..a {
                full[j] = match ch[j] {
                    Some(d) => d,
                    None => {
                        let words = kept.get(*kp..*kp + 4)?;
                        *kp += 4;
                        core::array::from_fn(|i| Goldilocks::new(words[i]))
                    }
                };
                state.as_mut()[j * 4..j * 4 + 4].copy_from_slice(&full[j]);
            }
            NH::hash(&mut state);
            let s = state.as_ref();
            next_nodes.push(nodes[first] / arity);
            next_digest.push([s[0], s[1], s[2], s[3]]);
            children.push(full);
        }
        for (q, n) in node_of.iter_mut().enumerate() {
            let (ch, curr) = (&children[run_of[*n]], (nodes[*n] % arity) as usize);
            for j in (0..a).filter(|&j| j != curr) {
                let at = (q * n_levels + level) * w + slot(j, curr) * 4;
                for (o, x) in out[at..at + 4].iter_mut().zip(ch[j]) {
                    *o = x.as_canonical_u64();
                }
            }
            *n = run_of[*n];
        }
        nodes = next_nodes;
        digest = next_digest;
    }
    Some(out)
}

/// The wire form of a flat proof (`[n_publics, publics…, proof…]`); `vk` seeds the transcript.
pub fn prune_proof<T: TranscriptLike<Goldilocks>>(proof: &[u64], vk: &[u64], layout: &ProofLayout) -> Option<Vec<u64>> {
    if vk.len() < 4 {
        return None;
    }
    let (split, sibling_blocks) = split(layout, proof, true)?;
    let indices = tree_indices::<T>(layout, &split, vk)?;
    let body = 1 + split.dense[0] as usize;
    let mut out = split.dense[..body].to_vec();
    out.push(MULTIPROOF_TAG);
    out.extend_from_slice(&split.dense[body..]);
    for ((idx, (_, _, n_levels)), sibs) in indices.iter().zip(trees(layout)).zip(sibling_blocks) {
        prune_tree(idx, n_levels, layout.arity, sibs, &mut out);
    }
    Some(out)
}

/// The flat proof a wire proof came from; `None` unless `wire` is exactly a wire form.
pub fn expand_proof<LH, NH, T>(wire: &[u64], vk: &[u64], layout: &ProofLayout) -> Option<Vec<u64>>
where
    LH: Hash<Goldilocks>,
    NH: Hash<Goldilocks>,
    T: TranscriptLike<Goldilocks>,
{
    if vk.len() < 4 {
        return None;
    }
    // A node hash must take exactly `arity` children.
    if !is_multiproof(wire) || layout.arity as usize * 4 != NH::WIDTH {
        return None;
    }
    let tag_at = 1 + wire[0] as usize;
    if wire.iter().enumerate().any(|(i, &w)| i != tag_at && w >= Goldilocks::ORDER_U64) {
        return None;
    }
    let pruned: Vec<u64> = wire[..tag_at].iter().chain(&wire[tag_at + 1..]).copied().collect();
    let (split, _) = split(layout, &pruned, false)?;
    let kept = &pruned[split.dense.len()..];
    let indices = tree_indices::<T>(layout, &split, vk)?;

    let mut out = Vec::with_capacity(pruned.len() * 2);
    let mut dp = 0usize;
    let mut kp = 0usize;
    for (tree, (idx, (width, n_queries, n_levels))) in indices.iter().zip(trees(layout)).enumerate() {
        let leaves_end = split.leaves[tree] + n_queries * width;
        let digests: Vec<[Goldilocks; 4]> = split.dense[split.leaves[tree]..leaves_end]
            .chunks(width)
            .map(|leaf| LH::linear_hash(&leaf.iter().map(|&w| Goldilocks::new(w)).collect::<Vec<_>>()))
            .collect();
        out.extend_from_slice(&split.dense[dp..leaves_end]);
        out.extend(expand_tree::<NH>(idx, n_levels, layout.arity, &digests, kept, &mut kp)?);
        dp = leaves_end;
    }
    if kp != kept.len() {
        return None;
    }
    out.extend_from_slice(&split.dense[dp..]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::{Blake3Transcript, Blake3_8, Poseidon2_16};

    /// Every level of a full tree over `leaves` digests, leaves first.
    fn build<NH: Hash<Goldilocks>>(leaves: &[[Goldilocks; 4]], arity: usize) -> Vec<Vec<[Goldilocks; 4]>> {
        let mut levels = vec![leaves.to_vec()];
        while levels.last().unwrap().len() > 1 {
            let next = levels
                .last()
                .unwrap()
                .chunks(arity)
                .map(|ch| {
                    let mut s = NH::State::default();
                    for (j, n) in ch.iter().enumerate() {
                        s.as_mut()[j * 4..j * 4 + 4].copy_from_slice(n);
                    }
                    NH::hash(&mut s);
                    let s = s.as_ref();
                    [s[0], s[1], s[2], s[3]]
                })
                .collect();
            levels.push(next);
        }
        levels
    }

    fn paths(levels: &[Vec<[Goldilocks; 4]>], idx: &[u64], arity: u64, n_levels: usize) -> Vec<u64> {
        let mut out = Vec::new();
        for &i in idx {
            let mut p = i;
            for level in levels.iter().take(n_levels) {
                let parent = p / arity;
                for j in (0..arity).filter(|&j| j != p % arity) {
                    out.extend(level[(parent * arity + j) as usize].iter().map(|x| x.as_canonical_u64()));
                }
                p = parent;
            }
        }
        out
    }

    fn round_trip<NH: Hash<Goldilocks>>(arity: u64, log_leaves: u32, idx: &[u64], n_levels: usize) -> usize {
        let leaves: Vec<[Goldilocks; 4]> =
            (0..1u64 << log_leaves).map(|i| core::array::from_fn(|j| Goldilocks::new(i * 7 + j as u64 + 1))).collect();
        let levels = build::<NH>(&leaves, arity as usize);
        let flat = paths(&levels, idx, arity, n_levels);
        let mut pruned = Vec::new();
        prune_tree(idx, n_levels, arity, &flat, &mut pruned);
        let digests: Vec<_> = idx.iter().map(|&i| leaves[i as usize]).collect();
        let mut kp = 0;
        assert_eq!(expand_tree::<NH>(idx, n_levels, arity, &digests, &pruned, &mut kp), Some(flat));
        assert_eq!(kp, pruned.len(), "every kept sibling is consumed");
        // One word short is rejected, not read past.
        let mut kp = 0;
        assert_eq!(expand_tree::<NH>(idx, n_levels, arity, &digests, &pruned[..pruned.len() - 1], &mut kp), None);
        pruned.len()
    }

    #[test]
    fn binary_paths_round_trip_and_share_the_top() {
        let idx = [3u64, 9, 9, 200, 201, 77, 1023, 0, 512];
        let full = idx.len() * 10 * 4;
        let kept = round_trip::<Blake3_8>(2, 10, &idx, 10);
        assert!(kept < full, "{kept} kept of {full}");
        // A cut-off path (a published last level) prunes the same way below the cut.
        round_trip::<Blake3_8>(2, 10, &idx, 6);
    }

    #[test]
    fn arity_four_paths_round_trip() {
        let idx = [5u64, 6, 63, 64, 1000, 4095, 2048, 5];
        round_trip::<Poseidon2_16>(4, 12, &idx, 6);
    }

    fn tiny_fri_layout() -> ProofLayout {
        ProofLayout {
            n_stages: 2,
            n_constants: 2,
            n_evals: 2,
            n_bits_ext: 6,
            num_vals: vec![2, 2, 2],
            arity: 2,
            last_level_verification: 0,
            hash_commits: true,
            ldt: LowDegreeLayout::Fri { steps: vec![6, 4, 2], n_queries: 3 },
        }
    }

    #[test]
    fn a_huge_public_count_is_rejected_not_a_panic() {
        let layout = tiny_fri_layout();
        for n_publics in [u64::MAX, u64::MAX - 1, 1 << 40] {
            let mut proof = vec![0u64; 64];
            proof[0] = n_publics;
            assert!(prune_proof::<Blake3Transcript<Goldilocks>>(&proof, &[0; 4], &layout).is_none());
            assert!(
                expand_proof::<Blake3_8, Blake3_8, Blake3Transcript<Goldilocks>>(&proof, &[0; 4], &layout).is_none()
            );
        }
    }

    #[test]
    fn an_arity_the_node_hash_cannot_take_is_refused() {
        let layout = ProofLayout { arity: 4, ..tiny_fri_layout() };
        let mut wire = vec![0u64; 4096];
        wire[1] = MULTIPROOF_TAG;
        assert!(expand_proof::<Blake3_8, Blake3_8, Blake3Transcript<Goldilocks>>(&wire, &[0; 4], &layout).is_none());
    }

    #[test]
    fn a_single_query_keeps_its_whole_path() {
        assert_eq!(round_trip::<Blake3_8>(2, 8, &[42], 8), 8 * 4);
    }
}
