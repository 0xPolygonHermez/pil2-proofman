//! The main's proof offsets (`proofSize`, `off*`, `OFF_*`, `L0_ROOT_PUBLICS`) against the native verifier's flat
//! proof: [`ProofLayout::segments`](proofman_verifier::ProofLayout::segments) in the order `stark_verify` reads
//! them, and `parse_stir_section` for the STIR section. That layout has neither airgroup/air values nor custom
//! commits, which the flat proof (`proof2pointer`) carries as: the values first, 3 words each, and each custom
//! commit's tree right after the fixed one, shaped like it. A BN128-hashed key has no native layout (1-word
//! digests): its offsets are not checked here.

use anyhow::{bail, ensure, Context, Result};
use proofman_common::{LowDegreeLayout, LowDegreeTest, PolMap, StarkInfo};
use proofman_fields::{CubicExtensionField, Goldilocks, PrimeField64};
use proofman_verifier::{parse_stir_section, stir_section_size_words, Segment};

use super::data::{l0_trees, L0Tree};
use super::syntax::Val;

/// A walk over the flat proof: where each section starts, -1 for one of no words.
struct Walk {
    at: usize,
    arity: u64,
}

impl Walk {
    fn take(&mut self, words: usize) -> i128 {
        let start = self.at;
        self.at += words;
        if words == 0 {
            -1
        } else {
            start as i128
        }
    }

    /// (values, siblings, last levels) of a tree segment and the last-levels segment after it.
    fn tree(&mut self, tree: &Segment, last: &Segment) -> Result<[i128; 3]> {
        let (&Segment::Tree { width, n_queries, .. }, &Segment::Data(last)) = (tree, last) else {
            bail!("the layout has no tree where the verifier reads one");
        };
        let values = width * n_queries;
        Ok([self.take(values), self.take(tree.words(self.arity) - values), self.take(last)])
    }
}

fn data(s: Option<&Segment>) -> Result<usize> {
    match s {
        Some(&Segment::Data(n)) => Ok(n),
        _ => bail!("the layout has no data segment where the verifier reads one"),
    }
}

/// The offsets `key_<name>(k)` must return for key `si`; none for a BN128-hashed key.
pub fn expected_offsets(si: &StarkInfo) -> Result<Vec<(&'static str, Val)>> {
    let ss = &si.stark_struct;
    if ss.verification_hash_type == "BN128" {
        return Ok(Vec::new());
    }
    let layout = si.proof_layout();
    let segs = layout.segments(0);
    // past the publics' count
    let mut segs = segs.iter().skip(1);
    let mut w = Walk { at: 0, arity: layout.arity };
    let len3 = |m: &Option<Vec<PolMap>>| 3 * m.as_ref().map_or(0, Vec::len);
    let off_airgroup_values = w.take(len3(&si.airgroupvalues_map));
    let off_air_values = w.take(len3(&si.airvalues_map));
    let mut extra = w.at;

    let q = si.n_stages as usize + 1;
    let roots = data(segs.next())?;
    ensure!(roots == 4 * q, "the layout has {roots} words of roots for {q} stages");
    let root_start = w.take(roots);
    let stage_root: Vec<i128> = (0..q).map(|s| root_start + 4 * s as i128).collect();
    let off_evals = w.take(data(segs.next())?);

    let (fixed_tree, fixed_last) = (segs.next().context("no fixed tree")?, segs.next().context("no last levels")?);
    let fixed = w.tree(fixed_tree, fixed_last)?;
    let &Segment::Tree { n_queries, n_levels, leaf_bits, .. } = fixed_tree else { bail!("no fixed tree") };
    let mut custom = Vec::new();
    for cc in &si.custom_commits {
        let width = si.map_sections_n.get(&format!("{}0", cc.name)).copied().unwrap_or(0) as usize;
        let start = w.at;
        custom.push(w.tree(&Segment::Tree { width, n_queries, n_levels, leaf_bits }, fixed_last)?);
        extra += w.at - start;
    }
    let mut stage = Vec::new();
    for _ in 0..q {
        stage.push(w.tree(segs.next().context("no stage tree")?, segs.next().context("no last levels")?)?);
    }

    let (mut fold_root, mut fold, mut ans, mut shake) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (off_final, off_nonce, off_betas, off_nonces);
    match &layout.ldt {
        LowDegreeLayout::Fri { steps, .. } => {
            let start = w.take(data(segs.next())?);
            fold_root = (0..steps.len() - 1).map(|f| start + 4 * f as i128).collect();
            for _ in 1..steps.len() {
                fold.push(w.tree(segs.next().context("no fold tree")?, segs.next().context("no last levels")?)?);
            }
            // the final polynomial, then the grinding nonce, which no verifier reads without grinding
            off_final = w.take(data(segs.next())? - 1);
            let nonce = w.take(1);
            let grinding = matches!(&ss.low_degree_test, LowDegreeTest::Fri(f) if f.grinding_bits_queries > 0);
            off_nonce = if grinding { nonce } else { -1 };
            (off_betas, off_nonces) = (-1, -1);
        }
        LowDegreeLayout::Stir(params) => {
            // word i of the proof is i: each field the verifier parses holds its own offset
            let proof: Vec<u64> = (0..(w.at + stir_section_size_words(params)) as u64).collect();
            let s = parse_stir_section(&proof, &mut w.at, params);
            let first = |v: Option<&Goldilocks>| v.map_or(-1, |g| g.as_canonical_u64() as i128);
            let e3 = |v: &[CubicExtensionField<Goldilocks>]| first(v.first().map(|e| &e.value[0]));
            let row = |v: &[Vec<Goldilocks>]| first(v.first().and_then(|r| r.first()));
            for f in 0..params.m() {
                fold_root.push(first(s.roots[f].first()));
                fold.push([row(&s.cosets[f]), row(&s.siblings[f]), first(s.last_levels[f].first())]);
            }
            ans = s.ans_coeffs.iter().map(|c| e3(c)).collect();
            shake = s.shake_coeffs.iter().map(|c| e3(c)).collect();
            (off_final, off_nonce, off_betas) = (e3(&s.final_pol), -1, e3(&s.betas));
            off_nonces = s.nonces.first().map_or(-1, |&n| n as i128);
        }
    }
    ensure!(
        w.at == layout.flat_proof_words() + extra,
        "the walk ends at {}, the layout has {} words and the values and custom commits {extra}",
        w.at,
        layout.flat_proof_words()
    );

    let (mut tree_values, mut tree_siblings, mut tree_last, mut tree_root, mut tree_publics) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (tree, _) in l0_trees(si) {
        let (offs, root, publics) = match tree {
            L0Tree::Stage(s) => (stage[s as usize - 1], stage_root[s as usize - 1], vec![0; 4]),
            L0Tree::Fixed => (fixed, -1, vec![0; 4]),
            L0Tree::Custom(c) => {
                let p: Vec<i128> = si.custom_commits[c].public_values.iter().map(|p| p.idx as i128).collect();
                ensure!(p.len() == 4, "custom commit {c}: its root needs 4 publics");
                (custom[c], -2, p)
            }
        };
        tree_values.push(offs[0]);
        tree_siblings.push(offs[1]);
        tree_last.push(offs[2]);
        tree_root.push(root);
        tree_publics.extend(publics);
    }
    let col = |v: &[[i128; 3]], i: usize| v.iter().map(|t| t[i]).collect::<Vec<_>>();
    let n = |x: i128| Val::Int(x);
    Ok(vec![
        ("proofSize", n(w.at as i128)),
        ("offAirgroupValues", n(off_airgroup_values)),
        ("offAirValues", n(off_air_values)),
        ("offEvals", n(off_evals)),
        ("offFinal", n(off_final)),
        ("offNonce", n(off_nonce)),
        ("offBetas", n(off_betas)),
        ("offNonces", n(off_nonces)),
        ("OFF_STAGE_ROOT", Val::ints(&stage_root)),
        ("OFF_L0_VALUES", Val::ints(&tree_values)),
        ("OFF_L0_SIBLINGS", Val::ints(&tree_siblings)),
        ("OFF_L0_LAST_LEVELS", Val::ints(&tree_last)),
        ("OFF_L0_ROOT", Val::ints(&tree_root)),
        ("L0_ROOT_PUBLICS", Val::ints(&tree_publics)),
        ("OFF_FOLD_ROOT", Val::ints(&fold_root)),
        ("OFF_FOLD_VALUES", Val::ints(&col(&fold, 0))),
        ("OFF_FOLD_SIBLINGS", Val::ints(&col(&fold, 1))),
        ("OFF_FOLD_LAST_LEVELS", Val::ints(&col(&fold, 2))),
        ("OFF_ANS_COEFFS", Val::ints(&ans)),
        ("OFF_SHAKE_COEFFS", Val::ints(&shake)),
    ])
}
