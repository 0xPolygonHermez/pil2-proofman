//! The main's DEEP tables against the native verifiers' grouping (`proofman_common::deep::fri_ev_groups`), decoded
//! as `deep.circom` (and its BN128 emulation) reads them. With A_r = vf1^{e_r} / (x - g^{o_r} xi) per opening rank
//! r, the circuit computes
//!
//! ```text
//! f_0(x) = sum_G S_G * sum_{r in R_G} A_r - sum_r A_r * K_r,
//! S_G = sum_{c in G} vf2^c * leaf_c(x),   K_r = sum_{k of r} vf2^{c_k} * evals[k],
//! ```
//!
//! the DEEP quotient of every evaluation exactly when the pairs (r, column, leaf) its column groups expand to are
//! those of the grouping, none twice, each evaluation k sits in its opening's K_r with its column's weight, and
//! g^{o_r} is the opening point's.

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};
use proofman_common::deep::fri_ev_groups;
use proofman_common::{EvMapEType, StarkInfo};
use proofman_fields::{Field, Goldilocks, PrimeField64};

use super::data::{l0_trees, L0Tree};
use super::syntax::{DataFn, Val};

/// Key `k`'s tables, as numbers.
struct Tables<'a> {
    fns: &'a BTreeMap<String, DataFn>,
    k: usize,
}

impl Tables<'_> {
    fn get(&self, name: &str) -> Result<&Val> {
        Ok(self.fns.get(name).with_context(|| format!("the main has no key_{name}"))?.value(self.k as u64))
    }

    fn int(&self, name: &str) -> Result<i128> {
        match self.get(name)? {
            Val::Int(x) => Ok(*x),
            v => bail!("key_{name}({}) is {v:?}, not a number", self.k),
        }
    }

    fn list(&self, name: &str) -> Result<Vec<i128>> {
        let Val::List(v) = self.get(name)? else { bail!("key_{name}({}) is not a list", self.k) };
        v.iter()
            .map(|x| match x {
                Val::Int(x) => Ok(*x),
                _ => bail!("key_{name}({}) is not a list of numbers", self.k),
            })
            .collect()
    }

    /// A list of `len` entries (circom writes an empty one as `[0]`).
    fn sized(&self, name: &str, len: usize) -> Result<Vec<i128>> {
        let v = self.list(name)?;
        match (len, v.as_slice()) {
            (0, [0]) => Ok(Vec::new()),
            _ if v.len() == len => Ok(v),
            _ => bail!("key_{name}({}) has {} entries, not {len}", self.k, v.len()),
        }
    }
}

fn index(x: i128, bound: usize, what: &str) -> Result<usize> {
    match usize::try_from(x) {
        Ok(i) if i < bound => Ok(i),
        _ => bail!("{what} {x} is outside 0..{bound}"),
    }
}

/// Key `k`'s DEEP tables in `fns` bind exactly key `si`'s evaluations.
pub fn check_deep(si: &StarkInfo, fns: &BTreeMap<String, DataFn>, k: usize) -> Result<()> {
    let t = Tables { fns, k };
    let groups = fri_ev_groups(si);
    let n = groups.len();
    ensure!(t.int("nOpenings")? == n as i128, "key_nOpenings({k}) is not the {n} openings with evaluations");

    let n_bits = si.stark_struct.n_bits as usize;
    let w = Goldilocks::new(Goldilocks::W[n_bits]);
    let (exponents, roots) = (t.sized("OPENING_EXPONENT", n)?, t.sized("OPENING_ROOT", n)?);
    for (r, g) in groups.iter().enumerate() {
        let o = si.opening_points[g.opening as usize];
        ensure!(exponents[r] == o as i128, "opening {r} has exponent {}, the grouping {o}", exponents[r]);
        // w has order 2^nBits
        let root = w.exp_u64(o as u64 & ((1u64 << n_bits) - 1)).as_canonical_u64();
        ensure!(roots[r] == root as i128, "opening {r} has root {}, g^{o} is {root}", roots[r]);
    }

    let refs = || groups.iter().enumerate().flat_map(|(r, g)| g.refs.iter().map(move |c| (r, c)));
    let n_pols = refs().map(|(_, c)| c.col as usize + 1).max().unwrap_or(0);
    ensure!(t.int("nPols")? == n_pols as i128, "key_nPols({k}) is not the {n_pols} columns evaluated");

    // K_r: evaluation k of opening r weighted vf2^c
    let n_evals = si.ev_map.len();
    let (counts, eval_column) = (t.sized("OPENING_NUM_EVALS", n)?, t.sized("EVAL_COLUMN", n_evals)?);
    let mut got = Vec::new();
    for (r, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            let e = got.len();
            got.push((r, e, *eval_column.get(e).context("OPENING_NUM_EVALS counts more evaluations than evMap")?));
        }
    }
    let want: Vec<(usize, usize, i128)> = refs().enumerate().map(|(e, (r, c))| (r, e, c.col as i128)).collect();
    if got != want {
        let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
        bail!(
            "the K sums bind (opening, evaluation, column) {:?} at {at}, the grouping {:?}",
            got.get(at),
            want.get(at)
        );
    }

    // the A sums: a source s < n is A_s, else prefix sum s - n, an earlier one plus A_rank
    let n_sums = index(t.int("nPrefixSums")?, n_evals * n + 1, "nPrefixSums")?;
    let sums = t.sized("PREFIX_SUMS", 2 * n_sums)?;
    let mut sum_ranks: Vec<Vec<usize>> = Vec::new();
    let source = |s: i128, sum_ranks: &[Vec<usize>]| -> Result<Vec<usize>> {
        match s < n as i128 {
            true => Ok(vec![index(s, n, "an A source")?]),
            false => Ok(sum_ranks[index(s - n as i128, sum_ranks.len(), "an earlier prefix sum")?].clone()),
        }
    };
    for j in 0..n_sums {
        let mut ranks = source(sums[2 * j], &sum_ranks)?;
        ranks.push(index(sums[2 * j + 1], n, "a prefix sum's opening")?);
        sum_ranks.push(ranks);
    }

    // S_G * sum A: (opening, column, leaf offset, dim) per column of each group, per opening of its sum
    let cols = t.list("COLUMN_GROUPS")?;
    let at = |p: usize| cols.get(p).copied().context("COLUMN_GROUPS ends early");
    let n_groups = index(t.int("nColumnGroups")?, n_pols + 1, "nColumnGroups")?;
    let (mut got, mut p) = (Vec::new(), 0);
    for _ in 0..n_groups {
        let mut ranks = source(at(p)?, &sum_ranks)?;
        let n_cols = index(at(p + 1)?, n_pols + 1, "a group's column count")?;
        p += 2;
        ranks.sort_unstable();
        ensure!(ranks.windows(2).all(|w| w[0] != w[1]), "a column group sums an opening twice: {ranks:?}");
        for _ in 0..n_cols {
            let (c, off, dim) = (at(p)?, at(p + 1)?, at(p + 2)?);
            p += 3;
            got.extend(ranks.iter().map(|&r| (r, c, off, dim)));
        }
    }
    ensure!(p == cols.len(), "COLUMN_GROUPS has {} entries past its {n_groups} groups", cols.len() - p);

    let trees = l0_trees(si);
    let mut start = vec![0u64];
    for (_, w) in &trees {
        start.push(start.last().unwrap() + w);
    }
    // The k-th ref is evMap[k]; its type names the tree, as the bucket of Q and of custom commit 0 coincide.
    let mut want = Vec::new();
    for ((r, c), ev) in refs().zip(&si.ev_map) {
        let tree = match ev.type_ {
            EvMapEType::Cm => L0Tree::Stage(u64::from(c.bucket)),
            EvMapEType::Const => L0Tree::Fixed,
            EvMapEType::Custom => L0Tree::Custom(ev.commit_id as usize),
        };
        let t = trees.iter().position(|(x, _)| *x == tree).with_context(|| format!("no L_0 tree {tree:?}"))?;
        want.push((r, c.col as i128, (start[t] + u64::from(c.offset)) as i128, i128::from(c.dim)));
    }
    got.sort_unstable();
    want.sort_unstable();
    if let Some(w) = got.windows(2).find(|w| w[0] == w[1]) {
        bail!("(opening, column, leaf, dim) {:?} is in the DEEP sums twice", w[0]);
    }
    if let Some(m) = want.iter().find(|x| got.binary_search(x).is_err()) {
        bail!("(opening, column, leaf, dim) {m:?} is in no DEEP column group");
    }
    if let Some(x) = got.iter().find(|x| want.binary_search(x).is_err()) {
        bail!("(opening, column, leaf, dim) {x:?} is in a DEEP column group but not evaluated there");
    }
    Ok(())
}
