//! Semantics-preserving optimization of the Q expression IR.
//!
//! 1. **CSE** — hash-consed DAG (value numbering) with commutative
//!    canonicalization of `add`/`mul` and folding of number-only ops.
//! 2. **Horner → powers** — the constraint combination `acc = vc·acc + cᵢ`
//!    is rewritten as `Σ vcʲ·cᵢ` with `vcʲ` read from a per-launch powers
//!    table (`Operand::Pow`). A base-field constraint then costs a `mul13`
//!    instead of a `mul33`, and the terms become order-independent.
//! 3. **Inner-chain folding** — the same rewrite for Horner chains over any
//!    challenge anywhere in the DAG (bus fingerprints), cost-gated; tried
//!    with and without, the variant with fewer cross-chunk slots wins.
//! 4. **Linear-term distribution** — affine terms `Σ inv_k·atom_k` are
//!    distributed over the powers so their row-invariant coefficients merge
//!    across terms and leave the per-row work.
//! 5. **Invariant hoisting** — row-invariant subexpressions are evaluated
//!    once per launch into the table (`Operand::Tab`) instead of per row.
//! 6. **Term clustering** — two candidate term orders (live-aware greedy,
//!    reverse Cuthill–McKee over shared features); each gets the full tail
//!    below on a DAG copy and the lower final max-live wins.
//! 7. **Rematerialization** — cheap shared values (≤ 3 ops) whose uses are far
//!    apart in that order are recomputed instead of kept live.
//! 8. **List scheduling** — topological order that prefers ops freeing
//!    registers, tie-broken by the clustered order.
//!
//! Field arithmetic is exact, so any topological order of the DAG computes
//! the same value; `check::equivalent` verifies every output anyway.

use crate::field;
use crate::ir::{Instr, Ir, Operand};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

type NodeId = usize;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Leaf(Operand),
    Op { op: u8, a: NodeId, b: NodeId },
}

const ADD: u8 = 0;
const SUB: u8 = 1;
const MUL: u8 = 2;

/// Diagnostic verbosity: EXPS_OPT_DEBUG=1 prints the per-AIR schedule/live
/// summary, 2 adds the longest-held shared values at the live peak. Unset or 0
/// = silent.
fn opt_debug() -> u32 {
    std::env::var("EXPS_OPT_DEBUG").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn op_code(op: &str) -> u8 {
    match op {
        "add" => ADD,
        "sub" => SUB,
        "mul" => MUL,
        other => panic!("unexpected op {other}"),
    }
}
fn op_name(code: u8) -> &'static str {
    match code {
        ADD => "add",
        SUB => "sub",
        _ => "mul",
    }
}

#[derive(Clone, Debug)]
struct Node {
    kind: Kind,
    dim: u64,
}

#[derive(Clone)]
struct Dag {
    nodes: Vec<Node>,
    memo: HashMap<Kind, NodeId>,
}

impl Dag {
    fn new() -> Self {
        Dag { nodes: Vec::new(), memo: HashMap::new() }
    }
    fn intern(&mut self, kind: Kind, dim: u64) -> NodeId {
        if let Some(&id) = self.memo.get(&kind) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(Node { kind: kind.clone(), dim });
        self.memo.insert(kind, id);
        id
    }
    fn leaf(&mut self, opnd: &Operand) -> NodeId {
        let dim = opnd.dim();
        self.intern(Kind::Leaf(opnd.clone()), dim)
    }
    fn num(&self, id: NodeId) -> Option<u64> {
        match &self.nodes[id].kind {
            Kind::Leaf(Operand::Num(v)) => Some(*v),
            _ => None,
        }
    }
    fn op(&mut self, op: u8, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.num(a), self.num(b)) {
            let v = match op {
                ADD => field::add(x, y),
                SUB => field::sub(x, y),
                _ => field::mul(x, y),
            };
            return self.leaf(&Operand::Num(v));
        }
        let (a, b) = if op != SUB && b < a { (b, a) } else { (a, b) };
        let dim = self.nodes[a].dim.max(self.nodes[b].dim);
        self.intern(Kind::Op { op, a, b }, dim)
    }
    fn children(&self, id: NodeId) -> Option<(NodeId, NodeId)> {
        match self.nodes[id].kind {
            Kind::Op { a, b, .. } => Some((a, b)),
            _ => None,
        }
    }
    fn is_leaf(&self, id: NodeId) -> bool {
        matches!(self.nodes[id].kind, Kind::Leaf(_))
    }
}

/// Summary of what the optimizer did to one expression.
#[derive(Debug, Clone, Default)]
pub struct OptStats {
    pub ops_before: usize,
    pub ops_after: usize,
    pub cost_before: u64,
    pub cost_after: u64,
    pub horner_terms: usize,
    /// Inner Horner chains folded (any challenge, anywhere but the root chain).
    pub inner_chains: usize,
    /// Whether the inner-fold variant was the one kept for this AIR.
    pub inner_selected: bool,
    pub max_live_before: usize,
    pub max_live_after: usize,
    pub remat: usize,
    /// ops moved to the once-per-launch table program, and table words exported
    pub tab_ops: usize,
    pub tab_words: u64,
    /// linear-term distribution: affine Horner terms rewritten, distinct atoms
    pub ltd_terms: usize,
    pub ltd_atoms: usize,
}

/// Instruction cost of one op by operand dims (cubic multiplies dominate).
/// Drives the remat candidate selection and the clustering weights, and the
/// reported cost totals.
fn cost(op: u8, ad: u64, bd: u64) -> u64 {
    match (op, ad, bd) {
        (MUL, 1, 1) => 4,
        (MUL, 3, 3) => 40,
        (MUL, _, _) => 12,
        (_, 3, 3) => 3,
        _ => 1,
    }
}

fn ir_cost(ir: &Ir) -> u64 {
    ir.instrs.iter().map(|i| cost(op_code(&i.op), i.a.dim(), i.b.dim())).sum()
}

/// Max number of simultaneously live tmps in a straight-line IR.
fn max_live(ir: &Ir) -> usize {
    let mut last_use: HashMap<u64, usize> = HashMap::new();
    for (i, ins) in ir.instrs.iter().enumerate() {
        for o in [&ins.a, &ins.b] {
            if let Some((t, _)) = o.as_tmp() {
                last_use.insert(t, i);
            }
        }
    }
    let mut live = 0usize;
    let mut best = 0usize;
    let mut dies_at: HashMap<usize, usize> = HashMap::new();
    for (i, ins) in ir.instrs.iter().enumerate() {
        live = live.saturating_sub(dies_at.remove(&i).unwrap_or(0));
        if ins.dst_is_tmp {
            let t = ins.dst_id.unwrap();
            match last_use.get(&t) {
                Some(&lu) if lu > i => {
                    live += 1;
                    *dies_at.entry(lu).or_default() += 1;
                }
                _ => {}
            }
        }
        best = best.max(live);
    }
    best
}

/// Build the hash-consed DAG from SSA straight-line IR. Returns the output
/// node, or `None` if the IR is not a single-output expression.
fn build_dag(ir: &Ir) -> Option<(Dag, NodeId)> {
    let mut dag = Dag::new();
    let mut tmp: HashMap<u64, NodeId> = HashMap::new();
    let mut out: Option<NodeId> = None;
    let n = ir.instrs.len();
    for (i, ins) in ir.instrs.iter().enumerate() {
        let resolve = |dag: &mut Dag, o: &Operand| -> Option<NodeId> {
            match o {
                Operand::Tmp { id, .. } => tmp.get(id).copied(),
                leaf => Some(dag.leaf(leaf)),
            }
        };
        let a = resolve(&mut dag, &ins.a)?;
        let b = resolve(&mut dag, &ins.b)?;
        let node = dag.op(op_code(&ins.op), a, b);
        if ins.dst_is_tmp {
            let id = ins.dst_id?;
            if tmp.contains_key(&id) {
                return None; // not SSA
            }
            tmp.insert(id, node);
        } else {
            if i + 1 != n {
                return None; // output store before the end
            }
            out = Some(node);
        }
    }
    // Non-Q expressions end in a tmp instead of an explicit store: root = that tmp.
    let out = match out {
        Some(o) => o,
        None => {
            let last = ir.instrs.last()?;
            let (id, _) = (last.dst_id?, last.dst_is_tmp);
            *tmp.get(&id)?
        }
    };
    Some((dag, out))
}

/// Detect the Horner chain `acc = vc·acc + c` ending at `root`. Returns the
/// challenge leaf and the terms as `(constraint_node, power)`.
fn horner_terms(dag: &Dag, root: NodeId, vc: NodeId) -> Option<Vec<(NodeId, u64)>> {
    let mut terms: Vec<(NodeId, u64)> = Vec::new();
    let mut node = root;
    let mut power = 0u64;
    loop {
        let mut next: Option<(NodeId, NodeId)> = None; // (constraint, prev_acc)
        if let Kind::Op { op: ADD, a, b } = dag.nodes[node].kind {
            for (m, c) in [(a, b), (b, a)] {
                if let Kind::Op { op: MUL, a: x, b: y } = dag.nodes[m].kind {
                    if x == vc && y != vc {
                        next = Some((c, y));
                        break;
                    }
                    if y == vc && x != vc {
                        next = Some((c, x));
                        break;
                    }
                }
            }
        }
        match next {
            Some((c, prev)) => {
                terms.push((c, power));
                power += 1;
                node = prev;
            }
            None => {
                terms.push((node, power));
                break;
            }
        }
    }
    if terms.len() >= 4 {
        Some(terms)
    } else {
        None
    }
}

/// The challenge leaf used by the constraint combination: the highest
/// challenge id referenced by the expression (`std_vc` is declared after every
/// challenge a constraint can mention).
fn combination_challenge(dag: &Dag) -> Option<NodeId> {
    let mut best: Option<(u64, NodeId)> = None;
    for (i, n) in dag.nodes.iter().enumerate() {
        if let Kind::Leaf(Operand::Ch { base }) = n.kind {
            if best.is_none_or(|(b, _)| base > b) {
                best = Some((base, i));
            }
        }
    }
    best.map(|(_, id)| id)
}

/// Reference counts (number of parent edges) of every node reachable from `root`.
fn refcounts(dag: &Dag, root: NodeId) -> Vec<u32> {
    let mut rc = vec![0u32; dag.nodes.len()];
    let mut seen = vec![false; dag.nodes.len()];
    let mut stack = vec![root];
    seen[root] = true;
    while let Some(n) = stack.pop() {
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                rc[c] += 1;
                if !seen[c] {
                    seen[c] = true;
                    stack.push(c);
                }
            }
        }
    }
    rc
}

/// Per-term sparse feature lists + feature weights. Features are shared Op
/// nodes (weight = clamped recompute cost, dim-aware) and column/const leaves
/// (a reload is one load). A shared subtree is one feature: don't descend.
fn term_feature_lists(dag: &Dag, terms: &[(NodeId, u64)], rc: &[u32], ex: &[u64]) -> (Vec<Vec<usize>>, Vec<i64>) {
    let mut feat_idx: HashMap<NodeId, usize> = HashMap::new();
    let mut weight: Vec<i64> = Vec::new();
    for (i, n) in dag.nodes.iter().enumerate() {
        let w = match &n.kind {
            Kind::Op { .. } if rc[i] > 1 => Some((ex[i].clamp(1, 200)) as i64),
            Kind::Leaf(Operand::Cm { .. }) | Kind::Leaf(Operand::Custom { .. }) | Kind::Leaf(Operand::Const { .. }) => {
                Some(2)
            }
            _ => None,
        };
        if let Some(w) = w {
            feat_idx.insert(i, weight.len());
            weight.push(w);
        }
    }
    let mut feats: Vec<Vec<usize>> = Vec::with_capacity(terms.len());
    for &(t, _) in terms {
        let mut fs: Vec<usize> = Vec::new();
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut stack = vec![t];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some(&fi) = feat_idx.get(&n) {
                fs.push(fi);
                if !dag.is_leaf(n) {
                    continue;
                }
            }
            if let Some((a, b)) = dag.children(n) {
                stack.push(a);
                stack.push(b);
            }
        }
        fs.sort_unstable();
        fs.dedup();
        feats.push(fs);
    }
    (feats, weight)
}

/// Live-set-aware term ordering. Nearest-neighbour clustering scores a
/// candidate only against the *previous* term, so it forgets every shared
/// value still in flight. Here each step picks the term that best consumes the
/// shared values already open (bonus for the use that closes one) and
/// penalizes opening new ones, weighted by recompute cost.
fn order_live(feats: &[Vec<usize>], weight: &[i64]) -> Vec<usize> {
    let mut uses_total = vec![0u32; weight.len()];
    for fs in feats {
        for &f in fs {
            uses_total[f] += 1;
        }
    }
    let mut uses_left = uses_total.clone();
    let mut opened = vec![false; weight.len()];
    let n = feats.len();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut used = vec![false; n];
    for _ in 0..n {
        let mut best = usize::MAX;
        let mut best_score = i64::MIN;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let mut sc = 0i64;
            for &f in &feats[j] {
                if uses_total[f] <= 1 {
                    continue;
                }
                if opened[f] {
                    sc += weight[f] * if uses_left[f] == 1 { 3 } else { 1 };
                } else {
                    sc -= weight[f];
                }
            }
            if sc > best_score || (sc == best_score && j < best) {
                best_score = sc;
                best = j;
            }
        }
        used[best] = true;
        order.push(best);
        for &f in &feats[best] {
            opened[f] = true;
            uses_left[f] -= 1;
        }
    }
    order
}

/// Reverse Cuthill-McKee over the term graph (terms adjacent when they share
/// a feature). The bandwidth of this order bounds how far apart any shared
/// value's users land, i.e. its live range -- the global property the greedy
/// scorers miss (Keccakf: theta values with 5 users spread ~3000 ops apart).
/// Hub features (more than `HUB` users) are dropped from adjacency: they
/// connect everything and flatten the BFS.
fn order_rcm(feats: &[Vec<usize>], nfeat: usize) -> Vec<usize> {
    const HUB: usize = 48;
    let n = feats.len();
    let mut by_feat: Vec<Vec<usize>> = vec![Vec::new(); nfeat];
    for (t, fs) in feats.iter().enumerate() {
        for &f in fs {
            by_feat[f].push(t);
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for ts in &by_feat {
        if ts.len() < 2 || ts.len() > HUB {
            continue;
        }
        for &a in ts {
            for &b in ts {
                if a != b {
                    adj[a].push(b);
                }
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let deg: Vec<usize> = adj.iter().map(|a| a.len()).collect();
    // BFS from `start` over unvisited nodes; returns (order, last level)
    let bfs = |start: usize, visited: &mut [bool]| -> (Vec<usize>, Vec<usize>) {
        let mut out = vec![start];
        visited[start] = true;
        let mut head = 0usize;
        let mut last_level: Vec<usize> = vec![start];
        while head < out.len() {
            let frontier_end = out.len();
            let mut next: Vec<usize> = Vec::new();
            while head < frontier_end {
                let u = out[head];
                head += 1;
                let mut nb: Vec<usize> = adj[u].iter().copied().filter(|&v| !visited[v]).collect();
                nb.sort_by_key(|&v| (deg[v], v));
                for v in nb {
                    if !visited[v] {
                        visited[v] = true;
                        out.push(v);
                        next.push(v);
                    }
                }
            }
            if !next.is_empty() {
                last_level = next;
            }
        }
        (out, last_level)
    };
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    for s0 in 0..n {
        if visited[s0] {
            continue;
        }
        // pseudo-peripheral start: farthest min-degree node of a probe BFS, twice
        let mut start = s0;
        for _ in 0..2 {
            let mut probe = visited.clone();
            let (_, last) = bfs(start, &mut probe);
            start = last.into_iter().min_by_key(|&v| (deg[v], v)).unwrap_or(start);
        }
        let (comp, _) = bfs(start, &mut visited);
        order.extend(comp);
    }
    order.reverse();
    order
}

/// Candidate term orders. EXPS_CLUSTER=live|rcm forces one; by default both
/// are returned and the caller keeps whichever yields the lower max-live once
/// the sum chain is actually built and scheduled (the cheap open-count proxy
/// picked wrong on Main and ArithEq, so the selection is exact).
fn cluster_candidates(dag: &Dag, terms: &[(NodeId, u64)], rc: &[u32], ex: &[u64]) -> Vec<(&'static str, Vec<usize>)> {
    let (feats, weight) = term_feature_lists(dag, terms, rc, ex);
    let nfeat = weight.len();
    match std::env::var("EXPS_CLUSTER").as_deref() {
        Ok("live") => vec![("live", order_live(&feats, &weight))],
        Ok("rcm") => vec![("rcm", order_rcm(&feats, nfeat))],
        _ => vec![("live", order_live(&feats, &weight)), ("rcm", order_rcm(&feats, nfeat))],
    }
}

/// Symbolic affine form of a row-dependent value: Σ coef_k·atom_k + cst, with
/// every `coef_k` and `cst` a row-invariant DAG node and every `atom_k` a
/// row-dependent base-field node (column, constant, Zi, or a base×base
/// product, kept opaque). `None` = not affine (e.g. a product with a cubic
/// column).
#[derive(Clone)]
struct LinForm {
    coef: Vec<(NodeId, NodeId)>, // (atom, coefficient), sorted by atom
    cst: Option<NodeId>,
}

struct LinCtx {
    memo: HashMap<NodeId, Option<LinForm>>,
    one: NodeId,
    zero: NodeId,
}

fn lin_scale(dag: &mut Dag, ctx: &LinCtx, f: &LinForm, m: NodeId) -> LinForm {
    let coef = f.coef.iter().map(|&(a, c)| (a, if c == ctx.one { m } else { dag.op(MUL, m, c) })).collect();
    let cst = f.cst.map(|c| if c == ctx.one { m } else { dag.op(MUL, m, c) });
    LinForm { coef, cst }
}

fn lin_combine(dag: &mut Dag, ctx: &LinCtx, op: u8, x: &LinForm, y: &LinForm) -> LinForm {
    // merge sorted coefficient lists
    let mut coef: Vec<(NodeId, NodeId)> = Vec::with_capacity(x.coef.len() + y.coef.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < x.coef.len() || j < y.coef.len() {
        let take_x = j >= y.coef.len() || (i < x.coef.len() && x.coef[i].0 < y.coef[j].0);
        let take_y = i >= x.coef.len() || (j < y.coef.len() && y.coef[j].0 < x.coef[i].0);
        if take_x && !take_y {
            coef.push(x.coef[i]);
            i += 1;
        } else if take_y && !take_x {
            let (a, c) = y.coef[j];
            coef.push((a, if op == ADD { c } else { dag.op(SUB, ctx.zero, c) }));
            j += 1;
        } else {
            let (a, cx) = x.coef[i];
            let (_, cy) = y.coef[j];
            coef.push((a, dag.op(op, cx, cy)));
            i += 1;
            j += 1;
        }
    }
    let cst = match (x.cst, y.cst) {
        (None, None) => None,
        (Some(c), None) => Some(c),
        (None, Some(c)) => Some(if op == ADD { c } else { dag.op(SUB, ctx.zero, c) }),
        (Some(cx), Some(cy)) => Some(dag.op(op, cx, cy)),
    };
    LinForm { coef, cst }
}

fn lin_form(dag: &mut Dag, ctx: &mut LinCtx, n: NodeId) -> Option<LinForm> {
    if let Some(v) = ctx.memo.get(&n) {
        return v.clone();
    }
    let v: Option<LinForm> = match dag.nodes[n].kind.clone() {
        Kind::Leaf(o) => {
            if invariant_leaf(&o) {
                Some(LinForm { coef: vec![], cst: Some(n) })
            } else if matches!(o, Operand::Cm { dim: 3, .. } | Operand::Custom { dim: 3, .. }) {
                None
            } else {
                Some(LinForm { coef: vec![(n, ctx.one)], cst: None })
            }
        }
        Kind::Op { op, a, b } => {
            let fa = lin_form(dag, ctx, a);
            let fb = lin_form(dag, ctx, b);
            match (fa, fb) {
                (Some(fa), Some(fb)) => {
                    let a_inv = fa.coef.is_empty();
                    let b_inv = fb.coef.is_empty();
                    if a_inv && b_inv {
                        Some(LinForm { coef: vec![], cst: Some(n) }) // invariant op: itself
                    } else if op == MUL {
                        if a_inv {
                            Some(lin_scale(dag, ctx, &fb, fa.cst.unwrap()))
                        } else if b_inv {
                            Some(lin_scale(dag, ctx, &fa, fb.cst.unwrap()))
                        } else if dag.nodes[n].dim == 1 {
                            Some(LinForm { coef: vec![(n, ctx.one)], cst: None })
                        // base x base: new atom
                        } else {
                            None
                        }
                    } else {
                        Some(lin_combine(dag, ctx, op, &fa, &fb))
                    }
                }
                _ => None,
            }
        }
    };
    ctx.memo.insert(n, v.clone());
    v
}

/// Linear-term distribution. Every Horner term that is affine in row values,
/// `c_j = Σ inv_jk·atom_k + inv_j0`, contributes
/// `vc^j·c_j = Σ (vc^j·inv_jk)·atom_k + vc^j·inv_j0`; summing over j gives
/// `Σ_k C_k·atom_k + C_0` with every C row-invariant (hoisted to the per-launch
/// table by `linearize`). The per-row work for the affine part drops to one
/// mul31 plus an add per distinct atom, and the per-term vc^j mul33 disappears.
/// Exact (distributivity over the field); the equivalence check verifies it.
/// Non-affine terms are returned unchanged.
fn distribute_linear(
    dag: &mut Dag,
    terms: &[(NodeId, u64)],
    vc: NodeId,
    base: u64,
) -> (Vec<(NodeId, u64)>, usize, usize) {
    let one = dag.leaf(&Operand::Num(1));
    let zero = dag.leaf(&Operand::Num(0));
    let mut ctx = LinCtx { memo: HashMap::new(), one, zero };
    let mut acc: Vec<(NodeId, NodeId)> = Vec::new(); // (atom, accumulated coefficient), first-seen order
    let mut acc_idx: HashMap<NodeId, usize> = HashMap::new();
    let mut acc_cst: Option<NodeId> = None;
    let mut kept: Vec<(NodeId, u64)> = Vec::new();
    let mut n_lin = 0usize;
    for &(c, p) in terms {
        let Some(f) = lin_form(dag, &mut ctx, c) else {
            kept.push((c, p));
            continue;
        };
        // Distribute only when it pays. As-is, a term costs its affine part
        // (atoms with a unit coefficient are free, base coefficients a mul11+add,
        // cubic ones a mul31+add33) plus the vc^j product (mul13 for a base
        // term, mul33 for a cubic one). Distributed, every atom costs a
        // mul31+add33 (merging across terms can only help). Base-coefficient
        // packing sums (Sha256f, Keccakf bit words) would get 3x dearer: keep.
        let mut before = 0u64;
        for &(_, coef) in &f.coef {
            before += if coef == one {
                0
            } else if dag.nodes[coef].dim == 1 {
                5
            } else {
                15
            };
        }
        let tdim = dag.nodes[c].dim;
        if p >= 1 {
            before += if tdim == 3 { 40 } else { 12 };
        }
        let after = 15 * f.coef.len() as u64;
        if after > before {
            kept.push((c, p));
            continue;
        }
        n_lin += 1;
        let m: Option<NodeId> = match p {
            0 => None,
            1 => Some(vc),
            j => Some(dag.leaf(&Operand::Pow { base, j })),
        };
        let f = match m {
            Some(m) => lin_scale(dag, &ctx, &f, m),
            None => f,
        };
        for (atom, coef) in f.coef {
            match acc_idx.get(&atom) {
                Some(&i) => {
                    let cur = acc[i].1;
                    acc[i].1 = dag.op(ADD, cur, coef);
                }
                None => {
                    acc_idx.insert(atom, acc.len());
                    acc.push((atom, coef));
                }
            }
        }
        if let Some(cst) = f.cst {
            acc_cst = Some(match acc_cst {
                None => cst,
                Some(a) => dag.op(ADD, a, cst),
            });
        }
    }
    let n_atoms = acc.len();
    let mut out: Vec<(NodeId, u64)> = Vec::with_capacity(n_atoms + 1 + kept.len());
    for (atom, coef) in acc {
        let t = if coef == one { atom } else { dag.op(MUL, coef, atom) };
        out.push((t, 0));
    }
    if let Some(cst) = acc_cst {
        out.push((cst, 0));
    }
    out.extend(kept);
    (out, n_lin, n_atoms)
}

/// Powers table needed by the `Pow` leaves read under `root`: one `(ch_base, n)`
/// region per challenge, sorted by base, sized to the largest exponent used.
fn powers_used(dag: &Dag, root: NodeId) -> Vec<(u64, u64)> {
    let mut seen = vec![false; dag.nodes.len()];
    let mut stack = vec![root];
    let mut mx: BTreeMap<u64, u64> = BTreeMap::new();
    while let Some(n) = stack.pop() {
        if seen[n] {
            continue;
        }
        seen[n] = true;
        match &dag.nodes[n].kind {
            Kind::Leaf(Operand::Pow { base, j }) => {
                let e = mx.entry(*base).or_insert(0);
                *e = (*e).max(*j);
            }
            Kind::Op { a, b, .. } => {
                stack.push(*a);
                stack.push(*b);
            }
            _ => {}
        }
    }
    mx.into_iter().map(|(b, mj)| (b, mj + 1)).collect()
}

/// Weighted cost of one op, in the same units as `distribute_linear`: a cubic
/// multiply is the expensive one, an extension-by-base far cheaper.
fn wcost(op: u8, da: u64, db: u64) -> u64 {
    let (lo, hi) = (da.min(db), da.max(db));
    if op == MUL {
        match (lo, hi) {
            (1, 1) => 1,
            (1, 3) => 3,
            _ => 6,
        }
    } else if hi == 3 {
        3
    } else {
        1
    }
}

/// Every maximal Horner chain over a single challenge, keyed by the node that
/// heads it: `(challenge leaf, [(exponent, term)])`. Generalizes `horner_terms`,
/// which only walks the one chain over `std_vc` that ends at the root, to the
/// whole DAG and every challenge -- which is what reaches the stage-2 bus
/// fingerprints `compress_exprs` builds (`e_j = sum_k a_k * alpha^k`), sitting
/// deep inside the expression rather than at its root.
///
/// A `SUB` never extends a chain (it would silently turn `a - b` into `a + b`),
/// and a chain over one challenge is never extended by a multiply against a
/// different one (that would invent a power that does not exist).
fn all_chains(dag: &Dag, order: &[NodeId]) -> HashMap<NodeId, (NodeId, Vec<(u64, NodeId)>)> {
    let mut chain: HashMap<NodeId, (NodeId, Vec<(u64, NodeId)>)> = HashMap::new();
    for &v in order {
        let Kind::Op { op, a, b } = dag.nodes[v].kind else { continue };
        if op == MUL {
            for (c, other) in [(a, b), (b, a)] {
                if !matches!(dag.nodes[c].kind, Kind::Leaf(Operand::Ch { .. })) || other == c {
                    continue;
                }
                let terms = match chain.get(&other) {
                    // extending this challenge's own chain: every exponent goes up by one
                    Some((ch, t)) if *ch == c => t.iter().map(|&(p, x)| (p + 1, x)).collect(),
                    _ => vec![(1u64, other)],
                };
                chain.insert(v, (c, terms));
                break;
            }
        } else if op == ADD {
            let (ca, cb) = (chain.get(&a).cloned(), chain.get(&b).cloned());
            match (ca, cb) {
                (Some((ch, mut t)), None) => {
                    t.push((0, b));
                    chain.insert(v, (ch, t));
                }
                (None, Some((ch, mut t))) => {
                    t.push((0, a));
                    chain.insert(v, (ch, t));
                }
                _ => {}
            }
        }
    }
    chain
}

/// Rewrite every maximal inner Horner chain into `sum_j Pow{ch,j} * term`, returning
/// the new root and how many chains fired. `skip` is the head of the chain the caller
/// handles itself (the root constraint combination), left untouched so its clustering,
/// rematerialization and scheduling still see the shape they expect.
///
/// Folding is refused unless it lowers the weighted cost: a chain of two variable terms
/// costs one `mul31` + one add as Horner, and two `mul31`s + one add folded, so it must
/// not fire there.
fn fold_inner_chains(dag: &mut Dag, root: NodeId, skip: Option<NodeId>) -> (NodeId, usize) {
    let order = post_order(dag, root);
    let chain = all_chains(dag, &order);
    let mut folded = 0usize;
    // children before parents, so a term is already rewritten when its chain is emitted
    let mut done: HashMap<NodeId, NodeId> = HashMap::new();
    for &v in &order {
        let worth = chain.get(&v).filter(|_| Some(v) != skip).and_then(|(ch, terms)| {
            let n = terms.len();
            if n < 2 {
                return None;
            }
            // Horner: the outermost term multiplies the raw operand, every later step
            // multiplies the cubic accumulator, and each remaining term costs one add.
            let head = terms.iter().max_by_key(|&&(p, _)| p).map(|&(_, x)| x).unwrap();
            let old = wcost(MUL, 3, dag.nodes[head].dim)
                + (n as u64 - 2) * wcost(MUL, 3, 3)
                + terms.iter().filter(|&&(_, x)| x != head).map(|&(_, x)| wcost(ADD, 3, dag.nodes[x].dim)).sum::<u64>();
            // Folded: one extension-by-operand multiply per non-zero exponent (`ch^0`
            // needs none) and one add per term after the first.
            let new: u64 =
                terms.iter().filter(|&&(p, _)| p > 0).map(|&(_, x)| wcost(MUL, 3, dag.nodes[x].dim)).sum::<u64>()
                    + (n as u64 - 1) * wcost(ADD, 3, 3);
            if new < old {
                Some((*ch, terms.clone()))
            } else {
                None
            }
        });
        let nv = match worth {
            Some((ch_node, terms)) => {
                let Kind::Leaf(Operand::Ch { base }) = dag.nodes[ch_node].kind else { unreachable!() };
                folded += 1;
                let mut acc: Option<NodeId> = None;
                for &(p, x) in &terms {
                    let xr = done[&x];
                    let term = match p {
                        0 => xr,
                        1 => dag.op(MUL, ch_node, xr),
                        j => {
                            let pw = dag.leaf(&Operand::Pow { base, j });
                            dag.op(MUL, pw, xr)
                        }
                    };
                    acc = Some(match acc {
                        None => term,
                        Some(a) => dag.op(ADD, a, term),
                    });
                }
                acc.expect("a folded chain always yields at least one term")
            }
            None => match dag.nodes[v].kind {
                Kind::Leaf(_) => v,
                Kind::Op { op, a, b } => {
                    let (ra, rb) = (done[&a], done[&b]);
                    if ra == a && rb == b {
                        v
                    } else {
                        dag.op(op, ra, rb)
                    }
                }
            },
        };
        done.insert(v, nv);
    }
    (done[&root], folded)
}

/// Children-before-parents order over the nodes reachable from `root`.
fn post_order(dag: &Dag, root: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    let mut seen = vec![false; dag.nodes.len()];
    let mut stack = vec![(root, false)];
    while let Some((v, expanded)) = stack.pop() {
        if expanded {
            out.push(v);
            continue;
        }
        if seen[v] {
            continue;
        }
        seen[v] = true;
        stack.push((v, true));
        if let Kind::Op { a, b, .. } = dag.nodes[v].kind {
            for c in [a, b] {
                if !seen[c] {
                    stack.push((c, false));
                }
            }
        }
    }
    out
}

/// Post-order sequence numbers of the nodes reachable from `root`,
/// visiting `a` before `b` (the clustered term order is baked into the chain).
fn sequence(dag: &Dag, root: NodeId) -> HashMap<NodeId, usize> {
    let mut seq: HashMap<NodeId, usize> = HashMap::new();
    let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];
    while let Some((n, expanded)) = stack.pop() {
        if seq.contains_key(&n) || dag.is_leaf(n) {
            continue;
        }
        if expanded {
            let k = seq.len();
            seq.insert(n, k);
        } else {
            stack.push((n, true));
            if let Some((a, b)) = dag.children(n) {
                stack.push((b, false));
                stack.push((a, false));
            }
        }
    }
    seq
}

/// Register-pressure-aware list scheduling. Returns the Op nodes in emission order.
fn schedule(dag: &Dag, root: NodeId, seq: &HashMap<NodeId, usize>) -> Vec<NodeId> {
    let rc = refcounts(dag, root);
    let mut remaining: Vec<u32> = rc.clone();
    let mut pending: Vec<u32> = vec![0; dag.nodes.len()]; // unscheduled Op children
    let mut parents: Vec<Vec<NodeId>> = vec![Vec::new(); dag.nodes.len()];
    for &n in seq.keys() {
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) {
                    pending[n] += 1;
                    parents[c].push(n);
                }
            }
        }
    }
    // heap of (score, -seq, node); stale entries are re-scored on pop
    let score = |n: NodeId, remaining: &[u32]| -> i64 {
        let mut freed = 0i64;
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) && remaining[c] == 1 {
                    freed += 1;
                }
            }
        }
        freed
    };
    // Key (score, -seq position, node): `seq` is a HashMap, so the unique node id
    // as the final tiebreak is what keeps the schedule deterministic.
    let mut heap: BinaryHeap<(i64, i64, NodeId)> = BinaryHeap::new();
    for (&n, &s) in seq.iter() {
        if pending[n] == 0 {
            heap.push((score(n, &remaining), -(s as i64), n));
        }
    }
    let mut done = vec![false; dag.nodes.len()];
    let mut order: Vec<NodeId> = Vec::with_capacity(seq.len());
    while let Some((sc, ns, n)) = heap.pop() {
        if done[n] {
            continue;
        }
        let cur = score(n, &remaining);
        if cur != sc {
            heap.push((cur, ns, n));
            continue;
        }
        done[n] = true;
        order.push(n);
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) {
                    remaining[c] -= 1;
                }
            }
        }
        for &p in &parents[n] {
            pending[p] -= 1;
            if pending[p] == 0 {
                heap.push((score(p, &remaining), -(seq[&p] as i64), p));
            }
        }
    }
    debug_assert_eq!(order.len(), seq.len());
    order
}

/// Layered order: Op nodes by longest path from the leaves, ascending, tie-broken
/// by post-order seq. Post-order is depth-first, which on a layered DAG (Poseidon's
/// uncommitted partial rounds: each round's 12 state values feed all 12 of the
/// next) keeps every layer's values alive at once; the frees-first greedy scores 0
/// for all but the last user of a value, so it degenerates to the same order.
/// Breadth-first by depth computes one layer, then the next: live ~2 layers.
/// Wrong for tree-shaped terms (leaf level first = huge live), so it is only a
/// candidate -- the caller keeps whichever order has the lowest max-live.
fn schedule_layered(dag: &Dag, seq: &HashMap<NodeId, usize>) -> Vec<NodeId> {
    let mut by_seq: Vec<(usize, NodeId)> = seq.iter().map(|(&n, &s)| (s, n)).collect();
    by_seq.sort_unstable();
    let mut depth: HashMap<NodeId, u32> = HashMap::with_capacity(seq.len());
    for &(_, n) in &by_seq {
        let mut d = 0u32;
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if let Some(&dc) = depth.get(&c) {
                    d = d.max(dc + 1);
                }
            }
        }
        depth.insert(n, d);
    }
    let mut keyed: Vec<(u32, usize, NodeId)> = by_seq.iter().map(|&(s, n)| (depth[&n], s, n)).collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, _, n)| n).collect()
}

/// Cost of recomputing `n` from leaves and from the shared nodes worth keeping
/// live ("expensive": shared and costlier than `cost_max`). Post-order over
/// `order` so children are costed first.
fn exclusive_costs(dag: &Dag, order: &[NodeId], rc: &[u32], cost_max: u64) -> Vec<u64> {
    let mut ex = vec![0u64; dag.nodes.len()];
    for &n in order {
        let Kind::Op { op, a, b } = dag.nodes[n].kind else { continue };
        let mut c = cost(op, dag.nodes[a].dim, dag.nodes[b].dim);
        for ch in [a, b] {
            if !dag.is_leaf(ch) && !(rc[ch] > 1 && ex[ch] > cost_max) {
                c += ex[ch];
            }
        }
        ex[n] = c;
    }
    ex
}

/// Fresh copy of the subtree under `n`, stopping at leaves and at expensive
/// shared nodes (which stay shared). Nodes beyond `rc`'s range are clones made
/// earlier in this pass: unshared by construction, so they are copied too.
fn clone_subtree(dag: &mut Dag, n: NodeId, rc: &[u32], ex: &[u64], cost_max: u64) -> NodeId {
    let Kind::Op { op, a, b } = dag.nodes[n].kind else { return n };
    let mut kids = [a, b];
    for k in kids.iter_mut() {
        let keep_shared = *k < rc.len() && rc[*k] > 1 && ex[*k] > cost_max;
        if !dag.is_leaf(*k) && !keep_shared {
            *k = clone_subtree(dag, *k, rc, ex, cost_max);
        }
    }
    let dim = dag.nodes[n].dim;
    let id = dag.nodes.len();
    dag.nodes.push(Node { kind: Kind::Op { op, a: kids[0], b: kids[1] }, dim });
    id
}

/// Number of Op nodes `clone_subtree` would create for `n` (same stopping
/// rule). Cost alone is a bad remat gate: a subtree of many cheap ops scores
/// low on `cost` yet duplicates a lot of instructions, which is what the
/// kernels actually issue. Memoized -- the DAG is shared, so the naive
/// recursion is exponential.
fn clone_size(dag: &Dag, n: NodeId, rc: &[u32], ex: &[u64], cost_max: u64, memo: &mut HashMap<NodeId, usize>) -> usize {
    if let Some(&v) = memo.get(&n) {
        return v;
    }
    let Kind::Op { a, b, .. } = dag.nodes[n].kind else { return 0 };
    let mut sz = 1usize;
    for k in [a, b] {
        let keep_shared = k < rc.len() && rc[k] > 1 && ex[k] > cost_max;
        if !dag.is_leaf(k) && !keep_shared {
            sz += clone_size(dag, k, rc, ex, cost_max, memo);
        }
    }
    memo.insert(n, sz);
    sz
}

/// Duplicate cheap shared subtrees for users that are far apart in the
/// schedule: recomputing them is cheaper than keeping them live (a register,
/// or a scratch slot across chunks). Returns the number of clones made.
/// Remat tier: (distance window, max recompute cost, max clone size). A use
/// further than the window from the previous use is served by a fresh clone if
/// the value is within the cost/size caps. Clones are appended to `dag.nodes`
/// directly and users are re-pointed in place, bypassing the hash-consing
/// `memo`: after this pass `Dag::op` must not be used to build new nodes (it
/// could CSE a clone back into its original), and nothing downstream does.
struct RematTier {
    window: usize,
    cost_max: u64,
    ops_max: usize,
}

fn rematerialize(dag: &mut Dag, root: NodeId, order: &[NodeId], tiers: &[RematTier]) -> usize {
    let pos: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let rc = refcounts(dag, root);
    let cost_max = tiers.iter().map(|t| t.cost_max).max().unwrap_or(0);
    let ops_max = tiers.iter().map(|t| t.ops_max).max().unwrap_or(0);
    let ex = exclusive_costs(dag, order, &rc, cost_max);
    // users of each shared node that some tier could clone, in schedule order
    let mut users: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &n in order {
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) && rc[c] > 1 && ex[c] <= cost_max {
                    users.entry(c).or_default().push(n);
                }
            }
        }
    }
    // Bound the instructions each clone duplicates, not just their cost.
    let mut szmemo: HashMap<NodeId, usize> = HashMap::new();
    let mut cands: Vec<NodeId> =
        users.keys().copied().filter(|&c| clone_size(dag, c, &rc, &ex, cost_max, &mut szmemo) <= ops_max).collect();
    cands.sort_by_key(|c| pos[c]);
    let mut clones = 0usize;
    for c in cands {
        let sz = szmemo[&c];
        let mut us = users[&c].clone();
        us.sort_by_key(|u| pos[u]);
        us.dedup();
        let mut current = c;
        let mut prev_pos = pos[&us[0]];
        for &u in &us[1..] {
            let p = pos[&u];
            let d = p - prev_pos;
            let clone_here = tiers.iter().any(|t| d > t.window && ex[c] <= t.cost_max && sz <= t.ops_max);
            if clone_here {
                current = clone_subtree(dag, c, &rc, &ex, cost_max);
                clones += 1;
            }
            prev_pos = p;
            if current != c {
                if let Kind::Op { a, b, .. } = &mut dag.nodes[u].kind {
                    if *a == c {
                        *a = current;
                    }
                    if *b == c {
                        *b = current;
                    }
                }
            }
        }
    }
    clones
}

/// Diagnostic (EXPS_OPT_DEBUG>=1): at the max-live point of `order`, bucket the
/// live values by how they are held: term-internal (single use, finishing the
/// current term) vs shared, and for shared ones by exclusive recompute cost and
/// by distance to their last use. Tells whether pressure comes from CSE keeping
/// expensive values in flight (clustering/remat problem) or from the term shape.
fn live_breakdown(dag: &Dag, root: NodeId, order: &[NodeId], label: &str) {
    let rc = refcounts(dag, root);
    let ex = exclusive_costs(dag, order, &rc, u64::MAX);
    let pos: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let mut last_use: HashMap<NodeId, usize> = HashMap::new();
    for (i, &n) in order.iter().enumerate() {
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) {
                    last_use.insert(c, i);
                }
            }
        }
    }
    let mut remaining = rc.clone();
    let mut live: HashSet<NodeId> = HashSet::new();
    let mut best = 0usize;
    let mut best_at = 0usize;
    let mut best_set: Vec<NodeId> = Vec::new();
    for (i, &n) in order.iter().enumerate() {
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) {
                    remaining[c] -= 1;
                    if remaining[c] == 0 {
                        live.remove(&c);
                    }
                }
            }
        }
        if n != root {
            live.insert(n);
        }
        if live.len() > best {
            best = live.len();
            best_at = i;
            best_set = live.iter().copied().collect();
        }
    }
    let mut single = 0usize;
    let mut shared_cheap = 0usize; // ex <= 16
    let mut shared_mid = 0usize; // 16 < ex <= 64
    let mut shared_exp = 0usize; // ex > 64
    let mut far = 0usize; // last use > 256 ops away
    let mut dist_sum = 0usize;
    for &n in &best_set {
        let d = last_use.get(&n).copied().unwrap_or(best_at).saturating_sub(best_at);
        dist_sum += d;
        if d > 256 {
            far += 1;
        }
        if rc[n] <= 1 {
            single += 1;
        } else if ex[n] <= 16 {
            shared_cheap += 1;
        } else if ex[n] <= 64 {
            shared_mid += 1;
        } else {
            shared_exp += 1;
        }
    }
    // EXPS_OPT_DEBUG>=2: structure of the longest-held shared values at the peak
    if opt_debug() >= 2 {
        let mut first_use: HashMap<NodeId, usize> = HashMap::new();
        for (i, &n) in order.iter().enumerate() {
            if let Some((a, b)) = dag.children(n) {
                for c in [a, b] {
                    if !dag.is_leaf(c) {
                        first_use.entry(c).or_insert(i);
                    }
                }
            }
        }
        let mut rows: Vec<(usize, NodeId)> =
            best_set.iter().filter(|&&n| rc[n] > 1).map(|&n| (last_use[&n].saturating_sub(pos[&n]), n)).collect();
        rows.sort_unstable_by(|a, b| b.cmp(a));
        let mut hist_users: HashMap<u32, usize> = HashMap::new();
        for &(_, n) in &rows {
            *hist_users.entry(rc[n]).or_default() += 1;
        }
        let mut hv: Vec<_> = hist_users.into_iter().collect();
        hv.sort_unstable();
        eprintln!("[exps-opt-debug]   shared-at-peak users histogram (users:count): {hv:?}");
        for &(span, n) in rows.iter().take(12) {
            let Kind::Op { op, a, b } = dag.nodes[n].kind else { continue };
            let mut upos: Vec<usize> = order
                .iter()
                .enumerate()
                .filter(|(_, &u)| matches!(dag.children(u), Some((x, y)) if x == n || y == n))
                .map(|(i, _)| i)
                .collect();
            upos.sort_unstable();
            let shown: Vec<usize> = upos.iter().copied().take(16).collect();
            eprintln!(
                "[exps-opt-debug]   node {n} {}(dim{} ,dim{}) ex={} users={} def@{} first_use@{} last_use@{} span={} uses_at={shown:?}{}",
                op_name(op), dag.nodes[a].dim, dag.nodes[b].dim, ex[n], rc[n], pos[&n], first_use.get(&n).copied().unwrap_or(0), last_use[&n], span,
                if upos.len() > 16 { "..." } else { "" }
            );
        }
    }
    eprintln!(
        "[exps-opt-debug] {label}: max-live {best} at op {best_at}/{}: single-use {single}, shared cheap(<=16) {shared_cheap}, mid(17..64) {shared_mid}, expensive(>64) {shared_exp}; far(>256 ops to last use) {far}, mean dist {}",
        order.len(),
        dist_sum.checked_div(best).unwrap_or(0)
    );
}

/// Max simultaneously live Op values for a given emission order.
fn live_of_order(dag: &Dag, root: NodeId, order: &[NodeId]) -> usize {
    let mut remaining = refcounts(dag, root);
    let mut live = 0usize;
    let mut best = 0usize;
    for &n in order {
        if let Some((a, b)) = dag.children(n) {
            for c in [a, b] {
                if !dag.is_leaf(c) {
                    remaining[c] -= 1;
                    if remaining[c] == 0 {
                        live -= 1;
                    }
                }
            }
        }
        if n != root {
            live += 1;
        }
        best = best.max(live);
    }
    best
}

/// Plain post-order (just-in-time) schedule.
fn schedule_seq(seq: &HashMap<NodeId, usize>) -> Vec<NodeId> {
    let mut v: Vec<(usize, NodeId)> = seq.iter().map(|(&n, &s)| (s, n)).collect();
    v.sort_unstable();
    v.into_iter().map(|(_, n)| n).collect()
}

/// Is this leaf the same for every row? Challenges, their powers, publics,
/// air/airgroup values and literals are; columns, constants and Zi are not.
fn invariant_leaf(o: &Operand) -> bool {
    matches!(
        o,
        Operand::Num(_)
            | Operand::Ch { .. }
            | Operand::Pow { .. }
            | Operand::Pub { .. }
            | Operand::Av { .. }
            | Operand::Agv { .. }
            | Operand::Tab { .. }
    )
}

/// Row-invariance of every node: a leaf per `invariant_leaf`, an op iff both
/// children are invariant. Computed along `order` (topological).
fn invariant_nodes(dag: &Dag, order: &[NodeId]) -> Vec<bool> {
    let mut inv = vec![false; dag.nodes.len()];
    for (i, n) in dag.nodes.iter().enumerate() {
        if let Kind::Leaf(o) = &n.kind {
            inv[i] = invariant_leaf(o);
        }
    }
    for &n in order {
        if let Kind::Op { a, b, .. } = dag.nodes[n].kind {
            inv[n] = inv[a] && inv[b];
        }
    }
    inv
}

/// Emit SSA IR for `order`. With hoisting (EXPS_HOIST != 0), every
/// row-invariant op node leaves the per-row program: it is computed in the
/// once-per-launch table program instead, and the invariant values that the
/// row-dependent ops actually consume are exported to table words and read
/// back as `Operand::Tab`. Measured on the ZisK key, Main's Q spends ~80% of
/// its instructions on such challenge-only arithmetic, repeated per row.
fn linearize(
    dag: &Dag,
    root: NodeId,
    order: &[NodeId],
    template: &Ir,
    pow: Vec<(u64, u64)>,
    stats: &mut OptStats,
    allow_hoist: bool,
) -> Ir {
    let hoist = allow_hoist && std::env::var("EXPS_HOIST").ok().is_none_or(|v| v != "0");
    let inv = invariant_nodes(dag, order);
    let hoisting = hoist && !inv[root];
    // table program: invariant op nodes in order; exports: those used by a row-dependent op
    let mut tab: Vec<Instr> = Vec::new();
    let mut tab_tmp: HashMap<NodeId, u64> = HashMap::new();
    let mut tab_out: Vec<(u64, u64, u64)> = Vec::new();
    let mut tab_word: HashMap<NodeId, u64> = HashMap::new();
    let mut tab_words = 0u64;
    if hoisting {
        let toperand = |n: NodeId, tab_tmp: &HashMap<NodeId, u64>| -> Operand {
            match &dag.nodes[n].kind {
                Kind::Leaf(o) => o.clone(),
                Kind::Op { .. } => Operand::Tmp { id: tab_tmp[&n], dim: dag.nodes[n].dim },
            }
        };
        for &n in order {
            let Kind::Op { op, a, b } = dag.nodes[n].kind else { unreachable!() };
            if !inv[n] {
                continue;
            }
            let id = tab.len() as u64;
            tab.push(Instr {
                op: op_name(op).to_string(),
                a: toperand(a, &tab_tmp),
                b: toperand(b, &tab_tmp),
                dst_is_tmp: true,
                dst_id: Some(id),
                ddim: dag.nodes[n].dim,
                idx: id as usize,
            });
            tab_tmp.insert(n, id);
        }
        for &n in order {
            let Kind::Op { a, b, .. } = dag.nodes[n].kind else { unreachable!() };
            if inv[n] {
                continue;
            }
            for c in [a, b] {
                if !dag.is_leaf(c) && inv[c] && !tab_word.contains_key(&c) {
                    let dim = dag.nodes[c].dim;
                    tab_word.insert(c, tab_words);
                    tab_out.push((tab_tmp[&c], tab_words, dim));
                    tab_words += dim;
                }
            }
        }
    }
    let mut tmp_of: HashMap<NodeId, u64> = HashMap::new();
    let mut instrs: Vec<Instr> = Vec::with_capacity(order.len());
    let operand = |n: NodeId, tmp_of: &HashMap<NodeId, u64>| -> Operand {
        match &dag.nodes[n].kind {
            Kind::Leaf(o) => o.clone(),
            Kind::Op { .. } => match tab_word.get(&n) {
                Some(&w) => Operand::Tab { idx: w, dim: dag.nodes[n].dim },
                None => Operand::Tmp { id: tmp_of[&n], dim: dag.nodes[n].dim },
            },
        }
    };
    let mut idx = 0usize;
    for &n in order {
        let Kind::Op { op, a, b } = dag.nodes[n].kind else { unreachable!() };
        if hoisting && inv[n] {
            continue;
        }
        let is_out = n == root;
        let id = idx as u64;
        instrs.push(Instr {
            op: op_name(op).to_string(),
            a: operand(a, &tmp_of),
            b: operand(b, &tmp_of),
            dst_is_tmp: !is_out,
            dst_id: if is_out { None } else { Some(id) },
            ddim: dag.nodes[n].dim,
            idx,
        });
        if !is_out {
            tmp_of.insert(n, id);
        }
        idx += 1;
    }
    stats.tab_ops = tab.len();
    stats.tab_words = tab_words;
    if opt_debug() >= 1 {
        let n_tab_opnd =
            instrs.iter().filter(|i| matches!(i.a, Operand::Tab { .. }) || matches!(i.b, Operand::Tab { .. })).count();
        let n_tmp_opnd = instrs.iter().filter(|i| i.a.as_tmp().is_some() || i.b.as_tmp().is_some()).count();
        let max_dst = instrs.iter().filter_map(|i| i.dst_id).max().unwrap_or(0);
        let max_use =
            instrs.iter().flat_map(|i| [i.a.as_tmp(), i.b.as_tmp()]).flatten().map(|(t, _)| t).max().unwrap_or(0);
        eprintln!(
            "[exps-opt-debug] linearize: instrs {} (order {}), hoisted {}, exported words {}, tab-operand instrs {n_tab_opnd}, tmp-operand instrs {n_tmp_opnd}, max dst id {max_dst}, max used tmp id {max_use}",
            instrs.len(),
            order.len(),
            tab.len(),
            tab_words
        );
    }
    Ir {
        instrs,
        ncols: template.ncols.clone(),
        n_constants: template.n_constants,
        n_bits: template.n_bits,
        pow,
        tab,
        tab_out,
        tab_words,
        pow_in_regs: false,
    }
}

/// Rematerialization: a shared value whose recompute cost is at most
/// `REMAT_COST_MAX` and whose next use is more than `REMAT_WINDOW` ops away is
/// recomputed for that use instead of kept live.
const REMAT_WINDOW: usize = 64;
const REMAT_COST_MAX: u64 = 16;
/// Max Op nodes one clone may duplicate.
const REMAT_OPS_MAX: usize = 3;
/// Tail of the pipeline for a fixed root: rematerialize far-apart cheap
/// shared values (twice: clones can expose new far-apart users), then keep the
/// candidate schedule with the lowest max-live. (A costlier "far" remat tier
/// was tried and measured worse: Keccakf 266 -> 447 ms/proof, issue-bound.) Returns (order, clones made,
/// max-live, schedule name).
fn finish(dag: &mut Dag, root: NodeId) -> (Vec<NodeId>, usize, usize, &'static str) {
    let mut order = schedule_seq(&sequence(dag, root));
    let mut remat = 0usize;
    for _ in 0..2 {
        let tiers = [RematTier { window: REMAT_WINDOW, cost_max: REMAT_COST_MAX, ops_max: REMAT_OPS_MAX }];
        let n = rematerialize(dag, root, &order, &tiers);
        remat += n;
        order = schedule_seq(&sequence(dag, root));
        if n == 0 {
            break;
        }
    }
    let seq = sequence(dag, root);
    let mut best_live = live_of_order(dag, root, &order);
    let mut winner = "postorder";
    for (name, cand) in [("greedy", schedule(dag, root, &seq)), ("layered", schedule_layered(dag, &seq))] {
        let l = live_of_order(dag, root, &cand);
        if l < best_live {
            best_live = l;
            order = cand;
            winner = name;
        }
    }
    (order, remat, best_live, winner)
}

/// Outcome of the tail for the chosen root: (dag, root, order, clones made,
/// max-live, schedule winner, cluster winner).
type Finished = (Dag, NodeId, Vec<NodeId>, usize, usize, &'static str, &'static str);

/// Optimize the Q IR. Returns the original (cloned) IR untouched if its shape
/// is not the expected single-output SSA expression.
pub fn optimize(ir: &Ir) -> (Ir, OptStats) {
    optimize_opts(ir, true, true)
}

/// `hoist`/`powers`: whether the target kernel has the per-launch table
/// (Q kernels do; the standalone non-Q expression kernels do not).
pub fn optimize_opts(ir: &Ir, hoist: bool, powers: bool) -> (Ir, OptStats) {
    let mut stats = OptStats {
        ops_before: ir.instrs.len(),
        cost_before: ir_cost(ir),
        max_live_before: max_live(ir),
        ..Default::default()
    };
    let Some((dag, out)) = build_dag(ir) else {
        let clone = linearize_identity(ir);
        stats.ops_after = stats.ops_before;
        stats.cost_after = stats.cost_before;
        stats.max_live_after = stats.max_live_before;
        return (clone, stats);
    };

    // The inner-chain fold is a per-AIR choice, made like the clustering and
    // schedule choices: both variants run the whole pipeline on a DAG copy and
    // the slot-aware rule below keeps one. Measured (1.2.0 key, 66-tx block):
    // folding lowered Main's cost 13.8% -> 10.8% but took its kernel from 0 to 6
    // cross-chunk slots -- the launcher grid shrinks with slots -- and the kernel
    // ran 4% slower; on Binary, Rom and the recursion AIRs it won 25-50%. Slots,
    // not ops, decide for a latency-bound single-chunk kernel.
    // EXPS_INNER=0 never folds, EXPS_INNER=1 always folds, unset = auto.
    let inner_mode = std::env::var("EXPS_INNER").ok();
    let candidates: Vec<bool> = match inner_mode.as_deref() {
        _ if !powers => vec![false],
        Some("0") => vec![false],
        Some(_) => vec![true],
        None => vec![false, true],
    };
    let mut best: Option<(Ir, OptStats, u64)> = None; // (ir, stats, slots at REF_CHUNK)
    for inner in candidates {
        let (cand, cstats) = optimize_variant(ir, dag.clone(), out, hoist, powers, inner, stats.clone());
        let slots = crate::ir::plan_chunks(&cand, REF_CHUNK, "variant").map(|p| p.total_slots).unwrap_or(u64::MAX);
        let take = match &best {
            None => true,
            Some((_, bstats, bslots)) => {
                // Prefer fewer slots; more slots only for a large cost win. A
                // variant that introduces slots into a slot-free kernel must
                // halve the cost to be taken.
                let cost_ratio = cstats.cost_after as f64 / bstats.cost_after.max(1) as f64;
                if slots < *bslots {
                    cost_ratio < 1.25
                } else if slots == *bslots {
                    cstats.cost_after < bstats.cost_after
                } else if *bslots == 0 {
                    cost_ratio < 0.5
                } else {
                    cost_ratio < 0.8
                }
            }
        };
        if take {
            best = Some((cand, cstats, slots));
        }
    }
    let (opt, stats, _) = best.expect("at least one variant");
    (opt, stats)
}

/// Reference chunk size for comparing the cross-chunk slot count of variants
/// (the autotuner picks the real size later; the relative comparison is the signal).
const REF_CHUNK: usize = 512;

/// One full pipeline run for a fixed inner-fold choice: Horner -> powers with
/// clustering/remat/schedule search on the root chain, then hoisting and SSA.
fn optimize_variant(
    ir: &Ir,
    mut dag: Dag,
    out: NodeId,
    hoist: bool,
    powers: bool,
    inner: bool,
    mut stats: OptStats,
) -> (Ir, OptStats) {
    // Horner -> sum of powers (optionally under a trailing `* Zi`).
    //
    // Two passes, in this order. First every INNER chain, over any challenge and anywhere
    // in the DAG -- that is what reaches the stage-2 bus fingerprints. Then the root
    // constraint combination below, which additionally gets clustering, remat and a
    // schedule search, so it is deliberately left for that machinery.
    let strip_zi = |dag: &Dag, out: NodeId| match dag.nodes[out].kind {
        Kind::Op { op: MUL, a, b } if matches!(dag.nodes[b].kind, Kind::Leaf(Operand::Zi)) => (a, Some(b)),
        Kind::Op { op: MUL, a, b } if matches!(dag.nodes[a].kind, Kind::Leaf(Operand::Zi)) => (b, Some(a)),
        _ => (out, None),
    };
    let mut out = out;
    stats.inner_selected = inner;
    if powers && inner {
        let root_chain_head = combination_challenge(&dag).map(|_| strip_zi(&dag, out).0);
        let (new_out, n) = fold_inner_chains(&mut dag, out, root_chain_head);
        stats.inner_chains = n;
        out = new_out;
    }
    let root = out;
    let mut finished: Option<Finished> = None;
    if let Some(vc) = combination_challenge(&dag).filter(|_| powers) {
        let (acc, zi) = strip_zi(&dag, out);
        if let Some(terms) = horner_terms(&dag, acc, vc) {
            stats.horner_terms = terms.len();
            let Kind::Leaf(Operand::Ch { base }) = dag.nodes[vc].kind else { unreachable!() };
            let terms = if std::env::var("EXPS_LTD").ok().is_none_or(|v| v != "0") {
                let (t, n_lin, n_atoms) = distribute_linear(&mut dag, &terms, vc, base);
                stats.ltd_terms = n_lin;
                stats.ltd_atoms = n_atoms;
                t
            } else {
                terms
            };
            // Refcounts / exclusive costs are taken on the pre-distribution
            // root: the clustering features are the subtrees the terms shared
            // in the setup's expression. Coefficients created by the
            // distribution are not features (they are row-invariant and end
            // up hoisted to the table, so they cost no per-row live range).
            let rc = refcounts(&dag, acc);
            let pre = schedule_seq(&sequence(&dag, acc));
            let ex = exclusive_costs(&dag, &pre, &rc, u64::MAX);
            let cands = cluster_candidates(&dag, &terms, &rc, &ex);
            // Every candidate term order gets the full tail (chain, remat,
            // schedule choice) on its own DAG copy; the one with the lowest
            // FINAL max-live wins (ties: fewer ops). Comparing before remat
            // picked wrong: the cheap values remat removes dominate there.
            let mut best_key: Option<(usize, usize)> = None;
            for (cname, order) in &cands {
                let mut d2 = dag.clone();
                let mut sum: Option<NodeId> = None;
                for &ti in order {
                    let (c, p) = terms[ti];
                    let term = match p {
                        0 => c,
                        1 => d2.op(MUL, vc, c),
                        j => {
                            let pw = d2.leaf(&Operand::Pow { base, j });
                            d2.op(MUL, pw, c)
                        }
                    };
                    sum = Some(match sum {
                        None => term,
                        Some(s) => d2.op(ADD, s, term),
                    });
                }
                let sum = sum.unwrap();
                let r = match zi {
                    Some(z) => d2.op(MUL, sum, z),
                    None => sum,
                };
                let (o, rm, live, sw) = finish(&mut d2, r);
                let key = (live, o.len());
                if best_key.is_none_or(|bk| key < bk) {
                    best_key = Some(key);
                    finished = Some((d2, r, o, rm, live, sw, cname));
                }
            }
        }
    }
    let (dag, root, order, remat, live, sched_winner, cluster_winner) = match finished {
        Some(f) => f,
        None => {
            let (o, rm, live, sw) = finish(&mut dag, root);
            (dag, root, o, rm, live, sw, "none")
        }
    };
    // Size the table from the powers actually read under the final root: coefficients
    // fold powers in, so term exponents alone under-count, and the inner pass emits
    // powers of challenges the root chain never touches.
    let pow = powers_used(&dag, root);
    stats.remat = remat;
    if opt_debug() >= 1 {
        eprintln!(
            "[exps-opt-debug] ops {} terms {} cluster winner: {cluster_winner}, schedule winner: {sched_winner} (max-live {live})",
            order.len(),
            stats.horner_terms
        );
        live_breakdown(&dag, root, &order, "chosen");
    }
    let opt = linearize(&dag, root, &order, ir, pow, &mut stats, hoist);
    stats.ops_after = opt.instrs.len();
    stats.cost_after = ir_cost(&opt);
    stats.max_live_after = max_live(&opt);
    (opt, stats)
}

/// Clone an IR verbatim (used when the optimizer declines).
fn linearize_identity(ir: &Ir) -> Ir {
    Ir {
        instrs: ir
            .instrs
            .iter()
            .map(|i| Instr {
                op: i.op.clone(),
                a: i.a.clone(),
                b: i.b.clone(),
                dst_is_tmp: i.dst_is_tmp,
                dst_id: i.dst_id,
                ddim: i.ddim,
                idx: i.idx,
            })
            .collect(),
        ncols: ir.ncols.clone(),
        n_constants: ir.n_constants,
        n_bits: ir.n_bits,
        pow: ir.pow.clone(),
        tab: Vec::new(),
        tab_out: Vec::new(),
        tab_words: 0,
        pow_in_regs: false,
    }
}
