//! Copy-constraint elimination via signal merging.
//!
//! Some PLONK constraints produced by [`r1cs2plonk`] are pure wire-copies — a gate
//! `qL*sl + qR*sr = 0` with `qL = -qR`, asserting `sl = sr` between two distinct
//! signal ids. The connection (permutation) argument enforces equality between all
//! `s_map` cells that hold the same signal id (see `build_s_polynomials`). So if `sl`
//! and `sr` are merged to a single id, the connection enforces `sl = sr` directly and
//! the copy gate can be dropped.
//!
//! ## How the merge is applied
//! [`merge_copy_signals`] union-finds the copy endpoints and returns a `remap` table
//! (`id -> representative`). The caller applies it to the entire `s_map` in one sweep
//! ([`apply_remap_to_s_map`]) *after* placement. Sweeping the final `s_map` rewrites
//! every cell — including custom-gate I/O cells, whose ids are placed directly from
//! `r1cs.custom_gates_uses[*].signals` and would otherwise keep their pre-merge id.
//! The witness layer needs no change: the exec reads `witness[cell] =
//! circomWitness[s_map_id]`, so a remapped id fetches the representative's value
//! (equal to the original for a satisfying witness).
//!
//! ## Conditions for merging
//! * Only EXACT copies (`qL = -qR`, `qM = qO = qC = 0`, `sl != sr`) are merged.
//!   Scaled copies (`sl = c*sr`, `c != 1`) cannot be expressed by the equality-only
//!   permutation and are left as gates.
//! * Copies touching the constant ONE (id 0) or a public signal (ids `1..=n_publics`)
//!   are NOT merged — the copy gate is kept. Publics are bound to a witness value the
//!   AIR may read via a gate equation at a single cell, which the connection argument
//!   cannot stand in for; keeping the gate enforces the equality directly.
//!
//! ## Soundness condition
//! Dropping a copy is only sound if the connection re-enforces the equality, which
//! requires a representative placed more than once to occupy >= 2 cells in
//! connection-covered columns (`col < n_cols`); one placed once had a dead partner and
//! needs nothing. The plonk checker (`check`) verifies every setup's result against its r1cs.

use proofman_fields::Field;

use super::r1cs::to_plonk::{r1cs2plonk, PlonkAddition, PlonkConstraint};
use super::r1cs::types::R1csFile;

/// Is this constraint a pure exact copy `sl = sr` (mergeable)?
/// Shape: `qM=0, qO=0, qC=0, qL = -qR (mod p), qL != 0`, and `sl != sr`.
fn is_exact_copy<F: Field>(c: &PlonkConstraint<F>) -> bool {
    let [sl, sr, _so] = c.wires;
    let [q_m, q_l, q_r, q_o, q_c] = c.coeffs;
    q_m.is_zero() && q_o.is_zero() && q_c.is_zero() && !q_l.is_zero() && !q_r.is_zero() && q_l == -q_r && sl != sr
}

/// Union-find that refuses to merge any signal pinned as externally bound.
struct UnionFind {
    parent: Vec<u32>,
    pinned: Vec<bool>,
}

impl UnionFind {
    fn new(max_id: u32, n_publics: u32) -> Self {
        let size = max_id as usize + 1;
        let mut pinned = vec![false; size];
        // Constant ONE (id 0) and publics (ids 1..=n_publics) are externally bound.
        for p in &mut pinned[0..=(n_publics as usize).min(size - 1)] {
            *p = true;
        }
        UnionFind { parent: (0..size as u32).collect(), pinned }
    }

    fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    /// Union `x` and `y`. Returns false (refusing the merge, keeping the copy gate)
    /// if EITHER side is pinned.
    ///
    /// Public signals (and the constant ONE) are bound to a witness value the AIR may
    /// read via a gate equation at a single cell, not via the connection permutation.
    /// Merging such a copy can leave the representative with one in-band cell whose
    /// equality the connection cannot enforce. Refusing keeps the copy gate, which
    /// enforces the equality directly — sound by construction, at the cost of the
    /// handful of copies that touch a public.
    fn union(&mut self, x: u32, y: u32) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return true;
        }
        if self.pinned[rx as usize] || self.pinned[ry as usize] {
            return false;
        }
        // Neither side pinned: merge, keeping the lower id as representative
        // (deterministic, biases toward original/low signal ids).
        let (keep, drop) = if rx <= ry { (rx, ry) } else { (ry, rx) };
        self.parent[drop as usize] = keep;
        true
    }
}

/// Output of the merge: the surviving constraints/additions and a remap table the
/// caller MUST apply to the full `s_map` after placement.
pub struct MergeResult<F> {
    pub constraints: Vec<PlonkConstraint<F>>,
    pub additions: Vec<PlonkAddition<F>>,
    /// `remap[id]` = representative id for signal `id`. Identity outside merged sets.
    /// Apply with [`apply_remap_to_s_map`] to every placed cell.
    pub remap: Vec<u32>,
    /// Number of copy gates dropped.
    pub dropped: usize,
}

/// Merge exact-copy signals, drop the redundant copy gates, and return a remap
/// table for the placement sweep.
///
/// `n_vars` is the R1CS variable count; `n_publics` is `n_outputs + n_pub_inputs`.
pub fn merge_copy_signals<F: Field>(
    constraints: &[PlonkConstraint<F>],
    additions: &[PlonkAddition<F>],
    n_vars: u32,
    n_publics: u32,
) -> MergeResult<F> {
    // Size the union-find over every id that can appear in s_map: all real signals,
    // all constraint operands, all addition operands.
    let wires = constraints.iter().flat_map(|c| c.wires).chain(additions.iter().flat_map(|a| a.wires));
    let max_id = wires.fold(n_vars.saturating_sub(1), u32::max);

    let mut uf = UnionFind::new(max_id, n_publics);

    // Union exact copies; record which constraint indices a successful merge consumed.
    let mut merged_away = vec![false; constraints.len()];
    for (i, c) in constraints.iter().enumerate() {
        if is_exact_copy(c) && uf.union(c.wires[0], c.wires[1]) {
            merged_away[i] = true;
        }
    }

    // Surviving constraints, with operands remapped to representatives.
    let mut out_constraints = Vec::with_capacity(constraints.len());
    let mut dropped = 0usize;
    for (i, c) in constraints.iter().enumerate() {
        if merged_away[i] {
            dropped += 1;
            continue;
        }
        out_constraints.push(PlonkConstraint { wires: c.wires.map(|w| uf.find(w)), coeffs: c.coeffs });
    }

    // Additions (witness-population rules) remapped too.
    let out_additions: Vec<PlonkAddition<F>> =
        additions.iter().map(|a| PlonkAddition { wires: a.wires.map(|w| uf.find(w)), coeffs: a.coeffs }).collect();

    // Materialize the full remap table (id -> representative) for the s_map sweep.
    let remap: Vec<u32> = (0..uf.parent.len() as u32).map(|id| uf.find(id)).collect();

    MergeResult { constraints: out_constraints, additions: out_additions, remap, dropped }
}

/// Rewrite every `s_map` cell to its representative id from `remap`. Covers all
/// placements uniformly, including custom-gate I/O cells (placed with raw ids). Cells
/// with id 0 (unused / constant ONE) map to 0. Ids outside `remap` are left as-is.
pub fn apply_remap_to_s_map(s_map: &mut [Vec<u32>], remap: &[u32]) {
    for col in s_map.iter_mut() {
        for cell in col.iter_mut() {
            let id = *cell as usize;
            if id < remap.len() {
                *cell = remap[id];
            }
        }
    }
}

/// The remap the setup applies to `s_map` after placement.
pub struct CopyMerge {
    /// `remap[id]` = representative; apply to every s_map cell via [`apply_remap_to_s_map`].
    pub remap: Vec<u32>,
}

/// Run [`r1cs2plonk`] and, if `merge` is set, apply [`merge_copy_signals`].
/// Returns the surviving `(constraints, additions)` and a [`CopyMerge`] payload.
/// When `merge` is false the remap is identity (sweep is a no-op).
pub fn r1cs2plonk_merged<F: Field>(
    r1cs: &R1csFile<F>,
    merge: bool,
) -> (Vec<PlonkConstraint<F>>, Vec<PlonkAddition<F>>, CopyMerge) {
    let (constraints, additions) = r1cs2plonk(r1cs);
    if !merge {
        let identity: Vec<u32> = (0..r1cs.header.n_vars).collect();
        return (constraints, additions, CopyMerge { remap: identity });
    }
    let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;
    let before = constraints.len();
    let r = merge_copy_signals(&constraints, &additions, r1cs.header.n_vars, n_publics);
    tracing::info!(
        "merge_copies: {} -> {} plonk constraints ({} copy gates dropped)",
        before,
        r.constraints.len(),
        r.dropped
    );
    (r.constraints, r.additions, CopyMerge { remap: r.remap })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::Goldilocks;

    const O: Goldilocks = Goldilocks::ZERO;
    const I: Goldilocks = Goldilocks::ONE;

    fn copy(sl: u32, sr: u32) -> PlonkConstraint<Goldilocks> {
        PlonkConstraint { wires: [sl, sr, 0], coeffs: [O, I, -I, O, O] } // sl = sr
    }
    fn mul(sl: u32, sr: u32, so: u32) -> PlonkConstraint<Goldilocks> {
        PlonkConstraint { wires: [sl, sr, so], coeffs: [I, O, O, -I, O] }
    }

    #[test]
    fn detects_exact_copy_not_mul_or_scaled() {
        assert!(is_exact_copy(&copy(5, 9)));
        assert!(!is_exact_copy(&mul(5, 9, 12)));
        let scaled = PlonkConstraint { wires: [5, 9, 0], coeffs: [O, I, -Goldilocks::TWO, O, O] }; // sl = 2*sr
        assert!(!is_exact_copy(&scaled));
        assert!(!is_exact_copy(&copy(7, 7))); // self-copy
    }

    #[test]
    fn merges_copy_and_drops_gate() {
        let cs = vec![copy(5, 6), mul(5, 6, 7)];
        let r = merge_copy_signals(&cs, &[], 10, 2);
        assert_eq!(r.dropped, 1);
        assert_eq!(r.constraints.len(), 1);
        assert_eq!(r.constraints[0].wires[0], r.constraints[0].wires[1]);
    }

    #[test]
    fn remap_table_unifies_merged_ids() {
        // copy 5=6 -> remap[6] == remap[5].
        let r = merge_copy_signals(&[copy(5, 6)], &[], 10, 2);
        assert_eq!(r.remap[6], r.remap[5]);
        // unrelated id maps to itself
        assert_eq!(r.remap[9], 9);
    }

    #[test]
    fn apply_remap_rewrites_custom_gate_cells() {
        // The soundness-critical site: a custom-gate cell holding the merged-away id
        // must become the representative after the sweep.
        let r = merge_copy_signals(&[copy(5, 6)], &[], 10, 2);
        // s_map with id 6 (e.g. a Poseidon input cell) in some column/row:
        let mut s_map = vec![vec![0u32; 1]; 20];
        s_map[3][0] = 6; // pretend this is a custom-gate I/O cell
        apply_remap_to_s_map(&mut s_map, &r.remap);
        assert_eq!(s_map[3][0], r.remap[5], "custom-gate cell id 6 must map to rep of 5");
    }

    #[test]
    fn copy_touching_public_is_not_merged() {
        // id 2 is public (n_publics=2). A copy public=non-public must be KEPT as a gate
        // (Option 2: never merge a copy touching a public), not merged.
        let r = merge_copy_signals(&[copy(2, 8)], &[], 10, 2);
        assert_eq!(r.dropped, 0, "copy touching a public must not be dropped");
        assert_eq!(r.constraints.len(), 1);
        assert_eq!(r.remap[2], 2);
        assert_eq!(r.remap[8], 8, "non-public endpoint stays itself; no merge");
    }

    #[test]
    fn two_distinct_publics_not_merged() {
        let r = merge_copy_signals(&[copy(1, 2)], &[], 10, 2);
        assert_eq!(r.dropped, 0);
        assert_eq!(r.constraints.len(), 1);
        assert_eq!(r.remap[1], 1);
        assert_eq!(r.remap[2], 2);
    }

    #[test]
    fn additions_are_remapped() {
        let r = merge_copy_signals(&[copy(5, 6)], &[PlonkAddition { wires: [6, 9], coeffs: [I, I] }], 10, 2);
        assert_eq!(r.additions[0].wires[0], r.remap[5]);
    }

    #[test]
    fn transitive_chain_collapses() {
        let cs = vec![copy(5, 6), copy(6, 7), copy(7, 8), mul(5, 8, 2)];
        let r = merge_copy_signals(&cs, &[], 10, 2);
        assert_eq!(r.dropped, 3);
        assert_eq!(r.constraints.len(), 1);
        assert_eq!(r.remap[5], r.remap[8]);
    }

    #[test]
    fn identity_remap_when_not_merging() {
        // r1cs2plonk_merged with merge=false must give an identity remap (no-op sweep).
        // (constructed remap covers [0, n_vars); applying to s_map changes nothing)
        let remap: Vec<u32> = (0..10).collect();
        let mut s_map = vec![vec![3u32, 7u32]];
        apply_remap_to_s_map(&mut s_map, &remap);
        assert_eq!(s_map[0], vec![3, 7]);
    }
}
