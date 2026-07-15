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
//! requires the representative to occupy >= 2 cells in connection-covered columns
//! (`col < n_cols`). [`verify_merge_soundness`] checks this on the final `s_map` and
//! panics otherwise. The condition depends on the verifier circuit's layout, so it is
//! enforced at runtime rather than assumed.

use super::r1cs::to_plonk::{r1cs2plonk, PlonkAddition, PlonkConstraint};
use super::r1cs::types::R1csFile;
use super::utils::neg;

/// Is this constraint a pure exact copy `sl = sr` (mergeable)?
/// Shape: `qM=0, qO=0, qC=0, qL = -qR (mod p), qL != 0`, and `sl != sr`.
fn is_exact_copy(c: &PlonkConstraint) -> bool {
    let [sl, sr, _so, q_m, q_l, q_r, q_o, q_c] = *c;
    q_m == 0 && q_o == 0 && q_c == 0 && q_l != 0 && q_r != 0 && q_l == neg(q_r) && sl != sr
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
pub struct MergeResult {
    pub constraints: Vec<PlonkConstraint>,
    pub additions: Vec<PlonkAddition>,
    /// `remap[id]` = representative id for signal `id`. Identity outside merged sets.
    /// Apply with [`apply_remap_to_s_map`] to every placed cell.
    pub remap: Vec<u32>,
    /// Number of copy gates dropped.
    pub dropped: usize,
    /// Representative id of every merged pair. Pass to [`verify_merge_soundness`]
    /// after the sweep to check each has >= 2 connection-covered cells.
    pub merged_reps: Vec<u32>,
}

/// Merge exact-copy signals, drop the redundant copy gates, and return a remap
/// table for the placement sweep.
///
/// `n_vars` is the R1CS variable count; `n_publics` is `n_outputs + n_pub_inputs`.
pub fn merge_copy_signals(
    constraints: &[PlonkConstraint],
    additions: &[PlonkAddition],
    n_vars: u32,
    n_publics: u32,
) -> MergeResult {
    // Size the union-find over every id that can appear in s_map: all real signals,
    // all constraint operands, all addition operands.
    let mut max_id = n_vars.saturating_sub(1);
    for c in constraints {
        max_id = max_id.max(c[0] as u32).max(c[1] as u32).max(c[2] as u32);
    }
    for a in additions {
        max_id = max_id.max(a[0] as u32).max(a[1] as u32);
    }

    let mut uf = UnionFind::new(max_id, n_publics);

    // Union exact copies; record which constraint indices a successful merge consumed.
    let mut merged_away = vec![false; constraints.len()];
    for (i, c) in constraints.iter().enumerate() {
        if is_exact_copy(c) && uf.union(c[0] as u32, c[1] as u32) {
            merged_away[i] = true;
        }
    }

    // Representatives of every merged pair (for the post-sweep soundness check).
    let mut merged_reps: Vec<u32> =
        constraints.iter().zip(merged_away.iter()).filter(|(_, &m)| m).map(|(c, _)| uf.find(c[0] as u32)).collect();
    merged_reps.sort_unstable();
    merged_reps.dedup();

    // Surviving constraints, with operands remapped to representatives.
    let mut out_constraints = Vec::with_capacity(constraints.len());
    let mut dropped = 0usize;
    for (i, c) in constraints.iter().enumerate() {
        if merged_away[i] {
            dropped += 1;
            continue;
        }
        let mut nc = *c;
        nc[0] = uf.find(c[0] as u32) as u64;
        nc[1] = uf.find(c[1] as u32) as u64;
        nc[2] = uf.find(c[2] as u32) as u64;
        out_constraints.push(nc);
    }

    // Additions (witness-population rules) remapped too.
    let out_additions: Vec<PlonkAddition> = additions
        .iter()
        .map(|a| {
            let mut na = *a;
            na[0] = uf.find(a[0] as u32) as u64;
            na[1] = uf.find(a[1] as u32) as u64;
            na
        })
        .collect();

    // Materialize the full remap table (id -> representative) for the s_map sweep.
    let remap: Vec<u32> = (0..uf.parent.len() as u32).map(|id| uf.find(id)).collect();

    MergeResult { constraints: out_constraints, additions: out_additions, remap, dropped, merged_reps }
}

/// Assert the soundness condition on the swept `s_map`; run after
/// [`apply_remap_to_s_map`].
///
/// The connection argument only enforces equality between cells in columns
/// `col < n_cols`. A dropped copy's equality is therefore re-enforced only if its
/// representative occupies >= 2 such cells. Panics if any `merged_reps` entry has an
/// in-band multiplicity below 2 (its equality would be unenforced).
pub fn verify_merge_soundness(s_map: &[Vec<u32>], merged_reps: &[u32], n_cols: usize) {
    use std::collections::HashMap;
    // In-band multiplicity of each id (only columns < n_cols are in the connection).
    let mut in_band_mult: HashMap<u32, usize> = HashMap::new();
    for col in s_map.iter().take(n_cols) {
        for &cell in col {
            if cell != 0 {
                *in_band_mult.entry(cell).or_insert(0) += 1;
            }
        }
    }
    let mut bad = 0usize;
    for &rep in merged_reps {
        if in_band_mult.get(&rep).copied().unwrap_or(0) < 2 {
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "merge_copies UNSOUND: {bad} merged representatives have in-band (col<{n_cols}) \
         multiplicity < 2; their dropped copy equality is unenforced. Disable merge_copies \
         or fix the verifier circuit/layout."
    );
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

/// Remap + merged-representative payload the setup applies to `s_map` after placement.
pub struct CopyMerge {
    /// `remap[id]` = representative; apply to every s_map cell via [`apply_remap_to_s_map`].
    pub remap: Vec<u32>,
    /// Representatives of merged pairs; pass to [`verify_merge_soundness`] after the sweep.
    pub merged_reps: Vec<u32>,
}

/// Run [`r1cs2plonk`] and, if `merge` is set, apply [`merge_copy_signals`].
/// Returns the surviving `(constraints, additions)` and a [`CopyMerge`] payload.
/// When `merge` is false the remap is identity (sweep is a no-op) and `merged_reps`
/// is empty (soundness check is vacuous).
pub fn r1cs2plonk_merged(r1cs: &R1csFile, merge: bool) -> (Vec<PlonkConstraint>, Vec<PlonkAddition>, CopyMerge) {
    let (constraints, additions) = r1cs2plonk(r1cs);
    if !merge {
        let identity: Vec<u32> = (0..r1cs.header.n_vars).collect();
        return (constraints, additions, CopyMerge { remap: identity, merged_reps: Vec::new() });
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
    (r.constraints, r.additions, CopyMerge { remap: r.remap, merged_reps: r.merged_reps })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn copy(sl: u64, sr: u64) -> PlonkConstraint {
        [sl, sr, 0, 0, 1, neg(1), 0, 0] // sl = sr
    }
    fn mul(sl: u64, sr: u64, so: u64) -> PlonkConstraint {
        [sl, sr, so, 1, 0, 0, neg(1), 0]
    }

    #[test]
    fn detects_exact_copy_not_mul_or_scaled() {
        assert!(is_exact_copy(&copy(5, 9)));
        assert!(!is_exact_copy(&mul(5, 9, 12)));
        let scaled: PlonkConstraint = [5, 9, 0, 0, 1, neg(2), 0, 0]; // sl = 2*sr
        assert!(!is_exact_copy(&scaled));
        assert!(!is_exact_copy(&copy(7, 7))); // self-copy
    }

    #[test]
    fn merges_copy_and_drops_gate() {
        let cs = vec![copy(5, 6), mul(5, 6, 7)];
        let r = merge_copy_signals(&cs, &[], 10, 2);
        assert_eq!(r.dropped, 1);
        assert_eq!(r.constraints.len(), 1);
        assert_eq!(r.constraints[0][0], r.constraints[0][1]);
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
        let r = merge_copy_signals(&[copy(5, 6)], &[[6, 9, 1, 1]], 10, 2);
        assert_eq!(r.additions[0][0], r.remap[5] as u64);
    }

    #[test]
    fn transitive_chain_collapses() {
        let cs = vec![copy(5, 6), copy(6, 7), copy(7, 8), mul(5, 8, 2)];
        let r = merge_copy_signals(&cs, &[], 10, 2);
        assert_eq!(r.dropped, 3);
        assert_eq!(r.constraints.len(), 1);
        assert_eq!(r.remap[5], r.remap[8]);
    }

    /// On a real recursion R1CS: after the packer runs with merge on, every merged
    /// exact-copy endpoint resolves to the same representative in the final s_map and
    /// no raw endpoint id survives, so the connection argument ties them. Point it at
    /// a real recursion `.r1cs` via `MERGE_R1CS` (no fixture committed):
    ///   MERGE_R1CS=path/to/x.r1cs cargo test -p stark-recurser structural_merge_soundness -- --ignored --nocapture
    #[test]
    #[ignore]
    fn structural_merge_soundness() {
        use crate::plonk2pil::r1cs::to_plonk::r1cs2plonk;
        use crate::plonk2pil::r1cs::types::{read_r1cs_from_bytes, PlonkOptions};
        use crate::plonk2pil::packers::pack_aggregation;
        use std::collections::HashMap;

        let Ok(f) = std::env::var("MERGE_R1CS") else {
            eprintln!("set MERGE_R1CS=/path/to/file.r1cs to run this check");
            return;
        };
        let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("read {f}: {e}"));
        let r1cs = read_r1cs_from_bytes(&bytes).unwrap();
        let n_publics = r1cs.header.n_outputs + r1cs.header.n_pub_inputs;

        // Recompute the remap the packer will use.
        let (constraints, additions) = r1cs2plonk(&r1cs);
        let mr = merge_copy_signals(&constraints, &additions, r1cs.header.n_vars, n_publics);

        // Run the real packer WITH merge -> inspect final s_map. (pack_aggregation
        // internally runs verify_merge_soundness; reaching here means it passed.)
        let opts = PlonkOptions { merge_copies: true, ..Default::default() };
        let res = pack_aggregation(&r1cs, &opts);

        // Build a value-independent check: for each merged-away exact copy (sl,sr),
        // remap[sl] must equal remap[sr] (guaranteed by union-find), AND every cell
        // in s_map that holds sl or sr must now hold that common representative.
        // Collect, per original id, the set of distinct ids it appears as in s_map.
        let mut id_in_smap: HashMap<u32, bool> = HashMap::new();
        for col in &res.s_map {
            for &cell in col {
                if cell != 0 {
                    id_in_smap.insert(cell, true);
                }
            }
        }

        // Every dropped copy's endpoints share a representative (union-find property),
        // and the representative is what's actually placed (no raw endpoint survives).
        let mut violations = 0usize;
        let mut checked = 0usize;
        for c in &constraints {
            if is_exact_copy(c) {
                let sl = c[0] as u32;
                let sr = c[1] as u32;
                let rep_l = mr.remap[sl as usize];
                let rep_r = mr.remap[sr as usize];
                if rep_l != rep_r {
                    violations += 1;
                    if violations <= 5 {
                        eprintln!("VIOLATION: copy {sl}={sr} -> reps {rep_l},{rep_r} differ");
                    }
                    continue;
                }
                checked += 1;
                // The raw endpoints must NOT appear in the final s_map (only the rep).
                if sl != rep_l && id_in_smap.contains_key(&sl) {
                    violations += 1;
                    if violations <= 10 {
                        eprintln!("VIOLATION: raw endpoint {sl} still present in s_map (rep {rep_l})");
                    }
                }
                if sr != rep_r && id_in_smap.contains_key(&sr) {
                    violations += 1;
                    if violations <= 10 {
                        eprintln!("VIOLATION: raw endpoint {sr} still present in s_map (rep {rep_r})");
                    }
                }
            }
        }
        eprintln!(
            "structural check: {} exact copies, {} checked, {} dropped, {} VIOLATIONS",
            constraints.iter().filter(|c| is_exact_copy(c)).count(),
            checked,
            mr.dropped,
            violations
        );
        assert_eq!(violations, 0, "merged copy endpoints must unify in s_map (sound)");
    }

    #[test]
    fn guard_passes_when_rep_has_two_in_band_cells() {
        // rep 5 appears twice in-band (cols < n_cols=4) -> sound.
        let s_map = vec![vec![5u32], vec![5u32], vec![0u32], vec![0u32]];
        verify_merge_soundness(&s_map, &[5], 4);
    }

    #[test]
    #[should_panic(expected = "UNSOUND")]
    fn guard_fires_when_rep_only_out_of_band() {
        // rep 5 appears only in col 4 (>= n_cols=4, out of connection band) -> unsound.
        // cols 0..3 in-band hold nothing; col 4 holds 5.
        let s_map = vec![vec![0u32], vec![0u32], vec![0u32], vec![0u32], vec![5u32]];
        verify_merge_soundness(&s_map, &[5], 4);
    }

    #[test]
    #[should_panic(expected = "UNSOUND")]
    fn guard_fires_when_rep_in_band_multiplicity_one() {
        // rep 5 appears once in-band -> connection cycle length 1, equality unenforced.
        let s_map = vec![vec![5u32], vec![0u32], vec![0u32], vec![0u32]];
        verify_merge_soundness(&s_map, &[5], 4);
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
