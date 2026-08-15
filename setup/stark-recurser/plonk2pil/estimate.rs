//! Cell estimation for the recursive verifier.
//!
//! Given only an R1CS file this computes — without compiling anything — how many
//! witness *cells* the verifier circuit actually uses, broken down per component
//! (PLONK, Poseidon Sponge, Poseidon Compression, CMul, FFT4, EvPol4,
//! TreeSelector, SelectVal1).
//!
//! A **cell** is a single value placed into the witness (`s_map`). The per-unit
//! cell costs below are read directly off the placement loops in compressor.rs /
//! aggregation.rs, so they match what the setup actually writes. The cell count
//! is the real footprint and is independent of how the rows are laid out.

use proofman_common::hash_family::GateRole;

use super::merge_copies::r1cs2plonk_merged;
use super::r1cs::to_plonk::{get_custom_gates_info, r1cs2plonk};
use super::r1cs::types::R1csFile;

/// Cells one PLONK constraint occupies: sl, sr, so.
pub const PLONK_CELLS: usize = 3;

/// Cells a single instance of `role` writes into `s_map`, read off the placement
/// loops in the setup files.
///
/// Poseidon (compressor.rs:147-213): the `0..16` loop writes 11 cells each
/// (input, round0-4, round26-29, output) = 176; the `0..11` loop writes im1 (11)
/// plus im2 (11) = 22, so the Sponge body = 198. The Compression variant additionally
/// writes fb and sb (`s_map[16]`/`s_map[17]`), giving 200.
///
/// `GateRole::TreeSelector` is NOT fixed-width: it covers TreeSelector4 (17 signals,
/// Poseidon2) and TreeSelector8 (30 signals, Poseidon1). Its cell count must be read from
/// the actual `CustomGateUse.signals.len()` (see [`tree_selector_cells`]), so this
/// function returns `None` for it and callers must resolve it from the r1cs.
pub fn cells_per_gate(role: GateRole) -> Option<usize> {
    match role {
        GateRole::PoseidonSponge => Some(198),
        GateRole::PoseidonCompression => Some(200),
        GateRole::CMul => Some(9),        // signals.len() == 9
        GateRole::EvPol4 => Some(21),     // take(21)
        GateRole::Fft4 => Some(24),       // take(24)
        GateRole::TreeSelector => None,   // 17 (TreeSelector4) or 30 (TreeSelector8) — resolve from r1cs
        GateRole::SelectVal1 => Some(22), // take(22)
    }
}

/// Cells one TreeSelector gate writes, read from the actual signal count of its first
/// `CustomGateUse` (TreeSelector4 = 17, TreeSelector8 = 30). Falls back to 17 when the
/// role has no gate uses in this r1cs (count is then 0, so the value is irrelevant).
fn tree_selector_cells(r1cs: &R1csFile, cgi: &super::r1cs::to_plonk::CustomGatesInfo) -> usize {
    use super::r1cs::to_plonk::filter_gate_uses;
    filter_gate_uses(&r1cs.custom_gates_uses, cgi.role_id(GateRole::TreeSelector))
        .first()
        .map(|u| u.signals.len())
        .unwrap_or(17)
}

/// Per-component slice of the estimate.
#[derive(Debug, Clone)]
pub struct ComponentCells {
    pub name: &'static str,
    /// Number of gates / constraints of this component.
    pub count: usize,
    /// Cells one unit of this component uses.
    pub cells_per: usize,
    /// Total cells this component uses = `count * cells_per`.
    pub cells: usize,
}

/// Full cell estimate for an R1CS verifier circuit.
#[derive(Debug, Clone)]
pub struct CellEstimate {
    /// Total cells used (sum of per-component `cells`).
    pub total_cells: usize,
    pub components: Vec<ComponentCells>,
}

impl CellEstimate {
    /// One-line-per-component human summary.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("cells_used = {}\n", self.total_cells));
        for c in &self.components {
            s.push_str(&format!(
                "  {:21} count={:>9}  cells/unit={:>4}  cells={:>11}  ({:.1}%)\n",
                c.name,
                c.count,
                c.cells_per,
                c.cells,
                if self.total_cells == 0 { 0.0 } else { 100.0 * c.cells as f64 / self.total_cells as f64 },
            ));
        }
        s
    }
}

/// Build the per-component breakdown for a given PLONK constraint count. The custom
/// gates are read from `r1cs` (merging only relabels ids, it never adds or removes
/// gate placements), so only the PLONK count varies between the raw and merged paths.
fn estimate_with_plonk_count(r1cs: &R1csFile, n_plonk: usize) -> CellEstimate {
    let cgi = get_custom_gates_info(r1cs);
    let tree_cells = tree_selector_cells(r1cs, &cgi);

    let mk_gate = |name, role| {
        let count = cgi.n(role);
        // Fixed-width roles come from cells_per_gate; TreeSelector is width-variable
        // (TreeSelector4=17 / TreeSelector8=30), resolved from the actual signal count.
        let cells_per = cells_per_gate(role).unwrap_or(tree_cells);
        ComponentCells { name, count, cells_per, cells: count * cells_per }
    };

    let components = vec![
        ComponentCells { name: "PLONK", count: n_plonk, cells_per: PLONK_CELLS, cells: n_plonk * PLONK_CELLS },
        mk_gate("Poseidon Sponge", GateRole::PoseidonSponge),
        mk_gate("Poseidon Compression", GateRole::PoseidonCompression),
        mk_gate("CMul", GateRole::CMul),
        mk_gate("FFT4", GateRole::Fft4),
        mk_gate("EvPol4", GateRole::EvPol4),
        mk_gate("TreeSelector", GateRole::TreeSelector),
        mk_gate("SelectVal1", GateRole::SelectVal1),
    ];

    let total_cells = components.iter().map(|c| c.cells).sum();
    CellEstimate { total_cells, components }
}

/// Estimate the cells the verifier circuit for `r1cs` actually uses, with a
/// per-component breakdown. Poseidon is split into Sponge and Compression.
pub fn estimate_cells(r1cs: &R1csFile) -> CellEstimate {
    let (plonk_constraints, _adds) = r1cs2plonk(r1cs);
    estimate_with_plonk_count(r1cs, plonk_constraints.len())
}

/// Same as [`estimate_cells`] but after copy-constraint merging
/// ([`r1cs2plonk_merged`]). Merging drops pure-copy PLONK gates, so the PLONK
/// component shrinks by `dropped * PLONK_CELLS`; all other components are unchanged.
pub fn estimate_cells_merged(r1cs: &R1csFile) -> CellEstimate {
    let (plonk_constraints, _adds, _merge) = r1cs2plonk_merged(r1cs, true);
    estimate_with_plonk_count(r1cs, plonk_constraints.len())
}

/// Side-by-side cell estimate for the raw vs. copy-merged verifier circuit.
#[derive(Debug, Clone)]
pub struct CellComparison {
    pub raw: CellEstimate,
    pub merged: CellEstimate,
}

impl CellComparison {
    /// Cells eliminated by merging (raw total minus merged total).
    pub fn cells_saved(&self) -> usize {
        self.raw.total_cells.saturating_sub(self.merged.total_cells)
    }

    /// Fraction of total cells eliminated by merging, in `[0, 1]`.
    pub fn fraction_saved(&self) -> f64 {
        if self.raw.total_cells == 0 {
            0.0
        } else {
            self.cells_saved() as f64 / self.raw.total_cells as f64
        }
    }

    /// Human summary: both breakdowns followed by the savings line.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("--- raw (no merge) ---\n");
        s.push_str(&self.raw.report());
        s.push_str("--- merged (copy-constraints eliminated) ---\n");
        s.push_str(&self.merged.report());
        s.push_str(&format!("saved = {} cells ({:.1}%)\n", self.cells_saved(), 100.0 * self.fraction_saved(),));
        s
    }
}

/// Estimate both the raw and copy-merged verifier circuits and return them together.
pub fn compare_cells(r1cs: &R1csFile) -> CellComparison {
    CellComparison { raw: estimate_cells(r1cs), merged: estimate_cells_merged(r1cs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plonk2pil::r1cs::types::{R1csConstraint, R1csFile, R1csHeader};
    use std::collections::HashMap;

    /// Minimal R1CS: `n_mul` multiplication constraints (a*b=c over fresh signals) and
    /// optional custom-gate uses, for exercising the estimator without a fixture file.
    fn synthetic_r1cs(n_mul: u32, gates: Vec<crate::plonk2pil::r1cs::types::CustomGateUse>) -> R1csFile {
        use crate::plonk2pil::r1cs::types::CustomGate;
        let mut constraints = Vec::new();
        let mut sid = 1u32;
        for _ in 0..n_mul {
            let mut a = HashMap::new();
            a.insert(sid, 1u64);
            let mut b = HashMap::new();
            b.insert(sid + 1, 1u64);
            let mut c = HashMap::new();
            c.insert(sid + 2, 1u64);
            constraints.push(R1csConstraint { a, b, c });
            sid += 3;
        }
        let custom_gates = if gates.is_empty() {
            vec![]
        } else {
            vec![CustomGate { template_name: "CMul".into(), parameters: vec![] }]
        };
        R1csFile {
            header: R1csHeader {
                n8: 8,
                prime_bytes: vec![],
                n_vars: sid + 100,
                n_outputs: 0,
                n_pub_inputs: 0,
                n_prv_inputs: 0,
                n_labels: 0,
                n_constraints: n_mul,
                use_custom_gates: !gates.is_empty(),
            },
            constraints,
            wire_to_label: vec![],
            custom_gates,
            custom_gates_uses: gates,
        }
    }

    #[test]
    fn cells_per_gate_matches_placement_widths() {
        assert_eq!(cells_per_gate(GateRole::PoseidonSponge), Some(198));
        assert_eq!(cells_per_gate(GateRole::PoseidonCompression), Some(200));
        assert_eq!(cells_per_gate(GateRole::CMul), Some(9));
        assert_eq!(cells_per_gate(GateRole::Fft4), Some(24));
        // TreeSelector is width-variable (TreeSelector4=17 / TreeSelector8=30) — resolved
        // from the r1cs, not a fixed constant.
        assert_eq!(cells_per_gate(GateRole::TreeSelector), None);
        assert_eq!(PLONK_CELLS, 3);
    }

    #[test]
    fn estimate_totals_and_breakdown_are_consistent() {
        use crate::plonk2pil::r1cs::types::CustomGateUse;
        // 4 mul constraints + 2 CMul gate uses (gate id 0 -> "CMul").
        let gates = vec![CustomGateUse { id: 0, signals: vec![0; 9] }, CustomGateUse { id: 0, signals: vec![0; 9] }];
        let r1cs = synthetic_r1cs(4, gates);
        let est = estimate_cells(&r1cs);

        // total == sum of components
        let sum: usize = est.components.iter().map(|c| c.cells).sum();
        assert_eq!(sum, est.total_cells);

        // each mul constraint becomes 1 PLONK constraint = 3 cells; CMul = 2*9 = 18.
        let plonk = est.components.iter().find(|c| c.name == "PLONK").unwrap();
        assert_eq!(plonk.cells, 4 * PLONK_CELLS);
        let cmul = est.components.iter().find(|c| c.name == "CMul").unwrap();
        assert_eq!(cmul.count, 2);
        assert_eq!(cmul.cells, 2 * 9);
    }

    #[test]
    fn merged_estimate_never_exceeds_raw() {
        // With no copy constraints to merge, merged == raw; in general merged <= raw.
        let r1cs = synthetic_r1cs(4, vec![]);
        let cmp = compare_cells(&r1cs);
        assert!(cmp.merged.total_cells <= cmp.raw.total_cells);
        assert_eq!(cmp.cells_saved(), cmp.raw.total_cells - cmp.merged.total_cells);
    }

    /// Reporting harness: print the raw-vs-merged cell breakdown for one R1CS.
    /// Point it at any `.r1cs` via the `ESTIMATE_R1CS` env var (no fixture committed):
    ///   ESTIMATE_R1CS=path/to/x.r1cs cargo test -p stark-recurser estimate_report -- --ignored --nocapture
    #[test]
    #[ignore]
    fn estimate_report() {
        use crate::plonk2pil::r1cs::types::read_r1cs_from_bytes;
        let Ok(f) = std::env::var("ESTIMATE_R1CS") else {
            eprintln!("set ESTIMATE_R1CS=/path/to/file.r1cs to run this report");
            return;
        };
        let bytes = std::fs::read(&f).unwrap_or_else(|e| panic!("read {f}: {e}"));
        let r1cs = read_r1cs_from_bytes(&bytes).unwrap();
        let cmp = compare_cells(&r1cs);
        eprintln!("\n=== {f}");
        eprint!("{}", cmp.report());
    }
}
