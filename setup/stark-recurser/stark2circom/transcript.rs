//! Rust port of the `Transcript` class used in `verify_global_challenge.circom.ejs`
//! and `verify_global_constraints.circom.ejs`.
//!
//! The JS class builds Circom signal declarations and wiring for a Poseidon2
//! transcript (merging public inputs / stage-1 hashes into a hash chain).
//! Each call to `put` (add one signal to the pending buffer) and `get_field`
//! (consume three outputs) drives the state machine forward; when the buffer
//! fills it emits a `Poseidon2(...)` signal declaration.

#![allow(dead_code)]

pub struct Transcript {
    /// Optional name suffix on emitted signal names (e.g. `"_bar"` → `"transcriptHash_bar_0"`).
    pub name: Option<String>,
    /// Number of rounds already emitted (index of the next `transcriptHash_N` signal).
    h_cnt: usize,
    /// Number of output values consumed from the current hash output.
    hi_cnt: usize,
    /// Index of the next available output field consumed by `get_fields1`.
    /// `out` acts as a FIFO; once exhausted a new Poseidon2 round is triggered.
    out: Vec<String>,
    /// Inputs pending for the next Poseidon2 call.
    pending: Vec<String>,
    /// The four Poseidon2 state elements.
    state: [String; 4],
    /// Generated code lines (will be joined with newlines by `get_code`).
    code: Vec<String>,
    /// Index of first code line not yet returned by `get_code`.
    last_code_printed: usize,
    /// Poseidon2 arity: how many input slots per round = `4 * (arity - 1)`.
    /// Output width = `4 * arity`.
    arity: usize,
    /// Per-round "used outputs" for the global-challenge transcript (see JS logic).
    used_vals_per_round_override: Option<Vec<usize>>,
    /// Early-rounds scheme: rounds < `early_rounds_threshold` use `early_used_vals`,
    /// remaining rounds use `late_used_vals`. None means use output_width (no drains).
    early_rounds_scheme: Option<(usize, usize, usize)>,
}

impl Transcript {
    /// Create a new Transcript for `merkle_tree_arity` trees.
    pub fn new(merkle_tree_arity: usize, name: Option<String>) -> Self {
        Self {
            name,
            h_cnt: 0,
            hi_cnt: 0,
            out: Vec::new(),
            pending: Vec::new(),
            state: ["0".into(), "0".into(), "0".into(), "0".into()],
            code: Vec::new(),
            last_code_printed: 0,
            arity: merkle_tree_arity,
            used_vals_per_round_override: None,
            early_rounds_scheme: None,
        }
    }

    /// Set a per-round override for how many output fields are "used" (all others
    /// get `_ <== ...` drain lines). Used by the global-challenge transcript.
    pub fn set_used_vals_override(&mut self, overrides: Vec<usize>) {
        self.used_vals_per_round_override = Some(overrides);
    }

    /// Set an early-rounds scheme:
    /// rounds < `threshold` → `early_used`, remaining → `late_used`.
    pub fn set_early_rounds_override(&mut self, threshold: usize, early_used: usize, late_used: usize) {
        self.early_rounds_scheme = Some((threshold, early_used, late_used));
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn signal_name(&self, idx: usize) -> String {
        match &self.name {
            Some(n) => format!("transcriptHash_{}_{idx}", n),
            None => format!("transcriptHash_{idx}"),
        }
    }

    /// Total input slots per Poseidon2 call = 4*(arity-1).
    fn input_width(&self) -> usize {
        4 * (self.arity - 1)
    }

    /// Total output width per Poseidon2 call = 4*arity.
    fn output_width(&self) -> usize {
        4 * self.arity
    }

    /// Flush pending inputs through Poseidon2 and fill `self.out`.
    fn update_state(&mut self, used_vals: Option<usize>) {
        let sig = self.signal_name(self.h_cnt);
        let out_w = self.output_width();

        // Determine how many output fields are "used" (rest get `_ <== ...`).
        let used = used_vals
            .or({
                // Early-rounds override scheme
                if let Some((threshold, early_used, late_used)) = self.early_rounds_scheme {
                    Some(if self.h_cnt < threshold { early_used } else { late_used })
                } else {
                    None
                }
            })
            .unwrap_or(out_w);

        self.code.push(format!(
            "\n    signal {sig}[{out_w}] <== Poseidon2({arity}, {out_w})([{pending}], [{state}]);",
            arity = self.arity,
            pending = self.pending.join(","),
            state = self.state.join(","),
        ));

        // Drain unused outputs.
        if used < out_w {
            self.code.push(format!("    for (var i = {used}; i < {out_w}; i++) {{"));
            self.code.push(format!("        _ <== {sig}[i]; // Unused transcript values"));
            self.code.push("    }\n".into());
        }

        // Collect all output slots.
        self.out = (0..out_w).map(|i| format!("{sig}[{i}]")).collect();

        // Update state to first 4 outputs.
        for i in 0..4 {
            self.state[i] = format!("{sig}[{i}]");
        }

        self.h_cnt += 1;
        self.hi_cnt = 0;
        self.pending.clear();
    }

    /// Consume one field from the output FIFO, triggering a new round if empty.
    fn get_fields1(&mut self) -> String {
        if self.out.is_empty() {
            // Pad pending to input_width with zeros.
            while self.pending.len() < self.input_width() {
                self.pending.push("0".into());
            }
            self.update_state(None);
        }
        let res = self.out.remove(0);
        self.hi_cnt += 1;
        res
    }

    fn _add1(&mut self, a: String) {
        self.out.clear();
        self.pending.push(a);
        if self.pending.len() == self.input_width() {
            self.update_state(None);
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Consume one field from the output FIFO (public version for use by other modules).
    pub fn get_fields1_pub(&mut self) -> String {
        self.get_fields1()
    }

    /// Add a single signal `a` to the pending inputs.
    pub fn put_single(&mut self, a: &str) {
        self._add1(a.into());
    }

    /// Add `n` elements `a[0]..a[n-1]` to pending inputs.
    pub fn put(&mut self, a: &str, n: usize) {
        for i in 0..n {
            self._add1(format!("{a}[{i}]"));
        }
    }

    /// Add a 2D slice `a[i][j]` for `i in 0..l`, `j in 0..m`.
    pub fn put_2d(&mut self, a: &str, l: usize, m: usize) {
        for i in 0..l {
            for j in 0..m {
                self._add1(format!("{a}[{i}][{j}]"));
            }
        }
    }

    /// Emit `v <== [f1, f2, f3];` consuming three output fields.
    pub fn get_field(&mut self, v: &str) {
        let f0 = self.get_fields1();
        let f1 = self.get_fields1();
        let f2 = self.get_fields1();
        self.code.push(format!("    {v} <== [{f0}, {f1}, {f2}];"));
    }

    /// Emit `v <== [f0, f1, f2, f3];` consuming four output fields.
    pub fn get_state(&mut self, v: &str) {
        let f0 = self.get_fields1();
        let f1 = self.get_fields1();
        let f2 = self.get_fields1();
        let f3 = self.get_fields1();
        self.code.push(format!("    {v} <== [{f0}, {f1}, {f2}, {f3}];"));
    }

    /// Return all code lines produced since the last `get_code()` call,
    /// indented by 4 spaces (matching the JS `getCode()` behaviour).
    pub fn get_code(&mut self) -> String {
        let lines: Vec<String> = self.code[self.last_code_printed..]
            .iter()
            .map(|l| if l.starts_with('\n') || l.is_empty() { l.clone() } else { format!("    {l}") })
            .collect();
        self.last_code_printed = self.code.len();
        lines.join("\n")
    }
}
