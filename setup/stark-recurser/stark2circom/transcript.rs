//! Rust port of the `Transcript` class used in `verify_global_challenge.circom.ejs`
//! and `verify_global_constraints.circom.ejs`.
//!
//! Builds Circom signal declarations and wiring for a (GL, width 16) transcript
//! hash chain. The hash family is selected via [`set_poseidon2`](Transcript::set_poseidon2):
//! Poseidon1 (`Poseidon(nOuts)`, default) or Poseidon2 (`Poseidon2(4, nOuts)`).
//! Each round absorbs 12 inputs into a 4-cell capacity, producing 16 output cells;
//! the first 4 cells become the next state and the remaining 12 are the FIFO
//! consumed by `get_field` / `get_state` callers.
//!
//! Each call to `put` (add one signal to the pending buffer) and `get_field`
//! (consume three outputs) drives the state machine forward; when the buffer
//! fills it emits a `Poseidon(...)` signal declaration.

#![allow(dead_code)]

pub struct Transcript {
    /// Optional name suffix on emitted signal names (e.g. `"_bar"` → `"transcriptHash_bar_0"`).
    pub name: Option<String>,
    /// Number of rounds already emitted (index of the next `transcriptHash_N` signal).
    h_cnt: usize,
    /// Number of output values consumed from the current hash output.
    hi_cnt: usize,
    /// Index of the next available output field consumed by `get_fields1`.
    /// `out` acts as a FIFO; once exhausted a new Poseidon1 round is triggered.
    out: Vec<String>,
    /// Inputs pending for the next Poseidon1 call.
    pending: Vec<String>,
    /// The four Poseidon1 capacity (state) elements.
    state: [String; 4],
    /// Generated code lines (will be joined with newlines by `get_code`).
    code: Vec<String>,
    /// Index of first code line not yet returned by `get_code`.
    last_code_printed: usize,
    /// Merkle-tree arity (consumer-side parameter; Poseidon1 widths are fixed).
    arity: usize,
    /// Per-round "used outputs" for the global-challenge transcript (see JS logic).
    used_vals_per_round_override: Option<Vec<usize>>,
    /// Early-rounds scheme: rounds < `early_rounds_threshold` use `early_used_vals`,
    /// remaining rounds use `late_used_vals`. None means use output_width (no drains).
    early_rounds_scheme: Option<(usize, usize, usize)>,
    /// Counter for `transcriptN2b_N` signal names emitted by `get_permutations`.
    n2b_cnt: usize,
    /// When true, drain unused outputs of the PREVIOUS hash round BEFORE emitting
    /// the next Poseidon2 signal — matching the GL `stark_verifier.circom.ejs`
    /// `Transcript.updateState()` behaviour.  The drain is emitted from
    /// `max(hi_cnt, 4)` (the first 4 outputs are always consumed as the new state).
    ///
    /// Default: false (matches `verify_global_challenge.circom.ejs` behaviour).
    drain_in_update_state: bool,
    /// Hash family of the emitted transcript chain. `false` → Poseidon1
    /// (`Poseidon(nOuts)` template); `true` → Poseidon2 (`Poseidon2(4, nOuts)`).
    /// Both are width-16 / rate-12 sponges, so only the emitted template name differs.
    poseidon2: bool,
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
            n2b_cnt: 0,
            drain_in_update_state: false,
            poseidon2: false,
        }
    }

    /// Select the hash family for the emitted transcript chain: `true` emits the
    /// Poseidon2 sponge (`Poseidon2(4, nOuts)`), `false` (default) emits Poseidon1
    /// (`Poseidon(nOuts)`). Must be set before any `put`/`get_*` call.
    pub fn set_poseidon2(&mut self, value: bool) {
        self.poseidon2 = value;
    }

    /// Enable draining unused outputs of the previous hash BEFORE emitting the
    /// next one (matches GL `stark_verifier.circom.ejs` Transcript behaviour).
    pub fn set_drain_in_update_state(&mut self, value: bool) {
        self.drain_in_update_state = value;
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

    /// Poseidon1 absorbs 12 inputs per round (in[12] + capacity[4] = state 16).
    fn input_width(&self) -> usize {
        12
    }

    /// Poseidon1 produces 16 output cells per round (full state).
    fn output_width(&self) -> usize {
        16
    }

    /// Flush pending inputs through Poseidon1 and fill `self.out`.
    fn update_state(&mut self, used_vals: Option<usize>) {
        let sig = self.signal_name(self.h_cnt);
        let out_w = self.output_width();

        // ── Drain outputs of PREVIOUS round (stark_verifier mode) ────────────
        // Mirrors EJS: if(hCnt > 0) { firstUnused = max(hiCnt,4); drain ... }
        // The first 4 outputs are always consumed as the new state, so we
        // drain from max(hi_cnt, 4) rather than hi_cnt.
        if self.drain_in_update_state && self.h_cnt > 0 {
            let first_unused = self.hi_cnt.max(4);
            if first_unused < out_w {
                let prev_sig = self.signal_name(self.h_cnt - 1);
                self.code.push(format!(
                    "for (var i = {first_unused}; i < {out_w}; i++) {{\n        _ <== {prev_sig}[i]; // Unused transcript values \n    }}"
                ));
            }
        }

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

        // Poseidon1 uses `Poseidon(nOuts)`; Poseidon2 uses `Poseidon2(arity, nOuts)`
        // with arity fixed to 4 (the width-16 / rate-12 sponge). Both take the same
        // `[in[12]], [capacity[4]]` arguments, so only the template head differs.
        let hash_call = if self.poseidon2 { format!("Poseidon2(4, {out_w})") } else { format!("Poseidon({out_w})") };
        self.code.push(format!(
            "\n    signal {sig}[{out_w}] <== {hash_call}([{pending}], [{state}]);",
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
        self.code.push(format!("{v} <== [{f0}, {f1}, {f2}];"));
    }

    /// Emit `v <== [f0, f1, f2, f3];` consuming four output fields.
    pub fn get_state(&mut self, v: &str) {
        let f0 = self.get_fields1();
        let f1 = self.get_fields1();
        let f2 = self.get_fields1();
        let f3 = self.get_fields1();
        self.code.push(format!("{v} <== [{f0}, {f1}, {f2}, {f3}];"));
    }

    /// Consume `n_fields` output fields from the transcript, emit a
    /// `Num2Bits_strict()` signal for each, drain any remaining outputs from
    /// the last hash round, then emit the bit-assignment loops that fill `v`.
    ///
    /// Mirrors `getPermutations(v, n, nBits)` from the GL EJS `Transcript` class.
    ///
    /// - `v`          — circom signal name of the 2-D output array (e.g. `"queriesL0"`)
    /// - `n_queries`  — first dimension count (e.g. `starkStruct.nQueries`)
    /// - `query_bits` — second dimension / bits per query (e.g. `starkStruct.steps[0].nBits`)
    pub fn get_permutations(&mut self, v: &str, n_queries: usize, query_bits: usize) {
        let total_bits = n_queries * query_bits;
        // NFields = floor((totalBits - 1) / 63) + 1
        let n_fields = (total_bits - 1) / 63 + 1;
        let out_w = self.output_width();

        // ── Emit one Num2Bits_strict per required field ───────────────────────
        let mut n2b_names: Vec<String> = Vec::with_capacity(n_fields);
        for _ in 0..n_fields {
            let f = self.get_fields1();
            let name = format!("transcriptN2b_{}", self.n2b_cnt);
            self.n2b_cnt += 1;
            self.code.push(format!("signal {{binary}} {name}[64] <== Num2Bits_strict()({f});"));
            n2b_names.push(name);
        }

        // ── Drain remaining unused outputs of the last hash round ─────────────
        // JS: if(this.hiCnt < 4*arity) { code.push(`for(var i = ${hiCnt}; ...`) }
        if self.hi_cnt < out_w {
            let prev_sig = self.signal_name(self.h_cnt - 1);
            self.code.push(format!(
                "for (var i = {}; i < {out_w}; i++) {{\n        _ <== {prev_sig}[i]; // Unused transcript values        \n    }}\n",
                self.hi_cnt
            ));
        }

        // ── Bit-assignment loops ──────────────────────────────────────────────
        self.code.push(
            "// From each transcript hash converted to bits, we assign those bits to queriesL0[q] to define the query positions"
                .into(),
        );
        self.code.push("var q = 0; // Query number ".into());
        self.code.push("var b = 0; // Bit number ".into());

        for (i, name) in n2b_names.iter().enumerate() {
            // Bits consumed from this field: 63 for all but the last, remainder for last.
            let bits_this_field = if i + 1 == n_fields { total_bits - 63 * i } else { 63 };

            self.code.push(format!(
                "for (var j = 0; j < {bits_this_field}; j++) {{\n        {v}[q][b] <== {name}[j];\n        b++;\n        if(b == {query_bits}) {{\n            b = 0; \n            q++;\n        }}\n    }}"
            ));

            if bits_this_field == 63 {
                self.code.push(format!("_ <== {name}[63]; // Unused last bit\n"));
            } else {
                self.code.push(format!(
                    "for (var j = {bits_this_field}; j < 64; j++) {{\n        _ <== {name}[j]; // Unused bits        \n    }}"
                ));
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── get_permutations: single small round ─────────────────────────────────

    #[test]
    fn get_permutations_small_single_round() {
        // Poseidon1: input_width=12, output_width=16
        // n_queries=1, query_bits=3 → total_bits=3, n_fields=1
        let mut t = Transcript::new(4, Some("friQueries".into()));
        // Prime with 3 items (doesn't fill input_width=12, so no Poseidon yet).
        t.put("challengeFRIQueries", 3);
        // Call get_permutations — triggers round 0 with pending padded to 12.
        t.get_permutations("queriesL0", 1, 3);
        let code = t.get_code();

        // Round 0 Poseidon must appear (pending was 3, padded to 12).
        assert!(code.contains("Poseidon(16)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],0,0,0,0,0,0,0,0,0], [0,0,0,0])"),
            "code:\n{code}");
        // N2b signal for field 0.
        assert!(
            code.contains("signal {binary} transcriptN2b_0[64] <== Num2Bits_strict()(transcriptHash_friQueries_0[0]);"),
            "code:\n{code}"
        );
        // Drain: hi_cnt=1, 1 < 16 → drain from 1 to 16.
        assert!(code.contains("for (var i = 1; i < 16; i++)"), "drain missing:\n{code}");
        // Last field partial: total_bits=3, n_fields=1, bits_this_field = 3-63*0 = 3.
        assert!(code.contains("for (var j = 0; j < 3; j++)"), "bit-loop missing:\n{code}");
        // Partial-bits drain: 3 < 63, so "Unused bits" drain.
        assert!(code.contains("for (var j = 3; j < 64; j++)"), "unused bits drain missing:\n{code}");
        assert!(!code.contains("Unused last bit"), "should be partial not last-bit:\n{code}");
    }

    // ── Poseidon2 family emits the Poseidon2 template, not Poseidon1 ─────────

    #[test]
    fn poseidon2_emits_poseidon2_template() {
        let mut t = Transcript::new(4, Some("friQueries".into()));
        t.set_poseidon2(true);
        t.put("challengeFRIQueries", 3);
        t.get_permutations("queriesL0", 1, 3);
        let code = t.get_code();
        // Same width-16/rate-12 geometry as Poseidon1, but the `Poseidon2(4, 16)` head.
        assert!(
            code.contains("Poseidon2(4, 16)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],0,0,0,0,0,0,0,0,0], [0,0,0,0])"),
            "code:\n{code}"
        );
        assert!(!code.contains("<== Poseidon(16)("), "must not emit the Poseidon1 template:\n{code}");
    }

    // ── get_permutations: last field exactly 63 bits → "Unused last bit" ─────

    #[test]
    fn get_permutations_last_field_is_63_bits() {
        // total_bits must = 63 * n_fields exactly → last field bits = 63.
        // n_queries=1, query_bits=63 → total_bits=63, n_fields=1.
        let mut t = Transcript::new(4, Some("friQueries".into()));
        t.get_permutations("queriesL0", 1, 63);
        let code = t.get_code();
        // bits_this_field=63 → _ <== transcriptN2b_0[63]; // Unused last bit
        assert!(code.contains("_ <== transcriptN2b_0[63]; // Unused last bit"), "code:\n{code}");
        assert!(!code.contains("for (var j = 63; j < 64; j++)"), "should NOT have partial drain:\n{code}");
    }

    // ── get_permutations: integration against FibonacciSquare ground truth ───

    #[test]
    fn get_permutations_fibonacci_square_ground_truth() {
        // Reproduces the calculateFRIQueries0 template for FibonacciSquare:
        //   nQueries=229, steps[0].nBits=23, powBits=16 (nonce present)
        // With Poseidon1: input_width=12, output_width=16.
        let mut t = Transcript::new(4, Some("friQueries".into()));
        t.put("challengeFRIQueries", 3);
        t.put_single("nonce");
        t.get_permutations("queriesL0", 229, 23);
        let code = t.get_code();

        // NFields = floor((229*23 - 1)/63) + 1 = floor(5266/63) + 1 = 83+1 = 84
        // 84 fields / 16 per round = 5.25 → 6 rounds (5 full + 4 of round 5)
        // → signals transcriptHash_friQueries_0 .. _5

        // Round 0: pending was [c[0],c[1],c[2],nonce] padded to 12 (8 zeros added).
        assert!(
            code.contains("Poseidon(16)([challengeFRIQueries[0],challengeFRIQueries[1],challengeFRIQueries[2],nonce,0,0,0,0,0,0,0,0], [0,0,0,0])"),
            "round-0 Poseidon mismatch:\n{}", &code[..code.len().min(2000)]
        );
        // transcriptN2b_0 through transcriptN2b_83 should all be present (84 total).
        assert!(code.contains("transcriptN2b_0[64]"), "N2b_0 missing");
        assert!(code.contains("transcriptN2b_83[64]"), "N2b_83 missing");
        assert!(!code.contains("transcriptN2b_84"), "N2b_84 should not exist");

        // Round 5 is the last (4 fields used: 80..83 of the 16 output cells).
        assert!(code.contains("transcriptHash_friQueries_5"), "round-5 signal missing");
        assert!(!code.contains("transcriptHash_friQueries_6"), "no round-6");

        // Last N2b_83 comes from round 5, field index 3 (84 mod 16 = 4 → indices 0..3).
        assert!(code.contains("Num2Bits_strict()(transcriptHash_friQueries_5[3])"), "N2b_83 source missing");

        // 84 mod 16 = 4 → drain from hi_cnt=4 to 16 (12 unused fields).
        assert!(code.contains("for (var i = 4; i < 16; i++)"), "end-of-round-5 drain missing");

        // Last field has 38 bits (5267 - 63*83 = 38).
        assert!(code.contains("for (var j = 0; j < 38; j++)"), "last-field 38-bit loop missing");
        assert!(code.contains("for (var j = 38; j < 64; j++)"), "unused-bits drain for last field missing");
        // All full fields (0..82) end with Unused last bit.
        assert!(code.contains("_ <== transcriptN2b_0[63]; // Unused last bit"), "unused-last-bit for N2b_0 missing");

        // Bit-assignment loop uses query_bits=23.
        assert!(code.contains("if(b == 23)"), "query_bits=23 check missing");

        // Comment line present.
        assert!(code.contains("From each transcript hash converted to bits"), "comment missing");
    }
}
