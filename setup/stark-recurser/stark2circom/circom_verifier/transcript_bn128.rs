//! Rust port of the `Transcript` class defined inside
//! `circuits.bn128/stark_verifier.circom.ejs`.
//!
//! The BN128 transcript differs from the GL one in several key ways:
//!
//! | Property            | GL (Poseidon2)          | BN128 (PoseidonEx)              |
//! |---------------------|-------------------------|---------------------------------|
//! | Hash primitive      | Poseidon2(arity, arity+1)| PoseidonEx(arity, arity+1)     |
//! | Custom variant      | —                       | CustomPoseidon(arity)           |
//! | Output width        | 4*arity                 | arity + 1                       |
//! | Input slots / round | 4*(arity-1)             | arity                           |
//! | State               | [s0,s1,s2,s3]           | single scalar `state`           |
//! | `getField(v)`       | `v <== [f0,f1,f2]`      | `v <== BN1toGL3()(field)`       |
//! | `getFieldHash(v)`   | —                       | `v <== field`                   |
//! | Field bits for N2b  | 63 (N2b[64])            | 253 (N2b[254], Num2Bits_strictT)|
//! | updateState drain   | from `hi_cnt`           | from `max(hi_cnt, 1)`           |
//! | getPermutations drain| from `hi_cnt`          | from `hi_cnt`                   |
//!
//! The `arity` parameter here is `transcriptArity` from the JS side, not
//! `merkleTreeArity`. They are equal only when `custom = true`; otherwise
//! `transcriptArity = 16` regardless of the Merkle arity.

#![allow(dead_code)]

pub struct TranscriptBn128 {
    /// Optional name suffix: `"publics"` → signal prefix `transcriptHash_publics_N`.
    pub name: Option<String>,
    /// Hash round index (next signal will be `transcriptHash_N[arity+1]`).
    h_cnt: usize,
    /// How many outputs have been consumed from the current `out` FIFO.
    hi_cnt: usize,
    /// N2b signal counter (`transcriptN2b_N`).
    n2b_cnt: usize,
    /// First code line not yet flushed by `get_code()`.
    last_code_printed: usize,
    /// Outputs available from the last hash call, in emission order.
    out: Vec<String>,
    /// Inputs pending for the next hash call.
    pending: Vec<String>,
    /// Current Poseidon state (a single scalar string, initially `"0"`).
    state: String,
    /// Accumulated code lines.
    code: Vec<String>,
    /// `transcriptArity`: inputs per hash round / pending-flush threshold.
    arity: usize,
    /// Whether to use the `custom` hash variant (`CustomPoseidon`).
    custom: bool,
}

impl TranscriptBn128 {
    /// Construct a new transcript for the BN128/Poseidon circuit.
    ///
    /// - `arity`  — `transcriptArity` from the JS context (16 unless `custom=true`)  
    /// - `custom` — if true emit `CustomPoseidon(arity)` instead of `PoseidonEx(arity, arity+1)`
    /// - `name`   — optional signal-name suffix (e.g. `"publics"`, `"evals"`, `"friQueries"`)
    pub fn new(arity: usize, custom: bool, name: Option<String>) -> Self {
        Self {
            name,
            h_cnt: 0,
            hi_cnt: 0,
            n2b_cnt: 0,
            last_code_printed: 0,
            out: Vec::new(),
            pending: Vec::new(),
            state: "0".into(),
            code: Vec::new(),
            arity,
            custom,
        }
    }

    // ── Naming helpers ────────────────────────────────────────────────────────

    fn signal_prefix(&self) -> String {
        match &self.name {
            Some(n) => format!("transcriptHash_{n}"),
            None => "transcriptHash".into(),
        }
    }

    fn signal_name(&self, idx: usize) -> String {
        format!("{}_{idx}", self.signal_prefix())
    }

    /// `arity + 1` outputs per hash round.
    fn output_width(&self) -> usize {
        self.arity + 1
    }

    // ── Internal state machine ────────────────────────────────────────────────

    /// Pad pending to `arity`, emit a hash signal, collect outputs and update state.
    /// Drain from `max(hi_cnt, 1)` (output [0] is always consumed as the new state).
    fn update_state(&mut self) {
        let sig = self.signal_name(self.h_cnt);
        let out_w = self.output_width();

        // Drain unused outputs of the *previous* hash round (if any).
        if self.h_cnt > 0 {
            let first_unused = self.hi_cnt.max(1);
            if first_unused < out_w {
                let prev_sig = self.signal_name(self.h_cnt - 1);
                self.code.push(format!(
                    "for(var i = {first_unused}; i < {out_w}; i++){{\n        _ <== {prev_sig}[i]; // Unused transcript values        \n    }}"
                ));
            }
        }

        // Pad pending to `arity`.
        while self.pending.len() < self.arity {
            self.pending.push("0".into());
        }

        let hash_fn = if self.custom {
            format!("CustomPoseidon({})", self.arity)
        } else {
            format!("PoseidonEx({}, {})", self.arity, out_w)
        };

        self.code.push(format!(
            "\n    signal {sig}[{out_w}] <== {hash_fn}([{}], {});",
            self.pending.join(","),
            self.state,
        ));

        // Build output FIFO and advance state.
        self.out = (0..out_w).map(|i| format!("{sig}[{i}]")).collect();
        self.state = format!("{sig}[0]");
        self.h_cnt += 1;
        self.hi_cnt = 0;
        self.pending.clear();
    }

    /// Consume one BN128 field element from the output FIFO, triggering a
    /// new hash round (with pending padded to `arity`) if the FIFO is empty.
    fn get_fields1(&mut self) -> String {
        if self.out.is_empty() {
            while self.pending.len() < self.arity {
                self.pending.push("0".into());
            }
            self.update_state();
        }
        let res = self.out.remove(0);
        self.hi_cnt += 1;
        res
    }

    /// `getFields253` is the same as `getFields1` for BN128 — it just gets
    /// the next scalar from the output FIFO.
    fn get_fields253(&mut self) -> String {
        self.get_fields1()
    }

    fn _add1(&mut self, a: String) {
        self.out.clear();
        self.pending.push(a);
        if self.pending.len() == self.arity {
            self.update_state();
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Consume one BN128 field and emit `v <== BN1toGL3()(field);`.
    /// Used for all regular challenge outputs.
    pub fn get_field(&mut self, v: &str) {
        let f = self.get_fields1();
        self.code.push(format!("{v} <== BN1toGL3()({f});"));
    }

    /// Consume one BN128 field and emit `v <== field;` (no GL3 conversion).
    /// Used inside `hashCommits` sub-transcripts where the output is itself
    /// fed back as a hash input.
    pub fn get_field_hash(&mut self, v: &str) {
        let f = self.get_fields1();
        self.code.push(format!("{v} <== {f};"));
    }

    /// Add the single signal `a` to the pending buffer.
    pub fn put_single(&mut self, a: &str) {
        self._add1(a.into());
    }

    /// Add signals `a[0]..a[n-1]` to the pending buffer.
    pub fn put(&mut self, a: &str, n: usize) {
        for i in 0..n {
            self._add1(format!("{a}[{i}]"));
        }
    }

    /// Add 2-D signals `a[i][j]` for `i in 0..l`, `j in 0..m`.
    pub fn put_2d(&mut self, a: &str, l: usize, m: usize) {
        for i in 0..l {
            for j in 0..m {
                self._add1(format!("{a}[{i}][{j}]"));
            }
        }
    }

    /// Generate bit-decomposition signals and FRI-query bit-assignment loops.
    ///
    /// Mirrors `getPermutations(v, n, nBits)` from the BN128 EJS Transcript class.
    ///
    /// BN128 uses 253-bit fields (`Num2Bits_strictT`, 254-element arrays) and
    /// `NFields = floor((n*nBits - 1) / 253) + 1`.
    ///
    /// - `v`          — 2-D signal name (e.g. `"queriesL0"`)
    /// - `n_queries`  — first dimension
    /// - `query_bits` — bits per query (`starkStruct.steps[0].nBits`)
    /// - `n_fields`   — pre-computed `NFields` (passed in because the JS compute it
    ///   outside the class, at the top of the template)
    pub fn get_permutations(&mut self, v: &str, n_queries: usize, query_bits: usize, n_fields: usize) {
        let total_bits = n_queries * query_bits;
        let out_w = self.output_width();

        // ── Emit one Num2Bits_strictT per required 253-bit field ──────────────
        let mut n2b_names: Vec<String> = Vec::with_capacity(n_fields);
        for _ in 0..n_fields {
            let f = self.get_fields253();
            let name = format!("transcriptN2b_{}", self.n2b_cnt);
            self.n2b_cnt += 1;
            self.code.push(format!("signal {{binary}} {name}[254] <== Num2Bits_strictT()({f});"));
            n2b_names.push(name);
        }

        // ── Drain remaining unused outputs of the last hash round ─────────────
        // JS: if(this.hiCnt < transcriptArity + 1) { ... }
        if self.hi_cnt < out_w {
            let prev_sig = self.signal_name(self.h_cnt - 1);
            self.code.push(format!(
                "for(var i = {}; i < {out_w}; i++){{\n        _ <== {prev_sig}[i]; // Unused transcript values           \n    }}\n",
                self.hi_cnt
            ));
        }

        // ── Bit-assignment preamble ───────────────────────────────────────────
        self.code.push(
            "// From each transcript hash converted to bits, we assign those bits to queriesL0[q] to define the query positions"
                .into(),
        );
        self.code.push("var q = 0; // Query number ".into());
        self.code.push("var b = 0; // Bit number ".into());

        // ── Bit-assignment loops ──────────────────────────────────────────────
        for (i, name) in n2b_names.iter().enumerate() {
            // Each field contributes 253 bits (or the remainder for the last one).
            let bits_this_field = if i + 1 == n_fields { total_bits - 253 * i } else { 253 };

            self.code.push(format!(
                "for(var j = 0; j < {bits_this_field}; j++) {{\n        {v}[q][b] <== {name}[j];\n        b++;\n        if(b == {query_bits}) {{\n            b = 0; \n            q++;\n        }}\n    }}"
            ));

            if bits_this_field == 253 {
                self.code.push(format!("_ <== {name}[253]; // Unused last bit\n"));
            } else {
                self.code.push(format!(
                    "for(var j = {bits_this_field}; j < 254; j++) {{\n        _ <== {name}[j]; // Unused bits\n    }}"
                ));
            }
        }
    }

    /// Return all code lines accumulated since the last `get_code()` call,
    /// each indented by 4 spaces (mirrors the JS `getCode()` behaviour).
    pub fn get_code(&mut self) -> String {
        let lines: Vec<String> = self.code[self.last_code_printed..]
            .iter()
            .map(|l| if l.starts_with('\n') || l.is_empty() { l.clone() } else { format!("    {l}") })
            .collect();
        self.last_code_printed = self.code.len();
        lines.join("\n")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers
    fn make_t(name: Option<&str>) -> TranscriptBn128 {
        TranscriptBn128::new(16, false, name.map(|s| s.into()))
    }

    // ── update_state / PoseidonEx emission ────────────────────────────────────

    #[test]
    fn first_hash_uses_poseidon_ex() {
        let mut t = make_t(None);
        // fill pending to 16 → triggers updateState
        for i in 0..16 {
            t._add1(format!("in[{i}]"));
        }
        let code = t.get_code();
        assert!(code.contains("PoseidonEx(16, 17)([in[0],in[1],"), "code:\n{code}");
        assert!(code.contains(", 0);"), "state must be 0: {code}");
        assert_eq!(t.h_cnt, 1);
    }

    #[test]
    fn custom_hash_uses_custom_poseidon() {
        let mut t = TranscriptBn128::new(16, true, None);
        for i in 0..16 {
            t._add1(format!("in[{i}]"));
        }
        let code = t.get_code();
        assert!(code.contains("CustomPoseidon(16)("), "code:\n{code}");
    }

    #[test]
    fn signal_name_includes_name_suffix() {
        let mut t = make_t(Some("publics"));
        for i in 0..16 {
            t._add1(format!("v[{i}]"));
        }
        let code = t.get_code();
        assert!(code.contains("transcriptHash_publics_0[17]"), "code:\n{code}");
    }

    #[test]
    fn state_updates_after_hash() {
        let mut t = make_t(None);
        for i in 0..16 {
            t._add1(format!("x[{i}]"));
        }
        // second call uses [0] of first hash as state
        for i in 0..16 {
            t._add1(format!("y[{i}]"));
        }
        let code = t.get_code();
        assert!(code.contains(", transcriptHash_0[0]);"), "state from prev round: {code}");
    }

    // ── Drain in updateState ──────────────────────────────────────────────────

    #[test]
    fn drain_starts_from_max_hi_cnt_1_in_update_state() {
        // Put 16 → hash. Put 16 → second hash. Before second hash, previous
        // round's drain should start from max(hiCnt, 1).
        let mut t = make_t(None);
        // fill → hash 0
        for i in 0..16 {
            t._add1(format!("a[{i}]"));
        }
        // consume 0 outputs before triggering hash 1 → hiCnt stays 0 → drain from max(0,1)=1
        for i in 0..16 {
            t._add1(format!("b[{i}]"));
        }
        let code = t.get_code();
        // drain from 1 (not 0) because max(0,1)=1
        assert!(code.contains("for(var i = 1; i < 17; i++)"), "expected drain from 1:\n{code}");
    }

    #[test]
    fn no_drain_if_all_outputs_consumed() {
        // Consume all 17 outputs before triggering the second hash.
        let mut t = make_t(None);
        for i in 0..16 {
            t._add1(format!("a[{i}]"));
        }
        // consume all 17 outputs
        for _ in 0..17 {
            t.get_fields1();
        }
        // trigger second hash by filling pending again
        for i in 0..16 {
            t._add1(format!("b[{i}]"));
        }
        let code = t.get_code();
        // second hash should have no drain (all 17 consumed → hi_cnt=17 ≥ out_w=17)
        // first hash drain: max(17,1)=17 which is NOT < 17, so no drain
        let drain_count = code.matches("Unused transcript values").count();
        assert_eq!(drain_count, 0, "expected no drain:\n{code}");
    }

    // ── get_field ─────────────────────────────────────────────────────────────

    #[test]
    fn get_field_emits_bn1_to_gl3() {
        let mut t = make_t(None);
        t.put_single("rootC");
        t.get_field("challengeQ[0]");
        let code = t.get_code();
        assert!(code.contains("challengeQ[0] <== BN1toGL3()(transcriptHash_0["), "expected BN1toGL3: {code}");
    }

    // ── get_field_hash ────────────────────────────────────────────────────────

    #[test]
    fn get_field_hash_emits_direct_assignment() {
        let mut t = make_t(Some("publics"));
        t.put_single("rootC");
        t.get_field_hash("publicsHash");
        let code = t.get_code();
        assert!(code.contains("publicsHash <== transcriptHash_publics_0["), "expected direct assignment: {code}");
        // must NOT contain BN1toGL3
        assert!(!code.contains("BN1toGL3"), "unexpected conversion: {code}");
    }

    // ── get_permutations ─────────────────────────────────────────────────────

    #[test]
    fn get_permutations_uses_num2bits_strict_t() {
        // NFields for 10 queries × 8 bits = 80 bits → floor(79/253)+1 = 1
        let mut t = make_t(Some("friQueries"));
        t.put("challengeFRIQueries", 3);
        t.get_permutations("queriesL0", 10, 8, 1);
        let code = t.get_code();
        assert!(code.contains("Num2Bits_strictT()"), "expected Num2Bits_strictT: {code}");
        assert!(code.contains("signal {binary} transcriptN2b_0[254]"), "expected 254-bit N2b: {code}");
    }

    #[test]
    fn get_permutations_253_bits_per_full_field() {
        // 2 fields: first contributes 253 bits, second the remainder.
        // Total: 1 query × 300 bits → NFields = floor(299/253)+1 = 2
        let mut t = make_t(None);
        t.put_single("challengeFRIQueries");
        t.get_permutations("queriesL0", 1, 300, 2);
        let code = t.get_code();
        // First field: 253 bits, unused last bit
        assert!(code.contains("for(var j = 0; j < 253; j++)"), "first field 253 bits: {code}");
        assert!(code.contains("_ <== transcriptN2b_0[253]; // Unused last bit"), "unused last bit: {code}");
        // Second field: 300 - 253 = 47 bits, unused j in 47..254
        assert!(code.contains("for(var j = 0; j < 47; j++)"), "remainder 47 bits: {code}");
        assert!(code.contains("for(var j = 47; j < 254; j++)"), "unused remainder: {code}");
    }

    #[test]
    fn get_permutations_drain_after_n2b() {
        // Fresh transcript: put 1 item → 15 left until flush. get_permutations
        // calls get_fields253(): triggers a hash (padded to 16), and hi_cnt=1.
        // After N2b for that 1 field, drain from 1 to 17.
        let mut t = make_t(None);
        t.put_single("rootC");
        t.get_permutations("queriesL0", 1, 63, 1);
        let code = t.get_code();
        assert!(code.contains("for(var i = 1; i < 17; i++)"), "drain: {code}");
    }

    // ── n2b_cnt increments across calls ──────────────────────────────────────

    #[test]
    fn n2b_cnt_increments_across_permutation_calls() {
        let mut t = make_t(None);
        // First get_permutations: 1 field → transcriptN2b_0
        t.put_single("a");
        t.get_permutations("q1", 1, 63, 1);
        // consume get_code to reset lastCodePrinted
        t.get_code();

        // Second get_permutations: 1 field → transcriptN2b_1
        t.put_single("b");
        t.get_permutations("q2", 1, 63, 1);
        let code = t.get_code();
        assert!(code.contains("transcriptN2b_1"), "second call should use n2b_cnt=1: {code}");
    }

    // ── Integration: small FRI query scenario ────────────────────────────────

    #[test]
    fn integration_small_transcript() {
        // Mirrors JS: put rootC, get challengeQ, get queriesL0
        // n_queries=2, step0_bits=4 → total=8 → NFields=1
        let mut t = make_t(None);

        t.put_single("rootC");
        t.get_field("challengeQ[0]");
        t.put_single("root1");
        t.get_permutations("queriesL0", 2, 4, 1);
        let code = t.get_code();

        // At least one PoseidonEx hash emitted
        assert!(code.contains("PoseidonEx(16, 17)"), "code:\n{code}");
        // challengeQ via BN1toGL3
        assert!(code.contains("challengeQ[0] <== BN1toGL3()"), "code:\n{code}");
        // N2b for queriesL0
        assert!(code.contains("Num2Bits_strictT()"), "code:\n{code}");
        // bit-assignment loop
        assert!(code.contains("queriesL0[q][b] <== transcriptN2b_0[j]"), "code:\n{code}");
    }
}
