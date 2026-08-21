//! Rust port of the `Transcript` class used in `verify_global_challenge.circom.ejs`
//! and `verify_global_constraints.circom.ejs`. Emits the circom signal
//! declarations for a GL Fiat-Shamir transcript.
//!
//! The family is fixed at construction and picks the [`Engine`]; everything else
//! here is family-independent. The two engines are different machines, not one
//! machine with a flag:
//!
//! * [`Sponge`] — Poseidon1/Poseidon2. Absorbs 12 into a 4-cell capacity per
//!   round, producing 16 cells: the first 4 are the next state, the rest a FIFO.
//! * [`Blake3`] — a real BLAKE3 chunk chain, so a squeeze roots a *copy* of the
//!   chain rather than permuting it. Mirrors `blake3core::Hasher`; see
//!   `circuits.gl/hash/blake3/linearhash.circom` for the same thing as a template.

/// Name suffix on emitted signals: `Some("friQueries")` makes `transcriptHash_0`
/// into `transcriptHash_friQueries_0`.
type Suffix<'a> = Option<&'a str>;

// ─────────────────────────────────────────────────────────────────────────────
// Poseidon sponge engine
// ─────────────────────────────────────────────────────────────────────────────

/// The width-16 / rate-12 Poseidon transcript sponge, for both Poseidon families.
/// `IN_W`/`OUT_W` live here, not on `Transcript`, so the blake3 path cannot reach them.
struct Sponge {
    /// `true` emits `Poseidon2(4, nOuts)`, `false` `Poseidon(nOuts)`. Both take
    /// the same `[in[12]], [capacity[4]]`, so only the template head differs.
    poseidon2: bool,
    /// Index of the next `transcriptHash_N` signal.
    h_cnt: usize,
    /// Output cells consumed from the current round.
    hi_cnt: usize,
    out: Vec<String>,
    pending: Vec<String>,
    state: [String; 4],
    /// Rounds < `threshold` use `early_used`, the rest `late_used`.
    early_rounds_scheme: Option<(usize, usize, usize)>,
    /// Drain the previous round's unread cells BEFORE emitting the next
    /// permutation, as GL `stark_verifier.circom.ejs` does. False matches
    /// `verify_global_challenge.circom.ejs`.
    drain_in_update_state: bool,
}

impl Sponge {
    // The arity-4 geometry written out: 4 * (arity - 1) absorbed, 4 * arity
    // produced. Both Poseidon families are arity 4 and blake3 does not use the
    // sponge at all, so no caller needs these parameterised -- see
    // proofman_common::hash_family for the general form.
    const IN_W: usize = 12;
    const OUT_W: usize = 16;

    fn new(poseidon2: bool) -> Self {
        Self {
            poseidon2,
            h_cnt: 0,
            hi_cnt: 0,
            out: Vec::new(),
            pending: Vec::new(),
            state: ["0".into(), "0".into(), "0".into(), "0".into()],
            early_rounds_scheme: None,
            drain_in_update_state: false,
        }
    }

    fn signal_name(name: Suffix, idx: usize) -> String {
        match name {
            Some(n) => format!("transcriptHash_{n}_{idx}"),
            None => format!("transcriptHash_{idx}"),
        }
    }

    fn absorb(&mut self, name: Suffix, code: &mut Vec<String>, a: String) {
        self.out.clear();
        self.pending.push(a);
        if self.pending.len() == Self::IN_W {
            self.update_state(name, code);
        }
    }

    fn squeeze1(&mut self, name: Suffix, code: &mut Vec<String>) -> String {
        if self.out.is_empty() {
            while self.pending.len() < Self::IN_W {
                self.pending.push("0".into());
            }
            self.update_state(name, code);
        }
        let res = self.out.remove(0);
        self.hi_cnt += 1;
        res
    }

    /// JS: `if(this.hiCnt < 4*arity) { code.push(`for(var i = ${hiCnt}; ...`) }`
    fn drain_unused(&mut self, name: Suffix, code: &mut Vec<String>) {
        if self.hi_cnt >= Self::OUT_W {
            return;
        }
        let out_w = Self::OUT_W;
        let prev_sig = Self::signal_name(name, self.h_cnt - 1);
        code.push(format!(
            "for(var i = {}; i < {out_w}; i++){{\n        _ <== {prev_sig}[i]; // Unused transcript values        \n    }}\n",
            self.hi_cnt
        ));
    }

    fn update_state(&mut self, name: Suffix, code: &mut Vec<String>) {
        let sig = Self::signal_name(name, self.h_cnt);
        let out_w = Self::OUT_W;

        // EJS: if(hCnt > 0) { firstUnused = max(hiCnt,4); drain ... }. From
        // max(hi_cnt, 4) because the first 4 cells are always the new state.
        if self.drain_in_update_state && self.h_cnt > 0 {
            let first_unused = self.hi_cnt.max(4);
            if first_unused < out_w {
                let prev_sig = Self::signal_name(name, self.h_cnt - 1);
                code.push(format!(
                    "for(var i = {first_unused}; i < {out_w}; i++){{\n        _ <== {prev_sig}[i]; // Unused transcript values \n    }}"
                ));
            }
        }

        // Cells beyond `used` get `_ <== ...` drain lines.
        let used = match self.early_rounds_scheme {
            Some((threshold, early, late)) => {
                if self.h_cnt < threshold {
                    early
                } else {
                    late
                }
            }
            None => out_w,
        };

        let hash_call = if self.poseidon2 { format!("Poseidon2(4, {out_w})") } else { format!("Poseidon({out_w})") };
        code.push(format!(
            "\n    signal {sig}[{out_w}] <== {hash_call}([{pending}], [{state}]);",
            pending = self.pending.join(","),
            state = self.state.join(","),
        ));

        if used < out_w {
            code.push(format!("    for (var i = {used}; i < {out_w}; i++) {{"));
            code.push(format!("        _ <== {sig}[i]; // Unused transcript values"));
            code.push("    }\n".into());
        }

        self.out = (0..out_w).map(|i| format!("{sig}[{i}]")).collect();
        for i in 0..4 {
            self.state[i] = format!("{sig}[{i}]");
        }

        self.h_cnt += 1;
        self.hi_cnt = 0;
        self.pending.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BLAKE3 engine
// ─────────────────────────────────────────────────────────────────────────────

const B3_IV: [u64; 8] =
    [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const B3_CHUNK_START: usize = 1;
const B3_CHUNK_END: usize = 2;
const B3_PARENT: usize = 4;
const B3_ROOT: usize = 8;

/// A real BLAKE3 hasher over the absorbed stream, mirroring `blake3core::Hasher`.
struct Blake3 {
    /// Fewer than 8 words, never emitted as a block until more input arrives, so
    /// the stream's final block is always still here when a squeeze roots it.
    buf: Vec<String>,
    /// The open chunk's chaining value; `None` means the IV.
    cv: Option<String>,
    /// Chaining values of completed subtrees.
    stack: Vec<String>,
    chunk_blocks: usize,
    chunk_counter: usize,
    chunks_done: usize,
    /// The current 8-word XOF output block, while still valid.
    xof: Option<String>,
    xof_offset: usize,
    ob: usize,
    sig_cnt: usize,
}

impl Blake3 {
    /// Words per compression. Not a sponge rate: the chaining value carries the
    /// capacity, the block does not.
    const BLOCK_WORDS: usize = 8;
    const CHUNK_BLOCKS: usize = 16;

    fn new() -> Self {
        Self {
            buf: Vec::new(),
            cv: None,
            stack: Vec::new(),
            chunk_blocks: 0,
            chunk_counter: 0,
            chunks_done: 0,
            xof: None,
            xof_offset: 0,
            ob: 0,
            sig_cnt: 0,
        }
    }

    fn signal_name(&mut self, name: Suffix, kind: &str) -> String {
        let n = self.sig_cnt;
        self.sig_cnt += 1;
        match name {
            Some(s) => format!("b3{kind}_{s}_{n}"),
            None => format!("b3{kind}_{n}"),
        }
    }

    fn cv_expr(&self) -> String {
        match &self.cv {
            Some(s) => s.clone(),
            None => format!("[{}]", B3_IV.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")),
        }
    }

    fn absorb(&mut self, name: Suffix, code: &mut Vec<String>, a: String) {
        // Emit a block only once more input is known to follow.
        if self.buf.len() == Self::BLOCK_WORDS {
            self.flush_block(name, code);
        }
        self.buf.push(a);
        // The stream changed, so XOF material from the old prefix is stale.
        self.xof = None;
        self.xof_offset = 0;
        self.ob = 0;
    }

    fn squeeze1(&mut self, name: Suffix, code: &mut Vec<String>) -> String {
        if self.xof.is_none() {
            self.ob = 0;
        } else if self.xof_offset == Self::BLOCK_WORDS {
            // Only the output-block counter advances; the root node is unchanged.
            self.ob += 1;
        }
        if self.xof.is_none() || self.xof_offset == Self::BLOCK_WORDS {
            let ob = self.ob;
            self.xof = Some(self.finalize(name, code, ob));
            self.xof_offset = 0;
        }
        let sig = self.xof.clone().expect("blake3 xof block");
        let i = self.xof_offset;
        self.xof_offset += 1;
        format!("{sig}[{i}]")
    }

    /// Emit the buffered words as a non-final block, closing the chunk at 16.
    fn flush_block(&mut self, name: Suffix, code: &mut Vec<String>) {
        let bi = self.chunk_blocks;
        let mut fl = 0;
        if bi == 0 {
            fl += B3_CHUNK_START;
        }
        if bi == Self::CHUNK_BLOCKS - 1 {
            // Every chunk's 16th block carries CHUNK_END, not only the stream's.
            fl += B3_CHUNK_END;
        }
        let cv = self.cv_expr();
        let words = self.buf.join(", ");
        let ctr = self.chunk_counter;
        let sig = self.signal_name(name, "blk");
        code.push(format!("\n    signal {sig}[8] <== Blake3AbsorbBlock()({cv}, [{words}], 64, {ctr}, {fl});"));
        self.buf.clear();
        self.chunk_blocks += 1;
        self.cv = Some(sig.clone());

        if self.chunk_blocks == Self::CHUNK_BLOCKS {
            // Push the chunk's cv, merging while the completed count is even.
            // That keeps the stack a canonical binary decomposition.
            let mut node = sig;
            let mut total = self.chunks_done + 1;
            while total.is_multiple_of(2) {
                let l = self.stack.pop().expect("blake3 cv stack underflow");
                node = self.emit_parent(name, code, &l, &node);
                total /= 2;
            }
            self.stack.push(node);
            self.chunks_done += 1;
            self.chunk_counter += 1;
            self.chunk_blocks = 0;
            self.cv = None;
        }
    }

    fn emit_parent(&mut self, name: Suffix, code: &mut Vec<String>, left: &str, right: &str) -> String {
        let p = self.signal_name(name, "par");
        code.push(format!("\n    signal {p}[8] <== Blake3Parent()({left}, {right}, {B3_PARENT});"));
        p
    }

    /// Root a **copy** of the chain and emit 8 XOF words at output block `ob`.
    /// Must not mutate the absorb state: ROOT is terminal in BLAKE3, but the
    /// transcript keeps absorbing after a challenge.
    fn finalize(&mut self, name: Suffix, code: &mut Vec<String>, ob: usize) -> String {
        let len = 8 * self.buf.len();
        let mut fl = B3_CHUNK_END;
        if self.chunk_blocks == 0 {
            fl += B3_CHUNK_START;
        }
        let mut words = self.buf.clone();
        while words.len() < Self::BLOCK_WORDS {
            words.push("0".into());
        }
        let w = words.join(", ");
        let cv = self.cv_expr();

        if self.stack.is_empty() {
            // Single chunk: the held-back block is the root node.
            let sig = self.signal_name(name, "fin");
            let root = fl + B3_ROOT;
            code.push(format!("\n    signal {sig}[8] <== Blake3FinalizeChunk()({cv}, [{w}], {len}, {root}, {ob});"));
            sig
        } else {
            // Multi-chunk: close this chunk without ROOT, merge a copy of the
            // stack, and the final parent is the root node.
            let close = self.signal_name(name, "cls");
            let ctr = self.chunk_counter;
            code.push(format!("\n    signal {close}[8] <== Blake3AbsorbBlock()({cv}, [{w}], {len}, {ctr}, {fl});"));
            let mut node = close;
            for si in (1..self.stack.len()).rev() {
                let l = self.stack[si].clone();
                node = self.emit_parent(name, code, &l, &node);
            }
            let l0 = self.stack[0].clone();
            let sig = self.signal_name(name, "fin");
            code.push(format!("\n    signal {sig}[8] <== Blake3FinalizeParent()({l0}, {node}, {ob});"));
            sig
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine dispatch
// ─────────────────────────────────────────────────────────────────────────────

enum Engine {
    Sponge(Sponge),
    Blake3(Blake3),
}

impl Engine {
    fn new(family: &str) -> Self {
        match family {
            "Poseidon1" => Engine::Sponge(Sponge::new(false)),
            "Poseidon2" => Engine::Sponge(Sponge::new(true)),
            "blake3" => Engine::Blake3(Blake3::new()),
            fam => panic!("unknown hash family: {fam}"),
        }
    }

    fn absorb(&mut self, name: Suffix, code: &mut Vec<String>, a: String) {
        match self {
            Engine::Sponge(s) => s.absorb(name, code, a),
            Engine::Blake3(b) => b.absorb(name, code, a),
        }
    }

    fn squeeze1(&mut self, name: Suffix, code: &mut Vec<String>) -> String {
        match self {
            Engine::Sponge(s) => s.squeeze1(name, code),
            Engine::Blake3(b) => b.squeeze1(name, code),
        }
    }

    /// blake3 has no output FIFO — unread XOF words are never materialised as
    /// signals — so there is nothing to drain.
    fn drain_unused(&mut self, name: Suffix, code: &mut Vec<String>) {
        match self {
            Engine::Sponge(s) => s.drain_unused(name, code),
            Engine::Blake3(_) => {}
        }
    }

    /// The four words of a digest-shaped read. Both engines consume; for blake3
    /// that reads `XOF[0..4]`, what the prover's `TranscriptGL::getState`
    /// returns. `getState` there does not consume, which is unobservable only
    /// while the read starts at offset 0 — the assert pins that.
    fn state_words(&mut self, name: Suffix, code: &mut Vec<String>) -> [String; 4] {
        if let Engine::Blake3(b) = self {
            debug_assert!(
                b.xof.is_none() && b.xof_offset == 0,
                "blake3 get_state must read a fresh XOF to match TranscriptGL::getState"
            );
        }
        [self.squeeze1(name, code), self.squeeze1(name, code), self.squeeze1(name, code), self.squeeze1(name, code)]
    }

    /// Sponge-only knobs are no-ops for blake3 rather than errors: callers set
    /// them uniformly, before knowing the family.
    fn sponge_mut(&mut self) -> Option<&mut Sponge> {
        match self {
            Engine::Sponge(s) => Some(s),
            Engine::Blake3(_) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transcript
// ─────────────────────────────────────────────────────────────────────────────

pub struct Transcript {
    pub name: Option<String>,
    engine: Engine,
    code: Vec<String>,
    /// First code line not yet returned by `get_code`.
    last_code_printed: usize,
    /// Counter for `transcriptN2b_N` names emitted by `get_permutations`.
    n2b_cnt: usize,
}

impl Transcript {
    pub fn new(name: Option<String>, family: &str) -> Self {
        Self { name, engine: Engine::new(family), code: Vec::new(), last_code_printed: 0, n2b_cnt: 0 }
    }

    /// See [`Sponge::drain_in_update_state`].
    pub fn set_drain_in_update_state(&mut self, value: bool) {
        if let Some(s) = self.engine.sponge_mut() {
            s.drain_in_update_state = value;
        }
    }

    /// See [`Sponge::early_rounds_scheme`].
    pub fn set_early_rounds_override(&mut self, threshold: usize, early_used: usize, late_used: usize) {
        if let Some(s) = self.engine.sponge_mut() {
            s.early_rounds_scheme = Some((threshold, early_used, late_used));
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn get_fields1_pub(&mut self) -> String {
        let name = self.name.as_deref();
        self.engine.squeeze1(name, &mut self.code)
    }

    pub fn put_single(&mut self, a: &str) {
        let name = self.name.as_deref();
        self.engine.absorb(name, &mut self.code, a.into());
    }

    pub fn put(&mut self, a: &str, n: usize) {
        for i in 0..n {
            self.put_single(&format!("{a}[{i}]"));
        }
    }

    pub fn put_2d(&mut self, a: &str, l: usize, m: usize) {
        for i in 0..l {
            for j in 0..m {
                self.put_single(&format!("{a}[{i}][{j}]"));
            }
        }
    }

    /// Emit `v <== [f0, f1, f2];`.
    pub fn get_field(&mut self, v: &str) {
        let f0 = self.get_fields1_pub();
        let f1 = self.get_fields1_pub();
        let f2 = self.get_fields1_pub();
        self.code.push(format!("{v} <== [{f0}, {f1}, {f2}];"));
    }

    /// Emit `v <== [f0, f1, f2, f3];`.
    pub fn get_state(&mut self, v: &str) {
        let name = self.name.as_deref();
        let [f0, f1, f2, f3] = self.engine.state_words(name, &mut self.code);
        self.code.push(format!("{v} <== [{f0}, {f1}, {f2}, {f3}];"));
    }

    /// Mirrors `getPermutations(v, n, nBits)` from the GL EJS `Transcript`:
    /// squeeze enough fields, `Num2Bits_strict` each, drain what the engine left
    /// unread, then emit the loops that fill `v`.
    ///
    /// - `v`          — 2-D output array, e.g. `"queriesFRI"`
    /// - `n_queries`  — first dimension, e.g. `starkStruct.nQueries`
    /// - `query_bits` — bits per query, e.g. `starkStruct.steps[0].nBits`
    pub fn get_permutations(&mut self, v: &str, n_queries: usize, query_bits: usize) {
        let total_bits = n_queries * query_bits;
        // NFields = floor((totalBits - 1) / 63) + 1
        let n_fields = (total_bits - 1) / 63 + 1;

        let mut n2b_names: Vec<String> = Vec::with_capacity(n_fields);
        for _ in 0..n_fields {
            let f = self.get_fields1_pub();
            let name = format!("transcriptN2b_{}", self.n2b_cnt);
            self.n2b_cnt += 1;
            self.code.push(format!("signal {{binary}} {name}[64] <== Num2Bits_strict()({f});"));
            n2b_names.push(name);
        }

        let name = self.name.as_deref();
        self.engine.drain_unused(name, &mut self.code);

        self.code.push(
            "// From each transcript hash converted to bits, we assign those bits to queriesFRI[q] to define the query positions"
                .into(),
        );
        self.code.push("var q = 0; // Query number ".into());
        self.code.push("var b = 0; // Bit number ".into());

        for (i, name) in n2b_names.iter().enumerate() {
            // 63 bits from each field but the last, the remainder from that one.
            let bits_this_field = if i + 1 == n_fields { total_bits - 63 * i } else { 63 };

            self.code.push(format!(
                "for(var j = 0; j < {bits_this_field}; j++) {{\n        {v}[q][b] <== {name}[j];\n        b++;\n        if(b == {query_bits}) {{\n            b = 0; \n            q++;\n        }}\n    }}"
            ));

            if bits_this_field == 63 {
                self.code.push(format!("_ <== {name}[63]; // Unused last bit\n"));
            } else {
                self.code.push(format!(
                    "for(var j = {bits_this_field}; j < 64; j++) {{\n        _ <== {name}[j]; // Unused bits        \n    }}"
                ));
            }
        }
    }

    /// Code lines produced since the last `get_code()`, indented 4 spaces to
    /// match the JS `getCode()`.
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
        let mut t = Transcript::new(Some("friQueries".into()), "Poseidon1");
        // Prime with 3 items (doesn't fill input_width=12, so no Poseidon yet).
        t.put("challengeFRIQueries", 3);
        // Call get_permutations — triggers round 0 with pending padded to 12.
        t.get_permutations("queriesFRI", 1, 3);
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
        assert!(code.contains("for(var i = 1; i < 16; i++)"), "drain missing:\n{code}");
        // Last field partial: total_bits=3, n_fields=1, bits_this_field = 3-63*0 = 3.
        assert!(code.contains("for(var j = 0; j < 3; j++)"), "bit-loop missing:\n{code}");
        // Partial-bits drain: 3 < 63, so "Unused bits" drain.
        assert!(code.contains("for(var j = 3; j < 64; j++)"), "unused bits drain missing:\n{code}");
        assert!(!code.contains("Unused last bit"), "should be partial not last-bit:\n{code}");
    }

    // ── Poseidon2 family emits the Poseidon2 template, not Poseidon1 ─────────

    #[test]
    fn poseidon2_emits_poseidon2_template() {
        let mut t = Transcript::new(Some("friQueries".into()), "Poseidon2");
        t.put("challengeFRIQueries", 3);
        t.get_permutations("queriesFRI", 1, 3);
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
        let mut t = Transcript::new(Some("friQueries".into()), "Poseidon1");
        t.get_permutations("queriesFRI", 1, 63);
        let code = t.get_code();
        // bits_this_field=63 → _ <== transcriptN2b_0[63]; // Unused last bit
        assert!(code.contains("_ <== transcriptN2b_0[63]; // Unused last bit"), "code:\n{code}");
        assert!(!code.contains("for(var j = 63; j < 64; j++)"), "should NOT have partial drain:\n{code}");
    }

    // ── get_permutations: integration against FibonacciSquare ground truth ───

    #[test]
    fn get_permutations_fibonacci_square_ground_truth() {
        // Reproduces the calculateFRIQueries0 template for FibonacciSquare:
        //   nQueries=229, steps[0].nBits=23, powBits=16 (nonce present)
        // With Poseidon1: input_width=12, output_width=16.
        let mut t = Transcript::new(Some("friQueries".into()), "Poseidon1");
        t.put("challengeFRIQueries", 3);
        t.put_single("nonce");
        t.get_permutations("queriesFRI", 229, 23);
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
        assert!(code.contains("for(var i = 4; i < 16; i++)"), "end-of-round-5 drain missing");

        // Last field has 38 bits (5267 - 63*83 = 38).
        assert!(code.contains("for(var j = 0; j < 38; j++)"), "last-field 38-bit loop missing");
        assert!(code.contains("for(var j = 38; j < 64; j++)"), "unused-bits drain for last field missing");
        // All full fields (0..82) end with Unused last bit.
        assert!(code.contains("_ <== transcriptN2b_0[63]; // Unused last bit"), "unused-last-bit for N2b_0 missing");

        // Bit-assignment loop uses query_bits=23.
        assert!(code.contains("if(b == 23)"), "query_bits=23 check missing");

        // Comment line present.
        assert!(code.contains("From each transcript hash converted to bits"), "comment missing");
    }

    // ── blake3 emitter: semantic validation ──────────────────────────────────

    /// Drive the blake3 emitter through a script that crosses a chunk boundary
    /// and squeezes several times, and check the shape of what comes out.
    ///
    /// The script is all `get_field` on purpose. `get_state` consumes in the
    /// emitter but not in `TranscriptGL`; that is unobservable in production
    /// because every `get_state` call site is terminal.
    #[test]
    fn blake3_emitter_emits_a_real_chunk_chain() {
        // (words to absorb, then that many challenge reads)
        const SCRIPT: &[(usize, usize)] = &[(3, 1), (5, 1), (120, 1), (8, 3), (1, 1)];

        let total: usize = SCRIPT.iter().map(|(n, _)| n).sum();
        // 137 words exceeds one 128-word BLAKE3 chunk, so the chain must go
        // multi-chunk partway through and the squeeze must move from rooting a
        // held-back block to rooting a parent.
        assert_eq!(total, 137);

        let mut t = Transcript::new(None, "blake3");

        let mut w = 0usize;
        let mut chal = 0usize;
        for (n, gets) in SCRIPT {
            for _ in 0..*n {
                t.put_single(&format!("w[{w}]"));
                w += 1;
            }
            for _ in 0..*gets {
                t.get_field(&format!("c{chal}"));
                chal += 1;
            }
        }
        let body = t.get_code();

        let absorbs = body.matches("Blake3AbsorbBlock()(").count();
        let fin_chunk = body.matches("Blake3FinalizeChunk()(").count();
        let fin_parent = body.matches("Blake3FinalizeParent()(").count();

        // Both squeeze shapes must appear: single-chunk early, parent-rooted
        // once the stream passes 128 words.
        assert!(fin_chunk > 0, "no single-chunk squeeze:\n{body}");
        assert!(fin_parent > 0, "no parent-rooted squeeze:\n{body}");

        // Absorption must dominate. A squeeze per absorbed word would mean the
        // rate-4 sponge shape had crept back in.
        assert!(absorbs > (fin_chunk + fin_parent) * 2, "absorbs={absorbs} finals={}", fin_chunk + fin_parent);

        // Every absorbed block is full and carries the chunk flags, never a
        // partial one -- the emitter holds the tail back for the root.
        assert!(body.contains(", 64, "), "absorbed blocks are not full:\n{body}");

        // The sponge forms must be gone entirely.
        assert!(!body.contains("Blake3Sponge"), "got:\n{body}");
        assert!(!body.contains("Poseidon"), "got:\n{body}");
    }
}
