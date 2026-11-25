//! `pilfflonk.verifier.sol`: the Solidity verifier of a vkey
//! (pilfflonk/docs/verifier.md#solidity-verifier), which `setup-pilfflonk --solidity` writes next
//! to `pilfflonk.vkey.json` and `proofman-setup pilfflonk-solidity` writes from a vkey alone.
//!
//! The contract is static, `sol/verifier_pilfflonk.sol` (`include_str!`), but for its
//! `// @generated <part>` lines, which this module replaces with what it computes from the vkey. It is
//! snarkjs 0.7.6's fflonk verifier (`templates/verifier_fflonk.sol.ejs`) generalised as the JS
//! verifier (`pilfflonk/js/src/verify.js`, pilfflonk/docs/verifier.md#js-verifier) generalises
//! snarkjs's `fflonk_verify.js`, and it accepts exactly the proofs the JS verifier accepts. The
//! parts:
//! - the constants: `[τ]₂`, the fixed commitments, `digest mod r` and the roots of unity;
//! - where each value is in the calldata ([`CalldataLayout`]) and in memory;
//! - the transcript (`challenges.js`, pilfflonk/docs/protocol.md#transcript) and the vkey's
//!   `qVerifier` (`qverifier.js`), as straight-line Yul;
//! - the steps over the vkey's layout (the `f_i`, their roots and boundaries), unrolled into
//!   straight-line Yul as snarkjs unrolls them: a loop over a table would cost gas.
//!
//! **Calldata.** `verifyProof(bytes32[W] calldata proof, uint256[P] calldata pubSignals)`, as
//! snarkjs's `FflonkVerifier`, with `P = nPublic` (no `pubSignals` if it is 0) and `W` the words of
//! [`CalldataLayout`]: the proof's bytes (`Proof::to_bytes`, pilfflonk/docs/formats.md#proof) as
//! 32-byte words, followed by one auxiliary inverse `1/(ξ − ω^j)` for each `firstRow` (`j = 0`) or
//! `lastRow` (`j = N − 1`) boundary, in the order of the boundaries
//! (pilfflonk/docs/formats.md#calldata). The contract checks each, as it checks `inv` and `invZh`.
//! A vkey without those boundaries, as every compiled PIL2 program
//! (pilfflonk/docs/protocol.md#constraint-polynomial), has none, and the calldata is the proof.
//! `proofman_pilfflonk::calldata` encodes it for a proof, and `proofman-cli pilfflonk calldata`
//! prints it.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use num_bigint::BigUint;
use proofman_pilfflonk::calldata::SELECTOR_BYTES;
use proofman_pilfflonk::field::r;
use proofman_pilfflonk::names::evaluation_name;
use proofman_pilfflonk::{Boundary, CalldataLayout, FrBytes, JsonFile, PolType, Vkey};
use serde_json::Value;

use crate::error::SetupError;

/// The file name of the verifier, in the backend directory of the `provingKey/`, next to the vkey
/// (pilfflonk/docs/formats.md#provingkey).
pub const VERIFIER_SOL_FILE: &str = "pilfflonk.verifier.sol";

/// The static contract (see the module), whose `// @generated <part>` lines [`contract`] replaces.
const VERIFIER_SOL: &str = include_str!("sol/verifier_pilfflonk.sol");

/// Bytes of a word of the calldata and of the memory.
const WORD: u64 = 32;

/// Where the verifier's memory starts: Solidity's first free address, 0x80. verifyProof allocates
/// nothing before its assembly block, which never returns to Solidity, so the addresses can be
/// constants and the compiler folds every offset into them.
const MEM_BASE: u64 = 0x80;

fn fail<T>(message: impl Into<String>) -> Result<T, SetupError> {
    Err(SetupError::Solidity(message.into()))
}

// ---------------------------------------------------------------------------------------------
// BN128's scalar field, with num-bigint: the constants the contract embeds
// ---------------------------------------------------------------------------------------------

fn fr_pow(base: &BigUint, exponent: u64) -> BigUint {
    base.modpow(&BigUint::from(exponent), r())
}

fn fr_inv(a: &BigUint) -> BigUint {
    a.modpow(&(r() - 2u32), r())
}

/// `r − a`, the negation of `a < r`, as the contract adds it: `addmod(x, r − a, r) = x − a`.
fn fr_neg(a: &BigUint) -> BigUint {
    (r() - a) % r()
}

/// `5^((r−1)/n)`, a primitive `n`-th root of unity for `n | r − 1` (`shplonk.js`, `rootOfUnity`).
fn root_of_unity(n: &BigUint) -> Result<BigUint, SetupError> {
    let order = r() - 1u32;
    if n == &BigUint::ZERO || &order % n != BigUint::ZERO {
        return fail(format!("there is no root of unity of order {n}: it does not divide r − 1"));
    }
    Ok(BigUint::from(5u32).modpow(&(order / n), r()))
}

/// `base^s`, the inverse of `base^|s|` for `s < 0` (`shplonk.js`, `signedPower`).
fn signed_pow(base: &BigUint, s: i64) -> BigUint {
    let power = fr_pow(base, s.unsigned_abs());
    if s < 0 {
        fr_inv(&power)
    } else {
        power
    }
}

// ---------------------------------------------------------------------------------------------
// The calldata
// ---------------------------------------------------------------------------------------------

/// The byte offset of `word` of the `proof` argument in the calldata, after the selector.
fn calldata_offset(word: u64) -> u64 {
    SELECTOR_BYTES as u64 + WORD * word
}

// ---------------------------------------------------------------------------------------------
// The memory
// ---------------------------------------------------------------------------------------------

/// The named regions of the verifier's memory, each a Solidity constant: an absolute address, from
/// [`MEM_BASE`].
#[derive(Default)]
struct Memory {
    next: u64,
    slots: Vec<Slot>,
}

struct Slot {
    name: String,
    offset: u64,
    comment: String,
}

impl Memory {
    /// A region of `words` words, `name` its constant.
    fn alloc(&mut self, name: String, words: u64, comment: String) -> String {
        self.slots.push(Slot { name: name.clone(), offset: self.next, comment: sanitize(&comment) });
        self.next += WORD * words;
        name
    }
}

/// `text` as a `//` comment of the contract can hold it: printable ASCII only. A name of the vkey
/// with a line terminator (Solidity ends a comment at any of LF, VT, FF, CR, NEL, LS or PS) would
/// otherwise end the comment and write into the code.
fn sanitize(text: &str) -> String {
    text.chars().map(|c| if c == ' ' || c.is_ascii_graphic() { c } else { '?' }).collect()
}

// ---------------------------------------------------------------------------------------------
// The transcript (`challenges.js`, pilfflonk/docs/protocol.md#transcript)
// ---------------------------------------------------------------------------------------------

/// The Yul of the transcript: its buffer `pT` holds what was absorbed since the last squeeze, which
/// is `keccak256` of it mod `r`, then the buffer is reset to the challenge (`transcript.js`,
/// `squeeze`).
#[derive(Default)]
struct TranscriptCode {
    len: u64,
    lines: Vec<String>,
}

impl TranscriptCode {
    fn at(&self) -> String {
        if self.len == 0 {
            "pT".to_string()
        } else {
            format!("add(pT, {})", self.len)
        }
    }

    fn comment(&mut self, text: &str) {
        self.lines.push(format!("// {}", sanitize(text)));
    }

    /// `addScalar` of a value the contract computes.
    fn word(&mut self, expr: &str) {
        self.lines.push(format!("mstore({}, {expr})", self.at()));
        self.len += WORD;
    }

    /// `addScalar` or `addPolCommitment` of `bytes` bytes of the calldata from `offset`, which the
    /// calldata holds big-endian, as the transcript encodes them.
    fn calldata(&mut self, offset: &str, bytes: u64) {
        if bytes > 0 {
            self.lines.push(format!("calldatacopy({}, {offset}, {bytes})", self.at()));
            self.len += bytes;
        }
    }

    fn squeeze(&mut self, slot: &str) {
        self.lines.push(format!("c := mod(keccak256(pT, {}), q)", self.len));
        self.lines.push(format!("mstore({slot}, c)"));
        self.lines.push("mstore(pT, c)".to_string());
        self.len = WORD;
    }
}

// ---------------------------------------------------------------------------------------------
// The vkey's qVerifier as straight-line Yul (`qverifier.js`, `executeCode`)
// ---------------------------------------------------------------------------------------------

/// What the operands of the `qVerifier` read, in the contract.
struct QOperands<'a> {
    eval: &'a [String],
    public: &'a [String],
    zi: &'a [String],
    /// `(stage, stageId)` → the challenge's memory constant (`challenges.js`, `challengeOf`).
    challenges: &'a HashMap<(u64, u64), String>,
    tmp: &'a str,
}

fn u64_field(value: &Value, key: &str, at: &str) -> Result<u64, SetupError> {
    match value.get(key).and_then(Value::as_u64) {
        Some(v) => Ok(v),
        None => fail(format!("qVerifier: {at} has no {key}")),
    }
}

fn index<'a>(list: &'a [String], i: u64, what: &str, at: &str) -> Result<&'a String, SetupError> {
    match usize::try_from(i).ok().and_then(|i| list.get(i)) {
        Some(e) => Ok(e),
        None => fail(format!("qVerifier: {at} reads {what} {i}, which there is not")),
    }
}

/// An operand that is not a temporary, as a Yul expression: an evaluation, a public, a number, a
/// challenge or a `Zi`, each fixed for the whole code.
fn q_leaf(value: &Value, at: &str, ops: &QOperands) -> Result<String, SetupError> {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or_default();
    Ok(match kind {
        "eval" => format!("calldataload({})", index(ops.eval, u64_field(value, "id", at)?, "evaluation", at)?),
        "public" => format!("calldataload({})", index(ops.public, u64_field(value, "id", at)?, "public", at)?),
        "number" => match value.get("value").and_then(Value::as_str) {
            Some(v) if FrBytes::from_decimal(v).is_ok() => v.to_string(),
            _ => return fail(format!("qVerifier: {at} is not a number below r")),
        },
        "challenge" => {
            let key = (u64_field(value, "stage", at)?, u64_field(value, "stageId", at)?);
            match ops.challenges.get(&key) {
                Some(slot) => format!("mload({slot})"),
                None => return fail(format!("qVerifier: {at} reads the challenge {key:?}, which there is not")),
            }
        }
        "Zi" => format!("mload({})", index(ops.zi, u64_field(value, "boundaryId", at)?, "boundary", at)?),
        other => return fail(format!("qVerifier: {at} is an operand {other:?}, which the verifier does not have")),
    })
}

/// A value of the `qVerifier` once its temporaries are values (each write a new one): a leaf, or
/// an operation on two earlier values.
enum QValue {
    Leaf(String),
    Op(&'static str, usize, usize),
}

/// How deep the expressions of values used once are nested before one is stored instead.
const Q_MAX_DEPTH: usize = 8;

/// The `qVerifier` of the vkey as Yul, its result stored to `pQ`, and the words of temporaries it
/// needs. Its temporaries are taken as values: a value used once is written into the expression
/// that uses it (up to [`Q_MAX_DEPTH`] deep), the others are stored, each in a slot of `pTmp` that is
/// reused once the value is last read; a value never read is not computed. The arithmetic is that
/// of the code, operation for operation: only where the intermediate values are kept changes.
fn q_code(q_verifier: &Value, ops: &QOperands) -> Result<(Vec<String>, u64), SetupError> {
    let Some(code) = q_verifier.get("code").and_then(Value::as_array) else {
        return fail("the vkey's qVerifier has no code");
    };
    let mut values: Vec<QValue> = Vec::new();
    let mut current: HashMap<u64, usize> = HashMap::new();
    let mut result = None;
    for (i, entry) in code.iter().enumerate() {
        let at = format!("code[{i}]");
        let op = entry.get("op").and_then(Value::as_str).unwrap_or_default();
        let src = entry.get("src").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut operands = Vec::with_capacity(src.len());
        for (j, s) in src.iter().enumerate() {
            let at = format!("{at}.src[{j}]");
            if s.get("type").and_then(Value::as_str) == Some("tmp") {
                let id = u64_field(s, "id", &at)?;
                match current.get(&id) {
                    Some(&v) => operands.push(v),
                    None => return fail(format!("qVerifier: {at} reads the temporary {id} before it is written")),
                }
            } else {
                values.push(QValue::Leaf(q_leaf(s, &at, ops)?));
                operands.push(values.len() - 1);
            }
        }
        let value = match (op, operands.as_slice()) {
            ("copy", &[a]) => a,
            (name @ ("add" | "sub" | "mul"), &[a, b]) => {
                let name = match name {
                    "add" => "add",
                    "sub" => "sub",
                    _ => "mul",
                };
                values.push(QValue::Op(name, a, b));
                values.len() - 1
            }
            _ => return fail(format!("qVerifier: {at} is op {op:?} of {} operands", operands.len())),
        };
        let dest = match entry.get("dest") {
            Some(d) if d.get("type").and_then(Value::as_str) == Some("tmp") => u64_field(d, "id", &at)?,
            _ => return fail(format!("qVerifier: {at} does not write a temporary")),
        };
        current.insert(dest, value);
        result = Some(value);
    }
    let Some(result) = result else {
        return fail("the vkey's qVerifier has no code");
    };

    // Which values are stored: those read twice or more, and those whose expression would nest too
    // deep; the result goes to pQ.
    let mut uses = vec![0usize; values.len()];
    for v in &values {
        if let QValue::Op(_, a, b) = *v {
            uses[a] += 1;
            uses[b] += 1;
        }
    }
    let mut stored = vec![false; values.len()];
    let mut depth = vec![0usize; values.len()];
    for (v, value) in values.iter().enumerate() {
        if let QValue::Op(_, a, b) = *value {
            let d = |o: usize| if stored[o] || matches!(values[o], QValue::Leaf(_)) { 0 } else { depth[o] };
            depth[v] = 1 + d(a).max(d(b));
            stored[v] = v != result && (uses[v] >= 2 || depth[v] > Q_MAX_DEPTH);
        }
    }

    // The statements, in the code's order, and the stored values each reads.
    fn reads(values: &[QValue], stored: &[bool], v: usize, out: &mut Vec<usize>) {
        if let QValue::Op(_, a, b) = values[v] {
            for o in [a, b] {
                if stored[o] {
                    out.push(o);
                } else {
                    reads(values, stored, o, out);
                }
            }
        }
    }
    let statements: Vec<usize> = (0..values.len()).filter(|&v| stored[v] || v == result).collect();
    let mut last_read = vec![0usize; values.len()];
    for (t, &v) in statements.iter().enumerate() {
        let mut r = Vec::new();
        reads(&values, &stored, v, &mut r);
        for w in r {
            last_read[w] = t;
        }
    }

    fn expr(values: &[QValue], stored: &[bool], slot: &[u64], tmp: &str, v: usize, top: bool) -> String {
        if !top && stored[v] {
            return format!("mload(add({tmp}, {}))", WORD * slot[v]);
        }
        match &values[v] {
            QValue::Leaf(e) => e.clone(),
            QValue::Op(op, a, b) => {
                let (a, b) = (expr(values, stored, slot, tmp, *a, false), expr(values, stored, slot, tmp, *b, false));
                match *op {
                    "add" => format!("addmod({a}, {b}, q)"),
                    "sub" => format!("addmod({a}, sub(q, {b}), q)"),
                    _ => format!("mulmod({a}, {b}, q)"),
                }
            }
        }
    }
    let mut slot = vec![0u64; values.len()];
    let mut free: Vec<u64> = Vec::new();
    let mut n_slots = 0u64;
    let mut lines = Vec::with_capacity(statements.len());
    for (t, &v) in statements.iter().enumerate() {
        let e = expr(&values, &stored, &slot, ops.tmp, v, true);
        // The slots this statement reads last are free for its own result: mstore evaluates first.
        let mut r = Vec::new();
        reads(&values, &stored, v, &mut r);
        r.sort_unstable();
        r.dedup();
        for w in r {
            if last_read[w] == t {
                free.push(slot[w]);
            }
        }
        if v == result {
            lines.push(format!("mstore(pQ, {e})"));
        } else {
            free.sort_unstable_by(|x, y| y.cmp(x));
            slot[v] = free.pop().unwrap_or_else(|| {
                n_slots += 1;
                n_slots - 1
            });
            lines.push(format!("mstore(add({}, {}), {e})", ops.tmp, WORD * slot[v]));
        }
    }
    Ok((lines, n_slots))
}

// ---------------------------------------------------------------------------------------------
// The generated parts of the contract
// ---------------------------------------------------------------------------------------------

fn offsets_text(offsets: &[i64]) -> String {
    format!("{offsets:?}")
}

/// The address `offset` bytes into the memory region `base`.
fn at(base: &str, offset: u64) -> String {
    if offset == 0 {
        base.to_string()
    } else {
        format!("add({base}, {offset})")
    }
}

/// Lines of the contract, each indented `indent` spaces.
struct Lines {
    indent: usize,
    out: String,
}

impl Lines {
    fn new(indent: usize) -> Self {
        Lines { indent, out: String::new() }
    }

    fn line(&mut self, text: impl AsRef<str>) {
        let text = text.as_ref();
        if !text.is_empty() {
            self.out.extend(std::iter::repeat_n(' ', self.indent));
            self.out.push_str(text);
        }
        self.out.push('\n');
    }

    /// The lines of `body` one level (4 spaces) deeper, in `{ }`.
    fn block(&mut self, body: impl FnOnce(&mut Lines)) {
        self.line("{");
        self.indent += 4;
        body(self);
        self.indent -= 4;
        self.line("}");
    }

    fn done(self) -> String {
        self.out
    }
}

/// `y^k − z`, with `y^k` at `p_yk` and `z` at `z`.
fn y_minus(p_yk: &str, z: &str) -> String {
    format!("addmod(mload({p_yk}), sub(q, mload({z})), q)")
}

/// The Horner evaluation at `y` of one offset of an `f_i`, into `v`: `sources` are its `p_j(ξ·ω^s)`.
fn horner(l: &mut Lines, sources: &[String]) {
    let mut from_top = sources.iter().rev();
    l.line(format!("v := {}", from_top.next().map_or("", String::as_str)));
    for e in from_top {
        l.line(format!("v := addmod(mulmod(v, y, q), {e}, q)"));
    }
}

/// The offsets of the `f_i` of one `(k, offsets)`, which every `f_i` of that shape shares: its
/// `Z_T(y)` and, with several offsets, where the CRT factor of each offset goes.
struct RootSet {
    k: u64,
    offsets: String,
    p_zt: String,
    p_l: Vec<Option<String>>,
}

/// The contract: [`VERIFIER_SOL`] with the parts the vkey gives (see the module).
fn contract(vkey: &Vkey) -> Result<String, SetupError> {
    let layout = &vkey.layout.0;
    let n_fixed = vkey.layout.n_fixed();
    let q_stage = vkey.layout.q_stage();
    let n_stages = q_stage.saturating_sub(1);
    let n_rows = 1u64 << vkey.power;
    let n_big = BigUint::from(n_rows);
    let omega_n = root_of_unity(&n_big)?;
    let cd = CalldataLayout::of(vkey);

    // --- The calldata (proof.rs, Proof::to_bytes; the module's "Calldata") ---------------------
    let mut calldata = Vec::new();
    let mut cd_slot = |name: String, word: u64, comment: String| -> String {
        calldata.push(Slot { name: name.clone(), offset: calldata_offset(word), comment: sanitize(&comment) });
        name
    };
    let mut commitment_of = BTreeMap::new();
    for (c, (i, f)) in layout.iter().enumerate().skip(n_fixed).enumerate() {
        let name = cd_slot(format!("pF{i}"), 2 * c as u64, format!("[f{i}]_1, stage {}", f.stage));
        commitment_of.insert(i, name);
    }
    let n_c = cd.n_commitments;
    cd_slot("pW".into(), 2 * n_c, "[W]_1".into());
    cd_slot("pWp".into(), 2 * n_c + 2, "[W']_1".into());

    // The evaluations, in the proof's order: the fixed columns' first, each in evMap order.
    let names: HashMap<(PolType, u64), &str> = layout
        .iter()
        .flat_map(|f| {
            let t = if f.stage == 0 { PolType::Const } else { PolType::Cm };
            f.pols.iter().map(move |p| ((t, p.id), p.name.as_str()))
        })
        .collect();
    let order: Vec<usize> = (0..vkey.ev_map.len())
        .filter(|&i| vkey.ev_map[i].pol_type == PolType::Const)
        .chain((0..vkey.ev_map.len()).filter(|&i| vkey.ev_map[i].pol_type == PolType::Cm))
        .collect();
    let mut eval = vec![String::new(); vkey.ev_map.len()];
    for (position, &i) in order.iter().enumerate() {
        let e = &vkey.ev_map[i];
        let Some(column) = names.get(&(e.pol_type, e.id)) else {
            return fail(format!("evMap[{i}] is {} {}, which no f of the layout packs", e.pol_type.as_str(), e.id));
        };
        eval[i] = cd_slot(
            format!("pEval{i}"),
            cd.first_scalar() + position as u64,
            format!("{} (evMap[{i}])", evaluation_name(column, e.prime)),
        );
    }
    // The pieces of a split Q, in the order of the layout (vkey.js, qPieceNames), which is that of
    // the calldata, and by piece.
    let mut piece_slots = BTreeMap::new();
    let mut pieces_in_calldata = Vec::new();
    if cd.n_q_pieces > 0 {
        let pieces = vkey.layout.q_entries().flat_map(|f| f.pols.iter());
        for (t, pol) in pieces.enumerate() {
            let piece: u64 = match pol.name.strip_prefix('Q').and_then(|i| i.parse().ok()) {
                Some(piece) => piece,
                None => return fail(format!("a piece of Q is named {:?}, not Q<i>", pol.name)),
            };
            let word = cd.first_scalar() + cd.n_evaluations + t as u64;
            let slot = cd_slot(format!("pEvalQ{piece}"), word, format!("Q{piece}(xi)"));
            pieces_in_calldata.push(slot.clone());
            piece_slots.insert(piece, slot);
        }
    }
    let after_pieces = cd.first_scalar() + cd.n_evaluations + cd.n_q_pieces;
    cd_slot("pInv".into(), after_pieces, "inv = 1/(prod_i Z_T_i(y), i >= 1, times the Lagrange denominators)".into());
    cd_slot("pInvZh".into(), after_pieces + 1, "invZh = 1/Z_H(xi)".into());
    let aux_slots: Vec<String> = cd
        .aux_rows
        .iter()
        .enumerate()
        .map(|(a, row)| {
            let comment = format!("1/(xi - w^{row}), auxiliary: not of the proof");
            cd_slot(format!("pAux{a}"), cd.proof_words() + a as u64, comment)
        })
        .collect();
    let mut aux_slots = aux_slots.into_iter();
    let p_public = calldata_offset(cd.words());
    let public: Vec<String> = (0..vkey.n_public).map(|i| at("pPublic", WORD * i)).collect();

    // --- checkInput: the points, those the transcript absorbs, the scalars and publics -------
    let mut check_input = Lines::new(16);
    for i in 0..n_c + 2 {
        check_input.line(format!("checkPointBelongsToBN128Curve({})", calldata_offset(2 * i)));
    }
    check_input.line("// Those the transcript absorbs: the commitments and W (challenges.js).");
    for i in 0..n_c + 1 {
        check_input.line(format!("checkAbsorbed({})", calldata_offset(2 * i)));
    }
    check_input.line("let r := q");
    check_input.line("let bad := 0");
    let scalars =
        (cd.first_scalar()..cd.words()).map(calldata_offset).chain((0..vkey.n_public).map(|i| p_public + WORD * i));
    for p in scalars {
        check_input.line(format!("bad := or(bad, iszero(lt(calldataload({p}), r)))"));
    }
    check_input.line("checkFields(bad)");

    // --- The memory -----------------------------------------------------------------------
    let mut mem = Memory { next: MEM_BASE, slots: Vec::new() };
    let mut challenges = HashMap::new();
    for s in 2..=n_stages {
        let count = vkey.num_challenges.get(s as usize - 1).copied().unwrap_or(0);
        for j in 0..count {
            let slot = mem.alloc(format!("pCh{s}_{j}"), 1, format!("challenge {j} of stage {s}"));
            challenges.insert((s, j), slot);
        }
    }
    challenges.insert((n_stages + 1, 0), mem.alloc("pStdVc".into(), 1, "std_vc".into()));
    challenges.insert((n_stages + 2, 0), mem.alloc("pXiSeed".into(), 1, "xiSeed (std_xi)".into()));
    mem.alloc("pAlpha".into(), 1, "alpha_S".into());
    mem.alloc("pY".into(), 1, "y".into());
    mem.alloc("pXi".into(), 1, "xi = xiSeed^powerW".into());
    mem.alloc("pXiN".into(), 1, "xi^N".into());
    mem.alloc("pZh".into(), 1, "Z_H(xi) = xi^N - 1".into());
    let zi_slots: Vec<String> = vkey
        .boundaries
        .iter()
        .enumerate()
        .map(|(b, boundary)| mem.alloc(format!("pZi{b}"), 1, format!("Zi of boundaries[{b}], {boundary:?}")))
        .collect();
    mem.alloc("pQ".into(), 1, "Q(xi)".into());

    // The power of y of each k (shplonk.js, computeZerofiers).
    let mut ks: Vec<u64> = layout.iter().map(|f| f.k).collect();
    ks.sort_unstable();
    ks.dedup();
    let mut p_yk = BTreeMap::new();
    let mut compute_z = Lines::new(16);
    for &k in &ks {
        let p = mem.alloc(format!("pYk{k}"), 1, format!("y^{k}"));
        let yk = if k == 1 { "y".to_string() } else { format!("powMod(y, {k})") };
        compute_z.line(format!("mstore({p}, {yk})"));
        p_yk.insert(k, p);
    }

    // The root sets: one per (k, offsets). The roots themselves are never computed
    // (computeInversions, computeR): what the verifier needs of them has a closed form in the
    // z_s = xi*w^s, whose k-th roots they are. For each set S, of t offsets, the product of the
    // k Lagrange denominators of an offset s (shplonk.js, computeInverseDenominators) is
    //   prod_j (y - x_j)*k*x_j^(k-1)*prod_{s' != s} (z_s - z_s')
    //     = sigma_k*k^k*z_s^(k-1)*prod_{s' != s} (z_s - z_s')^k*(y^k - z_s),
    // with prod_j (y - x_j) = y^k - z_s, prod_j x_j = (-1)^(k+1)*z_s and sigma_k its sign to the
    // k-1; and z_s = xi*w^s makes all but the y^k - z_s a constant times a power of xi: over the
    // offsets, C_S*xi^E_S*Z_T(y), E_S = t*(k-1) + k*t*(t-1).
    let mut root_sets: Vec<RootSet> = Vec::new();
    let mut set_of = Vec::with_capacity(layout.len());
    let mut set_consts: Vec<(BigUint, u64)> = Vec::new();
    let mut xi_neg_of: BTreeMap<u64, String> = BTreeMap::new();
    // computeInversions: each set's Z_T(y), then (after the f_i) the CRT factors of its offsets.
    let mut zt_of_sets = Lines::new(16);
    let mut crt = Lines::new(16);
    for f in layout {
        let offsets = offsets_text(&f.offsets);
        if let Some(set) = root_sets.iter().position(|s| s.k == f.k && s.offsets == offsets) {
            set_of.push(set);
            continue;
        }
        let index = root_sets.len();
        let k = f.k;
        let t = f.offsets.len() as u64;
        let comment = format!("k = {k}, offsets {offsets}");
        let p_z = mem.alloc(format!("pZ{index}"), t, format!("xi*w^s of each offset s of {comment}"));
        let p_zt = mem.alloc(format!("pZt{index}"), 1, format!("Z_T(y) of {comment}"));
        let p_l =
            (t > 1).then(|| mem.alloc(format!("pL{index}"), t, format!("the CRT factor of each offset of {comment}")));
        let p_xi_neg = (t > 1).then(|| {
            xi_neg_of
                .entry(t - 1)
                .or_insert_with(|| mem.alloc(format!("pXiNeg{}", t - 1), 1, format!("xi^-{}", t - 1)))
                .clone()
        });
        let p_yk = &p_yk[&k];
        let omegas: Vec<BigUint> = f.offsets.iter().map(|&s| signed_pow(&omega_n, s)).collect();
        let sigma = if k % 2 == 0 { r() - 1u32 } else { BigUint::from(1u32) };
        let k_pow_k = fr_pow(&BigUint::from(k), k);
        let mut c = BigUint::from(1u32);
        compute_z.line(format!("// {comment}"));
        zt_of_sets.line(format!("// {comment}"));
        zt_of_sets.line("d := 1");
        if p_xi_neg.is_some() {
            crt.line(format!("// {comment}: L_s = prod_{{s' != s}} (y^k - z_s')/(z_s - z_s')"));
        }
        for m in 0..f.offsets.len() {
            // prod_{s' != s} (w^s - w^s')
            let diff = omegas
                .iter()
                .enumerate()
                .filter(|&(l, _)| l != m)
                .fold(BigUint::from(1u32), |acc, (_, w)| acc * ((&omegas[m] + r() - w) % r()) % r());
            c = c * &sigma % r() * &k_pow_k % r() * fr_pow(&omegas[m], k - 1) % r() * fr_pow(&diff, k) % r();
            let z = at(&p_z, WORD * m as u64);
            let zs = if omegas[m] == BigUint::from(1u32) {
                "xi".to_string()
            } else {
                format!("mulmod(xi, {}, q)", omegas[m].to_str_radix(10))
            };
            compute_z.line(format!("mstore({z}, {zs})"));
            zt_of_sets.line(format!("d := mulmod(d, {}, q)", y_minus(p_yk, &z)));
            if let Some(p_xi_neg) = &p_xi_neg {
                crt.line(format!("d := mulmod({}, mload({p_xi_neg}), q)", fr_inv(&diff).to_str_radix(10)));
                for l in (0..t).filter(|&l| l != m as u64) {
                    crt.line(format!("d := mulmod(d, {}, q)", y_minus(p_yk, &at(&p_z, WORD * l))));
                }
                crt.line(format!("mstore({}, d)", p_l.as_ref().map(|p| at(p, WORD * m as u64)).unwrap_or_default()));
            }
        }
        zt_of_sets.line("// shplonk.js, verifyOpening: y is not a root of an f_i (inv·A·B = 1 checks it too).");
        zt_of_sets.line("if iszero(d) {");
        zt_of_sets.line("    fail()");
        zt_of_sets.line("}");
        zt_of_sets.line(format!("mstore({p_zt}, d)"));
        set_consts.push((c, t * (k - 1) + k * t * (t - 1)));
        let p_l = (0..t).map(|m| p_l.as_ref().map(|p| at(p, WORD * m))).collect();
        root_sets.push(RootSet { k, offsets, p_zt, p_l });
        set_of.push(index);
    }
    // B, the product of every f_i's Lagrange denominators: G*xi^E*prod_i Z_{T_i}(y).
    let (g, e) = set_of.iter().fold((BigUint::from(1u32), 0u64), |(g, e), &set| {
        let (c, e_s) = &set_consts[set];
        (g * c % r(), e + e_s)
    });
    let g = g.to_str_radix(10);

    // The Z_{T_i}(y), i >= 1, of the proof's inv with the Lagrange denominators, and their
    // inverses (pilfflonk/docs/protocol.md#inverses).
    let p_inv_zt: Vec<Option<String>> = (0..layout.len())
        .map(|i| (i > 0).then(|| mem.alloc(format!("pInvZt{i}"), 1, format!("Z_T(y) of f{i}, then its inverse"))))
        .collect();
    let p_r: Vec<String> = (0..layout.len()).map(|i| mem.alloc(format!("pR{i}"), 1, format!("r_{i}(y)"))).collect();
    mem.alloc("pF".into(), 2, "[F]_1".into());
    mem.alloc("pE".into(), 2, "[E]_1".into());
    mem.alloc("pJ".into(), 2, "[J]_1".into());

    // --- The transcript, challenges.js ----------------------------------------------------
    let mut t = TranscriptCode::default();
    t.comment("Step 1: digest mod r, the number of instances of the AIR (1), the publics");
    t.word("DIGEST");
    t.word("1");
    t.calldata("pPublic", WORD * vkey.n_public);
    let committed: Vec<(usize, u64)> = layout.iter().enumerate().skip(n_fixed).map(|(i, f)| (i, f.stage)).collect();
    let absorb_stage = |t: &mut TranscriptCode, s: u64| {
        let of_stage: Vec<usize> = committed.iter().filter(|(_, stage)| *stage == s).map(|(i, _)| *i).collect();
        if let Some(first) = of_stage.first() {
            t.calldata(&commitment_of[first], 2 * WORD * of_stage.len() as u64);
        }
    };
    for s in 1..=n_stages {
        t.comment(&format!("Step 2, stage {s}: the commitments of its f"));
        absorb_stage(&mut t, s);
        if s < n_stages {
            let count = vkey.num_challenges.get(s as usize).copied().unwrap_or(0);
            for j in 0..count {
                t.comment(&format!("challenge {j} of stage {}", s + 1));
                t.squeeze(&challenges[&(s + 1, j)]);
            }
        }
    }
    t.comment("Step 3: std_vc; the commitments of Q; xiSeed");
    t.squeeze("pStdVc");
    absorb_stage(&mut t, q_stage);
    t.squeeze("pXiSeed");
    t.comment("Step 4: the evaluations, in the order of the proof, and the pieces of Q");
    // From the first scalar of the calldata: the first evaluation or, with none, the first piece
    // of Q in the calldata's order (which need not be Q0).
    if let Some(first) = order.first().map(|&i| &eval[i]).or(pieces_in_calldata.first()) {
        t.calldata(first, WORD * (cd.n_evaluations + cd.n_q_pieces));
    }
    t.squeeze("pAlpha");
    t.comment("Step 5: [W]_1; y");
    t.calldata("pW", 2 * WORD);
    t.squeeze("pY");
    let mut compute_challenges = Lines::new(16);
    t.lines.iter().for_each(|line| compute_challenges.line(line));

    // --- Zi (qverifier.js, computeZi) -----------------------------------------------------
    let neg_row = |j: u64| fr_neg(&fr_pow(&omega_n, j)).to_str_radix(10);
    let mut compute_zi = Lines::new(16);
    for (b, boundary) in vkey.boundaries.iter().enumerate() {
        let slot = &zi_slots[b];
        // `firstRow`, `lastRow`: the calldata's auxiliary inverse, and `−ω^j` of its row.
        let row_aux = |aux_slots: &mut std::vec::IntoIter<String>, j: u64| aux_slots.next().map(|a| (a, neg_row(j)));
        let (kind, aux, neg_rows, comment) = match *boundary {
            Boundary::EveryRow => ("everyRow", None, vec![], "1/Z_H(xi): the proof's invZh".to_string()),
            Boundary::FirstRow => ("firstRow", row_aux(&mut aux_slots, 0), vec![], "Z_H(xi)/(xi - 1)".to_string()),
            Boundary::LastRow => {
                ("lastRow", row_aux(&mut aux_slots, n_rows - 1), vec![], format!("Z_H(xi)/(xi - w^{})", n_rows - 1))
            }
            Boundary::EveryFrame { offset_min, offset_max } => {
                let rows: Vec<u64> = (0..offset_min).chain(n_rows - offset_max..n_rows).collect();
                let comment =
                    format!("everyFrame {{{offset_min}, {offset_max}}}: prod of (xi - w^j) over rows {rows:?}");
                ("everyFrame", None, rows.into_iter().map(neg_row).collect(), comment)
            }
        };
        compute_zi.line(format!("// boundaries[{b}], {kind}: {}", sanitize(&comment)));
        match aux {
            _ if kind == "everyRow" => compute_zi.line(format!("mstore({slot}, calldataload(pInvZh))")),
            Some((aux, neg_root)) => compute_zi.block(|l| {
                l.line("// The calldata's 1/(xi - w^j), checked as inv is.");
                l.line(format!("let a := calldataload({aux})"));
                l.line(format!("if iszero(eq(mulmod(addmod(xi, {neg_root}, q), a, q), 1)) {{"));
                l.line("    fail()");
                l.line("}");
                l.line(format!("mstore({slot}, mulmod(zh, a, q))"));
            }),
            None => compute_zi.block(|l| {
                l.line("let z := 1");
                for neg in &neg_rows {
                    l.line(format!("z := mulmod(z, addmod(xi, {neg}, q), q)"));
                }
                l.line(format!("mstore({slot}, z)"));
            }),
        }
    }

    // --- The qVerifier (qverifier.js, executeCode) ----------------------------------------
    let (q_lines, n_tmp) = q_code(
        &vkey.q_verifier,
        &QOperands { eval: &eval, public: &public, zi: &zi_slots, challenges: &challenges, tmp: "pTmp" },
    )?;
    mem.alloc("pTmp".into(), n_tmp, "the stored values of the qVerifier".into());
    let mut compute_q = Lines::new(16);
    q_lines.iter().for_each(|line| compute_q.line(line));
    // A split Q: the function checkQPieces, and its call.
    let (check_q_pieces, check_q_pieces_call) = if cd.n_q_pieces > 0 {
        let mut l = Lines::new(12);
        l.line(format!("// STEP 5 (verify.js): Q is split in pieces of M = {}, and", vkey.max_q_degree));
        l.line("// sum_i xi^(i·M·N)·Q_i(xi) = Q(xi) (qverifier.js, joinQPieces).");
        l.line("function checkQPieces() {");
        l.line(format!("    let shift := powMod(mload(pXiN), {})", vkey.max_q_degree));
        l.line("    let acc := 0");
        // Horner's order: Q_{m−1} down to Q_0.
        for p in piece_slots.values().rev() {
            l.line(format!("    acc := addmod(mulmod(acc, shift, q), calldataload({p}), q)"));
        }
        l.line("    if iszero(eq(acc, mload(pQ))) {");
        l.line("        fail()");
        l.line("    }");
        l.line("}");
        (l.done(), "            checkQPieces()\n".to_string())
    } else {
        (String::new(), String::new())
    };

    // --- The f_i (shplonk.js: openingEvaluations of verify.js, computeR, computeF) ---------
    let mut inv_zt_of_fs = Lines::new(16);
    let mut compute_r = Lines::new(16);
    let mut compute_fej = Lines::new(16);
    for (i, f) in layout.iter().enumerate() {
        let set = &root_sets[set_of[i]];
        let t = if f.stage == 0 { PolType::Const } else { PolType::Cm };
        let is_q = f.stage == q_stage;
        // `f<i>` if fixed, the vkey's; otherwise its calldata constant.
        let commitment = if f.stage == 0 { format!("f{i}") } else { commitment_of[&i].clone() };
        if let Some(p_inv_zt) = &p_inv_zt[i] {
            inv_zt_of_fs.line(format!("d := mload({})", set.p_zt));
            inv_zt_of_fs.line(format!("mstore({p_inv_zt}, d)"));
            inv_zt_of_fs.line("a := mulmod(a, d, q)");
        }
        // Each offset's `p_j(ξ·ω^s)`, of `R_s(X) = Σ_j p_j(ξ·ω^s)·X^j`.
        let rows = f
            .offsets
            .iter()
            .map(|&s| {
                f.pols
                    .iter()
                    .map(|p| {
                        if is_q {
                            if cd.n_q_pieces > 0 {
                                let piece = p.name.strip_prefix('Q').and_then(|i| i.parse::<u64>().ok());
                                match piece.and_then(|piece| piece_slots.get(&piece)) {
                                    Some(slot) => Ok(format!("calldataload({slot})")),
                                    None => fail(format!("no piece of Q is named {:?}", p.name)),
                                }
                            } else {
                                Ok("mload(pQ)".to_string())
                            }
                        } else {
                            match vkey.ev_map.iter().position(|e| e.pol_type == t && e.id == p.id && e.prime == s) {
                                Some(e) => Ok(format!("calldataload({})", eval[e])),
                                None => fail(format!(
                                    "the evMap has no {} {} at offset {s}, which f{i} opens",
                                    t.as_str(),
                                    p.id
                                )),
                            }
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, SetupError>>()?;
        compute_r.line(format!("// f{i}: stage {}, k = {}, offsets {}", f.stage, f.k, set.offsets));
        if let [row] = rows.as_slice() {
            horner(&mut compute_r, row);
            compute_r.line(format!("mstore({}, v)", p_r[i]));
        } else {
            compute_r.block(|l| {
                l.line("let acc := 0");
                for (row, p_l) in rows.iter().zip(&set.p_l) {
                    horner(l, row);
                    l.line(format!(
                        "acc := addmod(acc, mulmod(v, mload({}), q), q)",
                        p_l.as_deref().unwrap_or_default()
                    ));
                }
                l.line(format!("mstore({}, acc)", p_r[i]));
            });
        }
        if i == 0 {
            compute_fej.line(format!("let zt0 := mload({})", set.p_zt));
            compute_fej.line(format!("let e := mload({})", p_r[i]));
            if f.stage == 0 {
                compute_fej.line(format!("mstore(pF, {commitment}x)"));
                compute_fej.line(format!("mstore(add(pF, 32), {commitment}y)"));
            } else {
                compute_fej.line(format!("calldatacopy(pF, {commitment}, 64)"));
            }
            continue;
        }
        compute_fej.line(format!("// f{i}"));
        compute_fej.line("alphaPow := mulmod(alphaPow, alpha, q)");
        let p_inv_zt = p_inv_zt[i].as_deref().unwrap_or_default();
        compute_fej.line(format!("quotient := mulmod(alphaPow, mulmod(zt0, mload({p_inv_zt}), q), q)"));
        if f.stage == 0 {
            compute_fej.line(format!("g1_mulAccC(pF, {commitment}x, {commitment}y, quotient)"));
        } else {
            compute_fej.line(format!("g1_mulAcc(pF, {commitment}, quotient)"));
        }
        compute_fej.line(format!("e := addmod(e, mulmod(quotient, mload({}), q), q)", p_r[i]));
    }

    // --- computeInversions: B = G·ξ^E·Π_i Z_{T_i}(y) and the ξ^(−(t−1)) of the root sets ----
    let mut compute_inversions = Lines::new(16);
    compute_inversions.out = zt_of_sets.done();
    compute_inversions.line("let a := 1");
    compute_inversions.out += &inv_zt_of_fs.done();
    compute_inversions.line(format!("let zts := mulmod(a, mload({}), q)", root_sets[set_of[0]].p_zt));
    let b = if e > 0 { format!("mulmod({g}, powMod(mload(pXi), {e}), q)") } else { g.clone() };
    compute_inversions.line(format!("let b := mulmod(zts, {b}, q)"));
    compute_inversions.line("let invB := mulmod(calldataload(pInv), a, q)");
    compute_inversions.line("checkInv(invB, b)");
    let n_inv_zt = layout.len() as u64 - 1;
    if n_inv_zt > 0 {
        compute_inversions.line(format!("batchInverse(pInvZt1, {n_inv_zt}, mulmod(calldataload(pInv), b, q))"));
    }
    if !xi_neg_of.is_empty() {
        compute_inversions.line(format!("let xiNegE := mulmod(invB, mulmod({g}, zts, q), q)"));
        for (&t1, p) in &xi_neg_of {
            let v = if e > t1 {
                format!("mulmod(xiNegE, powMod(mload(pXi), {}), q)", e - t1)
            } else {
                "xiNegE".to_string()
            };
            compute_inversions.line(format!("mstore({p}, {v})"));
        }
        compute_inversions.out += &crt.done();
    }

    // --- The constants: the vkey's, the calldata's and the memory's -------------------------
    let mut l = Lines::new(4);
    let aux = cd.words() - cd.proof_words();
    let aux = if aux > 0 { format!(", then {aux} auxiliary inverses") } else { String::new() };
    l.line(format!("// The vkey with digest {}: `proof` is {} words{aux}", vkey.digest.to_hex(), cd.proof_words()));
    l.line(format!("uint32 constant n = {n_rows}; // Domain size, N = 2^{}", vkey.power));
    l.line(format!("uint256 constant powerW = {}; // xi = xiSeed^powerW", vkey.power_w));
    l.line("");
    l.line("// Verifier preprocessed input x·[1]_2: the vkey's X_2, [tau]_2");
    let x2 = &vkey.x_2;
    for (name, v) in [("X2x1", &x2.x[0]), ("X2x2", &x2.x[1]), ("X2y1", &x2.y[0]), ("X2y2", &x2.y[1])] {
        l.line(format!("uint256 constant {name} = {};", v.to_decimal()));
    }
    l.line("");
    l.line("// The vkey's digest mod r, the first value of the transcript");
    l.line(format!("uint256 constant DIGEST = {};", vkey.digest.to_fr().to_decimal()));
    if !vkey.fixed_commitments.0.is_empty() {
        l.line("");
        l.line("// The commitments of the fixed f_i, the vkey's: never the proof's");
        for (index, p) in vkey.fixed_commitments.0.iter().enumerate() {
            l.line(format!("uint256 constant f{index}x = {};", p.x.to_decimal()));
            l.line(format!("uint256 constant f{index}y = {};", p.y.to_decimal()));
        }
    }
    l.line("");
    l.line("// Proof calldata: the byte offset of every value (proof.rs, Proof::to_bytes)");
    for s in &calldata {
        l.line(format!("uint256 constant {} = {}; // {}", s.name, s.offset, s.comment));
    }
    l.line("// Where the publics are, the unnamed `uint256[]` (pubSignals)");
    l.line(format!("uint256 constant pPublic = {p_public};"));
    l.line("");
    l.line("// Memory data, at absolute addresses from 0x80 (solidity.rs, MEM_BASE)");
    for s in &mem.slots {
        l.line(format!("uint256 constant {} = {}; // {}", s.name, s.offset, s.comment));
    }
    l.line(format!("uint256 constant lastMem = {};", mem.next));

    // `verifyProof`'s line: the ABI of the vkey (no `pubSignals` without publics).
    let publics = if vkey.n_public > 0 { format!(", uint256[{}] calldata", vkey.n_public) } else { String::new() };
    let signature =
        format!("    function verifyProof(bytes32[{}] calldata{publics}) public view returns (bool) {{\n", cd.words());
    // `xi^N`: `power` squarings of `xin`.
    let mut compute_zh = Lines::new(16);
    (0..vkey.power).for_each(|_| compute_zh.line("xin := mulmod(xin, xin, q)"));

    let parts = [
        ("constants", l.done()),
        ("signature", signature),
        ("checkInput", check_input.done()),
        ("computeChallenges", compute_challenges.done()),
        ("computeZh", compute_zh.done()),
        ("computeZi", compute_zi.done()),
        ("computeQ", compute_q.done()),
        ("checkQPieces", check_q_pieces),
        ("computeZ", compute_z.done()),
        ("computeInversions", compute_inversions.done()),
        ("computeR", compute_r.done()),
        ("computeFEJ", compute_fej.done()),
        ("checkQPiecesCall", check_q_pieces_call),
    ];
    splice(VERIFIER_SOL, &parts)
}

/// `src` with each `// @generated <part>` line replaced by that part's text, every part exactly once.
fn splice(src: &str, parts: &[(&str, String)]) -> Result<String, SetupError> {
    let mut used = vec![false; parts.len()];
    let mut out = String::with_capacity(src.len() * 4);
    // An empty part between blank lines takes the second along.
    let mut skip_blank = false;
    for line in src.split_inclusive('\n') {
        let Some(name) = line.trim().strip_prefix("// @generated ") else {
            if !(skip_blank && line.trim().is_empty()) {
                out.push_str(line);
            }
            skip_blank = false;
            continue;
        };
        match parts.iter().position(|(n, _)| *n == name) {
            Some(i) if !used[i] => {
                used[i] = true;
                skip_blank = parts[i].1.is_empty() && out.ends_with("\n\n");
                out.push_str(&parts[i].1);
            }
            _ => return fail(format!("the static verifier has an unknown or repeated part {name:?}")),
        }
    }
    if let Some(i) = used.iter().position(|u| !u) {
        return fail(format!("the static verifier has no part {:?}", parts[i].0));
    }
    Ok(out)
}

/// The Solidity verifier of `vkey` (see the module). Refuses a vkey the verifier would not read and
/// one whose digest is not that of its contents (the JS verifier accepts no proof of it).
pub fn verifier_sol(vkey: &Vkey) -> Result<String, SetupError> {
    vkey.validate()?;
    if vkey.compute_digest()? != vkey.digest {
        return fail(
            "the vkey's digest is not the digest of its contents (pilfflonk/docs/formats.md#digest): the JS verifier \
             accepts no proof of it",
        );
    }
    q_in_memory(&contract(vkey)?)
}

/// The memory slot `q` is kept in through `verifyProof`'s assembly: Solidity's zero slot, which the
/// assembly never hands back to Solidity (it ends in `return`).
const Q_SLOT: &str = "0x60";

/// `src` with every `q` of its assembly read from [`Q_SLOT`], stored there first. solc at
/// `optimize-runs 1` copies a 32-byte constant out of the code at each use, about 10 bytes a use: on a
/// verifier of thousands of field operations that is most of its bytecode (and gas).
fn q_in_memory(src: &str) -> Result<String, SetupError> {
    let at = src
        .find("        assembly {\n")
        .ok_or_else(|| SetupError::Solidity("the static verifier has no `assembly {` block to keep q in".into()))?;
    let (solidity, assembly) = src.split_at(at);
    if !assembly.contains("return(") {
        return Err(SetupError::Solidity(
            "the verifier's assembly does not end in return: q's slot is Solidity's".into(),
        ));
    }
    let mut out = String::with_capacity(src.len() + src.len() / 4);
    out.push_str(solidity);
    for (i, line) in assembly.split_inclusive('\n').enumerate() {
        let (code, comment) = line.find("//").map_or((line, ""), |c| line.split_at(c));
        let bytes = code.as_bytes();
        let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'.';
        let mut last = 0;
        for (j, &b) in bytes.iter().enumerate() {
            let alone = b == b'q' && (j == 0 || !ident(bytes[j - 1])) && (j + 1 == bytes.len() || !ident(bytes[j + 1]));
            if alone {
                out.push_str(&code[last..j]);
                out.push_str(&format!("mload({Q_SLOT})"));
                last = j + 1;
            }
        }
        out.push_str(&code[last..]);
        out.push_str(comment);
        if i == 0 {
            out.push_str(&format!("            mstore({Q_SLOT}, q)\n"));
        }
    }
    Ok(out)
}

/// Writes the Solidity verifier of `vkey` at `path`, replacing any file there.
pub fn write_verifier_sol(vkey: &Vkey, path: &Path) -> Result<(), SetupError> {
    let sol = verifier_sol(vkey)?;
    fs::write(path, sol).map_err(SetupError::io(path))
}

/// `proofman-setup pilfflonk-solidity`: the Solidity verifier of the vkey at `vkey_path`, written at
/// `out`.
pub fn export_verifier_sol(vkey_path: &Path, out: &Path) -> Result<(), SetupError> {
    let vkey = Vkey::read(vkey_path)?;
    write_verifier_sol(&vkey, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_comment_holds_no_line_terminator() {
        for terminator in ["\n", "\r", "\u{b}", "\u{c}", "\u{85}", "\u{2028}", "\u{2029}"] {
            let text = format!("Main.a{terminator}uint256 constant x = 1;");
            assert_eq!(sanitize(&text), "Main.a?uint256 constant x = 1;", "{terminator:?}");
        }
        assert_eq!(sanitize("Fibonacci.ImPol[0]w-1 (evMap[3])"), "Fibonacci.ImPol[0]w-1 (evMap[3])");
    }

    #[test]
    fn the_roots_of_unity_are_those_of_the_js_verifier() {
        // w_2 = −1, w_4^2 = −1, and ω_N of order N (shplonk.js, rootOfUnity).
        assert_eq!(root_of_unity(&BigUint::from(2u32)).unwrap(), r() - 1u32);
        assert_eq!(fr_pow(&root_of_unity(&BigUint::from(4u32)).unwrap(), 2), r() - 1u32);
        let omega = root_of_unity(&BigUint::from(1u64 << 8)).unwrap();
        assert_eq!(fr_pow(&omega, 1 << 8), BigUint::from(1u32));
        assert_ne!(fr_pow(&omega, 1 << 7), BigUint::from(1u32));
        // 5 or 7 do not divide r − 1: no root of those orders.
        assert!(root_of_unity(&BigUint::from(5u32)).is_err());
        assert!(root_of_unity(&BigUint::ZERO).is_err());
        // ω^(−s) is the inverse of ω^s, and −a + a = 0.
        let (w, w_inv) = (signed_pow(&omega, 3), signed_pow(&omega, -3));
        assert_eq!((w * w_inv) % r(), BigUint::from(1u32));
        assert_eq!((fr_neg(&omega) + &omega) % r(), BigUint::ZERO);
        assert_eq!(fr_neg(&BigUint::ZERO), BigUint::ZERO);
    }
}
