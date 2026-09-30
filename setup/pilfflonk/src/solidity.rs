//! `pilfflonk.verifier.sol`: the Solidity verifier of a vkey (spec §4.5, "Verificador Solidity";
//! Fase 4), which `setup-pilfflonk --solidity` writes next to `pilfflonk.vkey.json` and
//! `proofman-setup pilfflonk-solidity` writes from a vkey alone.
//!
//! The template, `tera/verifier_pilfflonk.sol.tera`, is rendered with `tera` as the STARK's Solidity
//! and circom templates are (`setup/stark-recurser/stark2circom/circuit_templates/templates.rs`:
//! `include_str!` and one `render`). It is snarkjs 0.7.6's fflonk verifier
//! (`templates/verifier_fflonk.sol.ejs`) generalised as the JS verifier (`pilfflonk/js/src/
//! verify.js`, D8) generalises snarkjs's `fflonk_verify.js`, and it accepts exactly the proofs the
//! JS verifier accepts. This module computes what the template takes from the vkey:
//! - the constants: `[τ]₂`, the fixed commitments, `digest mod r` and the roots of unity;
//! - where each value is in the calldata ([`CalldataLayout`]) and in memory;
//! - the transcript of A.4 (`challenges.js`) and the vkey's `qVerifier` (`qverifier.js`), as
//!   straight-line Yul.
//!
//! **Calldata.** `verifyProof(bytes32[W] calldata proof, uint256[P] calldata pubSignals)`, as snarkjs's
//! `FflonkVerifier`, with `P = nPublic` (no `pubSignals` if it is 0) and `W` the words of
//! [`CalldataLayout`]: the proof's bytes (A.6, `Proof::to_bytes`) as 32-byte words, followed by one
//! auxiliary inverse `1/(ξ − ω^j)` for each `firstRow` (`j = 0`) or `lastRow` (`j = N − 1`) boundary,
//! in the order of the boundaries (spec §4.5, "Calldata"). The contract checks each, as it checks
//! `inv` and `invZh`. A vkey without those boundaries, as every compiled PIL2 program (§3.4), has none,
//! and the calldata is the proof.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use num_bigint::BigUint;
use proofman_pilfflonk::names::evaluation_name;
use proofman_pilfflonk::{Boundary, G1Affine, JsonFile, PolType, Vkey, BN254_Q, BN254_R};
use serde::Serialize;
use serde_json::Value;
use tera::{Context as TeraContext, Tera};

use crate::digest::vkey_digest;
use crate::error::SetupError;

/// The file name of the verifier, in the backend directory of the `provingKey/`, next to the vkey
/// (spec §4.2.6).
pub const VERIFIER_SOL_FILE: &str = "pilfflonk.verifier.sol";

/// The template (see the module).
const VERIFIER_TEMPLATE: &str = include_str!("tera/verifier_pilfflonk.sol.tera");

/// Bytes of a word of the calldata and of the memory.
const WORD: u64 = 32;

/// The selector before the arguments of `verifyProof`.
const SELECTOR_BYTES: u64 = 4;

fn fail<T>(message: impl Into<String>) -> Result<T, SetupError> {
    Err(SetupError::Solidity(message.into()))
}

// ---------------------------------------------------------------------------------------------
// BN254's scalar field, with num-bigint: the constants the contract embeds
// ---------------------------------------------------------------------------------------------

fn big(decimal: &str) -> BigUint {
    BigUint::parse_bytes(decimal.as_bytes(), 10).unwrap_or_default()
}

fn r() -> &'static BigUint {
    static R: OnceLock<BigUint> = OnceLock::new();
    R.get_or_init(|| big(BN254_R))
}

fn q() -> &'static BigUint {
    static Q: OnceLock<BigUint> = OnceLock::new();
    Q.get_or_init(|| big(BN254_Q))
}

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

/// Whether a fixed commitment is a point of G1 or the point at infinity, `(0, 0)`, as the JS verifier
/// reads the vkey's (`vkey.js`, `g1FromObject` with `allowInfinity`): the cofactor of G1 is 1, and
/// the curve `y² = x³ + 3` is the group.
fn is_g1(point: &G1Affine) -> bool {
    if point.is_infinity() {
        return true;
    }
    let x = BigUint::from_bytes_le(&point.x.to_le_bytes());
    let y = BigUint::from_bytes_le(&point.y.to_le_bytes());
    (&y * &y) % q() == (&x * &x * &x + 3u32) % q()
}

// ---------------------------------------------------------------------------------------------
// The calldata
// ---------------------------------------------------------------------------------------------

/// Where the verifier finds each value in its `proof` argument, in words (see the module).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalldataLayout {
    /// The commitments of the non-fixed `f_i`, `x` and `y` each a word, from word 0.
    pub n_commitments: u64,
    /// The evaluations of the evMap, in the order of the proof: the fixed columns' first (A.4,
    /// step 4; A.6).
    pub n_evaluations: u64,
    /// The pieces `Q_i(ξ)` of a split `Q`, and 0 if it is whole.
    pub n_q_pieces: u64,
    /// The rows `j` of the auxiliary inverses `1/(ξ − ω^j)` after the proof: one per `firstRow`
    /// (`0`) and `lastRow` (`N − 1`) boundary, in the order of the boundaries.
    pub aux_rows: Vec<u64>,
}

impl CalldataLayout {
    pub fn of(vkey: &Vkey) -> Self {
        let n_fixed = vkey.layout.n_fixed() as u64;
        let n_rows = 1u64 << vkey.power;
        let q_stage = vkey.layout.0.last().map_or(0, |f| f.stage);
        let n_q = vkey.layout.0.iter().filter(|f| f.stage == q_stage).map(|f| f.k).sum::<u64>();
        let aux_rows = vkey
            .boundaries
            .iter()
            .filter_map(|b| match b {
                Boundary::FirstRow => Some(0),
                Boundary::LastRow => Some(n_rows - 1),
                Boundary::EveryRow | Boundary::EveryFrame { .. } => None,
            })
            .collect();
        CalldataLayout {
            n_commitments: vkey.layout.0.len() as u64 - n_fixed,
            n_evaluations: vkey.ev_map.len() as u64,
            n_q_pieces: if n_q > 1 { n_q } else { 0 },
            aux_rows,
        }
    }

    /// The words of the proof's bytes (A.6): the commitments, `W`, `W'`, the evaluations, the
    /// pieces of `Q`, `inv` and `invZh`. Format version 1 has no air, airgroup or proof values.
    pub fn proof_words(&self) -> u64 {
        2 * (self.n_commitments + 2) + self.n_evaluations + self.n_q_pieces + 2
    }

    /// The words of the `proof` argument: the proof's and the auxiliary inverses.
    pub fn words(&self) -> u64 {
        self.proof_words() + self.aux_rows.len() as u64
    }

    /// The word of the first scalar, the first evaluation.
    fn first_scalar(&self) -> u64 {
        2 * (self.n_commitments + 2)
    }

    /// The byte offset of `word` in the calldata, after the selector.
    fn offset(word: u64) -> u64 {
        SELECTOR_BYTES + WORD * word
    }
}

// ---------------------------------------------------------------------------------------------
// The memory
// ---------------------------------------------------------------------------------------------

/// The named regions of the verifier's memory, from `pMem`: each a Solidity constant.
#[derive(Default)]
struct Memory {
    next: u64,
    slots: Vec<Slot>,
}

#[derive(Serialize)]
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
// The template's context
// ---------------------------------------------------------------------------------------------

#[derive(Serialize)]
struct Context {
    digest: String,
    digest_fr: String,
    power: u64,
    n: u64,
    power_w: u64,
    n_public: u64,
    x2: [[String; 2]; 2],
    fixed: Vec<FixedCommitment>,

    // The calldata.
    words: u64,
    proof_words: u64,
    calldata: Vec<Slot>,
    p_points_end: u64,
    p_absorbed_end: u64,
    p_scalars: u64,
    p_scalars_end: u64,
    p_public: u64,

    // The memory.
    memory: Vec<Slot>,
    last_mem: u64,

    // The steps.
    transcript: Vec<String>,
    zi: Vec<Zi>,
    q_code: Vec<String>,
    q_split: Option<QSplit>,
    ks: Vec<KPowers>,
    root_sets: Vec<RootSet>,
    fs: Vec<F>,
    n_inv: u64,
}

#[derive(Serialize)]
struct FixedCommitment {
    index: usize,
    x: String,
    y: String,
}

/// `Zi(D)` at `ξ` for a boundary `D` (`qverifier.js`, `computeZi`).
#[derive(Serialize)]
struct Zi {
    index: usize,
    kind: &'static str,
    slot: String,
    /// `firstRow`, `lastRow`: the calldata's auxiliary inverse, and `−ω^j` of its row.
    aux: Option<String>,
    neg_root: Option<String>,
    /// `everyFrame`: `−ω^j` of each row it excludes.
    neg_rows: Vec<String>,
    comment: String,
}

/// `Σ_i ξ^(i·M·N)·Q_i(ξ) = Q(ξ)` (`qverifier.js`, `joinQPieces`).
#[derive(Serialize)]
struct QSplit {
    max_q_degree: u64,
    /// The calldata constants of the pieces, from `Q_{m−1}` down to `Q_0`: Horner's order.
    pieces_desc: Vec<String>,
}

/// The powers of `xiSeed` and `y` each `k` needs.
#[derive(Serialize)]
struct KPowers {
    k: u64,
    /// `powerW / k`: `seed_k = xiSeed^(powerW/k)`.
    exponent: u64,
    p_seed: String,
    p_yk: String,
}

/// The roots of the `f_i` of one `(k, offsets)` (`shplonk.js`, `computeRoots`), which every `f_i`
/// of that shape shares.
#[derive(Serialize)]
struct RootSet {
    index: usize,
    k: u64,
    offsets: String,
    k_minus_1: u64,
    p_seed: String,
    p_yk: String,
    w_k: String,
    p_zt: String,
    rows: Vec<RootRow>,
}

/// The `k` roots of one offset `s` of a root set: `x_j = xiSeed^(powerW/k)·ω_{kN}^s·w_k^j`, whose
/// `k`-th power is `z = ξ·ω_N^s`.
#[derive(Serialize)]
struct RootRow {
    s: i64,
    /// `ω_{kN}^s` and `ω_N^s`, `None` for `s = 0`.
    omega_kn_s: Option<String>,
    omega_n_s: Option<String>,
    /// Where its roots, its `z` and its Lagrange factor are, from `pMem`.
    roots: String,
    z: String,
    den1: String,
    /// Where the `z` of the other rows are.
    other_z: Vec<String>,
}

/// One `f_i` of the layout, in the global order of A.5.
#[derive(Serialize)]
struct F {
    index: usize,
    stage: u64,
    k: u64,
    offsets: String,
    /// `f<i>` if fixed, the vkey's; otherwise its calldata constant.
    fixed: bool,
    commitment: String,
    /// The slot of `Z_{T_i}(y)` in the inverted array, `i ≥ 1`.
    p_inv_zt: Option<String>,
    /// `Z_{T_i}(y)` of its root set, `w_k^(−1)`.
    p_zt: String,
    w_k_inv: String,
    p_r: String,
    rows: Vec<FRow>,
}

/// An offset of an `f_i`: the evaluations `p_j(ξ·ω^s)` its value at a root is made of,
/// `f_i(x) = Σ_j p_j(ξ·ω^s)·x^j`.
#[derive(Serialize)]
struct FRow {
    /// Where its roots and their Lagrange factor are (its root set's), and where its denominators
    /// go in the inverted array, from `pMem`.
    roots: String,
    den1: String,
    inv_den: String,
    /// Horner's: `p_{k−1}` first, then `p_{k−2} … p_0`.
    horner_first: String,
    horner_rest: Vec<String>,
}

// ---------------------------------------------------------------------------------------------
// The transcript (A.4, `challenges.js`)
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
        self.lines.push(format!("mstore(add(pMem, {slot}), c)"));
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

/// An operand as a Yul expression, and whether its value is fixed for the whole code (all but the
/// temporaries): a copy of such a value is the value itself.
fn q_operand(
    value: &Value,
    at: &str,
    ops: &QOperands,
    aliases: &HashMap<u64, String>,
) -> Result<(String, bool), SetupError> {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or_default();
    Ok(match kind {
        "tmp" => {
            let id = u64_field(value, "id", at)?;
            match aliases.get(&id) {
                Some(expr) => (expr.clone(), true),
                None => (format!("mload(add(pMem, add({}, {})))", ops.tmp, WORD * id), false),
            }
        }
        "eval" => (format!("calldataload({})", index(ops.eval, u64_field(value, "id", at)?, "evaluation", at)?), true),
        "public" => (format!("calldataload({})", index(ops.public, u64_field(value, "id", at)?, "public", at)?), true),
        "number" => match value.get("value").and_then(Value::as_str) {
            Some(v) if big(v) < *r() && big(v).to_str_radix(10) == v => (v.to_string(), true),
            _ => return fail(format!("qVerifier: {at} is not a number below r")),
        },
        "challenge" => {
            let key = (u64_field(value, "stage", at)?, u64_field(value, "stageId", at)?);
            match ops.challenges.get(&key) {
                Some(slot) => (format!("mload(add(pMem, {slot}))"), true),
                None => return fail(format!("qVerifier: {at} reads the challenge {key:?}, which there is not")),
            }
        }
        "Zi" => {
            let slot = index(ops.zi, u64_field(value, "boundaryId", at)?, "boundary", at)?;
            (format!("mload(add(pMem, {slot}))"), true)
        }
        other => return fail(format!("qVerifier: {at} is an operand {other:?}, which the verifier does not have")),
    })
}

/// The `qVerifier` of the vkey as Yul: every entry an `mstore` of its temporary, the last one's to
/// `pQ`. A `copy` of a value fixed for the whole code writes nothing: the temporary stands for the
/// value until it is written again (the code is straight-line, so this is exact).
fn q_code(q_verifier: &Value, ops: &QOperands) -> Result<Vec<String>, SetupError> {
    let Some(code) = q_verifier.get("code").and_then(Value::as_array) else {
        return fail("the vkey's qVerifier has no code");
    };
    let mut aliases: HashMap<u64, String> = HashMap::new();
    let mut lines = Vec::with_capacity(code.len());
    for (i, entry) in code.iter().enumerate() {
        let at = format!("code[{i}]");
        let op = entry.get("op").and_then(Value::as_str).unwrap_or_default();
        let src = entry.get("src").and_then(Value::as_array).cloned().unwrap_or_default();
        let operands = src
            .iter()
            .enumerate()
            .map(|(j, s)| q_operand(s, &format!("{at}.src[{j}]"), ops, &aliases))
            .collect::<Result<Vec<_>, _>>()?;
        let (expr, fixed) = match (op, operands.as_slice()) {
            ("add", [(a, _), (b, _)]) => (format!("addmod({a}, {b}, q)"), false),
            ("sub", [(a, _), (b, _)]) => (format!("addmod({a}, sub(q, {b}), q)"), false),
            ("mul", [(a, _), (b, _)]) => (format!("mulmod({a}, {b}, q)"), false),
            ("copy", [(a, fixed)]) => (a.clone(), *fixed),
            _ => return fail(format!("qVerifier: {at} is op {op:?} of {} operands", operands.len())),
        };
        let dest = match entry.get("dest") {
            Some(d) if d.get("type").and_then(Value::as_str) == Some("tmp") => u64_field(d, "id", &at)?,
            _ => return fail(format!("qVerifier: {at} does not write a temporary")),
        };
        if i + 1 == code.len() {
            lines.push(format!("mstore(add(pMem, pQ), {expr})"));
        } else if fixed {
            aliases.insert(dest, expr);
        } else {
            aliases.remove(&dest);
            lines.push(format!("mstore(add(pMem, add({}, {})), {expr})", ops.tmp, WORD * dest));
        }
    }
    Ok(lines)
}

// ---------------------------------------------------------------------------------------------
// Building the context
// ---------------------------------------------------------------------------------------------

fn offsets_text(offsets: &[i64]) -> String {
    format!("{offsets:?}")
}

/// The address `offset` bytes into the memory region `base`, as the template adds it to `pMem`.
fn at(base: &str, offset: u64) -> String {
    if offset == 0 {
        base.to_string()
    } else {
        format!("add({base}, {offset})")
    }
}

impl Context {
    fn new(vkey: &Vkey) -> Result<Self, SetupError> {
        let layout = &vkey.layout.0;
        let n_fixed = vkey.layout.n_fixed();
        let q_stage = layout.last().map_or(0, |f| f.stage);
        let n_stages = q_stage.saturating_sub(1);
        let n_rows = 1u64 << vkey.power;
        let n_big = BigUint::from(n_rows);
        let omega_n = root_of_unity(&n_big)?;
        let cd = CalldataLayout::of(vkey);

        // --- The calldata (proof.rs, Proof::to_bytes; the module's "Calldata") ---------------------
        let mut calldata = Vec::new();
        let mut cd_slot = |name: String, word: u64, comment: String| -> String {
            calldata.push(Slot {
                name: name.clone(),
                offset: CalldataLayout::offset(word),
                comment: sanitize(&comment),
            });
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
            let pieces = layout.iter().filter(|f| f.stage == q_stage).flat_map(|f| f.pols.iter());
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
        cd_slot("pInv".into(), after_pieces, "inv (A.5)".into());
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
        let p_public = CalldataLayout::offset(cd.words());
        let public: Vec<String> = (0..vkey.n_public).map(|i| at("pPublic", WORD * i)).collect();

        // --- The memory -----------------------------------------------------------------------
        let mut mem = Memory::default();
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

        // The powers of xiSeed and y of each k (shplonk.js, computeRoots and computeZerofiers).
        let mut ks: Vec<u64> = layout.iter().map(|f| f.k).collect();
        ks.sort_unstable();
        ks.dedup();
        let mut k_powers = BTreeMap::new();
        for &k in &ks {
            let p_seed = mem.alloc(format!("pSeed{k}"), 1, format!("xiSeed^(powerW/{k})"));
            let p_yk = mem.alloc(format!("pYk{k}"), 1, format!("y^{k}"));
            k_powers.insert(k, KPowers { k, exponent: vkey.power_w / k, p_seed, p_yk });
        }

        // The root sets: one per (k, offsets).
        let mut root_sets: Vec<RootSet> = Vec::new();
        let mut set_of = Vec::with_capacity(layout.len());
        for f in layout {
            if let Some(set) = root_sets.iter().position(|s| s.k == f.k && s.offsets == offsets_text(&f.offsets)) {
                set_of.push(set);
                continue;
            }
            let index = root_sets.len();
            let k = f.k;
            let t = f.offsets.len() as u64;
            let kn = BigUint::from(k) * &n_big;
            let omega_kn = root_of_unity(&kn)?;
            let w_k = root_of_unity(&BigUint::from(k))?;
            let comment = format!("k = {k}, offsets {}", offsets_text(&f.offsets));
            let p_roots = mem.alloc(format!("pRoots{index}"), k * t, format!("the roots T of {comment}"));
            let p_z = mem.alloc(format!("pZ{index}"), t, format!("xi*w^s of each offset s of {comment}"));
            let p_zt = mem.alloc(format!("pZt{index}"), 1, format!("Z_T(y) of {comment}"));
            let p_den1 =
                mem.alloc(format!("pDen1{index}"), t, format!("the Lagrange factor of each offset of {comment}"));
            let power = |value: BigUint| (value != BigUint::from(1u32)).then(|| value.to_str_radix(10));
            let rows = f
                .offsets
                .iter()
                .enumerate()
                .map(|(m, &s)| RootRow {
                    s,
                    omega_kn_s: power(signed_pow(&omega_kn, s)),
                    omega_n_s: power(signed_pow(&omega_n, s)),
                    roots: at(&p_roots, WORD * k * m as u64),
                    z: at(&p_z, WORD * m as u64),
                    den1: at(&p_den1, WORD * m as u64),
                    other_z: (0..t).filter(|&l| l != m as u64).map(|l| at(&p_z, WORD * l)).collect(),
                })
                .collect();
            let powers = &k_powers[&k];
            root_sets.push(RootSet {
                index,
                k,
                offsets: offsets_text(&f.offsets),
                k_minus_1: k - 1,
                p_seed: powers.p_seed.clone(),
                p_yk: powers.p_yk.clone(),
                w_k: w_k.to_str_radix(10),
                p_zt,
                rows,
            });
            set_of.push(index);
        }

        // The array the proof's inv inverts, in its order (A.5): Z_{T_i}(y) for i ≥ 1, then the
        // Lagrange denominators of each f_i.
        let n_inv = (layout.len() as u64 - 1) + layout.iter().map(|f| f.k * f.offsets.len() as u64).sum::<u64>();
        mem.alloc("pInvs".into(), 0, format!("the {n_inv} values the proof's inv inverts (A.5), below"));
        let p_inv_zt: Vec<Option<String>> = (0..layout.len())
            .map(|i| (i > 0).then(|| mem.alloc(format!("pInvZt{i}"), 1, format!("Z_T(y) of f{i}, then its inverse"))))
            .collect();
        let p_inv_den: Vec<String> = layout
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let words = f.k * f.offsets.len() as u64;
                mem.alloc(
                    format!("pInvDen{i}"),
                    words,
                    format!("the Lagrange denominators of f{i}, then their inverses"),
                )
            })
            .collect();
        let p_r: Vec<String> = (0..layout.len()).map(|i| mem.alloc(format!("pR{i}"), 1, format!("r_{i}(y)"))).collect();
        mem.alloc("pF".into(), 2, "[F]_1".into());
        mem.alloc("pE".into(), 2, "[E]_1".into());
        mem.alloc("pJ".into(), 2, "[J]_1".into());
        let tmp_used = vkey.q_verifier.get("tmpUsed").and_then(Value::as_u64).unwrap_or(0);
        let tmp = mem.alloc("pTmp".into(), tmp_used, "the temporaries of the qVerifier".into());

        // --- A.4, challenges.js ---------------------------------------------------------------
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

        // --- Zi (qverifier.js, computeZi) -----------------------------------------------------
        let neg_row = |j: u64| fr_neg(&fr_pow(&omega_n, j)).to_str_radix(10);
        let mut zi = Vec::with_capacity(vkey.boundaries.len());
        for (b, boundary) in vkey.boundaries.iter().enumerate() {
            let slot = zi_slots[b].clone();
            let (kind, aux, neg_root, neg_rows, comment) = match *boundary {
                Boundary::EveryRow => ("everyRow", None, None, vec![], "1/Z_H(xi): the proof's invZh".to_string()),
                Boundary::FirstRow => {
                    ("firstRow", aux_slots.next(), Some(neg_row(0)), vec![], "Z_H(xi)/(xi - 1)".to_string())
                }
                Boundary::LastRow => (
                    "lastRow",
                    aux_slots.next(),
                    Some(neg_row(n_rows - 1)),
                    vec![],
                    format!("Z_H(xi)/(xi - w^{})", n_rows - 1),
                ),
                Boundary::EveryFrame { offset_min, offset_max } => {
                    let rows: Vec<u64> = (0..offset_min).chain(n_rows - offset_max..n_rows).collect();
                    let comment =
                        format!("everyFrame {{{offset_min}, {offset_max}}}: prod of (xi - w^j) over rows {rows:?}");
                    ("everyFrame", None, None, rows.into_iter().map(neg_row).collect(), comment)
                }
            };
            zi.push(Zi { index: b, kind, slot, aux, neg_root, neg_rows, comment: sanitize(&comment) });
        }

        // --- The qVerifier (qverifier.js, executeCode) ----------------------------------------
        let q_code = q_code(
            &vkey.q_verifier,
            &QOperands { eval: &eval, public: &public, zi: &zi_slots, challenges: &challenges, tmp: &tmp },
        )?;
        let q_split = (cd.n_q_pieces > 0).then(|| QSplit {
            max_q_degree: vkey.max_q_degree,
            pieces_desc: piece_slots.values().rev().cloned().collect(),
        });

        // --- The f_i (shplonk.js: openingEvaluations of verify.js, computeR, computeF) ---------
        let fs = layout
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let set = &root_sets[set_of[i]];
                let t = if f.stage == 0 { PolType::Const } else { PolType::Cm };
                let is_q = f.stage == q_stage;
                let rows = f
                    .offsets
                    .iter()
                    .enumerate()
                    .map(|(m, &s)| {
                        let sources = f
                            .pols
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
                                        Ok("mload(add(pMem, pQ))".to_string())
                                    }
                                } else {
                                    match vkey
                                        .ev_map
                                        .iter()
                                        .position(|e| e.pol_type == t && e.id == p.id && e.prime == s)
                                    {
                                        Some(e) => Ok(format!("calldataload({})", eval[e])),
                                        None => fail(format!(
                                            "the evMap has no {} {} at offset {s}, which f{i} opens",
                                            t.as_str(),
                                            p.id
                                        )),
                                    }
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let mut horner = sources.into_iter().rev();
                        let horner_first = horner.next().unwrap_or_default();
                        let row = &set.rows[m];
                        Ok(FRow {
                            roots: row.roots.clone(),
                            den1: row.den1.clone(),
                            inv_den: at(&p_inv_den[i], WORD * f.k * m as u64),
                            horner_first,
                            horner_rest: horner.collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, SetupError>>()?;
                Ok(F {
                    index: i,
                    stage: f.stage,
                    k: f.k,
                    offsets: offsets_text(&f.offsets),
                    fixed: f.stage == 0,
                    commitment: if f.stage == 0 { format!("f{i}") } else { commitment_of[&i].clone() },
                    p_inv_zt: p_inv_zt[i].clone(),
                    p_zt: set.p_zt.clone(),
                    w_k_inv: fr_pow(&root_of_unity(&BigUint::from(f.k))?, f.k - 1).to_str_radix(10),
                    p_r: p_r[i].clone(),
                    rows,
                })
            })
            .collect::<Result<Vec<_>, SetupError>>()?;

        let fixed = vkey
            .fixed_commitments
            .0
            .iter()
            .enumerate()
            .map(|(index, p)| FixedCommitment { index, x: p.x.to_decimal(), y: p.y.to_decimal() })
            .collect();
        let x2 = &vkey.x_2;
        Ok(Context {
            digest: vkey.digest.to_hex(),
            digest_fr: vkey.digest.to_fr().to_decimal(),
            power: vkey.power,
            n: n_rows,
            power_w: vkey.power_w,
            n_public: vkey.n_public,
            x2: [[x2.x[0].to_decimal(), x2.x[1].to_decimal()], [x2.y[0].to_decimal(), x2.y[1].to_decimal()]],
            fixed,
            words: cd.words(),
            proof_words: cd.proof_words(),
            calldata,
            p_points_end: CalldataLayout::offset(2 * (n_c + 2)),
            p_absorbed_end: CalldataLayout::offset(2 * (n_c + 1)),
            p_scalars: CalldataLayout::offset(cd.first_scalar()),
            p_scalars_end: CalldataLayout::offset(cd.words()),
            p_public,
            memory: mem.slots,
            last_mem: mem.next,
            transcript: t.lines,
            zi,
            q_code,
            q_split,
            ks: k_powers.into_values().collect(),
            root_sets,
            fs,
            n_inv,
        })
    }
}

/// Renders a one-shot template, as `templates.rs` of the STARK's does.
fn render(template: &str, context: &TeraContext) -> Result<String, SetupError> {
    let mut tera = Tera::default();
    // Solidity, not HTML: nothing is escaped.
    tera.autoescape_on(vec![]);
    let error = |e: tera::Error| {
        let mut message = e.to_string();
        let mut source = std::error::Error::source(&e);
        while let Some(s) = source {
            message.push_str(&format!(": {s}"));
            source = s.source();
        }
        SetupError::Solidity(format!("the template: {message}"))
    };
    tera.add_raw_template("verifier_pilfflonk.sol", template).map_err(error)?;
    // The template starts with a comment of its own.
    Ok(tera.render("verifier_pilfflonk.sol", context).map_err(error)?.trim_start().to_string())
}

/// The Solidity verifier of `vkey` (see the module). Refuses a vkey the verifier would not read,
/// one whose digest is not that of its contents (the JS verifier accepts no proof of it) and one
/// with a fixed commitment off the curve.
pub fn verifier_sol(vkey: &Vkey) -> Result<String, SetupError> {
    vkey.validate()?;
    if vkey_digest(vkey)? != vkey.digest {
        return fail(
            "the vkey's digest is not the digest of its contents (A.6): the JS verifier accepts no proof of it",
        );
    }
    if let Some(i) = vkey.fixed_commitments.0.iter().position(|p| !is_g1(p)) {
        return fail(format!("the vkey's fixed commitment f{i} is not a point of G1"));
    }
    let context = TeraContext::from_serialize(Context::new(vkey)?).map_err(|e| SetupError::Solidity(e.to_string()))?;
    render(VERIFIER_TEMPLATE, &context)
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

    #[test]
    fn g1_is_the_curve_and_the_point_at_infinity() {
        let point = |x: u64, y: u64| G1Affine {
            x: proofman_pilfflonk::FqBytes::from_u64(x),
            y: proofman_pilfflonk::FqBytes::from_u64(y),
        };
        assert!(is_g1(&point(1, 2)));
        assert!(is_g1(&G1Affine::INFINITY));
        assert!(!is_g1(&point(1, 3)));
        assert!(!is_g1(&point(0, 1)));
    }
}
