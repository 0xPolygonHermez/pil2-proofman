//! The differential fuzzer of the Solidity verifier
//! (pilfflonk/docs/verifier.md#differential-fuzzer): on the proofs of a key, mutated in many ways,
//! the JS verifier (`pilfflonk/js`, the reference) and the contract the generator writes must say
//! the same, and the contract must return `false` for every case it refuses: only calldata shorter
//! than its arguments may revert.
//!
//! For a key, [`run_key`] makes cases in rounds of [`ROUND`] and, for each round:
//! 1. makes the calldata of each case: the calldata of an honest proof (`Calldata::encode`), with a
//!    mutation of one of the [`Family`]s;
//! 2. asks the JS verifier about the proof and publics the calldata holds, written as `proof.json`
//!    and `publics.json` hold them (any value of a word, in decimal), with one Node process for the
//!    round (`js_batch.mjs`, which calls `verify()` as `bin/verify.js` does; the first case of
//!    each family is also verified with `js_verifier::verify`, the CLI's, which must agree);
//! 3. runs, on Foundry, the key's verifier and its probe on every case ([`FuzzProject`]): the probe
//!    is a copy of the verifier, instrumented here ([`Probe`]), that returns the check that refused
//!    the case (each `fail()` of the contract is a site) and the gas left after each step;
//! 4. compares: the reference outcome is the JS verifier's verdict and, for the calldata only, the
//!    auxiliary inverses (they must be `1/(ξ − ω^j)`, each below `r`) and its length (shorter
//!    reverts); the contract's must be the same, and so must the check that refused the case, as the
//!    JS verifier's messages name it and as the probe's site does. Each family names the checks its
//!    cases must reach ([`FuzzCase::targets`]); a case refused by the pairing must cost at least
//!    90 % of the gas of the honest proof.
//!
//! A family with a disagreement is fuzzed no more ([`Stopped`]), on any key, and the test reports
//! the case with its evidence: the words it changed, the JS verifier's messages and the contract's
//! outcome, gas and check.
//!
//! The mutations of a proof that need its transcript (the "fixed-up" families, [`fixup`]) call the
//! transcript of the C++ core, which has no OpenMP: they may run on any thread.
//!
//! Include it with `foundry.rs` and `mutations.rs`, at the root of the test crate:
//! `#[path = ".../pilfflonk/tests/data/fuzz.rs"] mod fuzz;`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use pilfflonk_setup::test_ptau::g1_times;
use proofman_pilfflonk::calldata::{Word, SELECTOR_BYTES};
use proofman_pilfflonk::names::{INV, INV_ZH, W, WP};
use proofman_pilfflonk::{
    auxiliary_inverses, js_verifier, verifier_challenges, Calldata, CalldataLayout, FqBytes, FrBytes, G1Affine,
    JsonFile, Proof, ProofNames, Publics, Vkey,
};
use rand::rngs::Xoshiro256PlusPlus;
use rand::{RngExt, SeedableRng};
use serde_json::{json, Value};

use crate::foundry::{FuzzProject, FuzzRun, Outcome, Tools};
use crate::mutations::{be_word, big, fixup, fr, plus_one, q, r, rebalance_pieces};

/// The cases of a round: a batch of the JS verifier and of Foundry.
pub const ROUND: usize = 250;

/// A case refused by the pairing costs at least this fraction of the honest proof's gas.
const PAIRING_GAS: (u64, u64) = (9, 10);

// ---------------------------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------------------------

/// The check that refused a case, in the order the contract makes them
/// (pilfflonk/docs/verifier.md#steps), or what the call did otherwise. The JS verifier makes the
/// same checks, and its messages name them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Check {
    /// Accepted: `verifyProof` returned `true`.
    Accepted,
    /// A coordinate of a point not below `q` (`checkPointBelongsToBN128Curve`; `elements.js`).
    CoordinateNotBelowQ,
    /// A point `(0, 0)`, the point at infinity.
    Infinity,
    /// A point not on `y² = x³ + 3`.
    OffCurve,
    /// A point the transcript absorbs with a coordinate below `2^192` (`checkAbsorbed`;
    /// `transcript.js`, pilfflonk/docs/protocol.md#transcript).
    ShortCoordinate,
    /// A scalar, an auxiliary inverse or a public not below `r` (`checkFields`; `frFromObject`).
    ScalarNotBelowR,
    /// `Z_H(ξ)·invZh ≠ 1` (`computeZh`).
    InvZh,
    /// `(ξ − ω^j)·aux ≠ 1` (`computeZi`): the calldata's only, which the JS verifier does not see.
    AuxiliaryInverse,
    /// `Σ_i ξ^(i·M·N)·Q_i(ξ) ≠ Q(ξ)` (`checkQPieces`; `joinQPieces`).
    QPieces,
    /// `xiSeed = 0` (`computeZ`; `checkOpening`).
    XiSeedZero,
    /// `Z_T(y) = 0` (`computeInversions`; `verifyOpening`).
    ZeroZerofier,
    /// `inv·Π ≠ 1` (`checkInv`; `isValidInverse`).
    Inv,
    /// The pairing (`checkPairing`; `isValidPairing`).
    Pairing,
    /// A precompile that did not return what it must (`checkPointResult`).
    PrecompileResult,
    /// The call reverted: the ABI decoder, on calldata shorter than the arguments.
    Revert,
}

impl Check {
    pub fn name(self) -> &'static str {
        match self {
            Check::Accepted => "accepted",
            Check::CoordinateNotBelowQ => "coordinate >= q",
            Check::Infinity => "point at infinity",
            Check::OffCurve => "off the curve",
            Check::ShortCoordinate => "coordinate < 2^192",
            Check::ScalarNotBelowR => "scalar >= r",
            Check::InvZh => "invZh",
            Check::AuxiliaryInverse => "auxiliary inverse",
            Check::QPieces => "checkQPieces",
            Check::XiSeedZero => "xiSeed = 0",
            Check::ZeroZerofier => "Z_T(y) = 0",
            Check::Inv => "inv",
            Check::Pairing => "pairing",
            Check::PrecompileResult => "precompile result",
            Check::Revert => "revert",
        }
    }

    /// What the call must do if this is its check.
    fn outcome(self) -> Outcome {
        match self {
            Check::Accepted => Outcome::Accept,
            Check::Revert => Outcome::Revert,
            _ => Outcome::Reject,
        }
    }
}

/// The check a message of the JS verifier names: the reason it logs for a rejection
/// (`verify.js`, `elements.js`, `transcript.js`, `shplonk.js`).
fn js_check(message: &str) -> Option<Check> {
    let checks = [
        ("is not below q", Check::CoordinateNotBelowQ),
        ("the point at infinity", Check::Infinity),
        ("is not on the curve", Check::OffCurve),
        ("coordinate below 2^192", Check::ShortCoordinate),
        ("is not below r", Check::ScalarNotBelowR),
        ("invZh is not", Check::InvZh),
        ("pieces of Q do not add up", Check::QPieces),
        ("xiSeed is zero", Check::XiSeedZero),
        ("y is a root", Check::ZeroZerofier),
        ("inv is not the inverse", Check::Inv),
        ("Invalid SHPLONK opening", Check::Pairing),
    ];
    checks.iter().find(|(text, _)| message.contains(text)).map(|&(_, check)| check)
}

/// The check of a `fail()` of the contract, by the Yul function it is in and its place there.
fn site_check(function: &str, ordinal: usize) -> Check {
    match (function, ordinal) {
        ("checkPointBelongsToBN128Curve", 0) => Check::CoordinateNotBelowQ,
        ("checkPointBelongsToBN128Curve", 1) => Check::Infinity,
        ("checkPointBelongsToBN128Curve", 2) => Check::OffCurve,
        ("checkAbsorbed", 0) => Check::ShortCoordinate,
        ("checkFields", 0) => Check::ScalarNotBelowR,
        ("computeZh", 0) => Check::InvZh,
        ("computeZi", _) => Check::AuxiliaryInverse,
        ("checkQPieces", 0) => Check::QPieces,
        ("computeZ", 0) => Check::XiSeedZero,
        ("computeInversions", _) => Check::ZeroZerofier,
        ("checkInv", 0) => Check::Inv,
        ("checkPointResult", 0) => Check::PrecompileResult,
        _ => panic!("the verifier has a fail() of {function} (#{ordinal}) that the probe does not know: see fuzz.rs"),
    }
}

// ---------------------------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------------------------

/// The steps of the verifier's body, in order (the template's, `verifier_pilfflonk.sol.tera`):
/// `checkQPieces` only if `Q` is split.
pub const STEPS: [&str; 11] = [
    "checkInput",
    "computeChallenges",
    "computeZh",
    "computeZi",
    "computeQ",
    "checkQPieces",
    "computeZ",
    "computeInversions",
    "computeR",
    "computeFEJ",
    "checkPairing",
];

/// The words the probe returns: the verdict (0 or 1), the site of the `fail()` that refused the
/// case (0 for none), the gas left when it returned, the gas left when its body began, and after
/// each of [`STEPS`] (0 for a step it did not reach or that the key has not, and for the first two
/// if it did not reach the transcript).
const PROBE_WORDS: usize = 4 + STEPS.len();

/// The probe of a verifier: a copy of the contract, `PilfflonkProbe`, whose every `fail()` says
/// where it is and whose body records the gas left after each step, made from the generated
/// contract by rewriting its text. The verifier's logic is the same (the test checks that the probe
/// says what the verifier says of every case); its memory is the verifier's, and [`PROBE_WORDS`]
/// more words above it, which it returns. Made in the test's directory, never by the generator.
pub struct Probe {
    pub source: String,
    /// The check of each site, `fail(1)` first.
    sites: Vec<Check>,
}

impl Probe {
    pub fn of(sol: &str) -> Self {
        let bytes = 32 * PROBE_WORDS;
        let lines: Vec<&str> = sol.lines().collect();
        let indent = |line: &str| line[..line.len() - line.trim_start().len()].to_string();
        let mut out: Vec<String> = Vec::with_capacity(lines.len() + 32);
        let mut sites = Vec::new();
        let mut ordinals: BTreeMap<String, usize> = BTreeMap::new();
        let mut function = String::new();
        let (mut renamed, mut fail_defined, mut in_body, mut returned) = (0, 0, false, 0);
        let mut steps_found = [0usize; STEPS.len()];
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i];
            let text = line.trim();
            if let Some(rest) = text.strip_prefix("function ") {
                function = rest.split('(').next().unwrap_or_default().to_string();
            }
            if text == "contract PilfflonkVerifier {" {
                out.push(line.replace("PilfflonkVerifier", "PilfflonkProbe"));
                renamed += 1;
            } else if text == "function fail() {" {
                let body: Vec<&str> = lines[i + 1..i + 4].iter().map(|l| l.trim()).collect();
                assert_eq!(body, ["mstore(0, 0)", "return(0, 0x20)", "}"], "fail() of the template changed");
                let at = indent(line);
                out.push(format!("{at}function fail(site) {{"));
                out.push(format!("{at}    let pProbe := sub(mload(0x40), {bytes})"));
                out.push(format!("{at}    mstore(add(pProbe, 32), site)"));
                out.push(format!("{at}    mstore(add(pProbe, 64), gas())"));
                out.push(format!("{at}    return(pProbe, {bytes})"));
                out.push(format!("{at}}}"));
                fail_defined += 1;
                i += 4;
                continue;
            } else if text.contains("fail()") {
                assert_eq!(text, "fail()", "a fail() that is not a statement of its own: {line}");
                let ordinal = ordinals.entry(function.clone()).or_insert(0);
                sites.push(site_check(&function, *ordinal));
                *ordinal += 1;
                out.push(line.replace("fail()", &format!("fail({})", sites.len())));
            } else if text == "mstore(0x40, lastMem)" {
                // The body's first statement: the memory is the constants' up to lastMem, and the
                // probe's words go after it.
                let at = indent(line);
                out.push(format!("{at}mstore(0x40, add(lastMem, {bytes}))"));
                out.push(format!("{at}let pProbe := lastMem"));
                out.push(format!("{at}let probeStart := gas()"));
                in_body = true;
                i += 1;
                continue;
            } else if in_body && text == "mstore(0, isValid)" {
                assert_eq!(lines[i + 1].trim(), "return(0, 0x20)", "the end of the template changed");
                let at = indent(line);
                out.push(format!("{at}mstore(pProbe, isValid)"));
                out.push(format!("{at}mstore(add(pProbe, 64), gas())"));
                out.push(format!("{at}return(pProbe, {bytes})"));
                returned += 1;
                i += 2;
                continue;
            } else {
                out.push(line.to_string());
                if in_body {
                    let call = text.strip_prefix("let isValid := ").unwrap_or(text);
                    let name = call.split('(').next().unwrap_or_default();
                    if let Some(step) = STEPS.iter().position(|s| *s == name) {
                        steps_found[step] += 1;
                        let at = indent(line);
                        // The verifier first writes to its memory in the transcript, which grows it to
                        // all it uses: the gas of the start and after the checks of the input stay on
                        // the stack until then, so that growing the memory is the transcript's, as it is
                        // in the verifier.
                        match name {
                            "checkInput" => out.push(format!("{at}let probeInput := gas()")),
                            "computeChallenges" => {
                                out.push(format!("{at}mstore(add(pProbe, {}), gas())", 32 * (4 + step)));
                                out.push(format!("{at}mstore(add(pProbe, 96), probeStart)"));
                                out.push(format!("{at}mstore(add(pProbe, 128), probeInput)"));
                            }
                            _ => out.push(format!("{at}mstore(add(pProbe, {}), gas())", 32 * (4 + step))),
                        }
                    }
                }
            }
            i += 1;
        }
        assert_eq!((renamed, fail_defined, returned), (1, 1, 1), "the contract is not the template's");
        for (step, found) in STEPS.iter().zip(steps_found) {
            let expected = if *step == "checkQPieces" { found.min(1) } else { 1 };
            assert_eq!(found, expected, "the body calls {step} {found} times");
        }
        assert!(!sites.is_empty(), "the contract has no fail()");
        let mut source = out.join("\n");
        source.push('\n');
        Probe { source, sites }
    }

    /// What the probe's words say: its verdict, and the check that refused the case.
    fn check(&self, words: &[u64]) -> Check {
        assert_eq!(words.len(), PROBE_WORDS, "the probe returns {PROBE_WORDS} words");
        match (words[0], words[1]) {
            (1, 0) => Check::Accepted,
            (0, 0) => Check::Pairing,
            (0, site) => self.sites[site as usize - 1],
            other => panic!("the probe returned the verdict and site {other:?}"),
        }
    }
}

/// Where the gas of a call goes, from the probe's words: the gas of each of [`STEPS`] (0 for one the
/// key has not) and of the body of `verifyProof` (its assembly block, which the steps are), as the
/// probe measures them; and the gas of the call to the verifier (`total`) and to the probe
/// (`probe_total`), whose difference is what the probe adds: its `gas()` and `mstore` after each
/// step (a few gas each, which the steps include), its memory and the words it returns. The rest of
/// the verifier's, `total − body`, is the call, the dispatch, the ABI decoding of the arguments and
/// the return.
#[derive(Clone, Debug, Default)]
pub struct Breakdown {
    pub steps: [u64; STEPS.len()],
    pub body: u64,
    pub total: u64,
    pub probe_total: u64,
}

impl Breakdown {
    fn of(words: &[u64], total: u64, probe_total: u64) -> Self {
        let mut steps = [0; STEPS.len()];
        let mut left = words[3];
        for (s, step) in steps.iter_mut().enumerate() {
            let after = words[4 + s];
            if after != 0 {
                *step = left - after;
                left = after;
            }
        }
        Breakdown { steps, body: words[3] - words[2], total, probe_total }
    }
}

// ---------------------------------------------------------------------------------------------
// The mutations
// ---------------------------------------------------------------------------------------------

/// A family of mutations: what it changes of an honest proof's calldata, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Family {
    /// The proof as the prover makes it: the control, which both must accept.
    Honest,
    /// One bit of any word of the arguments flipped.
    BitFlip,
    /// Any word replaced by a random value: below `2^256`, `q` or `r`.
    RandomWord,
    /// Any word replaced by a value at a boundary of the fields: 0, 1, `r − 1`, `r`, `r + 1`,
    /// `q − 1`, `q`, `2^256 − 1`.
    Boundary,
    /// A commitment, `W` or `W'` replaced by another point of G1: a random multiple of `[1]₁`, the
    /// point negated, or another point of the proof.
    ValidPoint,
    /// A point replaced by one off the curve, with coordinates below `q`.
    OffCurve,
    /// A point replaced by `(0, 0)`.
    Infinity,
    /// A point replaced by `±[1]₁ = (1, ±2)`, of a coordinate below `2^192`.
    ShortPoint,
    /// Two scalars of the transcript's step 4 swapped: evaluations or pieces of `Q`.
    SwapEvaluations,
    /// Two points swapped: commitments, `W` and `W'`.
    SwapPoints,
    /// A public changed: `+ 1`, random, `p + r` (the same in `Fr`, which snarkjs does not refuse:
    /// pilfflonk/docs/verifier.md#differences-from-snarkjs), `r`, or any 256 bits.
    Public,
    /// `W` replaced by another point of G1.
    W,
    /// `W'` replaced by another point of G1 (it is not absorbed: only the pairing sees it).
    Wp,
    /// `inv` changed: `+ 1`, random below `r`, `+ r`.
    Inv,
    /// `invZh` changed, as `inv`.
    InvZh,
    /// An auxiliary inverse changed, as `inv`.
    Aux,
    /// A piece of a split `Q` changed, as `inv`.
    QPiece,
    /// The calldata cut short (which reverts) or with bytes after its arguments (which are not
    /// read).
    Length,
    /// An evaluation changed, then fixed up ([`fixup`]): to `checkQPieces` or the pairing.
    FixedEvaluation,
    /// A commitment replaced by a random point of G1, fixed up.
    FixedCommitment,
    /// A public changed, fixed up.
    FixedPublic,
    /// Two evaluations or two commitments swapped, fixed up.
    FixedSwap,
    /// `W` replaced by a random point of G1, fixed up: only `y` changes, to the pairing.
    FixedW,
    /// The pieces of a split `Q` changed without changing their sum, fixed up: to the pairing.
    FixedRebalance,
    /// A piece of a split `Q` changed, fixed up: to `checkQPieces`.
    FixedQPiece,
    /// `W' = y⁻¹·(E + J − F)` of a proof that gets past `checkQPieces` (unchanged, or fixed up with
    /// `W` changed and either the pieces of a split `Q` rebalanced or, if it is whole, an evaluation,
    /// a commitment or a public changed), which cancels the left side of the pairing: it must fail
    /// there, with an honest `X_2` (pilfflonk/docs/verifier.md#refused-vkeys).
    FixedForgery,
}

impl Family {
    pub const ALL: [Family; 26] = [
        Family::Honest,
        Family::BitFlip,
        Family::RandomWord,
        Family::Boundary,
        Family::ValidPoint,
        Family::OffCurve,
        Family::Infinity,
        Family::ShortPoint,
        Family::SwapEvaluations,
        Family::SwapPoints,
        Family::Public,
        Family::W,
        Family::Wp,
        Family::Inv,
        Family::InvZh,
        Family::Aux,
        Family::QPiece,
        Family::Length,
        Family::FixedEvaluation,
        Family::FixedCommitment,
        Family::FixedPublic,
        Family::FixedSwap,
        Family::FixedW,
        Family::FixedRebalance,
        Family::FixedQPiece,
        Family::FixedForgery,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Family::Honest => "honest",
            Family::BitFlip => "bit flip",
            Family::RandomWord => "random word",
            Family::Boundary => "field boundary",
            Family::ValidPoint => "point of G1",
            Family::OffCurve => "point off the curve",
            Family::Infinity => "point at infinity",
            Family::ShortPoint => "short point (1, ±2)",
            Family::SwapEvaluations => "two evaluations swapped",
            Family::SwapPoints => "two points swapped",
            Family::Public => "public",
            Family::W => "W",
            Family::Wp => "W'",
            Family::Inv => "inv",
            Family::InvZh => "invZh",
            Family::Aux => "auxiliary inverse",
            Family::QPiece => "piece of Q",
            Family::Length => "calldata length",
            Family::FixedEvaluation => "fixed up: evaluation",
            Family::FixedCommitment => "fixed up: commitment",
            Family::FixedPublic => "fixed up: public",
            Family::FixedSwap => "fixed up: swap",
            Family::FixedW => "fixed up: W",
            Family::FixedRebalance => "fixed up: Q rebalanced",
            Family::FixedQPiece => "fixed up: piece of Q",
            Family::FixedForgery => "fixed up: W' forged",
        }
    }

    /// Whether a key of this shape has what the family changes.
    fn applies(self, shape: &Shape) -> bool {
        match self {
            Family::Public | Family::FixedPublic => shape.n_public > 0,
            Family::Aux => !shape.layout.aux_rows.is_empty(),
            Family::QPiece | Family::FixedRebalance | Family::FixedQPiece => shape.split(),
            Family::FixedSwap => shape.n_evaluations >= 2 || shape.layout.n_commitments >= 2,
            _ => true,
        }
    }
}

/// How the length of the calldata changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Length {
    Exact,
    /// This many bytes cut from its end: shorter than the arguments, which the ABI decoder reverts.
    Truncated(usize),
    /// These bytes after it, which the contract does not read.
    Extended(Vec<u8>),
}

/// A case: the words of the arguments of `verifyProof` and the length of its calldata.
#[derive(Clone, Debug)]
pub struct FuzzCase {
    pub family: Family,
    pub label: String,
    /// The honest proof it mutates.
    pub base: usize,
    /// The words of `proof`: the proof's, then the auxiliary inverses.
    pub proof: Vec<Word>,
    pub publics: Vec<Word>,
    pub length: Length,
    /// The checks one of which must refuse it (or accept it, or revert): what its family is meant
    /// to reach. Empty for the random families, which may reach any.
    pub targets: Vec<Check>,
}

impl FuzzCase {
    fn calldata(&self, selector: &[u8; SELECTOR_BYTES]) -> Vec<u8> {
        let mut bytes = selector.to_vec();
        for word in self.proof.iter().chain(&self.publics) {
            bytes.extend_from_slice(word);
        }
        match &self.length {
            Length::Exact => {}
            Length::Truncated(n) => bytes.truncate(bytes.len() - n),
            Length::Extended(extra) => bytes.extend_from_slice(extra),
        }
        bytes
    }
}

/// The shape of a key's calldata, and what the mutations need of the vkey.
pub struct Shape {
    layout: CalldataLayout,
    n_public: usize,
    /// The evaluations of the evMap; the pieces of `Q` follow them in the proof.
    n_evaluations: usize,
}

impl Shape {
    fn of(vkey: &Vkey) -> Self {
        Shape { layout: CalldataLayout::of(vkey), n_public: vkey.n_public as usize, n_evaluations: vkey.ev_map.len() }
    }

    fn split(&self) -> bool {
        self.layout.n_q_pieces >= 2
    }

    /// The points of the proof: the commitments, `W` and `W'`.
    fn n_points(&self) -> usize {
        self.layout.n_commitments as usize + 2
    }

    fn w(&self) -> usize {
        self.layout.n_commitments as usize
    }

    fn wp(&self) -> usize {
        self.w() + 1
    }

    fn first_scalar(&self) -> usize {
        self.layout.first_scalar() as usize
    }

    /// The scalars of the transcript's step 4: the evaluations and the pieces of `Q`.
    fn n_step4(&self) -> usize {
        self.n_evaluations + self.layout.n_q_pieces as usize
    }

    fn inv(&self) -> usize {
        self.first_scalar() + self.n_step4()
    }

    fn inv_zh(&self) -> usize {
        self.inv() + 1
    }

    fn proof_words(&self) -> usize {
        self.layout.proof_words() as usize
    }

    /// The name of word `w` of the arguments, `proof`'s and then `pubSignals`'.
    fn word_name(&self, names: &ProofNames, w: usize) -> String {
        let points: Vec<&str> = names.commitments().iter().map(String::as_str).chain([W, WP]).collect();
        let scalars: Vec<&str> = names.evaluations().iter().map(String::as_str).chain([INV, INV_ZH]).collect();
        if w < 2 * self.n_points() {
            format!("{}.{}", points[w / 2], if w.is_multiple_of(2) { "x" } else { "y" })
        } else if w < self.proof_words() {
            scalars[w - self.first_scalar()].to_string()
        } else if w < self.layout.words() as usize {
            format!("aux[{}]", w - self.proof_words())
        } else {
            format!("publics[{}]", w - self.layout.words() as usize)
        }
    }
}

fn int(word: &Word) -> BigUint {
    BigUint::from_bytes_be(word)
}

fn point_words(point: &G1Affine) -> [Word; 2] {
    [point.x.to_be_bytes(), point.y.to_be_bytes()]
}

fn on_curve(x: &BigUint, y: &BigUint) -> bool {
    let q = q();
    (y * y) % &q == (x * x * x + 3u32) % &q
}

/// An honest proof of the key, its publics and its calldata.
pub struct Honest {
    pub label: String,
    pub proof: Proof,
    pub publics: Vec<FrBytes>,
    calldata: Calldata,
}

impl Honest {
    pub fn new(vkey: &Vkey, label: String, proof: Proof, publics: Vec<FrBytes>) -> Self {
        let calldata = Calldata::encode(vkey, &proof, &publics).unwrap();
        Honest { label, proof, publics, calldata }
    }

    fn proof_words(&self) -> Vec<Word> {
        self.calldata.proof.clone()
    }

    fn public_words(&self) -> Vec<Word> {
        self.publics.iter().map(FrBytes::to_be_bytes).collect()
    }
}

/// What a mutation makes: a case, or a proof that waits for the JS verifier to forge its `W'`
/// ([`Family::FixedForgery`]) to be one.
enum Made {
    Case(FuzzCase),
    Forgery(Box<Forgery>),
}

struct Forgery {
    label: String,
    base: usize,
    proof: Proof,
    publics: Vec<FrBytes>,
}

/// The mutations of the honest proofs of a key.
struct Generator<'a> {
    vkey: &'a Vkey,
    shape: &'a Shape,
    honest: &'a [Honest],
    rng: Xoshiro256PlusPlus,
}

/// The targets of a case refused by the deep checks of a fixed-up proof: `checkQPieces` if `Q` is
/// split (when the change reaches `Q(ξ)` or the pieces), and the pairing.
fn deep(shape: &Shape) -> Vec<Check> {
    if shape.split() {
        vec![Check::QPieces, Check::Pairing]
    } else {
        vec![Check::Pairing]
    }
}

impl Generator<'_> {
    fn below(&mut self, modulus: &BigUint) -> BigUint {
        let mut bytes = [0u8; 40];
        self.rng.fill(&mut bytes[..]);
        BigUint::from_bytes_be(&bytes) % modulus
    }

    fn word(&mut self) -> BigUint {
        let mut bytes = [0u8; 32];
        self.rng.fill(&mut bytes[..]);
        BigUint::from_bytes_be(&bytes)
    }

    fn index(&mut self, n: usize) -> usize {
        self.rng.random_range(0..n)
    }

    /// A random point of G1 other than the point at infinity: `s·[1]₁`.
    fn point(&mut self) -> G1Affine {
        loop {
            let s = self.below(&r());
            if s != BigUint::ZERO {
                return g1_times(&s);
            }
        }
    }

    /// `v + 1 mod r`, a random value below `r`, or `v + r` (not below `r`, and the same in `Fr`),
    /// for the scalar `v`: and the check that refuses the first two (`canonical`) or the third.
    fn scalar_change(&mut self, v: &BigUint, canonical: Check) -> (BigUint, Check, &'static str) {
        match self.index(3) {
            0 => ((v + 1u32) % r(), canonical, "+ 1"),
            1 => loop {
                let x = self.below(&r());
                if &x != v {
                    break (x, canonical, "random below r");
                }
            },
            _ => (v + r(), Check::ScalarNotBelowR, "+ r"),
        }
    }

    fn honest_base(&mut self) -> (usize, Vec<Word>, Vec<Word>) {
        let base = self.index(self.honest.len());
        (base, self.honest[base].proof_words(), self.honest[base].public_words())
    }

    /// A case of the calldata of an honest proof with its words changed.
    fn raw(
        &self,
        family: Family,
        label: String,
        base: usize,
        proof: Vec<Word>,
        publics: Vec<Word>,
        targets: Vec<Check>,
    ) -> Made {
        Made::Case(FuzzCase { family, label, base, proof, publics, length: Length::Exact, targets })
    }

    /// A case of a proof of the key, fixed up ([`fixup`]) and encoded as the CLI encodes it.
    fn fixed(
        &self,
        family: Family,
        label: String,
        base: usize,
        proof: Proof,
        publics: &[FrBytes],
        targets: Vec<Check>,
    ) -> FuzzCase {
        let proof = fixup(self.vkey, proof, publics);
        let calldata = Calldata::encode(self.vkey, &proof, publics).unwrap();
        let publics = publics.iter().map(FrBytes::to_be_bytes).collect();
        FuzzCase { family, label, base, proof: calldata.proof, publics, length: Length::Exact, targets }
    }

    /// A case of `family` (of the honest proof `honest_index` if it is [`Family::Honest`]), or the
    /// forgery that will be one.
    fn case(&mut self, family: Family, honest_index: usize) -> Made {
        let shape = self.shape;
        let n_points = shape.n_points();
        match family {
            Family::Honest => {
                let h = &self.honest[honest_index];
                let case = FuzzCase {
                    family,
                    label: h.label.clone(),
                    base: honest_index,
                    proof: h.proof_words(),
                    publics: h.public_words(),
                    length: Length::Exact,
                    targets: vec![Check::Accepted],
                };
                Made::Case(case)
            }
            Family::BitFlip => {
                let (base, mut proof, mut publics) = self.honest_base();
                let w = self.index(proof.len() + publics.len());
                let bit = self.index(256);
                let word = if w < proof.len() { &mut proof[w] } else { &mut publics[w - proof.len()] };
                word[31 - bit / 8] ^= 1 << (bit % 8);
                self.raw(family, format!("bit {bit} of word {w}"), base, proof, publics, vec![])
            }
            Family::RandomWord | Family::Boundary => {
                let (base, mut proof, mut publics) = self.honest_base();
                let w = self.index(proof.len() + publics.len());
                let word = if w < proof.len() { proof[w] } else { publics[w - proof.len()] };
                let (value, what) = if family == Family::RandomWord {
                    match self.index(3) {
                        0 => (self.word(), "random"),
                        1 => (self.below(&q()), "random below q"),
                        _ => (self.below(&r()), "random below r"),
                    }
                } else {
                    let max = (BigUint::from(1u32) << 256u32) - 1u32;
                    let values = [
                        (BigUint::ZERO, "0"),
                        (BigUint::from(1u32), "1"),
                        (r() - 1u32, "r - 1"),
                        (r(), "r"),
                        (r() + 1u32, "r + 1"),
                        (q() - 1u32, "q - 1"),
                        (q(), "q"),
                        (max, "2^256 - 1"),
                    ];
                    let start = self.index(values.len());
                    // A value the word does not hold already.
                    let pick =
                        (0..values.len()).map(|k| &values[(start + k) % values.len()]).find(|(v, _)| *v != int(&word));
                    let (v, what) = pick.unwrap();
                    (v.clone(), *what)
                };
                let target = if w < proof.len() { &mut proof[w] } else { &mut publics[w - proof.len()] };
                *target = be_word(&value);
                self.raw(family, format!("word {w} := {what}"), base, proof, publics, vec![])
            }
            Family::ValidPoint | Family::W | Family::Wp => {
                let (base, mut proof, publics) = self.honest_base();
                let slot = match family {
                    Family::W => shape.w(),
                    Family::Wp => shape.wp(),
                    _ => self.index(n_points),
                };
                let old = [proof[2 * slot], proof[2 * slot + 1]];
                let (new, what) = loop {
                    let candidate = match self.index(if family == Family::Wp { 4 } else { 3 }) {
                        0 => (point_words(&self.point()), "a random point"),
                        1 => {
                            let y = int(&old[1]);
                            ([old[0], be_word(&((q() - y) % q()))], "the point negated")
                        }
                        2 => {
                            let other = self.index(n_points);
                            ([proof[2 * other], proof[2 * other + 1]], "another point of the proof")
                        }
                        _ => (point_words(&G1Affine { x: FqBytes::from_u64(1), y: FqBytes::from_u64(2) }), "[1]_1"),
                    };
                    if candidate.0 != old {
                        break candidate;
                    }
                };
                proof[2 * slot] = new[0];
                proof[2 * slot + 1] = new[1];
                // A commitment changes xi; W, y; W', the pairing only.
                let target = if slot < shape.w() {
                    Check::InvZh
                } else if slot == shape.w() {
                    Check::Inv
                } else {
                    Check::Pairing
                };
                self.raw(family, format!("point {slot} := {what}"), base, proof, publics, vec![target])
            }
            Family::OffCurve | Family::Infinity | Family::ShortPoint => {
                let (base, mut proof, publics) = self.honest_base();
                let slot = self.index(n_points);
                let (x, y) = (int(&proof[2 * slot]), int(&proof[2 * slot + 1]));
                let ((nx, ny), what, target) = match family {
                    Family::Infinity => ((BigUint::ZERO, BigUint::ZERO), "(0, 0)", Check::Infinity),
                    Family::ShortPoint => {
                        let y = if self.index(2) == 0 { BigUint::from(2u32) } else { q() - 2u32 };
                        // W' is not absorbed: a short point of the curve is a point like another.
                        let target = if slot == shape.wp() { Check::Pairing } else { Check::ShortCoordinate };
                        ((BigUint::from(1u32), y), "(1, ±2)", target)
                    }
                    _ => loop {
                        let (candidate, what) = match self.index(4) {
                            0 => ((x.clone(), (&y + 1u32) % q()), "(x, y + 1)"),
                            1 => (((&x + 1u32) % q(), y.clone()), "(x + 1, y)"),
                            2 => ((x.clone(), self.below(&q())), "(x, random)"),
                            _ => ((self.below(&q()), self.below(&q())), "(random, random)"),
                        };
                        if !on_curve(&candidate.0, &candidate.1) && candidate != (BigUint::ZERO, BigUint::ZERO) {
                            break (candidate, what, Check::OffCurve);
                        }
                    },
                };
                proof[2 * slot] = be_word(&nx);
                proof[2 * slot + 1] = be_word(&ny);
                self.raw(family, format!("point {slot} := {what}"), base, proof, publics, vec![target])
            }
            Family::SwapEvaluations => {
                let (base, mut proof, publics) = self.honest_base();
                let first = shape.first_scalar();
                let n = shape.n_step4();
                let (a, b) = self.distinct_pair(n, |i, j| proof[first + i] != proof[first + j]);
                proof.swap(first + a, first + b);
                let targets = if shape.split() { vec![Check::QPieces, Check::Inv] } else { vec![Check::Inv] };
                self.raw(family, format!("scalars {a} and {b} of step 4 swapped"), base, proof, publics, targets)
            }
            Family::SwapPoints => {
                let (base, mut proof, publics) = self.honest_base();
                let (a, b) = self.distinct_pair(n_points, |i, j| proof[2 * i..2 * i + 2] != proof[2 * j..2 * j + 2]);
                proof.swap(2 * a, 2 * b);
                proof.swap(2 * a + 1, 2 * b + 1);
                // A commitment changes xi; W without a commitment, y.
                let target = if a.min(b) < shape.w() { Check::InvZh } else { Check::Inv };
                self.raw(family, format!("points {a} and {b} swapped"), base, proof, publics, vec![target])
            }
            Family::Public => {
                let (base, proof, mut publics) = self.honest_base();
                let i = self.index(publics.len());
                let v = int(&publics[i]);
                let (value, target, what) = match self.index(5) {
                    0 => ((&v + 1u32) % r(), Check::InvZh, "+ 1"),
                    1 => (self.below(&r()), Check::InvZh, "random below r"),
                    2 => (&v + r(), Check::ScalarNotBelowR, "+ r"),
                    3 => (r(), Check::ScalarNotBelowR, "r"),
                    _ => {
                        let w = self.word();
                        let target = if w < r() { Check::InvZh } else { Check::ScalarNotBelowR };
                        (w, target, "random")
                    }
                };
                if value == v {
                    return self.case(family, honest_index);
                }
                publics[i] = be_word(&value);
                self.raw(family, format!("publics[{i}] {what}"), base, proof, publics, vec![target])
            }
            Family::Inv | Family::InvZh | Family::Aux | Family::QPiece => {
                let (base, mut proof, publics) = self.honest_base();
                let (w, canonical) = match family {
                    Family::Inv => (shape.inv(), Check::Inv),
                    Family::InvZh => (shape.inv_zh(), Check::InvZh),
                    Family::Aux => {
                        (shape.proof_words() + self.index(shape.layout.aux_rows.len()), Check::AuxiliaryInverse)
                    }
                    _ => {
                        let piece = self.index(shape.layout.n_q_pieces as usize);
                        (shape.first_scalar() + shape.n_evaluations + piece, Check::QPieces)
                    }
                };
                let (value, target, what) = self.scalar_change(&int(&proof[w]), canonical);
                proof[w] = be_word(&value);
                self.raw(family, format!("word {w} {what}"), base, proof, publics, vec![target])
            }
            Family::Length => {
                let (base, proof, publics) = self.honest_base();
                let bytes = 32 * (proof.len() + publics.len());
                let mut case = FuzzCase {
                    family,
                    label: String::new(),
                    base,
                    proof,
                    publics,
                    length: Length::Exact,
                    targets: vec![],
                };
                if self.index(2) == 0 {
                    let cut = match self.index(4) {
                        0 => 1,
                        1 => 32,
                        2 => 33,
                        _ => 1 + self.index(bytes),
                    };
                    case.label = format!("{cut} bytes short");
                    case.length = Length::Truncated(cut);
                    case.targets = vec![Check::Revert];
                } else {
                    let mut extra = vec![0u8; 1 + self.index(64)];
                    self.rng.fill(&mut extra[..]);
                    case.label = format!("{} bytes more", extra.len());
                    case.length = Length::Extended(extra);
                    case.targets = vec![Check::Accepted];
                }
                Made::Case(case)
            }
            Family::FixedEvaluation
            | Family::FixedCommitment
            | Family::FixedPublic
            | Family::FixedSwap
            | Family::FixedW
            | Family::FixedRebalance
            | Family::FixedQPiece
            | Family::FixedForgery => self.fixed_case(family),
        }
    }

    /// Two indices `i ≠ j` below `n` that `differ` says differ, if there are any.
    fn distinct_pair(&mut self, n: usize, differ: impl Fn(usize, usize) -> bool) -> (usize, usize) {
        for _ in 0..64 {
            let (i, j) = (self.index(n), self.index(n));
            if i != j && differ(i, j) {
                return (i.min(j), i.max(j));
            }
        }
        (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .find(|&(i, j)| differ(i, j))
            .expect("two values that differ")
    }

    fn fixed_case(&mut self, family: Family) -> Made {
        let shape = self.shape;
        let base = self.index(self.honest.len());
        let mut proof = self.honest[base].proof.clone();
        let mut publics = self.honest[base].publics.clone();
        let change = |g: &mut Self, proof: &mut Proof, publics: &mut Vec<FrBytes>, kind: usize| -> String {
            match kind {
                0 => {
                    let i = g.index(shape.n_evaluations);
                    let v = big(&proof.evaluations[i]);
                    let new = if g.index(2) == 0 { plus_one(&proof.evaluations[i]) } else { fr(&g.below(&r())) };
                    if big(&new) == v {
                        proof.evaluations[i] = plus_one(&new);
                    } else {
                        proof.evaluations[i] = new;
                    }
                    format!("evaluation {i}")
                }
                1 => {
                    let i = g.index(proof.commitments.len());
                    proof.commitments[i] = g.point();
                    format!("commitment {i} := a random point")
                }
                _ => {
                    let i = g.index(publics.len());
                    publics[i] = if g.index(2) == 0 { plus_one(&publics[i]) } else { fr(&g.below(&r())) };
                    format!("publics[{i}]")
                }
            }
        };
        let (label, targets) = match family {
            Family::FixedEvaluation => (change(self, &mut proof, &mut publics, 0), deep(shape)),
            Family::FixedCommitment => (change(self, &mut proof, &mut publics, 1), deep(shape)),
            Family::FixedPublic => (change(self, &mut proof, &mut publics, 2), deep(shape)),
            Family::FixedSwap => {
                let evaluations = shape.n_evaluations >= 2
                    && (proof.commitments.len() < 2 || self.index(2) == 0)
                    && (0..shape.n_evaluations).any(|i| proof.evaluations[i] != proof.evaluations[0]);
                let label = if evaluations {
                    let values = proof.evaluations.clone();
                    let (a, b) = self.distinct_pair(shape.n_evaluations, |i, j| values[i] != values[j]);
                    proof.evaluations.swap(a, b);
                    format!("evaluations {a} and {b} swapped")
                } else {
                    let points = proof.commitments.clone();
                    let (a, b) = self.distinct_pair(points.len(), |i, j| points[i] != points[j]);
                    proof.commitments.swap(a, b);
                    format!("commitments {a} and {b} swapped")
                };
                (label, deep(shape))
            }
            Family::FixedW => {
                proof.w = self.point();
                ("W := a random point".to_string(), vec![Check::Pairing])
            }
            Family::FixedRebalance => {
                let xi_seed = big(&verifier_challenges(self.vkey, &proof, &publics).unwrap().xi_seed);
                let d = self.rng.random_range(1..u64::MAX);
                rebalance_pieces(self.vkey, &mut proof, &xi_seed, d);
                (format!("Q0 += xi^(M·N)·{d}, Q1 -= {d}"), vec![Check::Pairing])
            }
            Family::FixedQPiece => {
                let piece = self.index(shape.layout.n_q_pieces as usize);
                let at = shape.n_evaluations + piece;
                proof.evaluations[at] =
                    if self.index(2) == 0 { plus_one(&proof.evaluations[at]) } else { fr(&self.below(&r())) };
                (format!("piece {piece} of Q"), vec![Check::QPieces])
            }
            _ => {
                // A proof that gets past checkQPieces, so that the pairing is what sees W': unchanged,
                // with W changed (only y changes), and with Q's pieces rebalanced if it is split, or an
                // evaluation, a commitment or a public changed if it is whole (they change Q(ξ)).
                let label = match self.index(3) {
                    0 => "unchanged".to_string(),
                    1 => {
                        proof.w = self.point();
                        "W := a random point".to_string()
                    }
                    _ if shape.split() => {
                        let xi_seed = big(&verifier_challenges(self.vkey, &proof, &publics).unwrap().xi_seed);
                        let d = self.rng.random_range(1..u64::MAX);
                        rebalance_pieces(self.vkey, &mut proof, &xi_seed, d);
                        format!("Q0 += xi^(M·N)·{d}, Q1 -= {d}")
                    }
                    _ => {
                        let kind = self.index(if publics.is_empty() { 2 } else { 3 });
                        change(self, &mut proof, &mut publics, kind)
                    }
                };
                let proof = fixup(self.vkey, proof, &publics);
                return Made::Forgery(Box::new(Forgery { label: format!("W' forged, {label}"), base, proof, publics }));
            }
        };
        Made::Case(self.fixed(family, label, base, proof, &publics, targets))
    }
}

// ---------------------------------------------------------------------------------------------
// The JS verifier
// ---------------------------------------------------------------------------------------------

/// A file of `pilfflonk/tests/data` of the workspace this test is built in.
fn data_file(name: &str) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let found = manifest.ancestors().map(|dir| dir.join("pilfflonk/tests/data").join(name)).find(|p| p.is_file());
    found.unwrap_or_else(|| panic!("no pilfflonk/tests/data/{name} above {}", manifest.display()))
}

/// The JSON view of the proof whose words are `words` (`proof.json`,
/// pilfflonk/docs/formats.md#proof), with every value in decimal, below `r` or `q` or not: the JS
/// verifier checks it.
fn proof_json(shape: &Shape, names: &ProofNames, words: &[Word]) -> Value {
    let decimal = |w: &Word| int(w).to_str_radix(10);
    let mut polynomials = serde_json::Map::new();
    for (p, name) in names.commitments().iter().map(String::as_str).chain([W, WP]).enumerate() {
        polynomials.insert(name.to_string(), json!([decimal(&words[2 * p]), decimal(&words[2 * p + 1]), "1"]));
    }
    let mut evaluations = serde_json::Map::new();
    for (i, name) in names.evaluations().iter().map(String::as_str).chain([INV, INV_ZH]).enumerate() {
        evaluations.insert(name.to_string(), json!(decimal(&words[shape.first_scalar() + i])));
    }
    json!({"protocol": "pilfflonk", "curve": "bn128", "polynomials": polynomials, "evaluations": evaluations})
}

fn publics_json(words: &[Word]) -> Value {
    json!(words.iter().map(|w| int(w).to_str_radix(10)).collect::<Vec<_>>())
}

/// What the JS verifier says of a case.
#[derive(Clone, Debug)]
struct JsVerdict {
    verdict: bool,
    messages: Vec<String>,
}

impl JsVerdict {
    /// The check the JS verifier names, `None` for a message no check is.
    fn check(&self) -> Option<Check> {
        if self.verdict {
            Some(Check::Accepted)
        } else {
            self.messages.iter().find_map(|m| js_check(m))
        }
    }
}

/// Runs `js_batch.mjs` on `requests`, in `dir`.
fn run_js(vkey_path: &Path, dir: &Path, requests: &[Value]) -> Vec<Value> {
    fs::create_dir_all(dir).unwrap();
    let (input, output) = (dir.join("requests.jsonl"), dir.join("responses.jsonl"));
    let lines: String = requests.iter().map(|r| format!("{r}\n")).collect();
    fs::write(&input, lines).unwrap();
    let out = Command::new("node")
        .arg(data_file("js_batch.mjs"))
        .arg(vkey_path)
        .arg(&input)
        .arg(&output)
        .output()
        .expect("node runs");
    assert!(
        out.status.success(),
        "js_batch.mjs: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let responses: Vec<Value> =
        fs::read_to_string(&output).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(responses.len(), requests.len());
    responses
}

// ---------------------------------------------------------------------------------------------
// A key
// ---------------------------------------------------------------------------------------------

/// The key the fuzzer runs on: its vkey, the names of its proofs, its verifier, and a directory of
/// the test's for its files.
pub struct FuzzKey<'a> {
    pub name: &'a str,
    pub vkey: &'a Vkey,
    pub vkey_path: &'a Path,
    pub names: &'a ProofNames,
    pub sol: &'a Path,
    pub dir: &'a Path,
}

/// The families with a disagreement, which no key fuzzes any more.
pub type Stopped = Mutex<BTreeSet<Family>>;

/// What a family did on a key, or on all of them.
#[derive(Clone, Debug, Default)]
pub struct FamilyStats {
    pub cases: usize,
    /// The cases whose outcome is the reference's.
    pub agree: usize,
    /// The check that refused each case, the contract's (which is the JS verifier's when they agree).
    pub checks: BTreeMap<Check, usize>,
    pub gas: Option<(u64, u64)>,
}

impl FamilyStats {
    fn add(&mut self, check: Check, agree: bool, gas: u64) {
        self.cases += 1;
        self.agree += usize::from(agree);
        *self.checks.entry(check).or_insert(0) += 1;
        self.gas = Some(self.gas.map_or((gas, gas), |(lo, hi)| (lo.min(gas), hi.max(gas))));
    }

    fn merge(&mut self, other: &FamilyStats) {
        self.cases += other.cases;
        self.agree += other.agree;
        for (check, n) in &other.checks {
            *self.checks.entry(*check).or_insert(0) += n;
        }
        if let Some((lo, hi)) = other.gas {
            self.gas = Some(self.gas.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
        }
    }
}

/// What the fuzzer did on a key.
pub struct KeyReport {
    pub name: String,
    pub n_f: usize,
    pub words: u64,
    pub n_public: usize,
    pub cases: usize,
    /// The gas of the honest proof, and where it goes.
    pub honest: Breakdown,
    pub families: BTreeMap<Family, FamilyStats>,
    /// Every disagreement and every case that missed what it was meant to reach, with its evidence.
    pub problems: Vec<String>,
    pub times: [Duration; 3],
}

/// The evidence of a case: what it changed of its honest calldata, and what each verifier said.
fn evidence(
    key: &FuzzKey,
    shape: &Shape,
    honest: &[Honest],
    case: &FuzzCase,
    js: &JsVerdict,
    run: &FuzzRun,
    check: Check,
) -> String {
    let base: Vec<Word> = honest[case.base].proof_words().into_iter().chain(honest[case.base].public_words()).collect();
    let words: Vec<&Word> = case.proof.iter().chain(&case.publics).collect();
    let mut text = format!(
        "{}, {} ({}), of {}: length {:?}\n  JS: {} {:?}\n  Solidity: {} with {} gas, check {}\n  words changed:",
        key.name,
        case.family.name(),
        case.label,
        honest[case.base].label,
        case.length,
        if js.verdict { "accept" } else { "reject" },
        js.messages,
        run.outcome.as_str(),
        run.gas,
        check.name(),
    );
    for (w, (old, new)) in base.iter().zip(&words).enumerate() {
        if old != *new {
            let _ =
                write!(text, "\n    {w} {}: 0x{} -> 0x{}", shape.word_name(key.names, w), hex(&old[..]), hex(&new[..]));
        }
    }
    text
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The reference outcome of a case: the JS verifier's verdict on its proof and publics and, if it
/// accepts them, the calldata's own checks: each auxiliary inverse below `r` and `1/(ξ − ω^j)` of
/// the proof's `ξ`; and a calldata shorter than the arguments reverts.
fn reference(key: &FuzzKey, shape: &Shape, case: &FuzzCase, js: &JsVerdict) -> Option<Check> {
    if let Length::Truncated(_) = case.length {
        return Some(Check::Revert);
    }
    if !js.verdict {
        return js.check();
    }
    let aux = &case.proof[shape.proof_words()..];
    if aux.iter().any(|a| int(a) >= r()) {
        return Some(Check::ScalarNotBelowR);
    }
    let bytes: Vec<u8> = case.proof[..shape.proof_words()].iter().flatten().copied().collect();
    let proof = Proof::from_bytes(&bytes, &key.names.shape()).expect("a proof the JS verifier accepts is canonical");
    let publics: Vec<FrBytes> = case.publics.iter().map(|w| FrBytes::from_be_bytes(*w).unwrap()).collect();
    let xi_seed = verifier_challenges(key.vkey, &proof, &publics).unwrap().xi_seed;
    let expected: Vec<Word> =
        auxiliary_inverses(key.vkey, &xi_seed).unwrap().iter().map(FrBytes::to_be_bytes).collect();
    Some(if aux == expected.as_slice() { Check::Accepted } else { Check::AuxiliaryInverse })
}

/// The cases of the honest proofs of a key, and `n` more of the families that apply to it, by
/// rounds of [`ROUND`]; see the module. `honest[0]` is the proof whose gas the report gives.
pub fn run_key(tools: &Tools, key: &FuzzKey, honest: &[Honest], n: usize, seed: u64, stopped: &Stopped) -> KeyReport {
    let shape = Shape::of(key.vkey);
    let selector: [u8; SELECTOR_BYTES] =
        honest[0].calldata.to_abi_bytes().unwrap()[..SELECTOR_BYTES].try_into().unwrap();
    let probe = Probe::of(&fs::read_to_string(key.sol).unwrap());
    let project = FuzzProject::new(key.dir, key.sol, &probe.source);
    let mut generator =
        Generator { vkey: key.vkey, shape: &shape, honest, rng: Xoshiro256PlusPlus::seed_from_u64(seed) };
    let families: Vec<Family> =
        Family::ALL.iter().copied().filter(|f| *f != Family::Honest && f.applies(&shape)).collect();
    let offset = generator.index(families.len());

    // The JSON view the fuzzer writes of an honest proof is the prover's; and the JS verifier, the
    // CLI's (which installs its dependencies if it has to, before js_batch.mjs needs them), accepts
    // it.
    for h in honest {
        let view = proof_json(&shape, key.names, &h.proof_words());
        assert_eq!(view, serde_json::to_value(h.proof.to_json(key.names).unwrap()).unwrap(), "{}", key.name);
    }
    let dir = key.dir.join("honest");
    fs::create_dir_all(&dir).unwrap();
    honest[0].proof.to_json(key.names).unwrap().write(&dir.join("proof.json")).unwrap();
    Publics(honest[0].publics.clone()).write(&dir.join("publics.json")).unwrap();
    let verdict = js_verifier::verify(key.vkey_path, &dir.join("publics.json"), &dir.join("proof.json")).unwrap();
    assert!(verdict, "{}: the JS verifier rejects {}", key.name, honest[0].label);

    let mut report = KeyReport {
        name: key.name.to_string(),
        n_f: key.vkey.layout.0.len(),
        words: shape.layout.words(),
        n_public: shape.n_public,
        cases: 0,
        honest: Breakdown::default(),
        families: BTreeMap::new(),
        problems: Vec::new(),
        times: [Duration::ZERO; 3],
    };
    let mut cross_checked: BTreeSet<Family> = BTreeSet::new();
    let (mut made, mut next_family) = (0, 0);
    let total = honest.len() + n;
    while made < total {
        // 1. The cases of the round.
        let start = Instant::now();
        let mut cases = Vec::new();
        let mut forgeries = Vec::new();
        while made < total && cases.len() + forgeries.len() < ROUND {
            let family = if made < honest.len() {
                Family::Honest
            } else {
                let active: Vec<Family> = {
                    let stopped = stopped.lock().unwrap();
                    families.iter().copied().filter(|f| !stopped.contains(f)).collect()
                };
                if active.is_empty() {
                    break;
                }
                next_family += 1;
                active[(offset + next_family) % active.len()]
            };
            match generator.case(family, made) {
                Made::Case(case) => cases.push(case),
                Made::Forgery(forgery) => forgeries.push(*forgery),
            }
            made += 1;
        }
        if cases.is_empty() && forgeries.is_empty() {
            break;
        }
        if !forgeries.is_empty() {
            let requests: Vec<Value> = forgeries
                .iter()
                .map(|f| {
                    let words = Calldata::encode(key.vkey, &f.proof, &f.publics).unwrap().proof;
                    let publics: Vec<Word> = f.publics.iter().map(FrBytes::to_be_bytes).collect();
                    json!({"op": "forgeWp", "proof": proof_json(&shape, key.names, &words), "publics": publics_json(&publics)})
                })
                .collect();
            let responses = run_js(key.vkey_path, &key.dir.join("js"), &requests);
            for (forgery, response) in forgeries.into_iter().zip(responses) {
                let wp = &response["Wp"];
                let coordinate = |i: usize| FqBytes::from_decimal(wp[i].as_str().unwrap()).unwrap();
                let mut proof = forgery.proof;
                proof.wp = G1Affine { x: coordinate(0), y: coordinate(1) };
                let case = generator.fixed(
                    Family::FixedForgery,
                    forgery.label,
                    forgery.base,
                    proof,
                    &forgery.publics,
                    vec![Check::Pairing],
                );
                cases.push(case);
            }
        }
        report.times[0] += start.elapsed();

        // 2. The JS verifier.
        let start = Instant::now();
        let requests: Vec<Value> = cases
            .iter()
            .map(|c| {
                let words = &c.proof[..shape.proof_words()];
                json!({"op": "verify", "proof": proof_json(&shape, key.names, words), "publics": publics_json(&c.publics)})
            })
            .collect();
        let responses = run_js(key.vkey_path, &key.dir.join("js"), &requests);
        let mut verdicts = Vec::with_capacity(cases.len());
        for (case, response) in cases.iter().zip(&responses) {
            if let Some(threw) = response.get("threw") {
                report.problems.push(format!(
                    "{}, {} ({}): the JS verifier threw {threw}",
                    key.name,
                    case.family.name(),
                    case.label
                ));
            }
            let messages = response["messages"]
                .as_array()
                .map_or_else(Vec::new, |m| m.iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect());
            verdicts.push(JsVerdict { verdict: response["verdict"].as_bool() == Some(true), messages });
        }
        // The first case of each family, with the CLI's verifier too (bin/verify.js).
        for (case, js) in cases.iter().zip(&verdicts) {
            if case.length == Length::Exact && cross_checked.insert(case.family) {
                let dir = key.dir.join("cli").join(format!("{:?}", case.family));
                fs::create_dir_all(&dir).unwrap();
                let words = &case.proof[..shape.proof_words()];
                fs::write(dir.join("proof.json"), proof_json(&shape, key.names, words).to_string()).unwrap();
                fs::write(dir.join("publics.json"), publics_json(&case.publics).to_string()).unwrap();
                let cli =
                    js_verifier::verify(key.vkey_path, &dir.join("publics.json"), &dir.join("proof.json")).unwrap();
                if cli != js.verdict {
                    report.problems.push(format!(
                        "{}, {} ({}): bin/verify.js says {cli}, js_batch.mjs {}",
                        key.name,
                        case.family.name(),
                        case.label,
                        js.verdict
                    ));
                }
            }
        }
        report.times[1] += start.elapsed();

        // 3. Foundry.
        let start = Instant::now();
        let calldata: Vec<Vec<u8>> = cases.iter().map(|c| c.calldata(&selector)).collect();
        let runs = project.run(tools, report.cases, &calldata);
        report.times[2] += start.elapsed();

        // 4. The comparison.
        if report.cases == 0 {
            let words = runs[0].probe.as_ref().expect("the honest proof's probe returns");
            report.honest = Breakdown::of(words, runs[0].gas, runs[0].probe_gas);
        }
        let honest_gas = report.honest.total;
        for ((case, js), run) in cases.iter().zip(&verdicts).zip(&runs) {
            let solidity = match (run.outcome, &run.probe) {
                (Outcome::Revert, None) => Check::Revert,
                (Outcome::Accept | Outcome::Reject, Some(words)) => probe.check(words),
                _ => {
                    report.problems.push(format!(
                        "{}, {} ({}): the verifier's call {} and the probe's {}",
                        key.name,
                        case.family.name(),
                        case.label,
                        run.outcome.as_str(),
                        if run.probe.is_some() { "returned" } else { "reverted" }
                    ));
                    Check::Revert
                }
            };
            let expected = reference(key, &shape, case, js);
            let mut problems = Vec::new();
            match expected {
                None => problems.push("a message of the JS verifier names no check"),
                Some(expected) if expected.outcome() != run.outcome => {
                    problems.push("DISAGREEMENT: the outcomes differ")
                }
                Some(expected) if expected != solidity => problems.push("the checks that refused it differ"),
                Some(_) => {}
            }
            if solidity.outcome() != run.outcome {
                problems.push("the probe's verdict is not the verifier's");
            }
            if !case.targets.is_empty() && !case.targets.contains(&solidity) {
                problems.push("it does not reach the check its family targets");
            }
            if solidity == Check::Pairing && run.gas * PAIRING_GAS.1 < honest_gas * PAIRING_GAS.0 {
                problems.push("refused by the pairing for less than 90 % of the gas of the honest proof");
            }
            let agree = expected.is_some_and(|e| e.outcome() == run.outcome);
            report.families.entry(case.family).or_default().add(solidity, agree, run.gas);
            if !problems.is_empty() {
                if !agree {
                    stopped.lock().unwrap().insert(case.family);
                }
                report.problems.push(format!(
                    "{}:\n  {}",
                    problems.join("; "),
                    evidence(key, &shape, honest, case, js, run, solidity)
                ));
            }
        }
        report.cases += cases.len();
    }
    report
}

// ---------------------------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------------------------

/// Prints the report of every key: the cases of each family and what refused them, and where the
/// gas of each key's honest proof goes. Returns every problem.
pub fn print_report(reports: &[KeyReport], elapsed: Duration) -> Vec<String> {
    let mut families: BTreeMap<Family, FamilyStats> = BTreeMap::new();
    for report in reports {
        for (family, stats) in &report.families {
            families.entry(*family).or_default().merge(stats);
        }
    }
    let total: usize = reports.iter().map(|r| r.cases).sum();
    let agree: usize = families.values().map(|s| s.agree).sum();
    println!("\n| Family | Cases | Agree | Refused by (the contract's check, which is the JS verifier's) | Gas |");
    println!("|---|---|---|---|---|");
    for (family, stats) in &families {
        let checks: Vec<String> = stats.checks.iter().map(|(c, n)| format!("{} {n}", c.name())).collect();
        let (lo, hi) = stats.gas.unwrap_or_default();
        println!("| {} | {} | {} | {} | {lo}–{hi} |", family.name(), stats.cases, stats.agree, checks.join(", "));
    }
    println!("\n| Key | f | Words | Publics | Cases | Generation (s) | JS (s) | Foundry (s) |");
    println!("|---|---|---|---|---|---|---|---|");
    for r in reports {
        let [g, j, f] = r.times.map(|t| t.as_secs_f64());
        println!(
            "| `{}` | {} | {} | {} | {} | {g:.1} | {j:.1} | {f:.1} |",
            r.name, r.n_f, r.words, r.n_public, r.cases
        );
    }
    println!("\n| Key | `verifyProof` | Probe | Body | {} | Call and ABI |", STEPS.join(" | "));
    println!("|---|---|---|---|{}---|", "---|".repeat(STEPS.len()));
    for r in reports {
        let b = &r.honest;
        let steps: Vec<String> = b.steps.iter().map(u64::to_string).collect();
        let rest = b.total - b.body;
        println!("| `{}` | {} | {} | {} | {} | {rest} |", r.name, b.total, b.probe_total, b.body, steps.join(" | "));
    }
    let problems: Vec<String> = reports.iter().flat_map(|r| r.problems.iter().cloned()).collect();
    println!(
        "\n{total} cases on {} keys, {agree} agreements, {} problems, in {:.0} s",
        reports.len(),
        problems.len(),
        elapsed.as_secs_f64()
    );
    problems
}
