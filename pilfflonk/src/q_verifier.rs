//! The vkey's `qVerifier` (pilfflonk/docs/formats.md#expressionsinfo-and-verifierinfo): the code the
//! verifier runs over `Fr` to compute `Q(ξ)` from the evaluations, `pil-info`'s code block in the
//! STARK's format with every dimension 1. This crate does not run it. It checks that the JS verifier
//! can, with the rules of its `checkQVerifier` (`pilfflonk/js/src/qverifier.js`), so that no vkey
//! the verifier refuses is written:
//!
//! - an object `{tmpUsed, code}`: `tmpUsed` a count, `code` a list of at least one entry;
//! - each entry `{op, dest, src}`: `op` one of `add`, `sub`, `mul`, with two operands, and `copy`,
//!   with one; `dest` a temporary below `tmpUsed`, checked after the operands;
//! - each operand of dimension 1, and one of:
//!   - `tmp`, a temporary an earlier entry wrote (an entry cannot read its own destination first);
//!   - `eval`, an index of the evMap;
//!   - `public`, an index of the publics;
//!   - `number`, a canonical decimal string below `r`;
//!   - `challenge`, a `(stage, stageId)` of the AIR's challenges, with its position among them as
//!     its `id` ([`challenge_position`]);
//!   - `Zi`, the index `boundaryId` of a boundary.
//!
//! Air values, airgroup values and proof values do not exist in format version 1, nor do custom
//! commits (pilfflonk/docs/README.md#scope), and `xDivXSubXi` is FRI's: any other operand is
//! refused. Other keys, such as a block's `line` or a `Zi`'s `id`, are not looked at, as the
//! verifier does not.

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::error::{invalid, PilfflonkResult};
use crate::field::{parse_canonical_decimal, r};

/// What the operands of a vkey's `qVerifier` can refer to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QVerifierShape<'a> {
    /// The entries of the evMap.
    pub n_evaluations: usize,
    pub n_public: u64,
    pub n_boundaries: usize,
    /// The vkey's `numChallenges`, one entry per stage.
    pub num_challenges: &'a [u64],
}

/// The position of the challenge `(stage, stage_id)` among an AIR's, the order of its
/// challengesMap: the `numChallenges[s − 1]` challenges of each stage `s = 2 … nStages`, then
/// `std_vc` (stage `nStages + 1`) and `std_xi` (`nStages + 2`), each with `stageId` 0. Stage 1 has
/// none: the transcript squeezes no challenge before its commitments
/// (pilfflonk/docs/protocol.md#transcript). `None` if the AIR has no such challenge. `nStages` is
/// `num_challenges.len()`.
///
/// `challengesMap` of `qverifier.js`, without listing the challenges.
pub(crate) fn challenge_position(num_challenges: &[u64], stage: u64, stage_id: u64) -> Option<u64> {
    let counts = num_challenges.iter().skip(1).copied().chain([1, 1]);
    let mut position = 0u64;
    for (s, count) in (2u64..).zip(counts) {
        if s == stage {
            return if stage_id < count { position.checked_add(stage_id) } else { None };
        }
        position = position.checked_add(count)?;
    }
    None
}

/// A JSON value as an error message shows it.
fn show(value: Option<&Value>) -> String {
    value.map_or_else(|| "nothing".to_string(), Value::to_string)
}

/// `value` as a count, a non-negative integer.
fn as_count(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

/// Whether `value` is an index below `n`.
fn is_index(value: Option<&Value>, n: u64) -> bool {
    as_count(value).is_some_and(|i| i < n)
}

/// Checks `q_verifier` as the verifier checks it before running it (see the module).
pub(crate) fn check_q_verifier(q_verifier: &Value, shape: &QVerifierShape) -> PilfflonkResult<()> {
    let Some((block, code)) = q_verifier.as_object().and_then(|b| Some((b, b.get("code")?.as_array()?))) else {
        return invalid!("qVerifier: not a code block {{tmpUsed, code}}");
    };
    let Some(tmp_used) = as_count(block.get("tmpUsed")) else {
        return invalid!("qVerifier: tmpUsed = {} is not a count", show(block.get("tmpUsed")));
    };
    if code.is_empty() {
        return invalid!("qVerifier: no code: it computes no Q(ξ)");
    }

    let mut written = HashSet::new();
    for (i, entry) in code.iter().enumerate() {
        let Some((entry, src)) = entry.as_object().and_then(|e| Some((e, e.get("src")?.as_array()?))) else {
            return invalid!("qVerifier: code[{i}] is not {{op, dest, src}}");
        };
        let (op, arity) = match entry.get("op").and_then(Value::as_str) {
            Some(op @ ("add" | "sub" | "mul")) => (op, 2),
            Some(op @ "copy") => (op, 1),
            _ => return invalid!("qVerifier: code[{i}]: op {} is not add, sub, mul or copy", show(entry.get("op"))),
        };
        if src.len() != arity {
            return invalid!("qVerifier: code[{i}]: {op} takes {arity} operands");
        }
        for (j, value) in src.iter().enumerate() {
            check_operand(value, &format!("code[{i}].src[{j}]"), &written, shape)?;
        }
        let dest = entry.get("dest").and_then(Value::as_object);
        let dest_id = dest.filter(|d| is_temporary(d)).and_then(|d| as_count(d.get("id"))).filter(|&id| id < tmp_used);
        let Some(dest_id) = dest_id else {
            return invalid!("qVerifier: code[{i}]: the destination is not a temporary below tmpUsed = {tmp_used}");
        };
        written.insert(dest_id);
    }
    Ok(())
}

/// Whether `operand` is a temporary of dimension 1, its id aside.
fn is_temporary(operand: &Map<String, Value>) -> bool {
    operand.get("type").and_then(Value::as_str) == Some("tmp") && as_count(operand.get("dim")) == Some(1)
}

/// Checks one operand, `at` where it is in the code; `written` holds the temporaries the entries
/// before wrote.
fn check_operand(value: &Value, at: &str, written: &HashSet<u64>, shape: &QVerifierShape) -> PilfflonkResult<()> {
    let Some(operand) = value.as_object() else {
        return invalid!("qVerifier: {at} is not an operand");
    };
    let field = |key: &str| operand.get(key);
    if as_count(field("dim")) != Some(1) {
        return invalid!("qVerifier: {at} has dimension {}; over BN128 every operand has 1", show(field("dim")));
    }
    let id = field("id");
    let index = |n: u64, what: &str, kind: &str| {
        if is_index(id, n) {
            Ok(())
        } else {
            invalid!("qVerifier: {at}: {kind} {} is not one of the {n} {what}", show(id))
        }
    };
    match field("type").and_then(Value::as_str) {
        Some("tmp") => {
            if as_count(id).is_some_and(|id| written.contains(&id)) {
                Ok(())
            } else {
                invalid!("qVerifier: {at} reads temporary {} before it is written", show(id))
            }
        }
        Some("eval") => index(shape.n_evaluations as u64, "entries of the evMap", "eval"),
        Some("public") => index(shape.n_public, "publics", "public"),
        Some("number") => match field("value").and_then(Value::as_str).and_then(parse_canonical_decimal) {
            None => invalid!(
                "qVerifier: {at}: number: {} is not a decimal string without sign or leading zeros",
                show(field("value"))
            ),
            Some(v) if v >= *r() => invalid!("qVerifier: {at}: the number {v} is not below r"),
            Some(_) => Ok(()),
        },
        Some("challenge") => {
            let (stage, stage_id) = (field("stage"), field("stageId"));
            let position = as_count(stage)
                .zip(as_count(stage_id))
                .and_then(|(s, j)| challenge_position(shape.num_challenges, s, j));
            match position {
                None => {
                    invalid!("qVerifier: {at}: no challenge of stage {} has stageId {}", show(stage), show(stage_id))
                }
                Some(position) if as_count(id) != Some(position) => invalid!(
                    "qVerifier: {at}: the challenge of stage {}, stageId {} is {position}, not {}",
                    show(stage),
                    show(stage_id),
                    show(id)
                ),
                Some(_) => Ok(()),
            }
        }
        Some("Zi") => {
            if is_index(field("boundaryId"), shape.n_boundaries as u64) {
                Ok(())
            } else {
                invalid!(
                    "qVerifier: {at}: Zi of boundary {}, and there are {}",
                    show(field("boundaryId")),
                    shape.n_boundaries
                )
            }
        }
        Some(kind @ ("airvalue" | "airgroupvalue" | "proofvalue")) => {
            invalid!("qVerifier: {at}: {kind}s do not exist in format version 1 (pilfflonk/docs/README.md#scope)")
        }
        _ => invalid!("qVerifier: {at}: operand type {} is not one of the qVerifier's", show(field("type"))),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn operand(kind: &str, fields: Value) -> Value {
        let mut r = json!({"type": kind, "dim": 1});
        r.as_object_mut().unwrap().extend(fields.as_object().unwrap().clone());
        r
    }

    fn tmp(id: u64) -> Value {
        operand("tmp", json!({"id": id}))
    }

    fn eval(id: u64) -> Value {
        operand("eval", json!({"id": id}))
    }

    fn entry(op: &str, dest: u64, src: &[Value]) -> Value {
        json!({"op": op, "dest": tmp(dest), "src": src})
    }

    /// `SHAPE` of `pilfflonk/js/test/qverifier.test.js`: challenges `[0, 2]` of two stages.
    const NUM_CHALLENGES: [u64; 2] = [0, 2];
    const SHAPE: QVerifierShape<'static> =
        QVerifierShape { n_evaluations: 3, n_public: 2, n_boundaries: 2, num_challenges: &NUM_CHALLENGES };

    fn check(tmp_used: u64, code: Vec<Value>) -> PilfflonkResult<()> {
        check_q_verifier(&json!({"tmpUsed": tmp_used, "code": code}), &SHAPE)
    }

    /// `challengesMap` of `qverifier.test.js`: the positions of its lists.
    #[test]
    fn the_challenges_are_the_stages_then_std_vc_and_std_xi() {
        let positions = |num_challenges: &[u64]| -> Vec<(u64, u64, u64)> {
            let mut found = Vec::new();
            for stage in 0..8 {
                for stage_id in 0..4 {
                    if let Some(p) = challenge_position(num_challenges, stage, stage_id) {
                        found.push((p, stage, stage_id));
                    }
                }
            }
            found.sort();
            found
        };
        assert_eq!(positions(&[0]), [(0, 2, 0), (1, 3, 0)]);
        assert_eq!(positions(&[0, 2, 1]), [(0, 2, 0), (1, 2, 1), (2, 3, 0), (3, 4, 0), (4, 5, 0)]);
        // A stage 1 count is never a challenge; nor is a count that overflows a position.
        assert_eq!(positions(&[3]), [(0, 2, 0), (1, 3, 0)]);
        assert_eq!(challenge_position(&[0, u64::MAX, 1], 4, 0), None);
    }

    /// Every op and operand of `qverifier.test.js`'s accepted code.
    #[test]
    fn it_accepts_every_op_and_operand() {
        let code = vec![
            entry("add", 0, &[eval(0), operand("public", json!({"id": 1}))]),
            entry("sub", 1, &[tmp(0), operand("number", json!({"value": "7"}))]),
            entry("mul", 2, &[tmp(1), operand("challenge", json!({"id": 1, "stage": 2, "stageId": 1}))]),
            entry("copy", 3, &[operand("Zi", json!({"id": 0, "boundaryId": 1}))]),
            entry("mul", 3, &[tmp(3), tmp(2)]),
            entry("sub", 4, &[operand("challenge", json!({"id": 2, "stage": 3, "stageId": 0})), eval(2)]),
            entry("add", 5, &[tmp(3), tmp(4)]),
        ];
        check_q_verifier(&json!({"tmpUsed": 6, "code": code, "line": ""}), &SHAPE).unwrap();
        let r_minus_1 = "21888242871839275222246405745257275088548364400416034343698204186575808495616";
        check(1, vec![entry("copy", 0, &[operand("number", json!({"value": r_minus_1}))])]).unwrap();
    }

    /// The refusals of `qverifier.test.js`, with its reasons.
    #[test]
    fn it_refuses_what_the_verifier_refuses() {
        let r = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
        let refusals: Vec<(&str, Vec<Value>, &str)> = vec![
            ("no code", vec![], "no code"),
            ("an unknown op", vec![entry("div", 0, &[tmp(0), tmp(0)])], "op \"div\" is not add, sub, mul or copy"),
            (
                "sub_swap, which has no dimension to swap",
                vec![entry("sub_swap", 0, &[tmp(0), tmp(0)])],
                "op \"sub_swap\"",
            ),
            ("an op with too few operands", vec![entry("mul", 0, &[eval(0)])], "mul takes 2 operands"),
            ("a copy with two operands", vec![entry("copy", 0, &[eval(0), eval(1)])], "copy takes 1"),
            ("a temporary read before it is written", vec![entry("copy", 0, &[tmp(1)])], "reads temporary 1"),
            (
                "a temporary read by the op that writes it first",
                vec![entry("add", 0, &[tmp(0), tmp(0)])],
                "reads temporary 0",
            ),
            (
                "a destination that is not a temporary",
                vec![json!({"op": "copy", "dest": eval(0), "src": [eval(0)]})],
                "the destination is not a temporary",
            ),
            ("a destination above tmpUsed", vec![entry("copy", 9, &[eval(0)])], "below tmpUsed = 4"),
            ("an evaluation out of the evMap", vec![entry("copy", 0, &[eval(3)])], "eval 3 is not one of the 3"),
            (
                "a public out of range",
                vec![entry("copy", 0, &[operand("public", json!({"id": 2}))])],
                "public 2 is not one of the 2",
            ),
            (
                "a number not below r",
                vec![entry("copy", 0, &[operand("number", json!({"value": r}))])],
                "is not below r",
            ),
            (
                "a number that is not a decimal string",
                vec![entry("copy", 0, &[operand("number", json!({"value": 7}))])],
                "7 is not a decimal string",
            ),
            (
                "a number with a sign",
                vec![entry("copy", 0, &[operand("number", json!({"value": "-1"}))])],
                "without sign or leading zeros",
            ),
            (
                "an operand of dimension 3",
                vec![entry("copy", 0, &[json!({"type": "eval", "id": 0, "dim": 3})])],
                "dimension 3",
            ),
            (
                "a challenge of no stage",
                vec![entry("copy", 0, &[operand("challenge", json!({"id": 0, "stage": 1, "stageId": 0}))])],
                "no challenge of stage 1 has stageId 0",
            ),
            (
                "a challenge whose id is not its position",
                vec![entry("copy", 0, &[operand("challenge", json!({"id": 0, "stage": 3, "stageId": 0}))])],
                "the challenge of stage 3, stageId 0 is 2, not 0",
            ),
            (
                "a Zi of no boundary",
                vec![entry("copy", 0, &[operand("Zi", json!({"boundaryId": 2}))])],
                "Zi of boundary 2, and there are 2",
            ),
            (
                "an air value",
                vec![entry("copy", 0, &[operand("airvalue", json!({"id": 0}))])],
                "airvalues do not exist in format version 1",
            ),
            (
                "a proof value",
                vec![entry("copy", 0, &[operand("proofvalue", json!({"id": 0}))])],
                "proofvalues do not exist",
            ),
            (
                "an airgroup value",
                vec![entry("copy", 0, &[operand("airgroupvalue", json!({"id": 0}))])],
                "airgroupvalues do not exist",
            ),
            (
                "FRI's xDivXSubXi",
                vec![entry("copy", 0, &[operand("xDivXSubXi", json!({"id": 0}))])],
                "\"xDivXSubXi\" is not one of the qVerifier's",
            ),
            (
                "a column, which the verifier reads as an eval",
                vec![entry("copy", 0, &[operand("cm", json!({"id": 0}))])],
                "\"cm\" is not one",
            ),
        ];
        for (name, code, reason) in refusals {
            let err = check(4, code).expect_err(name).to_string();
            assert!(err.contains(reason), "{name}: {err}");
        }

        let good = || vec![entry("copy", 0, &[eval(0)])];
        for bad in [
            json!(null),
            json!([]),
            json!({"code": good()}),
            json!({"tmpUsed": -1, "code": good()}),
            json!({"tmpUsed": 1.5, "code": good()}),
            json!({"tmpUsed": 1, "code": {}}),
            json!({"tmpUsed": 1, "code": [null]}),
            json!({"tmpUsed": 1, "code": [{"op": "copy", "dest": tmp(0)}]}),
            json!({"tmpUsed": 1, "code": [{"op": "copy", "src": [eval(0)]}]}),
            json!({"tmpUsed": 1, "code": [{"op": "copy", "dest": {"type": "tmp", "id": 0, "dim": 3}, "src": [eval(0)]}]}),
            json!({"tmpUsed": 1, "code": [entry("copy", 0, &[json!("eval")])]}),
            json!({"tmpUsed": 1, "code": [entry("copy", 0, &[json!({"type": "eval", "id": 0})])]}),
            json!({"tmpUsed": 1, "code": [entry("copy", 0, &[json!({"type": "eval", "dim": 1})])]}),
            json!({"tmpUsed": 1, "code": [entry("copy", 0, &[json!({"type": "eval", "id": -1, "dim": 1})])]}),
            json!({"tmpUsed": 1, "code": [entry("copy", 0, &[operand("challenge", json!({"id": 1, "stage": 3}))])]}),
        ] {
            assert!(check_q_verifier(&bad, &SHAPE).is_err(), "{bad}");
        }
    }
}
