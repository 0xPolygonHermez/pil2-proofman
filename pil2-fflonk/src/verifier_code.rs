//! The constraint code a verifier evaluates at `xi`.
//!
//! `fflonkinfo.json` carries a straight-line program that computes the AIR's
//! combined constraint expression from the claimed openings. The verifier runs
//! it to obtain `cExp(xi)`, which [`crate::air`] turns into the quotient
//! evaluation the opening scheme needs.
//!
//! Only `verifierCode.first` is read. The `i` and `last` sections are
//! row-specialised and are not self-contained -- they reference temporaries
//! they never assign, so they cannot be evaluated on their own. `first` is the
//! general program, and it is the one that reproduces the prover's quotient.
//!
//! This is the pil1-era JSON encoding that pil-fflonk emits. pil2 expresses the
//! same thing as uint16 bytecode; when that path arrives it should produce a
//! `cExp(xi)` for [`crate::air`] to consume, leaving the relation between the
//! constraint and the quotient untouched.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use num_bigint::BigUint;
use serde_json::Value;

use crate::fr;
use crate::proof::{ShPlonkProof, evaluation_key};

/// Where an instruction reads from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operand {
    Tmp(usize),
    /// An opening, indexed into the `evMap`.
    Eval(usize),
    Number(BigUint),
    /// An AIR challenge, indexed in the order the protocol draws them.
    Challenge(usize),
    Public(usize),
    /// The point everything is evaluated at, `xi`.
    X,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Copy,
    MulAdd,
}

impl Op {
    /// How many operands the instruction takes. Fixed per op, so a program
    /// with the wrong arity is rejected at parse time rather than producing a
    /// wrong number.
    pub fn arity(self) -> usize {
        match self {
            Op::Copy => 1,
            Op::Add | Op::Sub | Op::Mul => 2,
            Op::MulAdd => 3,
        }
    }

    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "add" => Op::Add,
            "sub" => Op::Sub,
            "mul" => Op::Mul,
            "copy" => Op::Copy,
            "muladd" => Op::MulAdd,
            other => bail!("unknown verifier-code operation {other:?}"),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub op: Op,
    pub dest: usize,
    pub src: Vec<Operand>,
}

/// One entry of the `evMap`: which polynomial an `eval` operand refers to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalRef {
    /// `cm` or `const`, selecting which `polsMap` group names it.
    pub pol_type: String,
    pub id: u64,
    /// Whether this is the next-row opening.
    pub prime: bool,
}

impl EvalRef {
    /// The proof's key for this opening, given the key's `polsMap`.
    pub fn evaluation_key(&self, pols_map: &BTreeMap<String, BTreeMap<String, String>>) -> Result<String> {
        let group = pols_map
            .get(&self.pol_type)
            .with_context(|| format!("verification key has no polsMap group {:?}", self.pol_type))?;
        let name = group
            .get(&self.id.to_string())
            .with_context(|| format!("polsMap {:?} has no polynomial {}", self.pol_type, self.id))?;

        Ok(evaluation_key(name, if self.prime { 1 } else { 0 }))
    }
}

/// A parsed straight-line program plus the openings it indexes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierCode {
    pub instructions: Vec<Instruction>,
    /// Size of the temporary slab. Destinations are not written in order and
    /// may be reassigned, so the program needs a slab rather than a stack.
    pub tmp_used: usize,
    pub ev_map: Vec<EvalRef>,
}

fn operand(v: &Value) -> Result<Operand> {
    let ty = v["type"].as_str().context("operand has no type")?;
    let id =
        || -> Result<usize> { v["id"].as_u64().map(|i| i as usize).with_context(|| format!("{ty} operand has no id")) };

    Ok(match ty {
        "tmp" => Operand::Tmp(id()?),
        "eval" => Operand::Eval(id()?),
        "challenge" => Operand::Challenge(id()?),
        "public" => Operand::Public(id()?),
        "x" => Operand::X,
        "number" => {
            let raw = v["value"].as_str().context("number operand has no value")?;
            Operand::Number(fr::from_decimal(raw).context("verifier-code literal")?)
        }
        other => bail!("unknown verifier-code operand type {other:?}"),
    })
}

impl VerifierCode {
    /// Read `verifierCode.first` and `evMap` from a fflonkinfo document.
    pub fn from_json(info: &Value) -> Result<Self> {
        let code = info["verifierCode"]["first"].as_array().context("verifierCode.first is missing")?;

        let tmp_used = info["verifierCode"]["tmpUsed"].as_u64().context("verifierCode.tmpUsed is missing")? as usize;

        let mut instructions = Vec::with_capacity(code.len());
        for (i, raw) in code.iter().enumerate() {
            let op = Op::parse(raw["op"].as_str().with_context(|| format!("instruction {i} has no op"))?)?;

            let dest =
                raw["dest"]["id"].as_u64().with_context(|| format!("instruction {i} has no destination"))? as usize;
            if raw["dest"]["type"].as_str() != Some("tmp") {
                bail!("instruction {i} writes to {:?}, but only tmp is a valid destination", raw["dest"]["type"]);
            }
            if dest >= tmp_used {
                bail!("instruction {i} writes tmp {dest}, past the declared tmpUsed of {tmp_used}");
            }

            let src: Vec<Operand> = raw["src"]
                .as_array()
                .with_context(|| format!("instruction {i} has no src"))?
                .iter()
                .map(operand)
                .collect::<Result<_>>()
                .with_context(|| format!("instruction {i}"))?;

            if src.len() != op.arity() {
                bail!("instruction {i} is {op:?} with {} operands, want {}", src.len(), op.arity());
            }

            instructions.push(Instruction { op, dest, src });
        }

        if instructions.is_empty() {
            bail!("verifierCode.first is empty");
        }

        let ev_map = info["evMap"]
            .as_array()
            .context("evMap is missing")?
            .iter()
            .enumerate()
            .map(|(i, e)| -> Result<EvalRef> {
                Ok(EvalRef {
                    pol_type: e["type"].as_str().with_context(|| format!("evMap[{i}] has no type"))?.to_string(),
                    id: e["id"].as_u64().with_context(|| format!("evMap[{i}] has no id"))?,
                    prime: e["prime"].as_bool().unwrap_or(false),
                })
            })
            .collect::<Result<_>>()?;

        Ok(VerifierCode { instructions, tmp_used, ev_map })
    }

    /// The openings the program reads, in `evMap` order, taken from the proof.
    pub fn evaluations(
        &self,
        pols_map: &BTreeMap<String, BTreeMap<String, String>>,
        proof: &ShPlonkProof,
    ) -> Result<Vec<BigUint>> {
        self.ev_map
            .iter()
            .enumerate()
            .map(|(i, e)| -> Result<BigUint> {
                let key = e.evaluation_key(pols_map).with_context(|| format!("evMap[{i}]"))?;
                let raw = proof
                    .evaluations
                    .get(&key)
                    .with_context(|| format!("proof is missing evaluation {key:?}, read by evMap[{i}]"))?;
                fr::from_decimal(raw).with_context(|| format!("evaluation {key:?}"))
            })
            .collect()
    }

    /// Run the program, returning the value its last instruction writes.
    pub fn evaluate(&self, inputs: &Inputs) -> Result<BigUint> {
        let mut tmp: Vec<Option<BigUint>> = vec![None; self.tmp_used];

        for (i, ins) in self.instructions.iter().enumerate() {
            let mut s = Vec::with_capacity(ins.src.len());
            for operand in &ins.src {
                s.push(inputs.read(operand, &tmp).with_context(|| format!("instruction {i}"))?);
            }

            let value = match ins.op {
                Op::Add => fr::add(&s[0], &s[1]),
                Op::Sub => fr::sub(&s[0], &s[1]),
                Op::Mul => fr::mul(&s[0], &s[1]),
                Op::Copy => s[0].clone(),
                Op::MulAdd => fr::add(&fr::mul(&s[0], &s[1]), &s[2]),
            };

            tmp[ins.dest] = Some(value);
        }

        let last = self.instructions.last().expect("checked non-empty at parse").dest;
        Ok(tmp[last].clone().expect("the last instruction assigns its own destination"))
    }
}

/// Everything the program reads that does not come from the code itself.
pub struct Inputs<'a> {
    /// Openings in `evMap` order, from [`VerifierCode::evaluations`].
    pub evals: &'a [BigUint],
    /// The AIR's challenges, in the order the protocol draws them.
    pub challenges: &'a [BigUint],
    pub publics: &'a [BigUint],
    /// The evaluation point.
    pub x: &'a BigUint,
}

impl Inputs<'_> {
    fn read(&self, operand: &Operand, tmp: &[Option<BigUint>]) -> Result<BigUint> {
        let pick = |slice: &[BigUint], i: usize, what: &str| -> Result<BigUint> {
            slice.get(i).cloned().with_context(|| format!("{what} {i} is out of range ({} available)", slice.len()))
        };

        Ok(match operand {
            // Reading an unassigned temporary means the program is not
            // self-contained, which is how the `i` and `last` sections differ
            // from `first`.
            Operand::Tmp(i) => {
                tmp.get(*i).cloned().flatten().with_context(|| format!("reads tmp {i} before it is assigned"))?
            }
            Operand::Eval(i) => pick(self.evals, *i, "evaluation")?,
            Operand::Challenge(i) => pick(self.challenges, *i, "challenge")?,
            Operand::Public(i) => pick(self.publics, *i, "public input")?,
            Operand::Number(v) => v.clone(),
            Operand::X => self.x.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference;
    use serde_json::json;

    const INFO: &str = include_str!("../tests/fixtures/pilfflonk.verifierinfo.json");

    fn code() -> VerifierCode {
        VerifierCode::from_json(&serde_json::from_str(INFO).unwrap()).unwrap()
    }

    #[test]
    fn parses_the_reference_program() {
        let c = code();
        assert_eq!(c.instructions.len(), 145);
        assert_eq!(c.tmp_used, 145);
        assert_eq!(c.ev_map.len(), 38);

        // The program ends by writing the combined constraint expression.
        assert_eq!(c.instructions.last().unwrap().op, Op::MulAdd);
        assert_eq!(c.instructions.last().unwrap().dest, 144);
    }

    /// Every operand kind the reference uses is understood; an unhandled one
    /// would surface as a parse error rather than a wrong answer.
    #[test]
    fn uses_only_operands_the_evaluator_knows() {
        let c = code();
        let name = |o: &Operand| match o {
            Operand::Tmp(_) => "tmp",
            Operand::Eval(_) => "eval",
            Operand::Number(_) => "number",
            Operand::Challenge(_) => "challenge",
            Operand::Public(_) => "public",
            Operand::X => "x",
        };
        let kinds: std::collections::BTreeSet<&str> =
            c.instructions.iter().flat_map(|i| i.src.iter().map(name)).collect();

        assert_eq!(kinds, ["challenge", "eval", "number", "public", "tmp", "x"].into_iter().collect());
    }

    /// The evMap resolves to openings the proof actually carries.
    #[test]
    fn the_ev_map_resolves_against_the_proof() {
        let r = reference::load();
        let c = code();
        let evals = c.evaluations(&r.setup.pols_map, &r.proof).unwrap();

        assert_eq!(evals.len(), 38);
        // The first entry is a next-row opening, so it carries the `w` suffix.
        assert!(c.ev_map[0].prime);
        assert!(c.ev_map[0].evaluation_key(&r.setup.pols_map).unwrap().ends_with('w'));
    }

    /// Temporaries are written out of order and reused, so the program needs a
    /// slab rather than a stack. If this stopped being true the evaluator
    /// would still be correct, but the reason for the slab would be gone.
    #[test]
    fn destinations_are_not_written_in_order() {
        let c = code();
        let dests: Vec<usize> = c.instructions.iter().map(|i| i.dest).collect();
        let mut sorted = dests.clone();
        sorted.sort_unstable();
        assert_ne!(dests, sorted);
    }

    #[test]
    fn rejects_a_program_with_the_wrong_arity() {
        let bad = json!({
            "verifierCode": {"tmpUsed": 4, "first": [
                {"op": "add", "dest": {"type": "tmp", "id": 0},
                 "src": [{"type": "number", "value": "1"}]}
            ]},
            "evMap": []
        });
        let err = VerifierCode::from_json(&bad).unwrap_err().to_string();
        assert!(err.contains("operands"), "{err}");
    }

    #[test]
    fn rejects_a_program_that_writes_past_its_slab() {
        let bad = json!({
            "verifierCode": {"tmpUsed": 1, "first": [
                {"op": "copy", "dest": {"type": "tmp", "id": 9},
                 "src": [{"type": "number", "value": "1"}]}
            ]},
            "evMap": []
        });
        assert!(VerifierCode::from_json(&bad).is_err());
    }

    #[test]
    fn rejects_an_unknown_operation() {
        let bad = json!({
            "verifierCode": {"tmpUsed": 1, "first": [
                {"op": "invert", "dest": {"type": "tmp", "id": 0}, "src": []}
            ]},
            "evMap": []
        });
        assert!(VerifierCode::from_json(&bad).is_err());
    }

    /// Reading an unassigned temporary is an error, not a zero. This is
    /// exactly how `verifierCode.i` fails, and treating it as zero would turn
    /// an unusable program into a silently wrong answer.
    #[test]
    fn reading_an_unassigned_temporary_is_an_error() {
        let bad: Value = json!({
            "verifierCode": {"tmpUsed": 4, "first": [
                {"op": "copy", "dest": {"type": "tmp", "id": 0},
                 "src": [{"type": "tmp", "id": 3}]}
            ]},
            "evMap": []
        });
        let c = VerifierCode::from_json(&bad).unwrap();
        let zero = BigUint::from(0u32);
        // `{:#}` renders the whole context chain; the outermost frame only
        // names the instruction.
        let err =
            format!("{:#}", c.evaluate(&Inputs { evals: &[], challenges: &[], publics: &[], x: &zero }).unwrap_err());
        assert!(err.contains("before it is assigned"), "{err}");
    }

    /// A short input list is reported rather than read past.
    #[test]
    fn rejects_inputs_that_are_too_short() {
        let c = code();
        let r = reference::load();
        let evals = c.evaluations(&r.setup.pols_map, &r.proof).unwrap();

        let err = format!(
            "{:#}",
            c.evaluate(&Inputs { evals: &evals, challenges: &[], publics: &r.publics, x: &r.xi }).unwrap_err()
        );
        assert!(err.contains("challenge"), "{err}");
    }

    /// The arithmetic, on a program small enough to check by hand:
    /// `(3 + 4) * x` with `x = 5` is 35.
    #[test]
    fn evaluates_a_hand_checkable_program() {
        let doc = json!({
            "verifierCode": {"tmpUsed": 2, "first": [
                {"op": "add", "dest": {"type": "tmp", "id": 0},
                 "src": [{"type": "number", "value": "3"}, {"type": "number", "value": "4"}]},
                {"op": "mul", "dest": {"type": "tmp", "id": 1},
                 "src": [{"type": "tmp", "id": 0}, {"type": "x"}]}
            ]},
            "evMap": []
        });
        let c = VerifierCode::from_json(&doc).unwrap();
        let x = BigUint::from(5u32);
        let got = c.evaluate(&Inputs { evals: &[], challenges: &[], publics: &[], x: &x }).unwrap();
        assert_eq!(got, BigUint::from(35u32));
    }

    /// muladd is `a*b + c`, not `(a + b)*c`.
    #[test]
    fn muladd_multiplies_the_first_two() {
        let doc = json!({
            "verifierCode": {"tmpUsed": 1, "first": [
                {"op": "muladd", "dest": {"type": "tmp", "id": 0}, "src": [
                    {"type": "number", "value": "3"},
                    {"type": "number", "value": "4"},
                    {"type": "number", "value": "5"}]}
            ]},
            "evMap": []
        });
        let c = VerifierCode::from_json(&doc).unwrap();
        let zero = BigUint::from(0u32);
        let got = c.evaluate(&Inputs { evals: &[], challenges: &[], publics: &[], x: &zero }).unwrap();
        assert_eq!(got, BigUint::from(17u32));
    }
}
