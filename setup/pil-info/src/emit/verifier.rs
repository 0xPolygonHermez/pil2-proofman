//! Core verifier code -> the qVerifier `ExpressionCodeEntry` (the only conversion).

use crate::core::dag::lower_code::{CodeOp, Operand, Src, VerifierCode};
use crate::pil::gen_code::ExpressionCodeEntry;
use crate::types::output::{CodeEntry, CodeRef};

pub fn expression_code(v: &VerifierCode) -> ExpressionCodeEntry {
    let code = v
        .code
        .iter()
        .map(|i| CodeEntry {
            op: match i.op {
                CodeOp::Add => "add",
                CodeOp::Sub => "sub",
                CodeOp::Mul => "mul",
                CodeOp::Copy => "copy",
            }
            .to_string(),
            dest: code_ref(&Src { operand: Operand::Tmp(i.dest), dim: i.dim }),
            src: i.src.iter().map(code_ref).collect(),
        })
        .collect();
    ExpressionCodeEntry { tmp_used: v.tmp_used, code, exp_id: 0, stage: 0, dest: None, line: String::new() }
}

fn code_ref(s: &Src) -> CodeRef {
    let mut r = CodeRef {
        ref_type: String::new(),
        id: 0,
        dim: s.dim,
        prime: None,
        value: None,
        stage: None,
        stage_id: None,
        commit_id: None,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    };
    let (t, id) = match s.operand {
        Operand::Tmp(id) => ("tmp", id),
        Operand::Eval(id) => ("eval", id),
        Operand::Challenge { id, stage, stage_id } => {
            (r.stage, r.stage_id) = (Some(stage as usize), Some(stage_id as usize));
            ("challenge", id)
        }
        Operand::Number(ref v) => {
            r.value = Some(v.clone());
            ("number", 0)
        }
        Operand::Public(id) => ("public", id),
        Operand::ProofValue { id, stage } => {
            r.stage = Some(stage as usize);
            ("proofvalue", id)
        }
        Operand::AirValue { id, stage } => {
            r.stage = Some(stage as usize);
            ("airvalue", id)
        }
        Operand::AirgroupValue { id, stage } => {
            r.stage = Some(stage as usize);
            ("airgroupvalue", id)
        }
        Operand::Zi { boundary } => {
            r.boundary_id = Some(boundary as usize);
            ("Zi", 0)
        }
    };
    r.ref_type = t.to_string();
    r.id = id as usize;
    r
}
