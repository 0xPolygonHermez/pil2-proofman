//! Core verifier code -> the qVerifier `ExpressionCodeEntry` (the only conversion).

use crate::core::dag::build::{build_dag, dag_boundary};
use crate::core::dag::format::{DagBoundary, DagError};
use crate::core::dag::lower_code::{lower_to_code, CodeOp, Fold, LowerCtx, LowerTables, Operand, Src, VerifierCode};
use crate::core::im::ImChoice;
use crate::core::ir::ConstraintSystem;
use crate::core::layout::Layout;
use crate::core::openings::Openings;
use crate::pil::gen_code::ExpressionCodeEntry;
use crate::types::output::{CodeEntry, CodeRef};

/// pilfflonk's qVerifier: the constraint DAG lowered with `Fold::Shplonk`, dimensions checked.
pub fn shplonk_q_verifier(
    cs: &ConstraintSystem,
    im: &ImChoice,
    lay: &Layout,
    ops: &Openings,
) -> Result<ExpressionCodeEntry, DagError> {
    let dag = build_dag(cs, im)?;
    let boundaries: Vec<DagBoundary> = cs.boundaries().iter().map(dag_boundary).collect();
    let code = lower_to_code(&dag, &lower_ctx(cs, lay, ops)?, Fold::Shplonk { boundaries: &boundaries })?;
    code.check_dims(cs.field.ext_dim())?;
    Ok(expression_code(&code))
}

/// The layout's challenges and values, the openings' evMap.
pub fn lower_ctx(cs: &ConstraintSystem, lay: &Layout, ops: &Openings) -> Result<LowerCtx, DagError> {
    let stages = |m: &[crate::core::ir::NamedValue]| m.iter().map(|v| v.stage).collect();
    LowerCtx::new(LowerTables {
        ev_map: ops.ev_map.iter().map(|e| e.pol).collect(),
        challenges: lay.challenges_map.iter().map(|c| (c.stage, c.stage_id)).collect(),
        vc: lay.vc_index(),
        n_publics: lay.publics_map.len() as u32, // below the ingestion limit
        air_values_stage: stages(&lay.air_values_map),
        proof_values_stage: stages(&lay.proof_values_map),
        airgroup_values_stage: stages(&lay.airgroup_values_map),
        ext_dim: cs.field.ext_dim(),
    })
}

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
