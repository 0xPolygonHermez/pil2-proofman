//! Pilouts built in code for the core tests.
#![allow(dead_code)]

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, global_expression, global_operand, operand, SymbolType};

pub const R: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

pub fn op(o: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(o) })
}
pub fn witness_at(stage: u32, col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage, col_idx, row_offset }))
}
pub fn witness(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    witness_at(1, col_idx, row_offset)
}
pub fn fixed(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset: 0 }))
}
pub fn big_be(decimal: &str) -> Vec<u8> {
    BigUint::parse_bytes(decimal.as_bytes(), 10).expect("a decimal constant").to_bytes_be()
}
pub fn constant(decimal: &str) -> Option<pb::Operand> {
    op(operand::Operand::Constant(operand::Constant { value: big_be(decimal) }))
}
pub fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}
pub fn challenge(stage: u32, idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Challenge(operand::Challenge { stage, idx }))
}
pub fn public(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::PublicValue(operand::PublicValue { idx }))
}
pub fn air_value(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::AirValue(operand::AirValue { idx }))
}
pub fn add(l: Option<pb::Operand>, r: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Add(expression::Add { lhs: l, rhs: r })) }
}
pub fn sub(l: Option<pb::Operand>, r: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Sub(expression::Sub { lhs: l, rhs: r })) }
}
pub fn mul(l: Option<pb::Operand>, r: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Mul(expression::Mul { lhs: l, rhs: r })) }
}
pub fn neg(v: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Neg(expression::Neg { value: v })) }
}

pub fn every_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("constraint on expression {idx}")),
        })),
    }
}
pub fn first_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::FirstRow(constraint::FirstRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("firstRow {idx}")),
        })),
    }
}
pub fn last_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::LastRow(constraint::LastRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("lastRow {idx}")),
        })),
    }
}
pub fn every_frame(idx: u32, min: u32, max: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryFrame(constraint::EveryFrame {
            expression_idx: Some(operand::Expression { idx }),
            offset_min: min,
            offset_max: max,
            debug_line: Some(format!("everyFrame {idx}")),
        })),
    }
}
pub fn every_row_no_idx() -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: None,
            debug_line: Some("no idx".to_string()),
        })),
    }
}
pub fn empty_constraint() -> pb::Constraint {
    pb::Constraint { constraint: None }
}

pub fn symbol(name: &str, kind: SymbolType, stage: u32, id: u32) -> pb::Symbol {
    pb::Symbol {
        name: name.to_string(),
        air_group_id: Some(0),
        air_id: Some(0),
        r#type: kind as i32,
        id,
        stage: Some(stage),
        ..Default::default()
    }
}

/// One BN128-fielded air of 16 rows with one stage (`numChallenges = [0]`), as pil2com emits it.
pub fn pilout(
    witness_names: &[&str],
    fixed_names: &[&str],
    expressions: Vec<pb::Expression>,
    constraints: Vec<pb::Constraint>,
) -> pb::PilOut {
    let mut symbols: Vec<pb::Symbol> =
        fixed_names.iter().enumerate().map(|(i, n)| symbol(n, SymbolType::FixedCol, 0, i as u32)).collect();
    symbols.extend(witness_names.iter().enumerate().map(|(i, n)| symbol(n, SymbolType::WitnessCol, 1, i as u32)));
    let air = pb::Air {
        name: Some("Synthetic".to_string()),
        num_rows: Some(16),
        fixed_cols: fixed_names.iter().map(|_| pb::FixedCol { values: Vec::new() }).collect(),
        stage_widths: vec![witness_names.len() as u32],
        expressions,
        constraints,
        ..Default::default()
    };
    pb::PilOut {
        name: Some("synthetic".to_string()),
        base_field: big_be(R),
        air_groups: vec![pb::AirGroup { name: Some("Synthetic".to_string()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0],
        symbols,
        ..Default::default()
    }
}

// ---- global constraints ----
pub fn gop(o: global_operand::Operand) -> Option<pb::GlobalOperand> {
    Some(pb::GlobalOperand { operand: Some(o) })
}
pub fn gpublic(idx: u32) -> Option<pb::GlobalOperand> {
    gop(global_operand::Operand::PublicValue(global_operand::PublicValue { idx }))
}
pub fn gexp(idx: u32) -> Option<pb::GlobalOperand> {
    gop(global_operand::Operand::Expression(global_operand::Expression { idx }))
}
pub fn gconst(v: u64) -> Option<pb::GlobalOperand> {
    gop(global_operand::Operand::Constant(global_operand::Constant { value: BigUint::from(v).to_bytes_be() }))
}
pub fn gsub(l: Option<pb::GlobalOperand>, r: Option<pb::GlobalOperand>) -> pb::GlobalExpression {
    pb::GlobalExpression {
        operation: Some(global_expression::Operation::Sub(global_expression::Sub { lhs: l, rhs: r })),
    }
}
pub fn gmul(l: Option<pb::GlobalOperand>, r: Option<pb::GlobalOperand>) -> pb::GlobalExpression {
    pb::GlobalExpression {
        operation: Some(global_expression::Operation::Mul(global_expression::Mul { lhs: l, rhs: r })),
    }
}
pub fn gconstraint(idx: Option<u32>) -> pb::GlobalConstraint {
    pb::GlobalConstraint {
        expression_idx: idx.map(|idx| global_operand::Expression { idx }),
        debug_line: Some("global".to_string()),
    }
}

// ---- real pilouts ----
pub fn gl() -> pil_info::FieldCfg {
    pil_info::FieldCfg::goldilocks()
}

/// The example pilouts built under `pil2-components/test/*/build` (the loaded pilout and its path).
pub fn real_pilouts() -> Vec<(std::path::PathBuf, pb::PilOut)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pil2-components/test");
    let mut out = Vec::new();
    for d in std::fs::read_dir(&root).expect("pil2-components/test").flatten() {
        let Ok(files) = std::fs::read_dir(d.path().join("build")) else { continue };
        for f in files.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|e| e == "pilout")) {
            let Some(path) = f.to_str() else { continue };
            let Ok(proxy) = pil2_pilout::pilout_proxy::PilOutProxy::new(path) else { continue };
            out.push((f, proxy.pilout));
        }
    }
    assert!(!out.is_empty(), "no pilout under {root:?}");
    out
}
