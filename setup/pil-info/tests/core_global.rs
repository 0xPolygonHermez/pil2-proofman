//! `global_from_pilout` and the global constraints JSON.

mod common;

use common::*;
use pil2_pilout::pilout::{self as pb, global_operand as go};
use pil_info::core::from_pilout::PiloutError;
use pil_info::core::global::global_from_pilout;
use pil_info::core::ir::Expr;
use pil_info::FieldCfg;

fn global_pilout(exprs: Vec<pb::GlobalExpression>, cons: Vec<pb::GlobalConstraint>) -> pb::PilOut {
    pb::PilOut {
        num_challenges: vec![0],
        expressions: exprs,
        constraints: cons,
        num_public_values: 2,
        ..Default::default()
    }
}

#[test]
fn rejects_global_constraint_without_expression() {
    let p = global_pilout(vec![gsub(gpublic(0), gconst(1))], vec![gconstraint(Some(0)), gconstraint(None)]);
    assert_eq!(
        global_from_pilout(&p, &FieldCfg::goldilocks()).unwrap_err(),
        PiloutError::MissingExpressionIdx { constraint: 1 }
    );
}

/// The refusals are errors.rs's (dangling, cyclic); here a reference that resolves.
#[test]
fn global_refs_resolve() {
    let p = global_pilout(vec![gmul(gexp(1), gpublic(0)), gsub(gpublic(1), gconst(1))], vec![gconstraint(Some(0))]);
    let gs = global_from_pilout(&p, &FieldCfg::goldilocks()).unwrap();
    assert_eq!(gs.exprs.len(), 2);
    assert_eq!(gs.constraints.len(), 1);
}

#[test]
fn global_airgroup_values_carry_their_airgroup() {
    let agv = |g: u32, idx: u32| gop(go::Operand::AirGroupValue(go::AirGroupValue { air_group_id: g, idx }));
    let mut p = global_pilout(vec![gsub(agv(1, 0), agv(0, 1))], vec![gconstraint(Some(0))]);
    p.air_groups = vec![
        pb::AirGroup { air_group_values: vec![pb::AirGroupValue::default(); 2], ..Default::default() },
        pb::AirGroup { air_group_values: vec![pb::AirGroupValue::default(); 1], ..Default::default() },
    ];
    let gs = global_from_pilout(&p, &FieldCfg::goldilocks()).unwrap();
    assert_eq!(
        gs.exprs[0],
        Expr::Sub(
            Box::new(Expr::AirgroupValue { airgroup: Some(1), id: 0 }),
            Box::new(Expr::AirgroupValue { airgroup: Some(0), id: 1 })
        )
    );
    // id past the airgroup's own values (it would be valid if flattened), and an unknown airgroup
    p.expressions = vec![gsub(agv(1, 1), agv(0, 0))];
    assert!(matches!(global_from_pilout(&p, &FieldCfg::goldilocks()), Err(PiloutError::BadOperand { expr: 0, .. })));
    p.expressions = vec![gsub(agv(2, 0), agv(0, 0))];
    assert!(matches!(global_from_pilout(&p, &FieldCfg::goldilocks()), Err(PiloutError::BadOperand { expr: 0, .. })));
}

#[test]
fn global_rejects_out_of_range_operands() {
    let f = FieldCfg::goldilocks();
    let mut p = global_pilout(vec![gsub(gpublic(0), gconst(1))], vec![gconstraint(Some(0))]);
    p.num_public_values = 0;
    assert!(matches!(global_from_pilout(&p, &f), Err(PiloutError::BadOperand { .. })));
    p.num_public_values = 1;
    assert!(global_from_pilout(&p, &f).is_ok());
}
