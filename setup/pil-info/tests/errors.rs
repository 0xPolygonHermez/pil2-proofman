//! What the passes refuse (plan M27): a pilout that breaks its own format, or constraints they
//! cannot process, is a `PilInfoError` the setups report, where the passes used to `panic!` or
//! index out of range. The pilouts are built in code, each one well formed but for one piece.

use pil2_pilout::pilout::{self as pb, constraint, expression, global_expression, global_operand, operand, SymbolType};
use pil_info::output::global_constraints::build_global_constraints_json;
use pil_info::{FieldCfg, PilInfoCfg, PilInfoError, PilInfoResult};

fn op(operand: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(operand) })
}

fn witness(col_idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset: 0 }))
}

fn fixed(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset: 0 }))
}

fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}

fn mul(lhs: Option<pb::Operand>, rhs: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Mul(expression::Mul { lhs, rhs })) }
}

fn every_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: None,
        })),
    }
}

fn column_symbol(name: &str, kind: SymbolType, stage: u32, id: u32) -> pb::Symbol {
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

/// One air, `Synthetic`, of 16 rows and one stage, with witness columns `a` and `b` and fixed
/// column `L1`, and the one constraint `L1·a·b = 0`: what each test breaks one piece of.
fn pilout() -> pb::PilOut {
    let air = pb::Air {
        name: Some("Synthetic".to_string()),
        num_rows: Some(16),
        fixed_cols: vec![pb::FixedCol { values: Vec::new() }],
        stage_widths: vec![2],
        expressions: vec![mul(witness(0), witness(1)), mul(fixed(0), exp(0))],
        constraints: vec![every_row(1)],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("synthetic".to_string()),
        air_groups: vec![pb::AirGroup { name: Some("Synthetic".to_string()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0],
        symbols: vec![
            column_symbol("Synthetic.L1", SymbolType::FixedCol, 0, 0),
            column_symbol("Synthetic.a", SymbolType::WitnessCol, 1, 0),
            column_symbol("Synthetic.b", SymbolType::WitnessCol, 1, 1),
        ],
        ..Default::default()
    }
}

fn the_air(pilout: &mut pb::PilOut) -> &mut pb::Air {
    &mut pilout.air_groups[0].airs[0]
}

/// The STARK's passes (FRI) and pilfflonk's (SHPLONK).
fn cfgs() -> [PilInfoCfg; 2] {
    [PilInfoCfg::goldilocks(1), PilInfoCfg::bn254()]
}

fn run(pilout: &pb::PilOut, cfg: &PilInfoCfg) -> Result<PilInfoResult, PilInfoError> {
    pil_info::run(pilout, 0, 0, cfg, &Default::default())
}

fn run_err(pilout: &pb::PilOut, cfg: &PilInfoCfg) -> PilInfoError {
    match run(pilout, cfg) {
        Ok(_) => panic!("the passes accepted the pilout"),
        Err(err) => err,
    }
}

fn invalid_pilout(err: &PilInfoError) -> &str {
    match err {
        PilInfoError::InvalidPilout(message) => message,
        other => panic!("not an InvalidPilout: {other:?}"),
    }
}

#[test]
fn the_pilout_as_built_is_accepted() {
    for cfg in cfgs() {
        run(&pilout(), &cfg).unwrap();
    }
    build_global_constraints_json(&pilout(), &FieldCfg::goldilocks()).unwrap();
}

/// Was an index out of range on `air_groups` or `airs`.
#[test]
fn an_air_the_pilout_does_not_have_is_refused() {
    for (airgroup_id, air_id) in [(0, 1), (1, 0)] {
        let err = pil_info::run(&pilout(), airgroup_id, air_id, &PilInfoCfg::bn254(), &Default::default()).err();
        assert!(
            matches!(err, Some(PilInfoError::NoSuchAir { airgroup_id: a, air_id: b }) if (a, b) == (airgroup_id, air_id)),
            "{err:?}"
        );
    }
    let err = pil_info::run(&pilout(), 0, 1, &PilInfoCfg::bn254(), &Default::default()).err().unwrap();
    assert_eq!(err.to_string(), "the pilout has no air 1 in airgroup 0");
}

/// Was an index out of range on the expressions.
#[test]
fn a_constraint_on_an_expression_the_air_does_not_have_is_refused() {
    let mut pilout = pilout();
    the_air(&mut pilout).constraints.push(every_row(9));
    for cfg in cfgs() {
        let err = run_err(&pilout, &cfg);
        assert_eq!(invalid_pilout(&err), "constraint 1 of air Synthetic is expression 9, and there are 2");
    }
}

/// Was an index out of range on the expressions, once the passes followed the reference.
#[test]
fn a_reference_to_an_expression_the_air_does_not_have_is_refused() {
    let mut pilout = pilout();
    the_air(&mut pilout).expressions[1] = mul(fixed(0), exp(7));
    for cfg in cfgs() {
        let err = run_err(&pilout, &cfg);
        assert_eq!(invalid_pilout(&err), "expression 1 refers to expression 7, and the air has 2");
        assert_eq!(err.to_string(), "invalid pilout: expression 1 refers to expression 7, and the air has 2");
    }
}

/// Was an index out of range on the custom commits.
#[test]
fn a_custom_column_of_a_custom_commit_the_air_does_not_have_is_refused() {
    let mut pilout = pilout();
    let custom =
        op(operand::Operand::CustomCol(operand::CustomCol { commit_id: 2, stage: 0, col_idx: 0, row_offset: 0 }));
    the_air(&mut pilout).expressions[0] = mul(witness(0), custom);
    let err = run_err(&pilout, &PilInfoCfg::bn254());
    assert_eq!(invalid_pilout(&err), "a custom column of custom commit 2, and the air has 0 custom commits");
}

/// Was `panic!("Invalid stage {} for a custom commit")`.
#[test]
fn a_custom_column_of_a_stage_other_than_0_is_refused() {
    let mut pilout = pilout();
    pilout.symbols.push(pb::Symbol { commit_id: Some(0), ..column_symbol("Synthetic.c", SymbolType::CustomCol, 1, 0) });
    let err = run_err(&pilout, &PilInfoCfg::goldilocks(1));
    assert!(matches!(&err, PilInfoError::CustomColumnStage { name, stage: 1 } if name == "Synthetic.c"), "{err:?}");
}

fn hint_without_value(air: Option<u32>) -> pb::Hint {
    pb::Hint {
        name: "broken".to_string(),
        hint_fields: vec![pb::HintField { name: Some("reference".to_string()), value: None }],
        air_group_id: air,
        air_id: air,
    }
}

/// Was `panic!("Unknown hint field")`, for the air's hints and for the global ones.
#[test]
fn a_hint_field_without_a_value_is_refused() {
    let mut pilout = pilout();
    pilout.hints.push(hint_without_value(Some(0)));
    for cfg in cfgs() {
        let err = run_err(&pilout, &cfg);
        assert!(matches!(&err, PilInfoError::HintFieldWithoutValue { hint } if hint == "broken"), "{err:?}");
        assert_eq!(err.to_string(), "invalid pilout: a field of hint `broken` has no value");
    }

    let mut pilout = self::pilout();
    pilout.hints.push(hint_without_value(None));
    let err = build_global_constraints_json(&pilout, &FieldCfg::goldilocks()).unwrap_err();
    assert!(matches!(&err, PilInfoError::HintFieldWithoutValue { hint } if hint == "broken"), "{err:?}");
}

/// Was `panic!("Symbol not found for ev type=const id=0")`: the FRI polynomial needs the stage and
/// dimension of every column it opens, and SHPLONK builds no FRI polynomial.
#[test]
fn an_opened_column_without_a_symbol_is_refused_by_the_fri_opening() {
    let mut pilout = pilout();
    pilout.symbols.retain(|s| s.name != "Synthetic.L1");
    let err = run_err(&pilout, &PilInfoCfg::goldilocks(1));
    assert!(
        matches!(&err, PilInfoError::NoSymbolForEvaluation { entry_type, id: 0 } if entry_type == "const"),
        "{err:?}"
    );
    assert_eq!(err.to_string(), "invalid pilout: the constraints evaluate const 0, which has no symbol");
    run(&pilout, &PilInfoCfg::bn254()).unwrap();
}

fn global_ref(idx: u32) -> Option<pb::GlobalOperand> {
    Some(pb::GlobalOperand { operand: Some(global_operand::Operand::Expression(global_operand::Expression { idx })) })
}

/// Were indices out of range on the global expressions.
#[test]
fn global_references_to_nothing_are_refused() {
    let constant = Some(pb::GlobalOperand {
        operand: Some(global_operand::Operand::Constant(global_operand::Constant { value: vec![1] })),
    });
    let global_constraint =
        |idx| pb::GlobalConstraint { expression_idx: Some(global_operand::Expression { idx }), debug_line: None };

    let mut pilout = pilout();
    pilout.constraints.push(global_constraint(4));
    let err = build_global_constraints_json(&pilout, &FieldCfg::goldilocks()).unwrap_err();
    assert_eq!(invalid_pilout(&err), "constraint 0 of the global constraints is expression 4, and there are 0");

    let mut pilout = self::pilout();
    pilout.expressions.push(pb::GlobalExpression {
        operation: Some(global_expression::Operation::Mul(global_expression::Mul {
            lhs: constant,
            rhs: global_ref(9),
        })),
    });
    pilout.constraints.push(global_constraint(0));
    let err = build_global_constraints_json(&pilout, &FieldCfg::goldilocks()).unwrap_err();
    assert_eq!(invalid_pilout(&err), "global expression 0 refers to global expression 9, and the pilout has 1");
}

/// Was a recursion until the stack overflowed, which aborted the process (plan M26): expressions
/// that refer to each other in a cycle, which no pilout can mean. Refused before any pass runs,
/// whether a constraint reaches the cycle or not, for the air's expressions and the global ones.
#[test]
fn expressions_that_refer_to_each_other_in_a_cycle_are_refused() {
    // Expression 0 is a·exp(2) and the new expression 2 is b·exp(0): the constraint, L1·exp(0),
    // reaches the cycle 0 → 2 → 0.
    let mut pilout = pilout();
    let air = the_air(&mut pilout);
    air.expressions[0] = mul(witness(0), exp(2));
    air.expressions.push(mul(witness(1), exp(0)));
    for cfg in cfgs() {
        let err = run_err(&pilout, &cfg);
        assert_eq!(invalid_pilout(&err), "expression 0 refers to itself, through the references 0 → 2 → 0");
    }

    // An expression that refers to itself, which no constraint reaches.
    let mut pilout = self::pilout();
    the_air(&mut pilout).expressions.push(mul(witness(1), exp(2)));
    for cfg in cfgs() {
        let err = run_err(&pilout, &cfg);
        assert_eq!(invalid_pilout(&err), "expression 2 refers to itself, through the references 2 → 2");
    }

    // References that meet without a cycle are no cycle: 2 and 3 both refer to 0.
    let add = |lhs, rhs| pb::Expression { operation: Some(expression::Operation::Add(expression::Add { lhs, rhs })) };
    let mut pilout = self::pilout();
    let air = the_air(&mut pilout);
    air.expressions.push(add(exp(0), exp(0)));
    air.expressions.push(add(exp(2), exp(0)));
    air.constraints.push(every_row(3));
    for cfg in cfgs() {
        run(&pilout, &cfg).unwrap();
    }

    // The global expressions: 0 → 1 → 0.
    let constant = Some(pb::GlobalOperand {
        operand: Some(global_operand::Operand::Constant(global_operand::Constant { value: vec![1] })),
    });
    let global_mul = |lhs, rhs| pb::GlobalExpression {
        operation: Some(global_expression::Operation::Mul(global_expression::Mul { lhs, rhs })),
    };
    let mut pilout = self::pilout();
    pilout.expressions = vec![global_mul(constant.clone(), global_ref(1)), global_mul(constant, global_ref(0))];
    pilout
        .constraints
        .push(pb::GlobalConstraint { expression_idx: Some(global_operand::Expression { idx: 0 }), debug_line: None });
    let err = build_global_constraints_json(&pilout, &FieldCfg::goldilocks()).unwrap_err();
    assert_eq!(invalid_pilout(&err), "global expression 0 refers to itself, through the references 0 → 1 → 0");
}
