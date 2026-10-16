//! What the passes refuse: a pilout that breaks its own format, or constraints they cannot
//! process, is a `PilInfoError` the setups report, where the passes used to `panic!`. The pilouts
//! are built in code, each one well formed but for one piece.

use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
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

/// The STARK's passes, at a blowup of 2.
fn cfg() -> PilInfoCfg {
    PilInfoCfg::goldilocks(1)
}

fn run(pilout: &pb::PilOut) -> Result<PilInfoResult, PilInfoError> {
    pil_info::run(pilout, 0, 0, &cfg(), &Default::default())
}

fn run_err(pilout: &pb::PilOut) -> PilInfoError {
    match run(pilout) {
        Ok(_) => panic!("the passes accepted the pilout"),
        Err(err) => err,
    }
}

#[test]
fn the_pilout_as_built_is_accepted() {
    run(&pilout()).unwrap();
    build_global_constraints_json(&pilout(), &FieldCfg::goldilocks()).unwrap();
}

/// Was `panic!("Invalid stage {} for a custom commit")`.
#[test]
fn a_custom_column_of_a_stage_other_than_0_is_refused() {
    let mut pilout = pilout();
    pilout.symbols.push(pb::Symbol { commit_id: Some(0), ..column_symbol("Synthetic.c", SymbolType::CustomCol, 1, 0) });
    let err = run_err(&pilout);
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
    let err = run_err(&pilout);
    assert!(matches!(&err, PilInfoError::HintFieldWithoutValue { hint } if hint == "broken"), "{err:?}");
    assert_eq!(err.to_string(), "invalid pilout: a field of hint `broken` has no value");

    let mut pilout = self::pilout();
    pilout.hints.push(hint_without_value(None));
    let err = build_global_constraints_json(&pilout, &FieldCfg::goldilocks()).unwrap_err();
    assert!(matches!(&err, PilInfoError::HintFieldWithoutValue { hint } if hint == "broken"), "{err:?}");
}

/// Was `panic!("Symbol not found for ev type=const id=0")`: the DEEP polynomial needs the stage and
/// dimension of every column it opens.
#[test]
fn an_opened_column_without_a_symbol_is_refused() {
    let mut pilout = pilout();
    pilout.symbols.retain(|s| s.name != "Synthetic.L1");
    let err = run_err(&pilout);
    assert!(
        matches!(&err, PilInfoError::NoSymbolForEvaluation { entry_type, id: 0 } if entry_type == "const"),
        "{err:?}"
    );
    assert_eq!(err.to_string(), "invalid pilout: the constraints evaluate const 0, which has no symbol");
}
