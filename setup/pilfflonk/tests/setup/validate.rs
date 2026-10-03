//! What the setup refuses in a pilout (pilfflonk/docs/README.md#what-the-setup-refuses): one test
//! per error, on pilouts built in code from the valid one of `common`.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, global_expression, global_operand, SymbolType};
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::validate::{
    check_extended_domain, validate, PROVER_HINTS, SUPPORTED_PROVER_HINTS, WITNESS_AND_DEBUG_HINTS,
};
use pilfflonk_setup::SetupError;

use crate::common::*;

/// Goldilocks, `2^64 − 2^32 + 1`.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

fn refusal(pilout: &pb::PilOut) -> SetupError {
    validate(pilout).expect_err("the pilout must be refused")
}

#[test]
fn the_valid_pilout_passes() {
    let pilout = pilout();
    let air = validate(&pilout).unwrap();
    assert_eq!((air.airgroup_id, air.air_id), (0, 0));
    assert_eq!(air.air.name.as_deref(), Some("Sample"));
}

#[test]
fn its_air_is_found_in_any_airgroup() {
    let mut pilout = pilout();
    let air = pilout.air_groups.remove(0).airs.remove(0);
    pilout.air_groups = vec![
        pb::AirGroup { name: Some("Empty".into()), ..Default::default() },
        pb::AirGroup { name: Some("Group".into()), air_group_values: vec![], airs: vec![air] },
    ];
    let air = validate(&pilout).unwrap();
    assert_eq!((air.airgroup_id, air.air_id), (1, 0));
}

// --- The base field ------------------------------------------------------------------------

/// The pilout pil2com writes for `-P bn254.json` when it ignores `prime`, as the pinned compiler
/// does (pilfflonk/docs/README.md#compile-pil): over Goldilocks. The error says how to compile it.
#[test]
fn a_goldilocks_pilout_is_refused() {
    let mut pilout = pilout();
    pilout.base_field = BigUint::from(GOLDILOCKS).to_bytes_be();
    let err = refusal(&pilout);
    assert!(matches!(err, SetupError::GoldilocksPilout), "{err}");
    let message = err.to_string();
    assert!(message.contains("Goldilocks") && message.contains("PIL2C_EXEC") && message.contains("prime"), "{message}");
}

#[test]
fn a_pilout_over_any_other_field_is_refused() {
    let q = big("21888242871839275222246405745257275088696311157297823662689037894645226208583");
    for value in [q, BigUint::ZERO, r() + 1u32, r() - 1u32] {
        let mut pilout = pilout();
        pilout.base_field = be(&value);
        match refusal(&pilout) {
            SetupError::NotBn254 { base_field } => assert_eq!(base_field, value.to_str_radix(10)),
            other => panic!("{other}"),
        }
    }
}

// --- One instance of one AIR (pilfflonk/docs/README.md#scope) ------------------------------

#[test]
fn a_pilout_of_other_than_one_air_is_refused() {
    let one = pilout();
    let air = one.air_groups[0].airs[0].clone();

    let mut none = one.clone();
    none.air_groups[0].airs.clear();
    let mut two_in_a_group = one.clone();
    two_in_a_group.air_groups[0].airs.push(air.clone());
    let mut two_groups = one.clone();
    two_groups.air_groups.push(pb::AirGroup { name: Some("Other".into()), air_group_values: vec![], airs: vec![air] });

    for (pilout, n) in [(none, 0), (two_in_a_group, 2), (two_groups, 2)] {
        let err = refusal(&pilout);
        assert!(matches!(err, SetupError::AirCount { n_airs } if n_airs == n), "{err}");
        assert!(err.to_string().contains("exactly one instance of one AIR"), "{err}");
    }
}

/// Air values, airgroup values, proof values and global constraints are out of v1
/// (pilfflonk/docs/README.md#scope), whether the pilout declares them or only has their symbols.
#[test]
fn values_and_global_constraints_are_refused() {
    let value_symbol = |kind: SymbolType| pb::Symbol {
        name: "v".into(),
        r#type: kind as i32,
        stage: Some(1),
        air_group_id: Some(0),
        air_id: Some(0),
        ..Default::default()
    };

    let mut p = pilout();
    the_air(&mut p).air_values = vec![pb::AirValue { stage: 1 }; 2];
    let err = refusal(&p);
    assert!(matches!(&err, SetupError::AirValues { air, n: 2 } if air == "Sample"), "{err}");
    let mut p = pilout();
    p.symbols.push(value_symbol(SymbolType::AirValue));
    assert!(matches!(refusal(&p), SetupError::AirValues { n: 1, .. }));

    let mut p = pilout();
    p.air_groups[0].air_group_values = vec![pb::AirGroupValue { agg_type: 0, stage: 2 }];
    let err = refusal(&p);
    assert!(matches!(err, SetupError::AirgroupValues { n: 1 }), "{err}");
    let mut p = pilout();
    p.symbols.push(value_symbol(SymbolType::AirGroupValue));
    assert!(matches!(refusal(&p), SetupError::AirgroupValues { n: 1 }));

    let mut p = pilout();
    p.num_proof_values = vec![2, 1];
    let err = refusal(&p);
    assert!(matches!(err, SetupError::ProofValues { n: 3 }), "{err}");
    let mut p = pilout();
    p.symbols.push(value_symbol(SymbolType::ProofValue));
    assert!(matches!(refusal(&p), SetupError::ProofValues { n: 1 }));

    let mut p = pilout();
    p.constraints = vec![pb::GlobalConstraint { expression_idx: None, debug_line: None }; 3];
    let err = refusal(&p);
    assert!(matches!(err, SetupError::GlobalConstraints { n: 3 }), "{err}");
    assert!(err.to_string().contains("which pilfflonk does not support"), "{err}");
}

#[test]
fn an_air_whose_rows_are_not_a_power_of_two_up_to_2_28_is_refused() {
    for rows in [0, 12, 1 << 29] {
        let mut pilout = pilout();
        the_air(&mut pilout).num_rows = Some(rows);
        let err = refusal(&pilout);
        assert!(matches!(err, SetupError::NumRows { num_rows, .. } if num_rows == rows), "{err}");
    }
    let mut pilout = pilout();
    the_air(&mut pilout).num_rows = None;
    assert!(matches!(refusal(&pilout), SetupError::NumRows { num_rows: 0, .. }));
}

// --- Custom commits, periodic columns, public tables ---------------------------------------

#[test]
fn custom_commits_are_refused() {
    let mut pilout = pilout();
    the_air(&mut pilout).custom_commits =
        vec![pb::CustomCommit { name: Some("rom".into()), public_values: vec![], stage_widths: vec![1] }];
    let err = refusal(&pilout);
    assert!(matches!(&err, SetupError::CustomCommits { air, n: 1 } if air == "Sample"), "{err}");
}

#[test]
fn periodic_columns_are_refused() {
    let mut pilout = pilout();
    the_air(&mut pilout).periodic_cols = vec![pb::PeriodicCol { values: vec![vec![1], vec![]] }];
    let err = refusal(&pilout);
    assert!(matches!(&err, SetupError::PeriodicColumns { air, n: 1 } if air == "Sample"), "{err}");
}

#[test]
fn public_tables_are_refused() {
    let mut pilout = pilout();
    pilout.public_tables = vec![pb::PublicTable { num_cols: 1, max_rows: 4, agg_type: 0, row_expression_idx: None }; 2];
    let err = refusal(&pilout);
    assert!(matches!(err, SetupError::PublicTables { n: 2 }), "{err}");
}

// --- Hints ---------------------------------------------------------------------------------

/// The prover hints the setup supports (pilfflonk/docs/README.md#what-the-setup-refuses), `im_col`,
/// `gprod_col` and `gsum_col`, pass by name if they are of the AIR: what they give,
/// `check_prover_hints` checks after the passes (`tests/bytecode.rs`). Of the pilout, they are of
/// no AIR.
#[test]
fn the_supported_prover_hints_pass_by_name_if_they_are_of_the_air() {
    assert_eq!(SUPPORTED_PROVER_HINTS, ["im_col", "gprod_col", "gsum_col"], "the STARK's order");
    for name in SUPPORTED_PROVER_HINTS {
        assert!(PROVER_HINTS.contains(&name));
        let mut pilout = pilout();
        pilout.hints = vec![hint("range_def", true), hint(name, true)];
        validate(&pilout).unwrap();
        pilout.hints = vec![hint(name, false)];
        let err = refusal(&pilout);
        assert!(matches!(&err, SetupError::InvalidPilout(m) if m.contains("of no air")), "{name}: {err}");
    }
}

/// The other prover hint, `im_airval`, is refused, of the AIR or of the pilout, saying why: it
/// computes an air value (pilfflonk/docs/README.md#scope).
#[test]
fn the_other_prover_hints_are_refused() {
    for (of_air, where_) in [(true, "air Sample"), (false, "the pilout")] {
        let mut pilout = pilout();
        pilout.hints = vec![hint("range_def", true), hint("im_airval", of_air)];
        let err = refusal(&pilout);
        assert!(matches!(&err, SetupError::ImAirvalHint { location } if location == where_), "{err}");
        assert!(err.to_string().contains("which computes an air value, and pilfflonk has none"), "{err}");
    }
    assert_eq!(PROVER_HINTS.len(), SUPPORTED_PROVER_HINTS.len() + 1);
    assert!(!SUPPORTED_PROVER_HINTS.contains(&"im_airval"));
}

/// The hints are checked before the values, so that `im_airval` is told about rather than the air
/// value it computes.
#[test]
fn a_prover_hint_is_reported_before_the_values_it_brings() {
    let mut pilout = pilout();
    the_air(&mut pilout).air_values = vec![pb::AirValue { stage: 2 }];
    pilout.hints = vec![hint("im_airval", true)];
    assert!(matches!(refusal(&pilout), SetupError::ImAirvalHint { .. }));
    pilout.hints.clear();
    assert!(matches!(refusal(&pilout), SetupError::AirValues { n: 1, .. }));
}

/// `witness_bits`, the hint of a column declared with `bits(n)`, asks for packed trace rows, which
/// pilfflonk does not accept yet (pilfflonk/docs/README.md#scope): it is refused, saying so, of the
/// AIR or of the pilout.
#[test]
fn a_packed_trace_is_refused() {
    for (of_air, where_) in [(true, "air Sample"), (false, "the pilout")] {
        let mut pilout = pilout();
        pilout.hints = vec![hint("range_def", true), hint("witness_bits", of_air)];
        let err = refusal(&pilout);
        assert!(matches!(&err, SetupError::PackedTrace { location } if location == where_), "{err}");
        assert!(err.to_string().contains("pilfflonk does not accept packed traces yet"), "{err}");
    }
    assert!(!PROVER_HINTS.contains(&"witness_bits") && !WITNESS_AND_DEBUG_HINTS.contains(&"witness_bits"));
}

#[test]
fn unknown_hints_are_refused() {
    let mut pilout = pilout();
    pilout.hints = vec![hint("gsum_debug_data", true), hint("my_hint", true)];
    let err = refusal(&pilout);
    assert!(matches!(&err, SetupError::UnknownHint { name, .. } if name == "my_hint"), "{err}");
}

/// The witness and debug hints (pilfflonk/docs/README.md#what-the-setup-refuses) are not the
/// prover's: the setup ignores them.
#[test]
fn witness_and_debug_hints_are_ignored() {
    let mut pilout = pilout();
    pilout.hints = WITNESS_AND_DEBUG_HINTS.iter().flat_map(|name| [hint(name, true), hint(name, false)]).collect();
    assert_eq!(pilout.hints.len(), 24);
    validate(&pilout).unwrap();
}

#[test]
fn a_hint_of_an_air_the_pilout_does_not_have_is_refused() {
    let mut pilout = pilout();
    let mut stray = hint("gsum_col", true);
    stray.air_id = Some(3);
    pilout.hints = vec![stray];
    assert!(matches!(refusal(&pilout), SetupError::InvalidPilout(_)));
}

// --- Columns of stage 2 or above -----------------------------------------------------------

/// Stage-2 columns come from the prover hints of the std's buses: without any, they are refused.
#[test]
fn columns_of_stage_2_or_above_without_a_prover_hint_are_refused() {
    for (widths, stage, n) in [(vec![2, 1], 2, 1), (vec![2, 0, 3], 3, 3)] {
        let mut pilout = pilout();
        the_air(&mut pilout).stage_widths = widths;
        pilout.num_challenges = vec![0, 2, 1];
        let err = refusal(&pilout);
        assert!(
            matches!(err, SetupError::StageWithoutHint { stage: s, n_columns, .. } if s == stage && n_columns == n),
            "{err}"
        );
    }
    // An empty stage 2 is no column.
    let mut pilout = pilout();
    the_air(&mut pilout).stage_widths = vec![2, 0];
    validate(&pilout).unwrap();
}

/// With a supported hint, which columns it gives is `check_prover_hints`'s to say; an unsupported
/// one is reported before the columns it would give.
#[test]
fn a_prover_hint_is_reported_before_the_columns_it_would_produce() {
    let mut pilout = pilout();
    the_air(&mut pilout).stage_widths = vec![2, 1];
    pilout.num_challenges = vec![0, 2];
    for name in SUPPORTED_PROVER_HINTS {
        pilout.hints = vec![hint(name, true)];
        validate(&pilout).unwrap();
    }
    pilout.hints = vec![hint("im_airval", true)];
    assert!(matches!(refusal(&pilout), SetupError::ImAirvalHint { .. }));
}

// --- Constants below r ---------------------------------------------------------------------

#[test]
fn a_constant_of_an_expression_not_below_r_is_refused() {
    for value in [r(), r() + 1u32, BigUint::from(1u32) << 256] {
        let mut pilout = pilout();
        the_air(&mut pilout).expressions[1] = mul(exp(0), constant(&value));
        match refusal(&pilout) {
            SetupError::ConstantNotBelowR { location, value: shown } => {
                assert_eq!(shown, value.to_str_radix(10));
                assert!(location.contains("expression 1 of air Sample"), "{location}");
            }
            other => panic!("{other}"),
        }
    }
    // The negation of a constant, too: every operand is looked at.
    let mut pilout = pilout();
    the_air(&mut pilout).expressions.push(neg(constant(&r())));
    assert!(matches!(refusal(&pilout), SetupError::ConstantNotBelowR { .. }));
}

#[test]
fn a_constant_of_a_global_expression_not_below_r_is_refused() {
    let constant = |value: &BigUint| {
        Some(pb::GlobalOperand {
            operand: Some(global_operand::Operand::Constant(global_operand::Constant { value: be(value) })),
        })
    };
    let add = |lhs, rhs| pb::GlobalExpression {
        operation: Some(global_expression::Operation::Add(global_expression::Add { lhs, rhs })),
    };
    let mut pilout = pilout();
    pilout.expressions = vec![add(constant(&big(R_MINUS_ONE)), constant(&big(WIDE)))];
    validate(&pilout).unwrap();
    pilout.expressions.push(add(constant(&BigUint::from(1u32)), constant(&r())));
    match refusal(&pilout) {
        SetupError::ConstantNotBelowR { location, .. } => {
            assert!(location.contains("global expression 1"), "{location}")
        }
        other => panic!("{other}"),
    }
}

/// The fixed values are refused where they are decoded, as `<air>.const` is written.
#[test]
fn a_fixed_value_not_below_r_is_refused() {
    let mut pilout = pilout();
    let mut values = fixed_values();
    values[2][5] = r();
    the_air(&mut pilout).fixed_cols[2] = fixed_column(&values[2]);
    let err = FixedColumns::from_air(&pilout.air_groups[0].airs[0]).unwrap_err();
    match err {
        SetupError::ConstantNotBelowR { location, value } => {
            assert_eq!(location, "row 5 of fixed column 2");
            assert_eq!(value, r().to_str_radix(10));
        }
        other => panic!("{other}"),
    }
}

// --- What the passes and the layout decide -------------------------------------------------

#[test]
fn an_extended_domain_beyond_the_2_adicity_is_refused() {
    for n_bits_ext in [1, 25, 28] {
        check_extended_domain(n_bits_ext).unwrap();
    }
    let err = check_extended_domain(29).unwrap_err();
    assert!(matches!(err, SetupError::ExtendedDomain { n_bits_ext: 29 }), "{err}");
}

// The ptau with fewer powers than the layout's largest degree: `keys.rs`,
// `a_ptau_with_too_few_powers_is_refused`.
