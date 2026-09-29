//! The committed polynomials, their bounds, `nBitsExt` and the unpacked layout (spec §4.2.4,
//! A.1–A.3, plan R1), and the pilfflonkinfo they go into, from the passes run on pilouts built in
//! code: offsets `{−1, 0, 1, 2}`, im pols, a column never opened, `qDeg = 0`.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, SymbolType};
use pilfflonk_setup::air_info::{air_setup, q_piece_name, AirRef, AirSetup};
use pilfflonk_setup::global_info::global_info;
use pilfflonk_setup::layout::{committed_pols, max_degree, unpacked_layout, Committed, CommittedPol, Degrees};
use pilfflonk_setup::passes::run_passes;
use pilfflonk_setup::validate::validate;
use pilfflonk_setup::SetupError;
use proofman_pilfflonk::{EvMapEntry, JsonFile, PilfflonkInfo, PolMapEntry, PolType, ProofNames, SetupParams, WitnessShape};

use crate::common::*;

/// The passes and [`air_setup`] on the one AIR of `pilout`, searching degrees 2 to `max_degree`.
fn setup_air(pilout: &pb::PilOut, max_degree: u64) -> Result<AirSetup, SetupError> {
    let air = validate(pilout)?;
    let result = run_passes(pilout, air, max_degree)?;
    let name = air.air.name.clone().unwrap();
    air_setup(&result, AirRef { name: &name, airgroup_id: 0, air_id: 0 }, air.air, 0)
}

/// What the prover reads of a pilfflonkinfo must accept it: the names of the proof and the shape
/// of the witness (M12, M13).
fn check_readers(pilout: &pb::PilOut, info: &PilfflonkInfo) -> WitnessShape {
    let params = SetupParams { max_constraint_degree: 9, extra_muls: 0, max_q_degree: 0, packing: false };
    let gi = global_info(pilout, params).unwrap();
    ProofNames::new(&gi, &[info]).unwrap();
    // The file reads back as it was written.
    assert_eq!(&PilfflonkInfo::from_json_str(&info.to_json_string().unwrap()).unwrap(), info);
    WitnessShape::from_proving_key(&gi, &[info]).unwrap()
}

/// `(stage, id, name, offsets, degree)` of each `f` of the layout, which has `k = 1`.
fn layout(info: &PilfflonkInfo) -> Vec<(u64, u64, String, Vec<i64>, u64)> {
    info.layout
        .0
        .iter()
        .map(|f| {
            assert_eq!((f.k, f.pols.len()), (1, 1));
            (f.stage, f.pols[0].id, f.pols[0].name.clone(), f.offsets.clone(), f.degree)
        })
        .collect()
}

/// `L1`: 1 at the first of `rows` rows.
fn first_row_column(rows: u32) -> pb::FixedCol {
    fixed_column(&(0..rows).map(|i| BigUint::from(u32::from(i == 0))).collect::<Vec<_>>())
}

/// An AIR of `2^4` rows, witness columns `a`, `b` and `u` and the fixed `L1`, with the constraints
/// `a(−1)·a·a'·a(2) − b` (degree 4) and `L1·(b' − a)`. `u` is in none.
fn offsets_pilout() -> pb::PilOut {
    let rows = 16;
    let air = pb::Air {
        name: Some("Offsets".into()),
        num_rows: Some(rows),
        fixed_cols: vec![first_row_column(rows)],
        stage_widths: vec![3],
        expressions: vec![
            mul(witness(0, -1), witness(0, 0)), // 0
            mul(witness(0, 1), witness(0, 2)),  // 1
            mul(exp(0), exp(1)),                // 2: degree 4
            sub(exp(2), witness(1, 0)),         // 3
            sub(witness(1, 1), witness(0, 0)),  // 4
            mul(fixed(0), exp(4)),              // 5
        ],
        constraints: vec![every_row(3), every_row(5)],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("offsets".into()),
        base_field: r().to_bytes_be(),
        air_groups: vec![pb::AirGroup { name: Some("Offsets".into()), air_group_values: vec![], airs: vec![air] }],
        num_challenges: vec![0],
        symbols: vec![
            symbol("Offsets.L1", SymbolType::FixedCol, 0, Some(0), true),
            symbol("Offsets.a", SymbolType::WitnessCol, 0, Some(1), true),
            symbol("Offsets.b", SymbolType::WitnessCol, 1, Some(1), true),
            symbol("Offsets.u", SymbolType::WitnessCol, 2, Some(1), true),
        ],
        ..Default::default()
    }
}

/// Offsets `{−1, 0, 1, 2}`, and a degree-4 constraint that the search of D5 (2 to 9, the lowest
/// degree on a tie) brings down to degree 2 with two im pols: `qDeg = 1`.
#[test]
fn signed_offsets_and_im_pols_have_their_bounds() {
    let pilout = offsets_pilout();
    let AirSetup { info, committed } = setup_air(&pilout, 9).unwrap();
    assert_eq!(info.opening_points, [-1, 0, 1, 2]);
    assert_eq!(info.q_deg, 1);

    // The im pols are the array Offsets.ImPol, after the pilout's columns in stage 1.
    let im_pols: Vec<&PolMapEntry> = info.cm_pols_map.iter().filter(|p| p.im_pol).collect();
    assert_eq!(im_pols.len(), 2);
    for (k, p) in im_pols.iter().enumerate() {
        assert_eq!(
            (p.name.as_str(), p.lengths.as_slice(), p.stage, p.stage_id),
            ("Offsets.ImPol", &[k as u64][..], 1, 3 + k as u64)
        );
    }

    // N = 16, |O|_max = 4 (a): a has 16 + 4 + 1 coefficients, b 16 + 2 + 1, an im pol 16 + 1 + 1,
    // L1 16, and Q 1·16 + 2·4 + 1 = 25; the extended domain holds 25: 2^5.
    assert_eq!(committed.degrees, Degrees { n_bits: 4, max_openings: 4, q_coefficients: 25, n_bits_ext: 5 });
    let im = |k: usize| im_pols[k].pols_map_id;
    let s = |name: &str| name.to_string();
    assert_eq!(
        layout(&info),
        [
            (0, 0, s("Offsets.L1"), vec![0], 16),
            (1, 0, s("Offsets.a"), vec![-1, 0, 1, 2], 21),
            (1, 1, s("Offsets.b"), vec![0, 1], 19),
            (1, im(0), s("Offsets.ImPol[0]"), vec![0], 18),
            (1, im(1), s("Offsets.ImPol[1]"), vec![0], 18),
            (2, 5, q_piece_name(0), vec![0], 25),
        ]
    );
    assert_eq!(max_degree(&info.layout), 25);

    // u is in no constraint: it is not committed, but it is still a column of the witness.
    assert_eq!(committed.unopened, ["Offsets.u"]);
    assert!(info.cm_pols_map.iter().any(|p| p.name == "Offsets.u" && p.stage_id == 2));
    let shape = check_readers(&pilout, &info);
    assert_eq!(shape.airs()[0].n_cols, 3);
}

/// Searching degrees up to 4 only keeps the lowest one on the tie; up to 2, it is forced.
#[test]
fn the_degree_search_is_bounded_by_the_max_constraint_degree() {
    let pilout = offsets_pilout();
    for d in [2, 4] {
        let AirSetup { info, .. } = setup_air(&pilout, d).unwrap();
        assert_eq!((info.q_deg, info.cm_pols_map.iter().filter(|p| p.im_pol).count()), (1, 2), "D = {d}");
    }
}

/// The same constraints with more rows: the bounds scale with `N`, and `Q` needs `2N` points.
#[test]
fn the_bounds_scale_with_the_rows() {
    let mut pilout = offsets_pilout();
    let air = the_air(&mut pilout);
    air.num_rows = Some(1 << 10);
    air.fixed_cols = vec![first_row_column(1 << 10)];
    let AirSetup { info, committed } = setup_air(&pilout, 9).unwrap();
    assert_eq!(
        committed.degrees,
        Degrees { n_bits: 10, max_openings: 4, q_coefficients: 1024 + 2 * 4 + 1, n_bits_ext: 11 }
    );
    assert_eq!(info.layout.0.iter().map(|f| f.degree).collect::<Vec<_>>(), [1024, 1029, 1027, 1026, 1026, 1033]);
}

/// A constraint of degree 1 gives `qDeg = 0`: `pil-info` has no piece of `Q`, the setup its `Q0`,
/// of `|O|_max + 1` coefficients, and the extended domain is decided by the columns (M5).
#[test]
fn linear_constraints_give_q_deg_0() {
    let mut pilout = offsets_pilout();
    let air = the_air(&mut pilout);
    air.expressions = vec![sub(witness(0, 1), witness(1, 0))];
    air.constraints = vec![every_row(0)];
    let AirSetup { info, committed } = setup_air(&pilout, 9).unwrap();
    assert_eq!(info.q_deg, 0);
    assert_eq!(committed.degrees, Degrees { n_bits: 4, max_openings: 1, q_coefficients: 2, n_bits_ext: 5 });
    let s = |name: &str| name.to_string();
    assert_eq!(
        layout(&info),
        [(1, 0, s("Offsets.a"), vec![1], 18), (1, 1, s("Offsets.b"), vec![0], 18), (2, 3, s("Q0"), vec![0], 2)]
    );
    assert_eq!(committed.unopened, ["Offsets.L1", "Offsets.u"]);
    assert_eq!(info.map_sections_n.get("cm2"), Some(&1));
    check_readers(&pilout, &info);
}

/// Constraints on no column give `qDeg = −1` (A.1): there is no `Q` to commit to.
#[test]
fn constraints_on_no_column_are_refused() {
    let mut pilout = offsets_pilout();
    let air = the_air(&mut pilout);
    air.expressions = vec![sub(constant(&BigUint::from(3u32)), constant(&BigUint::from(3u32)))];
    air.constraints = vec![every_row(0)];
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(matches!(err, SetupError::QDegree(-1)), "{err}");
}

/// A column of the pilout without its symbol cannot be mapped: `pil-info`'s maps come from the
/// symbols.
#[test]
fn every_column_needs_its_symbol() {
    let mut pilout = offsets_pilout();
    pilout.symbols.retain(|s| s.name != "Offsets.L1");
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(matches!(&err, SetupError::InvalidPilout(m) if m.contains("1 fixed columns, and symbols for 0")), "{err}");

    let mut pilout = offsets_pilout();
    pilout.symbols.retain(|s| s.name != "Offsets.u");
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(
        matches!(&err, SetupError::InvalidPilout(m) if m.contains("3 witness columns, and symbols for 2")),
        "{err}"
    );
}

/// The passes recurse over the expressions: a chain of 2000 of them overflows the 2 MiB stack of
/// a test thread (in a debug build it does from about 1000), and runs on the passes' own.
#[test]
fn deep_expressions_run_on_the_stack_of_the_passes() {
    let depth = 2000u32;
    let mut pilout = offsets_pilout();
    let air = the_air(&mut pilout);
    air.expressions = vec![sub(witness(0, 0), witness(1, 0))];
    air.expressions.extend((1..depth).map(|i| sub(exp(i - 1), witness(i % 2, 0))));
    air.constraints = vec![every_row(depth - 1)];
    let AirSetup { info, .. } = setup_air(&pilout, 9).unwrap();
    assert_eq!(info.q_deg, 0);
}

/// A panic of the passes (plan R6) is an error of the setup, with its message.
#[test]
fn a_panic_of_the_passes_is_an_error() {
    let mut pilout = offsets_pilout();
    the_air(&mut pilout).constraints = vec![every_row(99)];
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(matches!(&err, SetupError::Passes(m) if !m.is_empty()), "{err}");
    assert!(err.to_string().starts_with("the symbolic passes (pil-info) failed: "), "{err}");
}

// ---------------------------------------------------------------------------------------------
// committed_pols and unpacked_layout on maps built by hand
// ---------------------------------------------------------------------------------------------

fn pol(stage: u64, name: &str, id: u64) -> PolMapEntry {
    PolMapEntry {
        stage,
        name: name.into(),
        dim: 1,
        pols_map_id: id,
        stage_id: id,
        lengths: vec![],
        im_pol: false,
        exp_id: None,
        stage_pos: id,
    }
}

fn ev(pol_type: PolType, id: u64, prime: i64) -> EvMapEntry {
    EvMapEntry { pol_type, id, prime, opening_pos: 0 }
}

fn committed(cm: &[PolMapEntry], ev_map: &[EvMapEntry]) -> Result<Committed, SetupError> {
    let consts = [pol(0, "F0", 0), pol(0, "F1", 1)];
    committed_pols(3, 2, 2, &consts, cm, ev_map)
}

#[test]
fn the_layout_goes_by_stage_and_index_with_q_last() {
    let cm = [pol(1, "a", 0), pol(1, "b", 1), pol(2, "Q0", 2), pol(1, "c", 3)];
    let ev_map = [
        ev(PolType::Cm, 3, 2),
        ev(PolType::Cm, 3, -1),
        ev(PolType::Const, 1, 0),
        ev(PolType::Cm, 0, 0),
        ev(PolType::Cm, 3, 0),
        ev(PolType::Cm, 3, 1),
        ev(PolType::Cm, 0, 1),
    ];
    let c = committed(&cm, &ev_map).unwrap();
    // N = 8, qDeg = 2, |O|_max = 4 (c): Q has 2·8 + 3·4 + 1 = 29, the domain 32.
    assert_eq!(c.degrees, Degrees { n_bits: 3, max_openings: 4, q_coefficients: 29, n_bits_ext: 5 });
    let pols = |stage, id, name: &str, offsets: &[i64], coefficients| CommittedPol {
        stage,
        id,
        name: name.into(),
        offsets: offsets.to_vec(),
        coefficients,
    };
    assert_eq!(
        c.pols,
        [
            pols(0, 1, "F1", &[0], 8),
            pols(1, 0, "a", &[0, 1], 11),
            pols(1, 3, "c", &[-1, 0, 1, 2], 13),
            pols(2, 2, "Q0", &[0], 29),
        ]
    );
    assert_eq!(c.unopened, ["F0", "b"]);
    let layout = unpacked_layout(&c.pols);
    assert_eq!(layout.0.len(), 4);
    assert!(layout.0.iter().zip(&c.pols).all(|(f, p)| f.stage == p.stage
        && f.k == 1
        && f.pols.len() == 1
        && f.pols[0].id == p.id
        && f.pols[0].name == p.name
        && f.offsets == p.offsets
        && f.degree == p.coefficients));
    assert_eq!(layout.power_w().unwrap(), 1);
    assert_eq!(max_degree(&layout), 29);
}

#[test]
fn what_the_layout_cannot_hold_is_refused() {
    let cm = [pol(1, "a", 0), pol(2, "Q0", 1)];
    let passes_output = |r: Result<Committed, SetupError>, what: &str| match r {
        Err(SetupError::PassesOutput(m)) => assert!(m.contains(what), "{m}"),
        other => panic!("{other:?}"),
    };
    // An evaluation of Q, which the verifier computes, and of a column that is not there.
    passes_output(committed(&cm, &[ev(PolType::Cm, 1, 0)]), "a piece of Q");
    passes_output(committed(&cm, &[ev(PolType::Cm, 2, 0)]), "not in its pol map");
    passes_output(committed(&cm, &[ev(PolType::Const, 2, 0)]), "not in its pol map");
    // Q not split is one piece.
    passes_output(committed(&cm[..1], &[]), "0 pieces");
    passes_output(committed(&[pol(1, "a", 0), pol(2, "Q0", 1), pol(2, "Q1", 2)], &[]), "2 pieces");
}
