//! The committed polynomials, their bounds, `nBitsExt` and their layout, unpacked (spec §4.2.4,
//! A.1–A.3, plan R1) and grouped (A.2, plan M22), and the pilfflonkinfo they go into, from the
//! passes run on pilouts built in code: offsets `{−1, 0, 1, 2}`, im pols, a column never opened,
//! `qDeg = 0`, fusions and the evaluations they add to the evMap.

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, SymbolType};
use pil_info::PilInfoError;
use pilfflonk_setup::air_info::{air_setup, q_piece_name, AirRef, AirSetup};
use pilfflonk_setup::global_info::global_info;
use pilfflonk_setup::grouping::GroupingError;
use pilfflonk_setup::layout::{
    committed_pols, ev_map_of, max_degree, unpacked_layout, Committed, CommittedPol, Degrees, Packing, QShape, QSplit,
};
use pilfflonk_setup::passes::run_passes;
use pilfflonk_setup::validate::validate;
use pilfflonk_setup::SetupError;
use proofman_pilfflonk::{
    EvMapEntry, JsonFile, Layout, LayoutEntry, LayoutPol, PilfflonkInfo, PolMapEntry, PolType, ProofNames, SetupParams,
    WitnessShape,
};

use crate::common::*;

/// The passes and [`air_setup`] on the one AIR of `pilout`, searching degrees 2 to `max_degree`,
/// laid out as `packing` says, with `Q` split by `--max-q-degree max_q_degree`.
fn setup_air_split(
    pilout: &pb::PilOut,
    max_degree: u64,
    packing: Packing,
    max_q_degree: u64,
) -> Result<AirSetup, SetupError> {
    let air = validate(pilout)?;
    let result = run_passes(pilout, air, max_degree)?;
    let name = air.air.name.clone().unwrap();
    air_setup(&result, AirRef { name: &name, airgroup_id: 0, air_id: 0 }, air.air, max_q_degree, packing)
}

/// [`setup_air_split`] with `Q` whole.
fn setup_air_with(pilout: &pb::PilOut, max_degree: u64, packing: Packing) -> Result<AirSetup, SetupError> {
    setup_air_split(pilout, max_degree, packing, 0)
}

/// [`setup_air_with`] unpacked, `--no-packing`.
fn setup_air(pilout: &pb::PilOut, max_degree: u64) -> Result<AirSetup, SetupError> {
    setup_air_with(pilout, max_degree, Packing::Unpacked)
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

/// What the passes refuse (M27: they return it rather than panic) is an error of the setup,
/// with the passes' own error: here a constraint on an expression the air does not have.
#[test]
fn what_the_passes_refuse_is_an_error() {
    let mut pilout = offsets_pilout();
    the_air(&mut pilout).constraints = vec![every_row(99)];
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(
        matches!(&err, SetupError::Passes(PilInfoError::InvalidPilout(m))
            if m == "constraint 0 of air Offsets is expression 99, and there are 6"),
        "{err}"
    );
    assert!(err.to_string().starts_with("the symbolic passes (pil-info) failed: invalid pilout: "), "{err}");
}

/// Expressions that refer to each other in a cycle (plan M26) are refused by the passes with an
/// error, where they used to recurse until the stack overflowed and the process aborted.
#[test]
fn expressions_in_a_cycle_are_an_error_not_a_stack_overflow() {
    let mut pilout = offsets_pilout();
    let air = the_air(&mut pilout);
    air.expressions.push(sub(exp(7), witness(0, 0))); // 6
    air.expressions.push(mul(fixed(0), exp(6))); // 7
    air.constraints.push(every_row(7));
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(
        matches!(&err, SetupError::Passes(PilInfoError::InvalidPilout(m))
            if m == "expression 6 refers to itself, through the references 6 → 7 → 6"),
        "{err}"
    );
}

/// An AIR of `2^4` rows whose columns share names, as the std's sum bus declares its `im_cluster`
/// and `im_single` in a loop (plan M34b): the fixed `F` twice, and the witness `a`, `x` twice and
/// `u` twice, with the constraints `x·x − a'` and `F·(a − F)`, the second `F` the second column.
/// The `u` are in none.
fn alike_pilout() -> pb::PilOut {
    let rows = 16;
    let air = pb::Air {
        name: Some("Alike".into()),
        num_rows: Some(rows),
        fixed_cols: vec![first_row_column(rows), first_row_column(rows)],
        stage_widths: vec![5],
        expressions: vec![
            mul(witness(1, 0), witness(2, 0)), // 0
            sub(exp(0), witness(0, 1)),        // 1
            sub(witness(0, 0), fixed(1)),      // 2
            mul(fixed(0), exp(2)),             // 3
        ],
        constraints: vec![every_row(1), every_row(3)],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("alike".into()),
        base_field: r().to_bytes_be(),
        air_groups: vec![pb::AirGroup { name: Some("Alike".into()), air_group_values: vec![], airs: vec![air] }],
        num_challenges: vec![0],
        symbols: vec![
            symbol("Alike.F", SymbolType::FixedCol, 0, Some(0), true),
            symbol("Alike.F", SymbolType::FixedCol, 1, Some(0), true),
            symbol("a", SymbolType::WitnessCol, 0, Some(1), true),
            symbol("x", SymbolType::WitnessCol, 1, Some(1), true),
            symbol("x", SymbolType::WitnessCol, 2, Some(1), true),
            symbol("u", SymbolType::WitnessCol, 3, Some(1), true),
            symbol("u", SymbolType::WitnessCol, 4, Some(1), true),
        ],
        ..Default::default()
    }
}

/// The columns of a pol map that share a name and have no `lengths` are the array of that name
/// (spec A.6, plan M34b), as the im pols are: the `k`-th in the map is `<name>[k]` in the pol map,
/// the layout and the proof, the fixed columns as the committed ones, and those not committed too.
/// Grouped or not, twice the same pilfflonkinfo.
#[test]
fn columns_named_alike_are_indexed_in_the_order_of_the_map() {
    let pilout = alike_pilout();
    let AirSetup { info, committed } = setup_air(&pilout, 9).unwrap();
    let names = |map: &[PolMapEntry]| -> Vec<(String, Vec<u64>)> {
        map.iter().map(|p| (p.name.clone(), p.lengths.clone())).collect()
    };
    let s = |name: &str, lengths: &[u64]| (name.to_string(), lengths.to_vec());
    assert_eq!(names(&info.const_pols_map), [s("Alike.F", &[0]), s("Alike.F", &[1])]);
    assert_eq!(
        names(&info.cm_pols_map),
        [s("a", &[]), s("x", &[0]), s("x", &[1]), s("u", &[0]), s("u", &[1]), s("Q0", &[])]
    );
    let in_layout: Vec<&str> = info.layout.0.iter().flat_map(|f| f.pols.iter().map(|p| p.name.as_str())).collect();
    assert_eq!(in_layout, ["Alike.F[0]", "Alike.F[1]", "a", "x[0]", "x[1]", "Q0"]);
    assert_eq!(committed.unopened, ["u[0]", "u[1]"]);

    let params = SetupParams { max_constraint_degree: 9, extra_muls: 0, max_q_degree: 0, packing: false };
    let names = ProofNames::new(&global_info(&pilout, params).unwrap(), &[&info]).unwrap();
    assert_eq!(names.evaluations(), ["Alike.F[0]", "Alike.F[1]", "a", "x[0]", "x[1]", "aw"]);
    check_readers(&pilout, &info);

    let grouped = setup_air_with(&pilout, 9, Packing::Grouped { extra_muls: 0 }).unwrap().info;
    assert_eq!((&grouped.const_pols_map, &grouped.cm_pols_map), (&info.const_pols_map, &info.cm_pols_map));
    let in_layout: Vec<&str> = grouped.layout.0.iter().flat_map(|f| f.pols.iter().map(|p| p.name.as_str())).collect();
    assert!(["x[0]", "x[1]"].iter().all(|name| in_layout.contains(name)), "{in_layout:?}");
    check_readers(&pilout, &grouped);

    let again = setup_air(&pilout, 9).unwrap().info;
    assert_eq!(again.to_json_string().unwrap(), info.to_json_string().unwrap());
}

/// Names that still collide once the columns named alike are indexed are refused, grouped or not,
/// before the layout: a column named as the setup names another (`x[0]` and the two `x`), two arrays
/// of the same name (their entries have `lengths` already), and a name that one column has with
/// `lengths` and two without (the setup indexes a name only when no column that has it has any).
#[test]
fn names_that_still_collide_are_refused() {
    let with_symbols = |symbols: &[(&str, u32, u32)]| {
        let mut pilout = alike_pilout();
        pilout.symbols.retain(|s| s.r#type != SymbolType::WitnessCol as i32);
        for &(name, id, length) in symbols {
            let column = match length {
                0 => symbol(name, SymbolType::WitnessCol, id, Some(1), true),
                length => array_symbol(name, SymbolType::WitnessCol, id, 1, length),
            };
            pilout.symbols.push(column);
        }
        pilout
    };
    let named_as_indexed = with_symbols(&[("x[0]", 0, 0), ("x", 1, 0), ("x", 2, 0), ("u", 3, 0), ("u", 4, 0)]);
    let two_arrays = with_symbols(&[("a", 0, 0), ("v", 1, 2), ("v", 3, 2)]);
    let an_array_and_two_alike = with_symbols(&[("a", 0, 0), ("x", 1, 0), ("x", 2, 0), ("x", 3, 2)]);
    for (pilout, name, first, second) in [
        (named_as_indexed, "x[0]", "cmPolsMap[0]", "cmPolsMap[1]"),
        (two_arrays, "v[0]", "cmPolsMap[1]", "cmPolsMap[3]"),
        (an_array_and_two_alike, "x", "cmPolsMap[1]", "cmPolsMap[2]"),
    ] {
        for packing in [Packing::Unpacked, Packing::Grouped { extra_muls: 0 }] {
            let err = setup_air_with(&pilout, 9, packing).unwrap_err();
            assert!(
                matches!(&err, SetupError::ColumnName { air, name: n, first: f, second: s }
                    if air == "Alike" && n == name && f == first && s == second),
                "{name}: {err}"
            );
            let message = err.to_string();
            assert!(message.contains(&format!("{first} and {second} are both named {name}")), "{message}");
        }
    }

    // A fixed column and a committed one are named alike in the proof too.
    let mut pilout = alike_pilout();
    pilout.symbols[2].name = "Alike.F[1]".into();
    let err = setup_air(&pilout, 9).unwrap_err();
    assert!(
        matches!(&err, SetupError::ColumnName { name, first, second, .. }
            if name == "Alike.F[1]" && first == "constPolsMap[1]" && second == "cmPolsMap[0]"),
        "{err}"
    );
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

/// `Q` of stage 2 and degree `q_deg`, whole.
fn whole(q_deg: u64) -> QShape {
    QShape { stage: 2, q_deg, max_q_degree: 0 }
}

fn committed(cm: &[PolMapEntry], ev_map: &[EvMapEntry]) -> Result<Committed, SetupError> {
    let consts = [pol(0, "F0", 0), pol(0, "F1", 1)];
    committed_pols(3, whole(2), &consts, cm, ev_map, Packing::Unpacked)
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
    assert_eq!(c.layout, layout, "--no-packing gives the unpacked layout");
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
    passes_output(committed(&cm[..1], &[]), "Q is made of 1 pieces (A.1), and cmPolsMap has 0");
    passes_output(committed(&[pol(1, "a", 0), pol(2, "Q0", 1), pol(2, "Q1", 2)], &[]), "and cmPolsMap has 2");
}

// ---------------------------------------------------------------------------------------------
// The grouped layout (A.2, plan M22): fusions, the bounds after them, and the evMap
// ---------------------------------------------------------------------------------------------

/// `(type, id, prime, openingPos)` of each entry of the evMap.
fn entries(ev_map: &[EvMapEntry]) -> Vec<(PolType, u64, i64, u64)> {
    ev_map.iter().map(|e| (e.pol_type, e.id, e.prime, e.opening_pos)).collect()
}

/// The offsets pilout grouped (`--extra-muls 2`): in stage 1, `a` is opened at `{−1, 0, 1, 2}`,
/// the union, and every other class is smaller than 3 and moves to it (A.2, rule 1): `b` from
/// `{0, 1}` and the two im pols from `{0}`. The four make one group, split as `[1, 1, 2]` (A.2,
/// rule 3), and the evMap gains the eight pairs the fusions open, after pil-info's. The group is
/// first met in the list of offset −1, where only `a` was and the moved ones follow it, `{0}`'s
/// class before `{0, 1}`'s: `a`, the im pols, `b`, and the composition is in reverse.
#[test]
fn the_grouped_layout_fuses_the_offsets_pilout() {
    let pilout = offsets_pilout();
    let unpacked = setup_air(&pilout, 9).unwrap();
    let AirSetup { info, committed } = setup_air_with(&pilout, 9, Packing::Grouped { extra_muls: 2 }).unwrap();
    assert_eq!(info.opening_points, [-1, 0, 1, 2]);

    // |O|_max is a's 4 already: the fusions do not change Q's bound, 16 + 2·4 + 1. A fused column
    // has the bound of its f's offsets: 16 + 4 + 1.
    assert_eq!(committed.degrees, unpacked.committed.degrees);
    assert_eq!(committed.pols, unpacked.committed.pols, "the grouping's input is the columns' own offsets");
    let union = vec![-1, 0, 1, 2];
    assert_eq!(
        f_shapes(&info.layout),
        [
            (0, vec!["Offsets.L1"], 1, vec![0], 16),
            (1, vec!["Offsets.b"], 1, union.clone(), 21),
            (1, vec!["Offsets.ImPol[1]"], 1, union.clone(), 21),
            (1, vec!["Offsets.ImPol[0]", "Offsets.a"], 2, union.clone(), 21 * 2 + 1),
            (2, vec!["Q0"], 1, vec![0], 25),
        ]
    );
    assert_eq!(info.layout, committed.layout);
    assert_eq!((info.layout.power_w().unwrap(), max_degree(&info.layout)), (2, 43));

    // The evMap: pil-info's entries where they were, then those of the fusions, by opening point,
    // then by id: b (1) and the im pols (3, 4) at −1, the im pols at 1, and b and the im pols at 2.
    let n = unpacked.info.ev_map.len();
    assert_eq!(info.ev_map[..n], unpacked.info.ev_map[..]);
    let cm = PolType::Cm;
    assert_eq!(
        entries(&info.ev_map[n..]),
        [
            (cm, 1, -1, 0),
            (cm, 3, -1, 0),
            (cm, 4, -1, 0),
            (cm, 3, 1, 2),
            (cm, 4, 1, 2),
            (cm, 1, 2, 3),
            (cm, 3, 2, 3),
            (cm, 4, 2, 3)
        ]
    );
    // The qVerifier reads pil-info's entries only, whose indices have not changed.
    let air = validate(&pilout).unwrap();
    let result = run_passes(&pilout, air, 9).unwrap();
    let evals: Vec<usize> = result
        .pil_code
        .verifier_info
        .q_verifier
        .code
        .iter()
        .flat_map(|c| &c.src)
        .filter(|r| r.ref_type == "eval")
        .map(|r| r.id)
        .collect();
    assert!(!evals.is_empty() && evals.iter().all(|&id| id < n), "{evals:?} of {n}");
    // The pilfflonkinfo is valid (Layout::check: the evMap is the layout's pairs) and the prover's
    // readers accept it, with the fused evaluations named as the others.
    let params = SetupParams { max_constraint_degree: 9, extra_muls: 2, max_q_degree: 0, packing: true };
    let gi = global_info(&pilout, params).unwrap();
    let names = ProofNames::new(&gi, &[&info]).unwrap();
    for name in ["Offsets.bw-1", "Offsets.bw2", "Offsets.ImPol[0]w", "Offsets.ImPol[1]w-1"] {
        assert!(names.evaluations().iter().any(|e| e == name), "{name} in {:?}", names.evaluations());
    }
    assert_eq!(&PilfflonkInfo::from_json_str(&info.to_json_string().unwrap()).unwrap(), &info);
    WitnessShape::from_proving_key(&gi, &[&info]).unwrap();
}

/// A fusion that raises `|O|_max` raises `Q`'s bound (A.1) and the extended domain with it: they
/// are those of the fused offsets, not of the evMap's (spec C.3.2, the old system's defect).
#[test]
fn the_bounds_are_those_of_the_fused_offsets() {
    // N = 8, qDeg = 2; a at {0} and b at {1}, alone in their classes: both move to {0, 1}.
    let consts = [pol(0, "F0", 0)];
    let cm = [pol(1, "a", 0), pol(1, "b", 1), pol(2, "Q0", 2)];
    let ev_map = [ev(PolType::Const, 0, 0), ev(PolType::Cm, 0, 0), ev(PolType::Cm, 1, 1)];
    let unpacked = committed_pols(3, whole(2), &consts, &cm, &ev_map, Packing::Unpacked).unwrap();
    let grouped = committed_pols(3, whole(2), &consts, &cm, &ev_map, Packing::Grouped { extra_muls: 0 }).unwrap();
    // |O|_max 1: Q has 2·8 + 3·1 + 1 = 20 coefficients; fused, |O|_max 2: 2·8 + 3·2 + 1 = 23.
    assert_eq!(unpacked.degrees, Degrees { n_bits: 3, max_openings: 1, q_coefficients: 20, n_bits_ext: 5 });
    assert_eq!(grouped.degrees, Degrees { n_bits: 3, max_openings: 2, q_coefficients: 23, n_bits_ext: 5 });
    assert_eq!(
        f_shapes(&grouped.layout),
        [
            (0, vec!["F0"], 1, vec![0], 8),
            (1, vec!["b", "a"], 2, vec![0, 1], 11 * 2 + 1),
            (2, vec!["Q0"], 1, vec![0], 23)
        ]
    );
    // The input of the grouping keeps each column's own offsets and bound; Q's is the fused one's.
    assert_eq!(grouped.pols[1].offsets, [0]);
    assert_eq!(grouped.pols[1].coefficients, 10);
    assert_eq!(grouped.pols[3].coefficients, 23);

    // What the grouping refuses is an error of the setup, with the grouping's message.
    let err = committed_pols(3, whole(2), &consts, &cm, &ev_map, Packing::Grouped { extra_muls: 2 }).unwrap_err();
    assert!(
        matches!(&err, SetupError::Grouping(GroupingError::TooManyExtraMuls { extra_muls: 2, n_pols: 4, n_groups: 3 })),
        "{err}"
    );
    assert!(err.to_string().contains("lower --extra-muls"), "{err}");
}

/// The rule of the evMap of a layout (`ev_map_of`): the passes' entries as they are, then the pairs
/// only the layout opens, by opening point, the fixed columns first, by id; Q's f adds nothing.
#[test]
fn the_pairs_of_the_fusions_are_appended_to_the_ev_map() {
    let f = |stage, pols: &[u64], offsets: &[i64]| LayoutEntry {
        stage,
        pols: pols.iter().map(|&id| LayoutPol { id, name: format!("p{id}") }).collect(),
        k: pols.len() as u64,
        offsets: offsets.to_vec(),
        degree: 100,
    };
    let layout = Layout(vec![f(0, &[0, 1], &[0]), f(0, &[2], &[0, 1]), f(1, &[3, 0], &[-1, 0, 1]), f(2, &[4], &[0])]);
    let opening_points = [-1, 0, 1];
    // The passes open const 2 at 1, cm 3 at 0 and cm 0 at −1 and 1, in their order.
    let passes = [
        EvMapEntry { pol_type: PolType::Cm, id: 0, prime: -1, opening_pos: 0 },
        EvMapEntry { pol_type: PolType::Const, id: 0, prime: 0, opening_pos: 1 },
        EvMapEntry { pol_type: PolType::Const, id: 1, prime: 0, opening_pos: 1 },
        EvMapEntry { pol_type: PolType::Cm, id: 3, prime: 0, opening_pos: 1 },
        EvMapEntry { pol_type: PolType::Const, id: 2, prime: 1, opening_pos: 2 },
        EvMapEntry { pol_type: PolType::Cm, id: 0, prime: 1, opening_pos: 2 },
    ];
    let ev_map = ev_map_of(&passes, &layout, 2, &opening_points).unwrap();
    assert_eq!(ev_map[..passes.len()], passes);
    let (cm, fixed) = (PolType::Cm, PolType::Const);
    assert_eq!(entries(&ev_map[passes.len()..]), [(cm, 3, -1, 0), (fixed, 2, 0, 1), (cm, 0, 0, 1), (cm, 3, 1, 2)]);
    // Nothing to add: the passes' evMap as it is.
    assert_eq!(ev_map_of(&ev_map, &layout, 2, &opening_points).unwrap(), ev_map);
    // An offset that is not an opening point of the AIR.
    let err = ev_map_of(&passes, &layout, 2, &[-1, 0]).unwrap_err();
    assert!(matches!(&err, SetupError::PassesOutput(m) if m.contains("at offset 1")), "{err}");
}

// ---------------------------------------------------------------------------------------------
// The pieces of Q (A.1, A.3, plan M33)
// ---------------------------------------------------------------------------------------------

/// `Q` of stage 2 and degree `q_deg`, split by `--max-q-degree max_q_degree`.
fn split(q_deg: u64, max_q_degree: u64) -> QShape {
    QShape { stage: 2, q_deg, max_q_degree }
}

/// `a` at `{0, 1}` and `c` at `{0}` of stage 1, `F1` at `{0}`, and the pieces of `Q` given, of stage
/// 2, at ids 2, 3, …
fn with_pieces(pieces: &[&str]) -> (Vec<PolMapEntry>, Vec<EvMapEntry>) {
    let mut cm = vec![pol(1, "a", 0), pol(1, "c", 1)];
    cm.extend(pieces.iter().enumerate().map(|(i, name)| pol(2, name, 2 + i as u64)));
    let ev_map = vec![ev(PolType::Const, 1, 0), ev(PolType::Cm, 0, 0), ev(PolType::Cm, 1, 0), ev(PolType::Cm, 0, 1)];
    (cm, ev_map)
}

/// `qDeg = 3` split by `M = 1` on `N = 8`, `|O|_max = 2`: `Q` has `3·8 + 4·2 + 1 = 33` coefficients,
/// and its three pieces `8 + 2`, `8 + 2` and `33 − 2·8 = 17`, which is what the grouping takes them
/// for. Unpacked, an `f` each, `Q0` first. Grouped, they are a group of their own (A.2, rule 2), in
/// reverse, `Q2, Q1, Q0` (rule 4): one `f` of `k = 3` with no extra mul, and with one, the split of
/// that group of least cost, `[Q2]` and `[Q1, Q0]` (rule 3), as the old system splits it
/// (`extraMuls` may split every group, `Q`'s too).
#[test]
fn the_pieces_of_q_have_the_bounds_of_a1_and_a_group_of_their_own() {
    let consts = [pol(0, "F0", 0), pol(0, "F1", 1)];
    let (cm, ev_map) = with_pieces(&["Q0", "Q1", "Q2"]);
    let unpacked = committed_pols(3, split(3, 1), &consts, &cm, &ev_map, Packing::Unpacked).unwrap();
    assert_eq!(unpacked.degrees, Degrees { n_bits: 3, max_openings: 2, q_coefficients: 33, n_bits_ext: 6 });
    assert_eq!(unpacked.q_split, QSplit { stride: 8, coefficients: vec![10, 10, 17] });
    let pieces: Vec<(u64, &str, u64)> =
        unpacked.pols.iter().filter(|p| p.stage == 2).map(|p| (p.id, p.name.as_str(), p.coefficients)).collect();
    assert_eq!(pieces, [(2, "Q0", 10), (3, "Q1", 10), (4, "Q2", 17)]);
    assert_eq!(
        f_shapes(&unpacked.layout),
        [
            (0, vec!["F1"], 1, vec![0], 8),
            (1, vec!["a"], 1, vec![0, 1], 11),
            (1, vec!["c"], 1, vec![0], 10),
            (2, vec!["Q0"], 1, vec![0], 10),
            (2, vec!["Q1"], 1, vec![0], 10),
            (2, vec!["Q2"], 1, vec![0], 17),
        ]
    );

    // Grouped: c fused to {0, 1} with a (rule 1), which leaves |O|_max, and so Q, as they were.
    let grouped = committed_pols(3, split(3, 1), &consts, &cm, &ev_map, Packing::Grouped { extra_muls: 0 }).unwrap();
    assert_eq!((grouped.degrees, &grouped.q_split), (unpacked.degrees, &unpacked.q_split));
    assert_eq!(
        f_shapes(&grouped.layout),
        [
            (0, vec!["F1"], 1, vec![0], 8),
            (1, vec!["c", "a"], 2, vec![0, 1], 11 * 2 + 1),
            (2, vec!["Q2", "Q1", "Q0"], 3, vec![0], 17 * 3),
        ]
    );
    assert_eq!(grouped.layout.power_w().unwrap(), 6);
    // One extra mul: splitting the pieces' group ([23, 21, 8]) is better than splitting a and c's
    // ([51, 11, 8]).
    let one = committed_pols(3, split(3, 1), &consts, &cm, &ev_map, Packing::Grouped { extra_muls: 1 }).unwrap();
    assert_eq!(
        f_shapes(&one.layout),
        [
            (0, vec!["F1"], 1, vec![0], 8),
            (1, vec!["c", "a"], 2, vec![0, 1], 23),
            (2, vec!["Q2"], 1, vec![0], 17),
            (2, vec!["Q1", "Q0"], 2, vec![0], 10 * 2 + 1),
        ]
    );

    // M = 2: two pieces, 16 + 2 and 33 − 16.
    let (cm, ev_map) = with_pieces(&["Q0", "Q1"]);
    let two = committed_pols(3, split(3, 2), &consts, &cm, &ev_map, Packing::Unpacked).unwrap();
    assert_eq!(two.q_split, QSplit { stride: 16, coefficients: vec![18, 17] });
    assert_eq!(max_degree(&two.layout), 18);
}

/// What the pieces of `Q` must be: as many as `qDeg` and `maxQDegree` make, `Q0 … Q<m−1>` in this
/// order; and as many as the grouping can split in chunks of `kN | r − 1`: five pieces are no
/// chunk, and need an extra mul.
#[test]
fn the_pieces_of_q_are_refused_unless_they_are_those_of_a1() {
    let consts = [pol(0, "F0", 0), pol(0, "F1", 1)];
    let passes_output = |r: Result<Committed, SetupError>, what: &str| match r {
        Err(SetupError::PassesOutput(m)) => assert!(m.contains(what), "{m}"),
        other => panic!("{other:?}"),
    };
    let (cm, ev_map) = with_pieces(&["Q0", "Q1"]);
    passes_output(
        committed_pols(3, split(3, 1), &consts, &cm, &ev_map, Packing::Unpacked),
        "Q is made of 3 pieces (A.1), and cmPolsMap has 2",
    );
    let (cm, ev_map) = with_pieces(&["Q0", "Q2", "Q1"]);
    passes_output(
        committed_pols(3, split(3, 1), &consts, &cm, &ev_map, Packing::Unpacked),
        "piece 1 of Q in cmPolsMap is Q2, not Q1",
    );
    // An evaluation of a piece, which the proof carries apart from the evMap.
    let (cm, mut ev_map) = with_pieces(&["Q0", "Q1", "Q2"]);
    ev_map.push(ev(PolType::Cm, 3, 0));
    passes_output(committed_pols(3, split(3, 1), &consts, &cm, &ev_map, Packing::Unpacked), "a piece of Q");

    let (cm, ev_map) = with_pieces(&["Q0", "Q1", "Q2", "Q3", "Q4"]);
    let err = committed_pols(3, split(5, 1), &consts, &cm, &ev_map, Packing::Grouped { extra_muls: 0 }).unwrap_err();
    assert!(matches!(&err, SetupError::Grouping(GroupingError::NoValidPartition { .. })), "{err}");
    assert!(err.to_string().contains("a larger --extra-muls"), "{err}");
    let one = committed_pols(3, split(5, 1), &consts, &cm, &ev_map, Packing::Grouped { extra_muls: 1 }).unwrap();
    let q_ks: Vec<u64> = one.layout.0.iter().filter(|f| f.stage == 2).map(|f| f.k).collect();
    assert_eq!(q_ks.iter().sum::<u64>(), 5);
    assert!(q_ks.iter().all(|&k| k != 5), "{q_ks:?}");
}

/// Two constraints of degree 3, `a³ − b` and `b³ − a`, on `2^4` rows: an im pol would bring each
/// down to degree 2 at the cost of the degree it saves, so the search keeps them, with `qDeg = 2`.
fn cubic_pilout() -> pb::PilOut {
    let mut pilout = offsets_pilout();
    let air = the_air(&mut pilout);
    air.expressions = vec![
        mul(witness(0, 0), witness(0, 0)), // 0: a²
        mul(exp(0), witness(0, 0)),        // 1: a³
        sub(exp(1), witness(1, 0)),        // 2
        mul(witness(1, 0), witness(1, 0)), // 3: b²
        mul(exp(3), witness(1, 0)),        // 4: b³
        sub(exp(4), witness(0, 0)),        // 5
    ];
    air.constraints = vec![every_row(2), every_row(5)];
    pilout
}

/// The pilfflonkinfo of `Q` split (`--max-q-degree 1` and `qDeg = 2`): `maxQDegree = 1`, the pieces
/// `Q0` and `Q1` at the end of `cmPolsMap`, at stageId and stagePos 0 and 1, and in the layout, of
/// `16 + 2` and `2·16 + 3·1 + 1 − 16 = 20` coefficients; the proof names their evaluations, after the
/// columns', in the order of the layout. A `--max-q-degree` of `qDeg` or more does not split `Q`, and
/// the pilfflonkinfo says `maxQDegree = 0`, whatever the option was.
#[test]
fn the_pilfflonkinfo_has_the_pieces_of_a_split_q() {
    let pilout = cubic_pilout();
    let whole = setup_air(&pilout, 9).unwrap();
    assert_eq!((whole.info.q_deg, whole.info.cm_pols_map.iter().filter(|p| p.im_pol).count()), (2, 0));
    for max_q_degree in [2, 3, 100] {
        let AirSetup { info, .. } = setup_air_split(&pilout, 9, Packing::Unpacked, max_q_degree).unwrap();
        assert_eq!(info, whole.info, "--max-q-degree {max_q_degree}");
        assert_eq!(info.max_q_degree, 0);
    }

    let params = SetupParams { max_constraint_degree: 9, extra_muls: 0, max_q_degree: 1, packing: false };
    let gi = global_info(&pilout, params).unwrap();
    for (packing, q_fs) in [
        (Packing::Unpacked, vec![(vec!["Q0"], 18), (vec!["Q1"], 20)]),
        (Packing::Grouped { extra_muls: 0 }, vec![(vec!["Q1", "Q0"], 20 * 2)]),
        (Packing::Grouped { extra_muls: 2 }, vec![(vec!["Q1"], 20), (vec!["Q0"], 18)]),
    ] {
        let AirSetup { info, committed } = setup_air_split(&pilout, 9, packing, 1).unwrap();
        assert_eq!((info.q_deg, info.max_q_degree), (2, 1), "{packing:?}");
        assert_eq!(committed.q_split, QSplit { stride: 16, coefficients: vec![18, 20] });
        assert_eq!(info.q_split().unwrap(), committed.q_split, "the prover derives the same pieces");
        let pieces: Vec<(&str, u64, u64)> = info
            .cm_pols_map
            .iter()
            .filter(|p| p.stage == 2)
            .map(|p| (p.name.as_str(), p.stage_id, p.stage_pos))
            .collect();
        assert_eq!(pieces, [("Q0", 0, 0), ("Q1", 1, 1)]);
        assert_eq!(info.map_sections_n.get("cm2"), Some(&2));
        let shapes: Vec<(Vec<&str>, u64)> =
            f_shapes(&info.layout).into_iter().filter(|f| f.0 == 2).map(|f| (f.1, f.4)).collect();
        assert_eq!(shapes, q_fs, "{packing:?}");
        // The proof's evaluations end with the pieces', in the order of the layout.
        let names = ProofNames::new(&gi, &[&info]).unwrap();
        let order: Vec<&str> = q_fs.iter().flat_map(|(pols, _)| pols.iter().copied()).collect();
        assert_eq!(names.evaluations()[names.evaluations().len() - 2..], order[..], "{packing:?}");
        check_readers(&pilout, &info);
    }
}
