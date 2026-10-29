//! pilout v2 constraint kinds to boundaries; fixtures: `node src/pil.js <pil> -o tests/fixtures/<name>.pilout`.

use pil2_pilout::pilout::{self as pb, constraint};
use pil_info::pil::constraint_poly::Boundary;
use pil_info::pil::domain::Domain;
use pil_info::pil::prepare::{prepare_pil, PrepareOptions};
use pil_info::types::pilout_info::{format_constraints, ConstraintInfo};
use pil_info::PilInfoCfg;
use prost::Message;

fn load(name: &str) -> pb::PilOut {
    let path = format!("{}/tests/fixtures/{name}.pilout", env!("CARGO_MANIFEST_DIR"));
    pb::PilOut::decode(std::fs::read(path).unwrap().as_slice()).unwrap()
}

fn air_constraints(file: &str, air: &str) -> Vec<ConstraintInfo> {
    let pilout = load(file);
    let air = pilout.air_groups.iter().flat_map(|g| &g.airs).find(|a| a.name.as_deref() == Some(air)).unwrap();
    format_constraints(&air.constraints, &air.domains).unwrap()
}

#[test]
fn domain_fixture_maps_every_constraint() {
    let cs = air_constraints("domain", "DomainAir");
    assert_eq!(cs.len(), 7);
    // the compiler emits the constraint without domain first
    let limits = |c| domain(4, &[0, 15], c);
    assert_eq!(cs[0].boundary, Boundary::EveryRow);
    assert_eq!(cs[1].boundary, limits(false));
    assert_eq!(cs[2].boundary, limits(false));
    assert_eq!(cs[3].boundary, limits(false));
    assert_eq!(cs[4].boundary, limits(true));
    assert_eq!(cs[5].boundary, domain(8, &[0], false));
    assert_eq!(cs[6].boundary, domain(8, &[0], true));
}

#[test]
fn multi_air_domains_are_per_air() {
    let cycles = air_constraints("domain_multi", "Cycles");
    assert!(cycles.iter().any(|c| c.boundary == domain(1, &[1], false)));
    let fl = air_constraints("domain_multi", "FirstLast");
    assert!(fl.iter().any(|c| c.boundary == domain(8, &[255], false)));
    assert!(!fl.iter().any(|c| c.boundary == domain(1, &[1], false)));
}

#[test]
fn no_domain_pilout_is_all_every_row() {
    let cs = air_constraints("no_domains", "NoDom");
    assert_eq!(cs.len(), 2);
    assert!(cs.iter().all(|c| c.boundary == Boundary::EveryRow));
}

#[test]
fn unknown_constraint_kind_is_an_error() {
    let err = format_constraints(&[pb::Constraint { constraint: None }], &[]).unwrap_err();
    assert!(err.to_string().contains("constraint has no kind"), "{err}");
}

#[test]
fn out_of_range_domain_idx_is_an_error() {
    let c = pb::Constraint {
        constraint: Some(constraint::Constraint::DomainRows(constraint::DomainRows {
            domain_idx: 3,
            expression_idx: Some(pb::operand::Expression { idx: 0 }),
            debug_line: None,
        })),
    };
    assert!(format_constraints(&[c], &[]).is_err());
}

fn run_air(file: &str, air: usize) -> pil_info::PilInfoResult {
    pil_info::run(&load(file), 0, air, &PilInfoCfg::goldilocks(1), &PrepareOptions::default()).unwrap()
}

fn domain(cycle_bits: u32, offsets: &[u32], complement: bool) -> Boundary {
    Boundary::Domain(Domain { cycle_bits, offsets: offsets.to_vec(), complement })
}

#[test]
fn domain_air_starkinfo_boundaries() {
    let bs = run_air("domain", 0).boundaries;
    assert_eq!(bs[0], Boundary::EveryRow);
    let expected =
        [domain(4, &[0, 15], false), domain(4, &[0, 15], true), domain(8, &[0], false), domain(8, &[0], true)];
    assert_eq!(bs.len(), 1 + expected.len(), "{bs:?}");
    for b in &expected {
        assert_eq!(bs.iter().filter(|x| *x == b).count(), 1, "{b:?} in {bs:?}");
    }
}

#[test]
fn first_row_constraint_degree() {
    // `a*a - 1` (degree 2): FIRST selects 1 row, so its zerofier adds 1; !FIRST's 255 rows + 2 > N adds 0.
    let deg = |air| prepare_pil(&load("first_degree"), 0, air, &PilInfoCfg::goldilocks(1)).unwrap();
    assert_eq!(deg(0).constraint_poly.initial_q_degree - 1, 2);
    assert_eq!(deg(1).constraint_poly.initial_q_degree - 1, 1);
    // At a bound of 3, FIRST ties (q_deg 2) with promoting the constraint (q_deg 1 + 1 im pol); ties take the lower.
    let im_pols = |r: &pil_info::PilInfoResult| r.setup.cm_pols_map.iter().filter(|p| p.im_pol).count();
    let not_first = run_air("first_degree", 1);
    assert_eq!((not_first.q_deg, im_pols(&not_first)), (1, 0));
    let first = run_air("first_degree", 0);
    assert_eq!((first.q_deg, im_pols(&first)), (1, 1));
}

#[test]
fn promoted_domain_constraint_keeps_its_debug_code() {
    // FIRST's whole `a*a - 1` becomes an im pol; its constraint's code must still read that column
    let r = run_air("first_degree", 0);
    let im = r.setup.cm_pols_map.iter().position(|p| p.im_pol).expect("an im pol");
    let cs = &r.pil_code.expressions_info.constraints;
    let first = cs.iter().find(|c| c.boundary != Boundary::EveryRow).unwrap();
    assert_eq!(first.code.len(), 1, "{:?}", first.code);
    let op = &first.code[0];
    assert_eq!((op.op.as_str(), op.dest.ref_type.as_str()), ("copy", "tmp"));
    assert_eq!((op.src[0].ref_type.as_str(), op.src[0].id), ("cm", im));
    assert!(cs.iter().all(|c| !c.code.is_empty()));
}

#[test]
fn missing_expression_is_an_error() {
    let c = pb::Constraint {
        constraint: Some(constraint::Constraint::AllRows(constraint::AllRows {
            expression_idx: None,
            debug_line: None,
        })),
    };
    let err = format_constraints(&[c], &[]).unwrap_err().to_string();
    assert!(err.contains("constraint 0 has no expression"), "{err}");
}

#[test]
fn cycle_larger_than_n_names_the_domain_symbol() {
    let err = prepare_pil(&load("big_cycle"), 0, 0, &PilInfoCfg::goldilocks(1)).unwrap_err().to_string();
    assert!(err.contains("Small.BIG") && err.contains("cycle of 16 rows exceeds N=8"), "{err}");
}
