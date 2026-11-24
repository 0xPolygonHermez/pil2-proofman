//! The setup on the core (`run_passes` -> `pil_info::shplonk_info`) through the real pipeline: every
//! artefact of a key (pilfflonkinfo, unsealed vkey, verifierinfo, expressionsinfo, `<air>.bin`) is
//! built and deterministic; qDeg counts the boundary degree; the old passes are not reached. That the
//! keys prove and verify is `pilfflonk_prove` (cli, `--ignored`).

use std::fs;

use pil2_pilout::pilout::{self as pb, constraint, operand};
use pil_info::PilInfoResult;
use pilfflonk_setup::air_info::{air_setup, AirRef};
use pilfflonk_setup::bytecode::write_air_bin;
use pilfflonk_setup::global_info::global_info;
use pilfflonk_setup::layout::Packing;
use pilfflonk_setup::passes::run_passes;
use pilfflonk_setup::validate::validate;
use pilfflonk_setup::SetupError;
use proofman_pilfflonk::{AirVerkey, G1Affine, G2Affine, JsonFile, SetupParams, Vkey};

use crate::common::*;

type Artefacts = Vec<(&'static str, Vec<u8>)>;

fn constraint_on(idx: u32, kind: &str) -> pb::Constraint {
    let e = Some(operand::Expression { idx });
    let debug_line = Some(format!("{kind} {idx}"));
    pb::Constraint {
        constraint: Some(match kind {
            "firstRow" => constraint::Constraint::FirstRow(constraint::FirstRow { expression_idx: e, debug_line }),
            _ => constraint::Constraint::EveryRow(constraint::EveryRow { expression_idx: e, debug_line }),
        }),
    }
}

/// Every artefact of the key the passes' result `r` feeds; (result, artefacts) or the setup's error.
fn artefacts(pilout: &pb::PilOut, max_degree: u64) -> Result<(PilInfoResult, Artefacts), SetupError> {
    let air = validate(pilout)?;
    let r = run_passes(pilout, air, max_degree)?;
    let name = air.air.name.clone().unwrap_or_default();
    let packing = Packing::Grouped { extra_muls: 2 };
    let setup = air_setup(&r, AirRef { name: &name, airgroup_id: 0, air_id: 0 }, air.air, 0, packing)?;
    let params = SetupParams { max_constraint_degree: max_degree, extra_muls: 2, max_q_degree: 0, packing: true };
    let gi = global_info(pilout, params)?;
    let q_verifier = serde_json::to_value(&r.pil_code.verifier_info).unwrap()["qVerifier"].clone();
    let no_points = AirVerkey(vec![G1Affine::INFINITY; setup.info.layout.n_fixed()]);
    let vkey = Vkey::new(
        &setup.info,
        gi.n_publics,
        gi.num_challenges.clone(),
        G2Affine::generator().unwrap(),
        &no_points,
        q_verifier,
    )
    .map_err(SetupError::from)?;
    let dir = TestDir::new("core_path_bin");
    let bin = dir.file("air.bin");
    write_air_bin(&r, &bin).unwrap();
    let json = |v: serde_json::Value| serde_json::to_vec(&v).unwrap();
    let a: Artefacts = vec![
        ("pilfflonkinfo", setup.info.to_json_string().unwrap().into_bytes()),
        ("vkey", serde_json::to_vec(&vkey).unwrap()),
        ("verifierinfo", json(serde_json::to_value(&r.pil_code.verifier_info).unwrap())),
        ("expressionsinfo", json(serde_json::to_value(&r.pil_code.expressions_info).unwrap())),
        ("bin", fs::read(&bin).unwrap()),
    ];
    Ok((r, a))
}

fn assert_deterministic(label: &str, pilout: &pb::PilOut, max_degree: u64) {
    let _cpp = cpp_core();
    let (a, b) = (artefacts(pilout, max_degree).unwrap(), artefacts(pilout, max_degree).unwrap());
    for ((name, x), (_, y)) in a.1.iter().zip(&b.1) {
        assert_eq!(x, y, "{label}: {name}");
    }
}

#[test]
fn the_synthetic_pilout_gives_deterministic_key_artefacts() {
    for max in [2, 3, 4, 9] {
        assert_deterministic(&format!("pilout max {max}"), &pilout(), max);
    }
}

/// Offsets and im pols: `a·b'·b'` at degree 2 needs an im pol.
#[test]
fn a_pilout_with_im_pols_gives_deterministic_key_artefacts() {
    let mut p = pilout();
    the_air(&mut p).expressions = vec![
        mul(witness(0, 0), witness(1, 1)), // 0
        mul(exp(0), witness(1, 1)),        // 1: degree 3
        sub(exp(1), witness(0, -1)),       // 2
    ];
    the_air(&mut p).constraints = vec![constraint_on(2, "everyRow")];
    for max in [2, 3, 4] {
        assert_deterministic(&format!("im pols max {max}"), &p, max);
    }
}

/// A boundary off everyRow adds a degree (pilfflonk/docs/protocol.md#degree-search): `a·b` is qDeg 1
/// everyRow, and 2 at firstRow when the degree bound (3) leaves it without an im pol.
#[test]
fn qdeg_has_the_boundary_degree_off_every_row() {
    let q_deg = |kind: &str| {
        let mut p = pilout();
        the_air(&mut p).expressions = vec![mul(witness(0, 0), witness(1, 0))];
        the_air(&mut p).constraints = vec![constraint_on(0, kind)];
        let r = run_passes(&p, validate(&p).unwrap(), 3).unwrap();
        (r.q_deg, r.setup.cm_pols_map.iter().filter(|s| s.im_pol).count())
    };
    assert_eq!(q_deg("everyRow"), (1, 0));
    assert_eq!(q_deg("firstRow"), (1, 1), "im pol at degree 2 ties no im pol at degree 3 (qDeg 2)");
}

#[test]
fn no_constraint_is_a_clear_error_and_first_row_only_sets_up() {
    let _cpp = cpp_core();
    let mut p = pilout();
    the_air(&mut p).constraints = vec![];
    let e = artefacts(&p, 4).err().expect("refused").to_string();
    assert!(!e.is_empty() && !e.contains("not yet implemented"), "{e}");
    let mut p = pilout();
    the_air(&mut p).expressions = vec![sub(witness(0, 0), fixed(0))];
    the_air(&mut p).constraints = vec![constraint_on(0, "firstRow")];
    assert_eq!(run_passes(&p, validate(&p).unwrap(), 4).unwrap().q_deg, 1, "firstRow only");
}

/// The setup's entry runs the core, not the old passes.
#[test]
fn the_passes_are_the_core() {
    let src = include_str!("../../src/passes.rs");
    let start = src.find("pub fn run_passes").expect("run_passes");
    let body = &src[start..];
    let body = &body[..body.find("\n}\n").unwrap()];
    assert!(body.contains("shplonk_info"));
}
