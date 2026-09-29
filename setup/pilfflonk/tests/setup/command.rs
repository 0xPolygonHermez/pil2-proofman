//! `setup-pilfflonk` as `proofman-setup` runs it, `run_setup_pilfflonk`, on `common`'s pilout: its
//! arguments, `pilout.globalInfo.json`, and the files of the `provingKey/` it writes (spec
//! §4.2.6). The same through the binary, and on the compiled Fibonacci fixture, is in
//! `setup/pil2-stark/tests/setup_pilfflonk.rs`.

use std::fs;
use std::path::Path;

use pil2_pilout::pilout::{self as pb, SymbolType};
use pilfflonk_setup::bytecode::Bytecode;
use pilfflonk_setup::command::{DEFAULT_EXTRA_MULS, DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_MAX_Q_DEGREE, PROVING_KEY_DIR};
use pilfflonk_setup::digest::keccak256;
use pilfflonk_setup::fixed::FixedColumns;
use pilfflonk_setup::global_info::global_info;
use pilfflonk_setup::test_ptau::write_tau_one_ptau;
use pilfflonk_setup::{run_setup_pilfflonk, SetupError, SetupPilfflonkOptions};
use prost::Message;
use proofman_pilfflonk::global_info::GLOBAL_CONSTRAINTS_FILE;
use proofman_pilfflonk::{
    AggType, AirFile, AirVerkey, FqBytes, G2Affine, GlobalInfoAir, JsonFile, NameStageEntry, PilfflonkGlobalInfo,
    PilfflonkInfo, ProofNames, SetupParams, Vkey, WitnessShape,
};
use proofman_starks_lib_c::PilFflonkSrs;

use crate::common::*;

fn options(dir: &TestDir) -> SetupPilfflonkOptions {
    SetupPilfflonkOptions {
        airout_path: dir.file("synthetic.pilout"),
        build_dir: dir.file("build"),
        powers_of_tau: dir.file("tau_one.ptau"),
        max_constraint_degree: DEFAULT_MAX_CONSTRAINT_DEGREE,
        extra_muls: DEFAULT_EXTRA_MULS,
        max_q_degree: DEFAULT_MAX_Q_DEGREE,
        no_packing: true,
    }
}

/// Writes `pilout` and a ptau with `τ = 1` of `n_g1` powers where [`options`] expects them.
fn inputs(dir: &TestDir, pilout: &pb::PilOut, n_g1: usize) -> SetupPilfflonkOptions {
    let opts = options(dir);
    fs::write(&opts.airout_path, pilout.encode_to_vec()).unwrap();
    write_tau_one_ptau(&opts.powers_of_tau, n_g1).unwrap();
    opts
}

/// Every file under `dir`, relative to it, sorted.
fn files(dir: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.push(path.strip_prefix(root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

#[test]
fn the_defaults_are_those_of_spec_4_2() {
    assert_eq!((DEFAULT_MAX_CONSTRAINT_DEGREE, DEFAULT_EXTRA_MULS, DEFAULT_MAX_Q_DEGREE), (9, 2, 0));
    let dir = TestDir::new("defaults");
    let opts = options(&dir);
    assert_eq!(
        opts.setup_params(),
        SetupParams { max_constraint_degree: 9, extra_muls: 2, max_q_degree: 0, packing: false }
    );
    opts.check().unwrap();
}

/// What the arguments cannot ask for: a degree search below 2, and, until they are implemented,
/// splitting Q (plan R3) and packing (plan R1).
#[test]
fn the_arguments_the_setup_cannot_do_are_refused() {
    let dir = TestDir::new("arguments");
    for (change, expected) in [
        ((|o| o.max_constraint_degree = 1) as fn(&mut SetupPilfflonkOptions), "--max-constraint-degree 1"),
        (|o| o.max_q_degree = 3, "--max-q-degree 3"),
        (|o| o.no_packing = false, "--no-packing"),
    ] {
        let mut opts = options(&dir);
        change(&mut opts);
        let err = opts.check().unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
        // The command refuses them before it reads anything.
        let err = run_setup_pilfflonk(&opts).unwrap_err();
        assert!(err.to_string().contains(expected), "{err}");
    }
    let mut opts = options(&dir);
    opts.max_constraint_degree = 2;
    opts.extra_muls = 0;
    opts.check().unwrap();
    assert!(matches!(SetupPilfflonkOptions { max_q_degree: 1, ..opts }.check(), Err(SetupError::QSplitting(1))));
}

#[test]
fn the_global_info_has_the_common_part_and_pilfflonks_fields() {
    let mut pilout = pilout();
    pilout.num_proof_values = vec![1];
    pilout.symbols.push(pb::Symbol {
        name: "pv".into(),
        r#type: SymbolType::ProofValue as i32,
        id: 0,
        stage: Some(1),
        ..Default::default()
    });
    pilout.air_groups[0].air_group_values = vec![pb::AirGroupValue { agg_type: 1, stage: 2 }];
    let params = SetupParams { max_constraint_degree: 5, extra_muls: 0, max_q_degree: 0, packing: false };
    let gi = global_info(&pilout, params.clone()).unwrap();

    let name_stage = |name: &str| NameStageEntry { name: name.into(), stage: 1, lengths: vec![] };
    assert_eq!(gi.name, "Synthetic");
    assert_eq!(gi.airs, vec![vec![GlobalInfoAir { name: "Sample".into(), num_rows: N as u64 }]]);
    assert_eq!(gi.air_groups, vec!["Group".to_string()]);
    assert_eq!(gi.agg_types, vec![vec![AggType { agg_type: 1, stage: 2 }]]);
    assert_eq!(gi.setup_params, params);
    assert_eq!((gi.n_publics, gi.num_challenges.clone(), gi.num_proof_values.clone()), (2, vec![0], vec![1]));
    assert_eq!(gi.publics_map, vec![name_stage("in"), name_stage("out")]);
    assert_eq!(gi.proof_values_map, vec![name_stage("pv")]);
    // It is a valid file, read back as it was written.
    let text = gi.to_json_string().unwrap();
    assert_eq!(PilfflonkGlobalInfo::from_json_str(&text).unwrap(), gi);

    // A pilout without challenges has [0] of them, as in the STARK's; unnamed airgroups and AIRs
    // get the STARK setup's directory names.
    pilout.num_challenges.clear();
    pilout.name = None;
    pilout.air_groups[0].name = None;
    pilout.air_groups[0].airs[0].name = None;
    let gi = global_info(&pilout, params.clone()).unwrap();
    assert_eq!(gi.num_challenges, vec![0]);
    assert_eq!(
        (gi.name.as_str(), gi.air_groups[0].as_str(), gi.airs[0][0].name.as_str()),
        ("pilout", "airgroup_0", "air_0")
    );

    pilout.air_groups[0].air_group_values[0].agg_type = -1;
    assert!(matches!(global_info(&pilout, params), Err(SetupError::InvalidPilout(_))));
}

/// The `provingKey/` of spec §4.2.6, every file readable by its type, the same bytes on a second
/// run.
#[test]
fn the_command_writes_the_files_of_the_proving_key() {
    let dir = TestDir::new("command");
    let opts = inputs(&dir, &pilout(), 64);
    run_setup_pilfflonk(&opts).unwrap();

    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    assert_eq!(
        files(&proving_key),
        [
            "Synthetic/Group/airs/Sample/air/Sample.bin",
            "Synthetic/Group/airs/Sample/air/Sample.const",
            "Synthetic/Group/airs/Sample/air/Sample.expressionsinfo.json",
            "Synthetic/Group/airs/Sample/air/Sample.pilfflonkinfo.json",
            "Synthetic/Group/airs/Sample/air/Sample.verifierinfo.json",
            "Synthetic/Group/airs/Sample/air/Sample.verkey.json",
            "Synthetic/pilfflonk/pilfflonk.srs.bin",
            "Synthetic/pilfflonk/pilfflonk.vkey.json",
            "pilout.globalConstraints.json",
            "pilout.globalInfo.json",
        ]
    );

    // The globalInfo is the one of the pilout and the arguments, and names the other paths.
    let gi = PilfflonkGlobalInfo::from_proving_key(&proving_key).unwrap();
    assert_eq!(gi, global_info(&pilout(), opts.setup_params()).unwrap());
    let air_file = |file| gi.air_file(&proving_key, 0, 0, file).unwrap();

    let fixed = FixedColumns::read_const(&air_file(AirFile::Const), N_BITS, 4).unwrap();
    assert_eq!(fixed, FixedColumns::from_air(&pilout().air_groups[0].airs[0]).unwrap());

    // The layout: every opened column in an f of its own, U (never opened) in none, Q last. The
    // degrees (A.1–A.3): N = 8 for the fixed ones, N + |O| + 1 = 10 for a and b, and qDeg = 1
    // (a·b'), |O|_max = 1: 8 + 2·1 + 1 = 11 for Q.
    let info = PilfflonkInfo::read(&air_file(AirFile::PilfflonkInfo)).unwrap();
    let layout: Vec<(u64, u64, &str, Vec<i64>, u64)> = info
        .layout
        .0
        .iter()
        .map(|f| (f.stage, f.pols[0].id, f.pols[0].name.as_str(), f.offsets.clone(), f.degree))
        .collect();
    assert_eq!(
        layout,
        [
            (0, 0, "Sample.L1", vec![0], 8),
            (0, 1, "Sample.C[0]", vec![0], 8),
            (0, 2, "Sample.C[1]", vec![0], 8),
            (1, 0, "Sample.a", vec![0], 10),
            (1, 1, "Sample.b", vec![1], 10),
            (2, 2, "Q0", vec![0], 11),
        ]
    );
    assert!(info.layout.0.iter().all(|f| f.k == 1 && f.pols.len() == 1));
    assert_eq!((info.q_deg, info.q_dim, info.max_q_degree, info.n_bits), (1, 1, 0, N_BITS));
    assert_eq!(
        info.const_pols_map.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        ["Sample.L1", "Sample.C", "Sample.C", "Sample.U"]
    );
    WitnessShape::from_proving_key(&gi, &[&info]).unwrap();
    ProofNames::new(&gi, &[&info]).unwrap();

    // The SRS holds exactly the largest degree of the layout, Q's 11 powers.
    let srs_bytes = fs::read(gi.srs_path(&proving_key)).unwrap();
    assert_eq!(srs_bytes.len(), 12 + 3 * 12 + 88 + 11 * 64 + 2 * 128);
    let srs = PilFflonkSrs::load(&gi.srs_path(&proving_key)).unwrap();
    assert!(srs.commit_fixed(N_BITS, 1, &vec![[0u8; 32]; N]).is_ok());

    // A commitment per fixed f of the layout, τ = 1: the first rows of L1, C[0] and C[1], 1, 2
    // and 3. U, whose first row is 0, is not committed.
    let verkey = AirVerkey::read(&air_file(AirFile::Verkey)).unwrap();
    assert_eq!(verkey, AirVerkey(vec![multiple_of_g(1), multiple_of_g(2), multiple_of_g(3)]));

    // The code, in the STARK's formats: the verifierinfo has the qVerifier only.
    let json = |file| -> serde_json::Value { serde_json::from_slice(&fs::read(air_file(file)).unwrap()).unwrap() };
    let expressions_info = json(AirFile::ExpressionsInfo);
    let keys = |v: &serde_json::Value| {
        let mut keys = v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        keys.sort();
        keys
    };
    assert_eq!(keys(&expressions_info), ["constraints", "expressionsCode", "hintsInfo"]);
    let verifier_info = json(AirFile::VerifierInfo);
    assert_eq!(keys(&verifier_info), ["qVerifier"]);
    let bytecode = Bytecode::read(&air_file(AirFile::Bin)).unwrap();
    assert_eq!(bytecode.expressions.len(), expressions_info["expressionsCode"].as_array().unwrap().len());
    let global_constraints: serde_json::Value =
        serde_json::from_slice(&fs::read(proving_key.join(GLOBAL_CONSTRAINTS_FILE)).unwrap()).unwrap();
    assert_eq!(global_constraints, serde_json::json!({"constraints": [], "hints": []}));

    // The vkey: the pilfflonkinfo's evMap, layout and degrees, the verkey's points, [τ]₂ of the
    // SRS (the generator of G2 with τ = 1), the qVerifier of the verifierinfo, and its digest.
    let vkey = Vkey::read(&gi.vkey_path(&proving_key)).unwrap();
    assert!(vkey.digest_matches(|data| keccak256(data).unwrap()).unwrap());
    assert_eq!(
        (vkey.ev_map.clone(), vkey.layout.clone(), vkey.boundaries.clone()),
        (info.ev_map.clone(), info.layout.clone(), info.boundaries.clone())
    );
    assert_eq!((vkey.power, vkey.power_w, vkey.n_public, vkey.q_deg, vkey.max_q_degree), (N_BITS, 1, 2, 1, 0));
    assert_eq!(vkey.num_challenges, gi.num_challenges);
    assert_eq!(vkey.fixed_commitments.0, verkey.0);
    assert_eq!(vkey.q_verifier, verifier_info["qVerifier"]);
    let g2 = G2_GENERATOR.map(|c| FqBytes::from_decimal(c).unwrap());
    assert_eq!(vkey.x_2, G2Affine { x: [g2[0], g2[1]], y: [g2[2], g2[3]] });

    // Running again gives the same bytes.
    let before: Vec<Vec<u8>> = files(&proving_key).iter().map(|f| fs::read(proving_key.join(f)).unwrap()).collect();
    run_setup_pilfflonk(&opts).unwrap();
    let after: Vec<Vec<u8>> = files(&proving_key).iter().map(|f| fs::read(proving_key.join(f)).unwrap()).collect();
    assert_eq!(before, after);
}

/// The command stops at the first refusal, with the pilout's path and the reason, and writes
/// nothing.
#[test]
fn the_command_refuses_a_pilout_the_setup_does_not_support() {
    let dir = TestDir::new("command_refuses");
    let mut goldilocks = pilout();
    goldilocks.base_field = 0xFFFF_FFFF_0000_0001u64.to_be_bytes().to_vec();
    let mut big_fixed = pilout();
    the_air(&mut big_fixed).fixed_cols[1].values[4] = be(&r());

    for (pilout, expected) in [(goldilocks, "Goldilocks"), (big_fixed, "row 4 of fixed column 1")] {
        let opts = inputs(&dir, &pilout, 64);
        let err = run_setup_pilfflonk(&opts).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains(expected) && message.contains("synthetic.pilout"), "{message}");
        assert!(!opts.build_dir.exists());
    }

    let opts = inputs(&dir, &pilout(), 64);
    fs::write(&opts.airout_path, b"not a pilout").unwrap();
    let err = run_setup_pilfflonk(&opts).unwrap_err();
    assert!(format!("{err:#}").contains("is not a pilout"), "{err:#}");
}

/// A pilout whose vkey the verifier would refuse (`Vkey::validate`) is refused before any file is
/// written: here, a challenge of stage 1, which A.4 never squeezes.
#[test]
fn the_command_refuses_a_vkey_the_verifier_would_refuse() {
    let dir = TestDir::new("command_refuses_vkey");
    let mut pilout = pilout();
    pilout.num_challenges = vec![1];
    let opts = inputs(&dir, &pilout, 64);
    let err = run_setup_pilfflonk(&opts).unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("A.4 squeezes no challenge of stage 1") && message.contains("synthetic.pilout"),
        "{message}"
    );
    assert!(!opts.build_dir.exists());
}

/// A ptau with fewer powers than the largest degree of the layout, Q's 11, is refused before any
/// file is written; 11 are enough.
#[test]
fn the_command_refuses_a_ptau_too_small_for_the_layout() {
    let dir = TestDir::new("command_small_ptau");
    let opts = inputs(&dir, &pilout(), 10);
    let err = run_setup_pilfflonk(&opts).unwrap_err();
    let message = format!("{err:#}");
    assert!(message.contains("fewer than the 11 requested"), "{message}");
    assert_eq!(files(&opts.build_dir), Vec::<String>::new());

    let opts = inputs(&dir, &pilout(), 11);
    run_setup_pilfflonk(&opts).unwrap();
}
