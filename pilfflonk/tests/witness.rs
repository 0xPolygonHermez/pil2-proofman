//! The witness directory of `proofman_pilfflonk::witness`
//! (pilfflonk/docs/formats.md#witness-directory): the files a `Witness` writes, the round trip
//! through `FileWitnessSource`, and every directory it refuses. The shapes are the made-up `Sample`
//! AIR of `tests/fixtures/pilfflonkinfo/` (16 rows, 4 stage-1 columns, a stage-1 air value, 2
//! publics and a stage-1 proof value) and a made-up set of three AIRs.

use std::fs;
use std::path::{Path, PathBuf};

use proofman_pilfflonk::tag::{Backend, Field, Modulus, Transcript};
use proofman_pilfflonk::witness::{instance_file_name, INSTANCES_FILE, PROOF_VALUES_FILE};
use proofman_pilfflonk::{
    AggType, AirInstanceRef, AirShape, FileWitnessSource, FrBytes, GlobalInfoAir, InstanceWitness, JsonFile,
    NameStageEntry, PilfflonkGlobalInfo, PilfflonkInfo, SetupParams, Stage1Witness, Witness, WitnessShape,
    WitnessSource, FORMAT_VERSION,
};

const R: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
const R_MINUS_1: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495616";

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(path)
}

/// A fresh, empty directory for a test.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("witness").join(name);
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    dir
}

fn name_stage(name: &str, stage: u64) -> NameStageEntry {
    NameStageEntry { name: name.into(), stage, lengths: vec![] }
}

fn sample_info() -> PilfflonkInfo {
    PilfflonkInfo::read(&fixture("pilfflonkinfo/Sample.pilfflonkinfo.json")).unwrap()
}

fn sample_global_info() -> PilfflonkGlobalInfo {
    PilfflonkGlobalInfo {
        name: "build".into(),
        airs: vec![vec![GlobalInfoAir { name: "Sample".into(), num_rows: 16 }]],
        air_groups: vec!["Sample".into()],
        agg_types: vec![vec![AggType { agg_type: 0, stage: 2 }]],
        backend: Backend,
        format_version: FORMAT_VERSION,
        field: Field,
        modulus: Modulus,
        transcript: Transcript,
        setup_params: SetupParams { max_constraint_degree: 9, extra_muls: 2, max_q_degree: 0, packing: true },
        n_publics: 2,
        num_challenges: vec![0, 2],
        num_proof_values: vec![1, 1],
        proof_values_map: vec![name_stage("pv", 1), name_stage("pv2", 2)],
        publics_map: vec![name_stage("in", 1), name_stage("out", 1)],
    }
}

fn sample_shape() -> WitnessShape {
    WitnessShape::from_proving_key(&sample_global_info(), &[&sample_info()]).unwrap()
}

fn fr(v: u64) -> FrBytes {
    FrBytes::from_u64(v)
}

fn r_minus_1() -> FrBytes {
    FrBytes::from_decimal(R_MINUS_1).unwrap()
}

const SAMPLE: AirInstanceRef = AirInstanceRef { airgroup_id: 0, air_id: 0 };

/// A trace whose values are all different, `r - 1` among them.
fn trace(n_rows: usize, n_cols: usize, seed: u64, air_values: Vec<FrBytes>) -> Stage1Witness {
    let columns: Vec<Vec<FrBytes>> = (0..n_cols)
        .map(|c| {
            (0..n_rows)
                .map(|row| {
                    if (row, c) == (n_rows - 1, 0) {
                        r_minus_1()
                    } else {
                        fr(seed * 1000 + (row * n_cols + c) as u64)
                    }
                })
                .collect()
        })
        .collect();
    Stage1Witness::from_columns(n_rows, &columns, air_values).unwrap()
}

fn sample_witness() -> Witness {
    Witness {
        instances: vec![InstanceWitness { air: SAMPLE, stage1: trace(16, 4, 1, vec![fr(42)]) }],
        publics: vec![fr(1), r_minus_1()],
        proof_values: vec![fr(7)],
    }
}

fn file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> =
        fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    names
}

// ---------------------------------------------------------------------------------------------
// The shape
// ---------------------------------------------------------------------------------------------

#[test]
fn the_shape_of_a_proving_key_counts_stage_1_only() {
    let shape = sample_shape();
    assert_eq!(shape.airs(), [AirShape { airgroup_id: 0, air_id: 0, n_bits: 4, n_cols: 4, n_air_values: 1 }]);
    assert_eq!(shape.n_publics(), 2);
    assert_eq!(shape.n_proof_values(), 1, "pv2 is of stage 2: the prover's");
}

#[test]
fn the_shape_of_a_proving_key_leaves_the_im_pols_of_stage_1_out() {
    let mut info = sample_info();
    // Make Sample.gsum and Sample.im (stageIds 0 and 1 of stage 2) stage-1 columns 4 and 5.
    for p in info.cm_pols_map.iter_mut().filter(|p| p.stage == 2) {
        p.stage = 1;
        p.stage_id += 4;
        p.stage_pos += 4;
    }
    let shape = WitnessShape::from_proving_key(&sample_global_info(), &[&info]).unwrap();
    assert_eq!(shape.airs()[0].n_cols, 5, "Sample.im is an im pol");

    // An im pol before a witness column: the file's columns would not be the first stageIds.
    let im = info.cm_pols_map.iter().position(|p| p.im_pol).unwrap();
    info.cm_pols_map[im].stage_id = 0;
    info.cm_pols_map[0].stage_id = 5;
    let err = WitnessShape::from_proving_key(&sample_global_info(), &[&info]).unwrap_err();
    assert!(err.to_string().contains("stageIds [5, 1, 2, 3, 4]"), "{err}");
}

#[test]
fn the_shape_of_a_proving_key_matches_the_global_info() {
    let mut global = sample_global_info();
    global.airs[0][0].num_rows = 32;
    let err = WitnessShape::from_proving_key(&global, &[&sample_info()]).unwrap_err();
    assert!(err.to_string().contains("of 32 rows in the globalInfo"), "{err}");

    let mut global = sample_global_info();
    global.airs[0][0].name = "Other".into();
    assert!(WitnessShape::from_proving_key(&global, &[&sample_info()]).is_err());

    let mut info = sample_info();
    info.air_id = 1;
    let err = WitnessShape::from_proving_key(&sample_global_info(), &[&info]).unwrap_err();
    assert!(err.to_string().contains("no air 1 in airgroup 0"), "{err}");
}

#[test]
fn a_shape_has_each_air_once_and_at_most_2_28_rows() {
    let air = AirShape { airgroup_id: 0, air_id: 0, n_bits: 4, n_cols: 1, n_air_values: 0 };
    assert!(WitnessShape::new(vec![air, air], 0, 0).unwrap_err().to_string().contains("twice"));
    let big = AirShape { n_bits: 29, ..air };
    assert!(WitnessShape::new(vec![big], 0, 0).unwrap_err().to_string().contains("above 2^28"));
    assert!(WitnessShape::new(vec![AirShape { n_bits: 28, ..air }], 0, 0).is_ok());
}

// ---------------------------------------------------------------------------------------------
// Round trips
// ---------------------------------------------------------------------------------------------

#[test]
fn a_witness_round_trips_through_its_directory() {
    let dir = scratch("round_trip");
    let witness = sample_witness();
    let shape = sample_shape();
    witness.write(&dir, &shape).unwrap();

    assert_eq!(file_names(&dir), ["instance_0_0_0.bin", "instances.json", "proof_values.json", "publics.json"]);
    let bin = fs::read(dir.join("instance_0_0_0.bin")).unwrap();
    assert_eq!(bin.len(), 16 * 4 * 32);
    assert_eq!(bin, witness.instances[0].stage1.trace_bytes());
    // Row 1, column 2 is value 1000 + 1·4 + 2, at byte (1·4 + 2)·32, little-endian.
    assert_eq!(&bin[6 * 32..6 * 32 + 3], &[0xee, 0x03, 0x00], "1006 = 0x3ee");
    assert_eq!(
        fs::read_to_string(dir.join(INSTANCES_FILE)).unwrap(),
        "[\n {\n  \"airgroupId\": 0,\n  \"airId\": 0,\n  \"airValues\": [\n   \"42\"\n  ]\n }\n]"
    );
    assert_eq!(fs::read_to_string(dir.join("publics.json")).unwrap(), format!("[\n \"1\",\n \"{R_MINUS_1}\"\n]"));
    assert_eq!(fs::read_to_string(dir.join(PROOF_VALUES_FILE)).unwrap(), "[\n \"7\"\n]");

    let source = FileWitnessSource::open(&dir, &shape).unwrap();
    assert_eq!(source.instances(), [SAMPLE]);
    assert_eq!(Witness::from_source(&source).unwrap(), witness);
    let err = source.stage1(1).unwrap_err();
    assert!(err.to_string().contains("no instance 1"), "{err}");

    // Deterministic: the same witness gives the same bytes.
    let again = scratch("round_trip_again");
    witness.write(&again, &shape).unwrap();
    for name in file_names(&dir) {
        assert_eq!(fs::read(dir.join(&name)).unwrap(), fs::read(again.join(&name)).unwrap(), "{name}");
    }
}

fn three_airs() -> WitnessShape {
    let air = |airgroup_id, air_id, n_bits, n_cols, n_air_values| AirShape {
        airgroup_id,
        air_id,
        n_bits,
        n_cols,
        n_air_values,
    };
    WitnessShape::new(vec![air(1, 0, 2, 1, 0), air(0, 1, 3, 2, 2), air(0, 0, 1, 0, 0)], 0, 0).unwrap()
}

fn instance(airgroup_id: u64, air_id: u64, stage1: Stage1Witness) -> InstanceWitness {
    InstanceWitness { air: AirInstanceRef { airgroup_id, air_id }, stage1 }
}

#[test]
fn several_instances_are_named_by_air_and_index_in_canonical_order() {
    let dir = scratch("several");
    let shape = three_airs();
    let witness = Witness {
        instances: vec![
            instance(0, 0, trace(2, 0, 0, vec![])),
            instance(0, 1, trace(8, 2, 1, vec![fr(1), fr(2)])),
            instance(0, 1, trace(8, 2, 2, vec![fr(3), fr(4)])),
            instance(1, 0, trace(4, 1, 3, vec![])),
        ],
        publics: vec![],
        proof_values: vec![],
    };
    witness.write(&dir, &shape).unwrap();
    assert_eq!(
        file_names(&dir),
        [
            "instance_0_0_0.bin",
            "instance_0_1_0.bin",
            "instance_0_1_1.bin",
            "instance_1_0_0.bin",
            "instances.json",
            "proof_values.json",
            "publics.json"
        ]
    );
    assert!(fs::read(dir.join("instance_0_0_0.bin")).unwrap().is_empty(), "an AIR without stage-1 columns");
    assert_eq!(fs::read(dir.join("instance_0_1_1.bin")).unwrap(), witness.instances[2].stage1.trace_bytes());
    let source = FileWitnessSource::open(&dir, &shape).unwrap();
    assert_eq!(Witness::from_source(&source).unwrap(), witness);
    assert_eq!(instance_file_name(AirInstanceRef { airgroup_id: 0, air_id: 1 }, 1), "instance_0_1_1.bin");
}

// ---------------------------------------------------------------------------------------------
// What is refused
// ---------------------------------------------------------------------------------------------

/// Writes the sample witness to a fresh directory, lets `edit` break it, and returns why `open`
/// refuses it.
fn refused_on_open(name: &str, edit: impl FnOnce(&Path)) -> String {
    let dir = scratch(name);
    sample_witness().write(&dir, &sample_shape()).unwrap();
    edit(&dir);
    match FileWitnessSource::open(&dir, &sample_shape()) {
        Ok(_) => panic!("{name}: the directory was accepted"),
        Err(e) => format!("{e:?} | {e}"),
    }
}

fn assert_contains(what: &str, text: &str, expected: &str) {
    assert!(text.contains(expected), "{what}: expected {expected:?} in {text}");
}

fn replace_in(path: PathBuf, from: &str, to: &str) {
    let text = fs::read_to_string(&path).unwrap();
    assert!(text.contains(from), "{} has no {from:?}", path.display());
    fs::write(&path, text.replacen(from, to, 1)).unwrap();
}

#[test]
fn open_refuses_json_files_that_break_the_format() {
    let r_quoted = format!("\"{R}\"");
    let cases: Vec<(&str, &str, &str, &str, &str)> = vec![
        // (case, file, from, to, expected)
        ("unknown_field", INSTANCES_FILE, "\"airId\": 0,", "\"airId\": 0, \"stage\": 1,", "unknown field `stage`"),
        ("air_id_string", INSTANCES_FILE, "\"airId\": 0", "\"airId\": \"0\"", "invalid type"),
        ("air_value_number", INSTANCES_FILE, "\"42\"", "42", "invalid type"),
        ("air_value_leading_zero", INSTANCES_FILE, "\"42\"", "\"042\"", "not a FrBytes"),
        ("air_value_r", INSTANCES_FILE, "\"42\"", &r_quoted, "not a FrBytes"),
        ("public_r", "publics.json", "\"1\"", &r_quoted, "not a FrBytes"),
        ("public_negative", "publics.json", "\"1\"", "\"-1\"", "not a FrBytes"),
        ("proof_value_hex", PROOF_VALUES_FILE, "\"7\"", "\"0x7\"", "not a FrBytes"),
        ("not_an_array", PROOF_VALUES_FILE, "[\n \"7\"\n]", "{}", "invalid type"),
    ];
    for (case, file, from, to, expected) in cases {
        let why = refused_on_open(case, |dir| replace_in(dir.join(file), from, to));
        assert_contains(case, &why, expected);
        assert_contains(case, &why, file);
    }
}

#[test]
fn open_refuses_counts_other_than_the_shapes() {
    let cases = [
        ("no_instance", INSTANCES_FILE, "[]", "one instance at least"),
        (
            "two_air_values",
            INSTANCES_FILE,
            r#"[{"airgroupId": 0, "airId": 0, "airValues": ["42", "43"]}]"#,
            "instance 0 has 2 air values, and its air 0/0 has 1 of stage 1",
        ),
        ("one_public", "publics.json", r#"["1"]"#, "the witness has 1 publics, and nPublics is 2"),
        ("three_publics", "publics.json", r#"["1", "2", "3"]"#, "3 publics"),
        ("no_proof_value", PROOF_VALUES_FILE, "[]", "0 proof values, and there are 1 of stage 1"),
    ];
    for (case, file, content, expected) in cases {
        let why = refused_on_open(case, |dir| fs::write(dir.join(file), content).unwrap());
        assert_contains(case, &why, expected);
    }
}

#[test]
fn open_refuses_instances_of_other_airs_or_out_of_order() {
    let dir = scratch("out_of_order");
    let shape = three_airs();
    let witness = Witness {
        instances: vec![instance(0, 1, trace(8, 2, 1, vec![fr(1), fr(2)])), instance(1, 0, trace(4, 1, 3, vec![]))],
        publics: vec![],
        proof_values: vec![],
    };
    witness.write(&dir, &shape).unwrap();
    let instances = dir.join(INSTANCES_FILE);
    let text = fs::read_to_string(&instances).unwrap();

    // Swapped: (1, 0) before (0, 1).
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let swapped = serde_json::to_string(&[&entries[1], &entries[0]]).unwrap();
    fs::write(&instances, swapped).unwrap();
    let err = FileWitnessSource::open(&dir, &shape).unwrap_err().to_string();
    assert_contains("swapped", &err, "not in canonical order: instance 1 is of air 0/1, after one of air 1/0");

    // An AIR the shape does not have.
    fs::write(&instances, text.replace("\"airgroupId\": 1", "\"airgroupId\": 2")).unwrap();
    let err = FileWitnessSource::open(&dir, &shape).unwrap_err().to_string();
    assert_contains("unknown air", &err, "air 2/0, which is not one of its AIRs");

    // The writer refuses the same witnesses.
    let mut swapped = witness.clone();
    swapped.instances.reverse();
    assert!(swapped.write(&scratch("out_of_order_write"), &shape).unwrap_err().to_string().contains("canonical"));
    let mut unknown = witness;
    unknown.instances[1].air.airgroup_id = 2;
    assert!(unknown.write(&scratch("unknown_air_write"), &shape).unwrap_err().to_string().contains("not one of"));
}

#[test]
fn open_refuses_files_missing_or_too_many() {
    for file in [INSTANCES_FILE, "publics.json", PROOF_VALUES_FILE, "instance_0_0_0.bin"] {
        let why = refused_on_open(&format!("missing_{file}"), |dir| fs::remove_file(dir.join(file)).unwrap());
        assert_contains(file, &why, "Io");
        assert_contains(file, &why, file);
    }
    for extra in ["instance_0_0_1.bin", "instance_0_1_0.bin", "README", ".hidden"] {
        let why = refused_on_open(&format!("extra_{extra}"), |dir| fs::write(dir.join(extra), b"").unwrap());
        assert_contains(extra, &why, &format!("\"{extra}\" is not a file of this witness directory"));
    }
    let why = refused_on_open("extra_dir", |dir| fs::create_dir(dir.join("sub")).unwrap());
    assert_contains("extra_dir", &why, "\"sub\" is not a file");
    let why = refused_on_open("bin_is_a_dir", |dir| {
        fs::remove_file(dir.join("instance_0_0_0.bin")).unwrap();
        fs::create_dir(dir.join("instance_0_0_0.bin")).unwrap();
    });
    assert_contains("bin_is_a_dir", &why, "must be a file");
}

#[test]
fn open_refuses_a_trace_of_the_wrong_size() {
    for (case, len) in
        [("short", 16 * 4 * 32 - 1), ("long", 16 * 4 * 32 + 32), ("empty", 0), ("one_row_less", 15 * 4 * 32)]
    {
        let why = refused_on_open(case, |dir| {
            let path = dir.join("instance_0_0_0.bin");
            let mut bytes = fs::read(&path).unwrap();
            bytes.resize(len, 0);
            fs::write(&path, bytes).unwrap();
        });
        assert_contains(case, &why, &format!("the trace has {len} bytes, and one of this AIR has 2048"));
        assert_contains(case, &why, "instance_0_0_0.bin");
    }
}

#[test]
fn a_trace_value_not_below_r_is_refused_when_it_is_read() {
    let dir = scratch("value_r");
    let shape = sample_shape();
    sample_witness().write(&dir, &shape).unwrap();
    let path = dir.join("instance_0_0_0.bin");
    let mut bytes = fs::read(&path).unwrap();
    // Row 3, column 1: r itself, little-endian.
    let r = num_bigint::BigUint::parse_bytes(R.as_bytes(), 10).unwrap().to_bytes_le();
    let at = (3 * 4 + 1) * 32;
    bytes[at..at + 32].copy_from_slice(&r);
    fs::write(&path, bytes).unwrap();

    let source = FileWitnessSource::open(&dir, &shape).unwrap();
    let err = source.stage1(0).unwrap_err();
    assert_contains("value_r", &format!("{err:?} | {err}"), "row 3, column 1 is not below r");
    assert_contains("value_r", &err.to_string(), "instance_0_0_0.bin");
}

#[test]
fn the_writer_checks_the_shape_and_wants_an_empty_directory() {
    let shape = sample_shape();
    let cases: Vec<(&str, Witness, &str)> = vec![
        (
            "rows",
            Witness {
                instances: vec![InstanceWitness { air: SAMPLE, stage1: trace(8, 4, 1, vec![fr(1)]) }],
                ..sample_witness()
            },
            "instance 0 has 8 rows and 4 columns, and its air 0/0 has 16 and 4",
        ),
        (
            "cols",
            Witness {
                instances: vec![InstanceWitness { air: SAMPLE, stage1: trace(16, 3, 1, vec![fr(1)]) }],
                ..sample_witness()
            },
            "has 16 rows and 3 columns",
        ),
        (
            "air_values",
            Witness {
                instances: vec![InstanceWitness { air: SAMPLE, stage1: trace(16, 4, 1, vec![]) }],
                ..sample_witness()
            },
            "0 air values",
        ),
        ("publics", Witness { publics: vec![], ..sample_witness() }, "0 publics"),
        ("proof_values", Witness { proof_values: vec![fr(1), fr(2)], ..sample_witness() }, "2 proof values"),
        ("no_instance", Witness { instances: vec![], ..sample_witness() }, "one instance at least"),
    ];
    for (case, witness, expected) in cases {
        let dir = scratch(&format!("write_{case}"));
        let err = witness.write(&dir, &shape).unwrap_err().to_string();
        assert_contains(case, &err, expected);
        assert!(!dir.exists(), "{case}: nothing is written for a witness that does not fit");
        assert!(shape.check(&witness).is_err());
    }

    let dir = scratch("write_not_empty");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("stale.bin"), b"").unwrap();
    let err = sample_witness().write(&dir, &shape).unwrap_err().to_string();
    assert_contains("not_empty", &err, "not empty");
    assert_eq!(file_names(&dir), ["stale.bin"]);
}
