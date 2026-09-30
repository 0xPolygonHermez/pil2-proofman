//! The pilfflonk file types of spec A.6, through the crate's public API: deterministic JSON, round
//! trips, what each type refuses, the vkey's digest preimage and the proof's two forms. Also the
//! Rust side of the Rust → file → C++ round trip of `<air>.pilfflonkinfo.json`, and the check that
//! the STARK loader refuses a pilfflonk globalInfo.

use std::fs;
use std::path::{Path, PathBuf};

use proofman_pilfflonk::global_info::GLOBAL_INFO_FILE;
use proofman_pilfflonk::tag::{Backend, Field, Modulus, Transcript};
use proofman_pilfflonk::{
    canonical_json, AggType, AirVerkey, Boundary, ChallengeMapEntry, EvMapEntry, FqBytes, FrBytes, G1Affine, G2Affine,
    GlobalInfoAir, JsonFile, Layout, LayoutEntry, LayoutPol, NameStageEntry, PilfflonkGlobalInfo, PilfflonkInfo,
    PolMapEntry, PolType, Proof, ProofJson, ProofNames, Publics, SetupParams, Vkey, DIGEST_DOMAIN, FORMAT_VERSION,
    G2_GENERATOR,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------------------------
// Samples. The pilfflonkinfo is a made-up AIR of 16 rows and two stages that uses every field
// and every optional one: an array column, an im pol, a negative offset, the four kinds of
// boundary but lastRow, air and airgroup values, and an f of each stage.
// ---------------------------------------------------------------------------------------------

fn pol(stage: u64, name: &str, id: u64, stage_id: u64, lengths: &[u64]) -> PolMapEntry {
    PolMapEntry {
        stage,
        name: name.into(),
        dim: 1,
        pols_map_id: id,
        stage_id,
        lengths: lengths.to_vec(),
        im_pol: false,
        exp_id: None,
        stage_pos: stage_id,
    }
}

fn ev(pol_type: PolType, id: u64, prime: i64, opening_pos: u64) -> EvMapEntry {
    EvMapEntry { pol_type, id, prime, opening_pos }
}

fn f(stage: u64, pols: &[(u64, &str)], offsets: &[i64], degree: u64) -> LayoutEntry {
    LayoutEntry {
        stage,
        pols: pols.iter().map(|&(id, name)| LayoutPol { id, name: name.into() }).collect(),
        k: pols.len() as u64,
        offsets: offsets.to_vec(),
        degree,
    }
}

fn name_stage(name: &str, stage: u64) -> NameStageEntry {
    NameStageEntry { name: name.into(), stage, lengths: vec![] }
}

fn sample_info() -> PilfflonkInfo {
    use PolType::{Cm, Const};
    PilfflonkInfo {
        name: "Sample".into(),
        airgroup_id: 0,
        air_id: 0,
        n_bits: 4,
        n_stages: 2,
        n_constants: 3,
        cm_pols_map: vec![
            pol(1, "Sample.a", 0, 0, &[]),
            pol(1, "Sample.b", 1, 1, &[]),
            pol(1, "Sample.c", 2, 2, &[0]),
            pol(1, "Sample.c", 3, 3, &[1]),
            pol(2, "Sample.gsum", 4, 0, &[]),
            PolMapEntry { im_pol: true, exp_id: Some(3), ..pol(2, "Sample.im", 5, 1, &[]) },
            pol(3, "Q0", 6, 0, &[]),
        ],
        const_pols_map: vec![
            pol(0, "Sample.L1", 0, 0, &[]),
            pol(0, "Sample.C", 1, 1, &[0]),
            pol(0, "Sample.C", 2, 2, &[1]),
        ],
        challenges_map: vec![
            ChallengeMapEntry { name: "std_alpha".into(), stage: 2, dim: 1, stage_id: 0 },
            ChallengeMapEntry { name: "std_gamma".into(), stage: 2, dim: 1, stage_id: 1 },
            ChallengeMapEntry { name: "std_vc".into(), stage: 3, dim: 1, stage_id: 0 },
            ChallengeMapEntry { name: "std_xi".into(), stage: 4, dim: 1, stage_id: 0 },
        ],
        air_values_map: vec![name_stage("Sample.av", 1)],
        airgroup_values_map: vec![name_stage("Sample.gsum_result", 2)],
        map_sections_n: [("const", 3), ("cm1", 4), ("cm2", 2), ("cm3", 1)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        opening_points: vec![-1, 0, 1],
        boundaries: vec![Boundary::EveryRow, Boundary::FirstRow, Boundary::EveryFrame { offset_min: 1, offset_max: 2 }],
        ev_map: vec![
            ev(Cm, 4, -1, 0),
            ev(Cm, 5, -1, 0),
            ev(Const, 0, 0, 1),
            ev(Const, 1, 0, 1),
            ev(Const, 2, 0, 1),
            ev(Cm, 0, 0, 1),
            ev(Cm, 1, 0, 1),
            ev(Cm, 2, 0, 1),
            ev(Cm, 3, 0, 1),
            ev(Cm, 4, 0, 1),
            ev(Cm, 5, 0, 1),
            ev(Cm, 0, 1, 2),
            ev(Cm, 1, 1, 2),
        ],
        q_deg: 2,
        q_dim: 1,
        max_q_degree: 0,
        c_exp_id: 7,
        layout: Layout(vec![
            f(0, &[(0, "Sample.L1"), (1, "Sample.C[0]"), (2, "Sample.C[1]")], &[0], 50),
            f(1, &[(0, "Sample.a"), (1, "Sample.b")], &[0, 1], 39),
            f(1, &[(2, "Sample.c[0]"), (3, "Sample.c[1]")], &[0], 37),
            f(2, &[(4, "Sample.gsum"), (5, "Sample.im")], &[-1, 0], 39),
            f(3, &[(6, "Q0")], &[0], 39),
        ]),
    }
}

/// The sample with `Q` split in two pieces (A.1): `maxQDegree = 1 < qDeg = 2`.
fn split_q_info() -> PilfflonkInfo {
    let mut info = sample_info();
    info.max_q_degree = 1;
    info.cm_pols_map.push(pol(3, "Q1", 7, 1, &[]));
    info.map_sections_n.insert("cm3".into(), 2);
    info.layout.0[4] = f(3, &[(6, "Q0"), (7, "Q1")], &[0], 34);
    info
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
        num_proof_values: vec![1],
        proof_values_map: vec![name_stage("pv", 1)],
        publics_map: vec![name_stage("in", 1), name_stage("out", 1)],
    }
}

fn point(seed: u64) -> G1Affine {
    G1Affine { x: FqBytes::from_u64(seed), y: FqBytes::from_u64(seed + 1) }
}

/// The largest scalar, `r - 1`, so that big values are exercised too.
fn r_minus_1() -> FrBytes {
    FrBytes::from_decimal("21888242871839275222246405745257275088548364400416034343698204186575808495616").unwrap()
}

fn q_verifier() -> Value {
    json!({
        "tmpUsed": 1,
        "code": [{
            "op": "mul",
            "dest": {"type": "tmp", "id": 0, "dim": 1},
            "src": [
                {"type": "eval", "id": 5, "dim": 1},
                {"type": "challenge", "id": 2, "stageId": 0, "dim": 1, "stage": 3},
            ],
        }],
    })
}

/// A stand-in for Keccak-256, which is M15's: these tests only need a function of the preimage.
fn fake_hash(preimage: &[u8]) -> [u8; 32] {
    let mut h = [0u8; 32];
    for (i, b) in preimage.iter().enumerate() {
        h[i % 32] = h[i % 32].rotate_left(3) ^ b;
    }
    h
}

fn sample_vkey() -> Vkey {
    // A point of G2, as X_2 must be (Vkey::validate): [1]₂.
    let x_2 = G2Affine::generator().unwrap();
    let verkey = AirVerkey(vec![point(100)]);
    Vkey::new(&sample_info(), 2, vec![0, 2], x_2, &verkey, q_verifier()).unwrap().seal(fake_hash).unwrap()
}

/// A temporary directory of its own for a test, removed when it is dropped.
struct TempDir(PathBuf);

impl TempDir {
    fn new(test: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("proofman_pilfflonk_{test}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The keys of the top-level object of a file the crate wrote, in the order they are written
/// (the crate indents with one space).
fn top_level_keys(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.strip_prefix(" \""))
        .filter_map(|rest| rest.split_once("\":").map(|(key, _)| key.to_string()))
        .collect()
}

/// Writes `value` as JSON with the keys of every object in reverse order, whatever serde_json's
/// Map keeps.
fn reversed_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| b.0.cmp(a.0));
            let body: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), reversed_json(v)))
                .collect();
            format!("{{{}}}", body.join(", "))
        }
        Value::Array(items) => format!("[{}]", items.iter().map(reversed_json).collect::<Vec<_>>().join(", ")),
        scalar => scalar.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Deterministic JSON and round trips.
// ---------------------------------------------------------------------------------------------

/// Serialising twice gives the same bytes, and reading them back gives the same value, which
/// serialises to the same bytes again.
fn assert_round_trip<T: JsonFile + PartialEq + std::fmt::Debug>(value: &T) -> String {
    let text = value.to_json_string().unwrap();
    assert_eq!(value.to_json_string().unwrap(), text, "serialising twice must give the same bytes");
    let back = T::from_json_str(&text).unwrap();
    assert_eq!(&back, value);
    assert_eq!(back.to_json_string().unwrap(), text);
    text
}

#[test]
fn every_file_type_round_trips_byte_for_byte() {
    assert_round_trip(&sample_global_info());
    assert_round_trip(&sample_info());
    assert_round_trip(&split_q_info());
    assert_round_trip(&AirVerkey(vec![point(1), point(3)]));
    assert_round_trip(&sample_vkey());
    assert_round_trip(&Publics(vec![FrBytes::from_u64(1), r_minus_1()]));
    let names = ProofNames::new(&sample_global_info(), &[&sample_info()]).unwrap();
    assert_round_trip(&sample_proof(&names).to_json(&names).unwrap());
}

#[test]
fn the_files_are_read_from_and_written_to_disk() {
    let dir = TempDir::new("disk");
    let path = dir.0.join("Sample.pilfflonkinfo.json");
    sample_info().write(&path).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), sample_info().to_json_string().unwrap());
    assert_eq!(PilfflonkInfo::read(&path).unwrap(), sample_info());

    let missing = dir.0.join("missing.json");
    let err = PilfflonkInfo::read(&missing).unwrap_err().to_string();
    assert!(err.contains("missing.json"), "{err}");
    fs::write(&path, "{").unwrap();
    let err = PilfflonkInfo::read(&path).unwrap_err().to_string();
    assert!(err.contains("Sample.pilfflonkinfo.json") && err.contains("JSON"), "{err}");
}

#[test]
fn the_pilfflonkinfo_has_the_fields_of_a6_in_a_fixed_order() {
    let text = sample_info().to_json_string().unwrap();
    assert_eq!(
        top_level_keys(&text),
        [
            "name",
            "airgroupId",
            "airId",
            "nBits",
            "nStages",
            "nConstants",
            "cmPolsMap",
            "constPolsMap",
            "challengesMap",
            "airValuesMap",
            "airgroupValuesMap",
            "mapSectionsN",
            "openingPoints",
            "boundaries",
            "evMap",
            "qDeg",
            "qDim",
            "maxQDegree",
            "cExpId",
            "layout",
        ]
    );
    for absent in ["starkStruct", "friExpId", "customCommits", "customCommitsMap", "publicsMap", "security"] {
        assert!(!text.contains(&format!("\"{absent}\"")), "{absent}");
    }
    // A layout entry, with its fields in order.
    assert_eq!(
        serde_json::to_string(&sample_info().layout.0[3]).unwrap(),
        r#"{"stage":2,"pols":[{"id":4,"name":"Sample.gsum"},{"id":5,"name":"Sample.im"}],"k":2,"offsets":[-1,0],"degree":39}"#
    );
}

#[test]
fn the_vkey_has_the_fields_of_a6_in_a_fixed_order() {
    let text = sample_vkey().to_json_string().unwrap();
    assert_eq!(
        top_level_keys(&text),
        [
            "protocol",
            "curve",
            "formatVersion",
            "nPublic",
            "power",
            "powerW",
            "X_2",
            "numChallenges",
            "evMap",
            "layout",
            "boundaries",
            "f0",
            "qDeg",
            "maxQDegree",
            "qVerifier",
            "digest",
        ]
    );
    let value: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["protocol"], "pilfflonk");
    assert_eq!(value["curve"], "bn128");
    assert_eq!(value["power"], 4);
    assert_eq!(value["powerW"], 6, "the lcm of k = 3, 2, 2, 2, 1");
    assert_eq!(value["f0"], json!(["100", "101"]));
    let [x_c0, x_c1, y_c0, y_c1] = G2_GENERATOR;
    assert_eq!(value["X_2"], json!([[x_c0, x_c1], [y_c0, y_c1]]));
    assert!(value["digest"].as_str().unwrap().starts_with("0x"));
    // qVerifier is written with its keys sorted, whatever order it came in.
    let q = text.find("\"qVerifier\"").unwrap();
    assert!(text[q..].find("\"code\"").unwrap() < text[q..].find("\"tmpUsed\"").unwrap());
}

// ---------------------------------------------------------------------------------------------
// What each type refuses.
// ---------------------------------------------------------------------------------------------

fn assert_refused(info: PilfflonkInfo, why: &str) {
    let err = info.validate().expect_err(why);
    // Nor is it written.
    assert!(info.to_json_string().is_err(), "{why}");
    println!("{why}: {err}");
}

#[test]
fn the_pilfflonkinfo_refuses_what_a6_and_the_layout_rules_forbid() {
    use PolType::{Cm, Const};
    assert!(sample_info().validate().is_ok() && split_q_info().validate().is_ok());

    let with = |change: &dyn Fn(&mut PilfflonkInfo)| {
        let mut info = sample_info();
        change(&mut info);
        info
    };
    assert_refused(with(&|i| i.q_dim = 3), "qDim is 1");
    assert_refused(with(&|i| i.cm_pols_map[0].dim = 3), "every dim is 1");
    assert_refused(with(&|i| i.challenges_map[0].dim = 3), "every challenge dim is 1");
    assert_refused(with(&|i| i.n_bits = 29), "nBits is at most 28");
    assert_refused(with(&|i| i.n_stages = u64::MAX), "nStages is the sections of mapSectionsN, less one");
    assert_refused(with(&|i| i.n_stages = 3), "nStages is the sections of mapSectionsN, less one");
    assert_refused(with(&|i| i.n_constants = 2), "nConstants is the fixed columns");
    assert_refused(with(&|i| i.cm_pols_map[1].pols_map_id = 0), "polsMapId is the index");
    assert_refused(with(&|i| i.cm_pols_map[1].stage_id = 0), "stageId numbers the columns of a stage");
    assert_refused(with(&|i| i.cm_pols_map[1].stage_pos = 7), "stagePos numbers the columns of a stage");
    assert_refused(with(&|i| i.cm_pols_map[5].exp_id = None), "an im pol has its expression");
    assert_refused(with(&|i| i.cm_pols_map[2].stage = 5), "a column is of stage 1 … nStages + 1");
    assert_refused(with(&|i| i.const_pols_map[1].stage = 1), "a fixed column is of stage 0");
    assert_refused(with(&|i| *i.map_sections_n.get_mut("cm1").unwrap() = 3), "mapSectionsN is the columns");
    assert_refused(
        with(&|i| {
            i.map_sections_n.insert("cm4".into(), 0);
        }),
        "no section beyond Q's",
    );
    assert_refused(
        with(&|i| {
            i.map_sections_n.remove("const");
        }),
        "the const section is there",
    );
    assert_refused(with(&|i| i.challenges_map[3].stage = 5), "no challenge beyond std_xi's stage");
    assert_refused(with(&|i| i.air_values_map[0].stage = 3), "an air value is of a stage of the AIR");
    assert_refused(with(&|i| i.opening_points = vec![0, -1, 1]), "openingPoints are increasing");
    assert_refused(with(&|i| i.boundaries.push(Boundary::EveryRow)), "a boundary is there once");

    // The evMap.
    assert_refused(with(&|i| i.ev_map[0].opening_pos = 1), "openingPos is the position of prime");
    assert_refused(with(&|i| i.ev_map.push(ev(Cm, 6, 0, 1))), "Q is not evaluated: the verifier computes it");
    assert_refused(with(&|i| i.ev_map.push(ev(Cm, 9, 0, 1))), "an evaluation is of a column of the map");
    assert_refused(
        with(&|i| {
            i.ev_map.pop();
        }),
        "every (column, offset) of the layout is evaluated",
    );
    assert_refused(with(&|i| i.ev_map.push(ev(Const, 0, 1, 2))), "every evaluation is opened by the layout");
    assert_refused(with(&|i| i.ev_map.push(ev(Cm, 0, 1, 2))), "an evaluation is there once");

    // The layout.
    assert_refused(with(&|i| i.layout.0[1].k = 3), "k is the number of polynomials");
    assert_refused(with(&|i| i.n_bits = 28), "k·N divides r - 1: k = 2 does not with N = 2^28");
    assert_refused(with(&|i| i.layout.0[1].offsets = vec![1, 0]), "offsets are increasing");
    assert_refused(with(&|i| i.layout.0[1].offsets.clear()), "an f is opened somewhere");
    // As the JS's checkLayout (plan M40, review): each offset a row, |s| < N = 16, and no two the
    // same row modulo N.
    assert_refused(with(&|i| i.layout.0[1].offsets = vec![0, 16]), "an offset is below N in absolute value");
    assert_refused(with(&|i| i.layout.0[1].offsets = vec![-16, 0]), "an offset is below N in absolute value");
    assert_refused(with(&|i| i.layout.0[1].offsets = vec![-1, 15]), "no two offsets are the same row modulo N");
    assert_refused(with(&|i| i.layout.0[1].degree = 0), "an f has a degree");
    assert_refused(with(&|i| i.layout.0[1].pols[1].name = "Sample.x".into()), "the name is the column's");
    assert_refused(with(&|i| i.layout.0[2].pols[0].name = "Sample.c".into()), "an array column carries its index");
    assert_refused(with(&|i| i.layout.0[2].pols[0].id = 1), "a column is in one f only");
    assert_refused(with(&|i| i.layout.0[2].pols[0].id = 4), "a column is in an f of its stage");
    assert_refused(with(&|i| i.layout.0.swap(1, 3)), "the layout goes by ascending stage");
    assert_refused(
        with(&|i| {
            i.layout.0.pop();
        }),
        "Q has an f, the last one",
    );
    assert_refused(with(&|i| i.layout.0[4].offsets = vec![0, 1]), "Q is opened at ξ only");
    assert_refused(with(&|i| i.max_q_degree = 1), "Q split in two needs two pieces");
    for max_q_degree in [2, 3] {
        assert_refused(with(&|i| i.max_q_degree = max_q_degree), "maxQDegree is 0 unless it splits Q (plan M33)");
    }
    // Split, the pieces are Q0 and Q1 in this order, at stageId and stagePos 0 and 1.
    let with_split = |change: &dyn Fn(&mut PilfflonkInfo)| {
        let mut info = split_q_info();
        change(&mut info);
        info
    };
    assert_refused(
        with_split(&|i| {
            i.cm_pols_map[6].name = "Q1".into();
            i.cm_pols_map[7].name = "Q0".into();
            i.layout.0[4].pols[0].name = "Q1".into();
            i.layout.0[4].pols[1].name = "Q0".into();
        }),
        "piece i of Q is named Q<i>",
    );
    assert_refused(
        with_split(&|i| {
            i.cm_pols_map[7].name = "Q2".into();
            i.layout.0[4].pols[1].name = "Q2".into();
        }),
        "the pieces of Q are Q0 … Q<m−1>",
    );
    assert_refused(with_split(&|i| i.cm_pols_map[7].lengths = vec![0]), "a piece of Q is no array");
    assert_refused(
        with_split(&|i| {
            i.cm_pols_map[6].stage_pos = 1;
            i.cm_pols_map[7].stage_pos = 0;
        }),
        "piece i of Q is at stagePos i",
    );
}

#[test]
fn the_pilfflonkinfo_refuses_unknown_fields_and_stark_ones() {
    let text = sample_info().to_json_string().unwrap();
    for (from, to) in [
        ("\"nBits\": 4", "\"nBits\": 4, \"starkStruct\": {}"),
        ("\"type\": \"const\"", "\"type\": \"custom\""),
        ("\"prime\": 1,", "\"prime\": 1, \"commitId\": 0,"),
        ("\"name\": \"everyRow\"", "\"name\": \"everyRow\", \"offsetMin\": 0"),
        ("\"name\": \"firstRow\"", "\"name\": \"someRow\""),
        ("\"nBits\": 4", "\"nBits\": -4"),
        ("\"degree\": 50", "\"degree\": 50.5"),
    ] {
        assert!(text.contains(from), "{from}");
        assert!(PilfflonkInfo::from_json_str(&text.replacen(from, to, 1)).is_err(), "{to}");
    }
}

#[test]
fn the_vkey_refuses_what_does_not_match_its_layout() {
    let with = |change: &dyn Fn(&mut Vkey)| {
        let mut vkey = sample_vkey();
        change(&mut vkey);
        vkey
    };
    assert!(sample_vkey().validate().is_ok());
    for (vkey, why) in [
        (with(&|v| v.power_w = 12), "powerW is the lcm of the k"),
        (with(&|v| v.fixed_commitments.0.clear()), "one commitment per fixed f"),
        (with(&|v| v.fixed_commitments.0.push(point(7))), "one commitment per fixed f"),
        (with(&|v| v.format_version = 2), "format version 1"),
        (with(&|v| v.power = 29), "power is at most 28"),
        (
            with(&|v| {
                v.ev_map.pop();
            }),
            "the evMap is what the layout opens",
        ),
        (with(&|v| v.q_verifier = json!([])), "qVerifier is a code block"),
        (with(&|v| v.layout.0[1].k = 1), "k is the number of polynomials"),
        (with(&|v| v.max_q_degree = 2), "maxQDegree is 0 unless it splits Q"),
        (with(&|v| v.max_q_degree = 1), "Q split in two needs two pieces"),
    ] {
        assert!(vkey.validate().is_err(), "{why}");
    }
    // The offsets as the JS's checkLayout reads them (plan M40, review), with the reason.
    for (offsets, reason) in [
        (vec![0, 17], "offset 17, and an offset must be below N = 16 in absolute value"),
        (vec![-17, 0], "offset -17, and an offset must be below N = 16 in absolute value"),
        (vec![-1, 15], "two of them the same row modulo N = 16"),
    ] {
        let vkey = with(&|v| v.layout.0[1].offsets = offsets.clone());
        let err = vkey.validate().unwrap_err().to_string();
        assert!(err.contains(reason), "{offsets:?}: {err}");
    }

    let text = sample_vkey().to_json_string().unwrap();
    for (from, to) in [
        ("\"f0\": [", "\"f1\": ["),
        ("\"f0\": [", "\"f00\": ["),
        ("\"qDeg\": 2", "\"qDeg\": 2, \"f1\": [\"1\", \"2\"]"),
        ("\"qDeg\": 2", "\"qDeg\": 2, \"qdeg\": 2"),
        ("\"protocol\": \"pilfflonk\"", "\"protocol\": \"fflonk\""),
        ("\"curve\": \"bn128\"", "\"curve\": \"bls12381\""),
    ] {
        assert!(text.contains(from), "{from}");
        assert!(Vkey::from_json_str(&text.replacen(from, to, 1)).is_err(), "{to}");
    }
}

/// What the JS verifier refuses in a vkey the layout alone allows (`pilfflonk/js/src/vkey.js`,
/// `fromObjectVk`), refused here too, with the reason: neither read nor written. The sample has
/// two stages, 16 rows and the boundaries everyRow, firstRow and everyFrame {1, 2}.
#[test]
fn the_vkey_refuses_what_the_verifier_refuses() {
    let with = |change: &dyn Fn(&mut Vkey)| {
        let mut vkey = sample_vkey();
        change(&mut vkey);
        vkey
    };
    // A qVerifier that copies `operand` to its one temporary.
    let code = |operand: Value| {
        let dest = json!({"type": "tmp", "id": 0, "dim": 1});
        json!({"tmpUsed": 1, "code": [{"op": "copy", "dest": dest, "src": [operand]}]})
    };
    // A point of the twist outside G2, its r-torsion group: x = 2 + u (found with ffjavascript).
    let twist_not_g2 = G2Affine {
        x: [FqBytes::from_u64(2), FqBytes::from_u64(1)],
        y: [
            FqBytes::from_decimal("7292567877523311580221095596750716176434782432868683424513645834767876293070")
                .unwrap(),
            FqBytes::from_decimal("19659275751359636165940301690575149581329631496732780143538578556285923319774")
                .unwrap(),
        ],
    };
    for (vkey, reason) in [
        // X_2 as elements.js, g2FromObject, reads it (plan M40, review): the point at infinity, with
        // which anyone could forge a proof on the Solidity verifier, a point off the twist, and one
        // on it but not in G2.
        (with(&|v| v.x_2 = G2Affine::default()), "X_2 is not a point of G2 other than the point at infinity"),
        (with(&|v| v.x_2 = G2Affine::default()), "the point at infinity of G2"),
        (with(&|v| v.x_2.y[1] = FqBytes::from_u64(1)), "not on the twist"),
        (with(&|v| v.x_2 = twist_not_g2), "not in G2"),
        (with(&|v| v.num_challenges = vec![0]), "numChallenges has 1 stages, and the layout 2"),
        (with(&|v| v.num_challenges = vec![0, 2, 0]), "numChallenges has 3 stages, and the layout 2"),
        (with(&|v| v.num_challenges = vec![1, 2]), "A.4 squeezes no challenge of stage 1"),
        (with(&|v| v.boundaries.clear()), "boundaries[0] must be everyRow"),
        (with(&|v| v.boundaries.swap(0, 1)), "boundaries[0] must be everyRow"),
        (
            with(&|v| v.boundaries.push(Boundary::EveryFrame { offset_min: 8, offset_max: 8 })),
            "boundaries[3] is everyFrame {8, 8}: no row of 16 is left",
        ),
        (
            with(&|v| v.boundaries[2] = Boundary::EveryFrame { offset_min: u64::MAX, offset_max: 1 }),
            "no row of 16 is left",
        ),
        (with(&|v| v.q_verifier = json!({"tmpUsed": 1, "code": []})), "qVerifier: no code"),
        (with(&|v| v.q_verifier = code(json!({"type": "eval", "id": 13, "dim": 1}))), "eval 13 is not one of the 13"),
        (with(&|v| v.q_verifier = code(json!({"type": "public", "id": 2, "dim": 1}))), "public 2 is not one of the 2"),
        (
            with(&|v| v.q_verifier = code(json!({"type": "Zi", "id": 0, "boundaryId": 3, "dim": 1}))),
            "Zi of boundary 3, and there are 3",
        ),
        // std_vc is challenge 2 of challengesMap [0, 2]: after the two of stage 2.
        (
            with(&|v| v.q_verifier = code(json!({"type": "challenge", "id": 0, "stage": 3, "stageId": 0, "dim": 1}))),
            "the challenge of stage 3, stageId 0 is 2, not 0",
        ),
        (
            with(&|v| v.q_verifier = code(json!({"type": "challenge", "id": 2, "stage": 2, "stageId": 2, "dim": 1}))),
            "no challenge of stage 2 has stageId 2",
        ),
    ] {
        let err = vkey.validate().expect_err(reason).to_string();
        assert!(err.contains(reason), "{reason}: {err}");
        assert!(vkey.to_json_string().is_err(), "{reason}: it is not written");
    }

    // What they allow: an everyFrame that leaves one row, and every challenge at its position.
    let everyframe = with(&|v| v.boundaries.push(Boundary::EveryFrame { offset_min: 8, offset_max: 7 }));
    assert!(everyframe.validate().is_ok());
    for (id, stage, stage_id) in [(0, 2, 0), (1, 2, 1), (2, 3, 0), (3, 4, 0)] {
        let challenge = json!({"type": "challenge", "id": id, "stage": stage, "stageId": stage_id, "dim": 1});
        assert!(with(&|v| v.q_verifier = code(challenge.clone())).validate().is_ok(), "challenge {id}");
    }

    // And on reading.
    let text = sample_vkey().to_json_string().unwrap();
    for (from, to) in [
        ("\"numChallenges\": [\n  0,\n  2\n ]", "\"numChallenges\": [1, 2]"),
        ("\"boundaries\": [\n  {\n   \"name\": \"everyRow\"\n  },", "\"boundaries\": ["),
        ("\"op\": \"mul\"", "\"op\": \"div\""),
    ] {
        assert!(text.contains(from), "{from}");
        assert!(Vkey::from_json_str(&text.replacen(from, to, 1)).is_err(), "{to}");
    }
}

// ---------------------------------------------------------------------------------------------
// The digest's preimage (A.6).
// ---------------------------------------------------------------------------------------------

#[test]
fn the_digest_preimage_is_the_canonical_vkey_without_its_digest() {
    let vkey = sample_vkey();
    let preimage = vkey.digest_preimage().unwrap();
    assert!(preimage.starts_with(DIGEST_DOMAIN) && DIGEST_DOMAIN == b"pilfflonk-v1");
    let canonical = std::str::from_utf8(&preimage[DIGEST_DOMAIN.len()..]).unwrap();
    assert!(!canonical.contains("digest") && !canonical.contains(' ') && !canonical.contains('\n'));
    let [x_c0, x_c1, y_c0, y_c1] = G2_GENERATOR;
    let x_2 = format!("{{\"X_2\":[[\"{x_c0}\",\"{x_c1}\"],[\"{y_c0}\",\"{y_c1}\"]],\"boundaries\":");
    assert!(canonical.starts_with(&x_2), "{canonical}");

    // The canonical form of the file, parsed and less its digest, is the same text.
    let mut value: Value = serde_json::from_str(&vkey.to_json_string().unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("digest");
    assert_eq!(canonical_json(&value).unwrap(), canonical);

    assert!(vkey.digest_matches(fake_hash).unwrap());
    let mut other = vkey.clone();
    other.digest.0[0] ^= 1;
    assert_eq!(other.digest_preimage().unwrap(), preimage, "the digest is not part of its own preimage");
    assert!(!other.digest_matches(fake_hash).unwrap());
}

#[test]
fn the_digest_preimage_changes_with_every_field() {
    let base = sample_vkey().digest_preimage().unwrap();
    let changes: [fn(&mut Vkey); 12] = [
        |v| v.n_public = 3,
        |v| v.power = 5,
        |v| v.power_w = 12,
        |v| v.x_2.y[1] = FqBytes::from_u64(1),
        |v| v.num_challenges = vec![0, 3],
        |v| v.ev_map[0].prime = -2,
        |v| v.layout.0[0].degree = 51,
        |v| {
            v.boundaries.pop();
        },
        |v| v.fixed_commitments.0[0] = point(200),
        |v| v.q_deg = 3,
        |v| v.max_q_degree = 1,
        |v| v.q_verifier["tmpUsed"] = json!(2),
    ];
    for (i, change) in changes.iter().enumerate() {
        let mut vkey = sample_vkey();
        change(&mut vkey);
        assert_ne!(vkey.digest_preimage().unwrap(), base, "change {i}");
    }
}

#[test]
fn the_digest_preimage_does_not_depend_on_the_order_of_the_keys_of_the_file() {
    let vkey = sample_vkey();
    let value: Value = serde_json::from_str(&vkey.to_json_string().unwrap()).unwrap();
    let reordered = reversed_json(&value);
    assert!(reordered.starts_with("{\"qVerifier\""), "{}", &reordered[..40]);
    let back = Vkey::from_json_str(&reordered).unwrap();
    assert_eq!(back, vkey);
    assert_eq!(back.digest_preimage().unwrap(), vkey.digest_preimage().unwrap());
    assert_eq!(
        canonical_json(&serde_json::from_str::<Value>(&reordered).unwrap()).unwrap(),
        canonical_json(&value).unwrap()
    );
}

// ---------------------------------------------------------------------------------------------
// The proof: names, bytes and JSON view.
// ---------------------------------------------------------------------------------------------

fn sample_proof(names: &ProofNames) -> Proof {
    let shape = names.shape();
    let scalars = |base: u64, n: usize| (0..n as u64).map(|i| FrBytes::from_u64(base + i)).collect();
    let mut evaluations: Vec<FrBytes> = scalars(1000, shape.n_evaluations);
    evaluations[0] = r_minus_1();
    Proof {
        commitments: (0..shape.n_commitments as u64).map(|i| point(10 * i + 1)).collect(),
        w: point(500),
        wp: point(600),
        evaluations,
        air_values: scalars(2000, shape.n_air_values),
        airgroup_values: scalars(3000, shape.n_airgroup_values),
        proof_values: scalars(4000, shape.n_proof_values),
        inv: FrBytes::from_u64(7),
        inv_zh: FrBytes::from_u64(8),
    }
}

#[test]
fn a_proof_of_one_instance_has_pil_fflonks_names() {
    let names = ProofNames::new(&sample_global_info(), &[&sample_info()]).unwrap();
    // f0 is fixed: it is the vkey's.
    assert_eq!(names.commitments(), ["f1", "f2", "f3", "f4"]);
    assert_eq!(
        names.evaluations(),
        [
            "Sample.L1",
            "Sample.C[0]",
            "Sample.C[1]",
            "Sample.gsumw-1",
            "Sample.imw-1",
            "Sample.a",
            "Sample.b",
            "Sample.c[0]",
            "Sample.c[1]",
            "Sample.gsum",
            "Sample.im",
            "Sample.aw",
            "Sample.bw",
        ]
    );
    assert_eq!(names.air_values(), ["Sample.av"]);
    assert_eq!(names.airgroup_values(), ["Sample.gsum_result"]);
    assert_eq!(names.proof_values(), ["pv"]);

    let split = ProofNames::new(&sample_global_info(), &[&split_q_info()]).unwrap();
    assert_eq!(split.evaluations()[13..], ["Q0", "Q1"], "the Q_i(ξ) go after the other evaluations");
}

#[test]
fn a_proof_round_trips_through_its_bytes_and_its_json_view() {
    for info in [sample_info(), split_q_info()] {
        let names = ProofNames::new(&sample_global_info(), &[&info]).unwrap();
        let proof = sample_proof(&names);
        let bytes = proof.to_bytes();
        assert_eq!(bytes.len(), names.shape().byte_len());
        assert_eq!(Proof::from_bytes(&bytes, &names.shape()).unwrap(), proof);

        let json = proof.to_json(&names).unwrap();
        let text = json.to_json_string().unwrap();
        let back = Proof::from_json(&ProofJson::from_json_str(&text).unwrap(), &names).unwrap();
        assert_eq!(back, proof);
        assert_eq!(back.to_bytes(), bytes, "bytes → JSON → bytes is the identity");
    }
}

#[test]
fn the_proof_bytes_follow_a6() {
    let names = ProofNames::new(&sample_global_info(), &[&sample_info()]).unwrap();
    let proof = sample_proof(&names);
    let bytes = proof.to_bytes();
    // f1: x = 1, y = 2, 32 bytes big-endian each.
    assert_eq!(bytes[..64], point(1).to_be_bytes());
    assert_eq!((bytes[31], bytes[63]), (1, 2));
    // Then f2 … f4, W and W'.
    let scalars_at = 64 * (4 + 2);
    assert_eq!(bytes[64 * 4..64 * 5], point(500).to_be_bytes());
    assert_eq!(bytes[64 * 5..scalars_at], point(600).to_be_bytes());
    // The evaluations, then the air, airgroup and proof values, then inv and invZh.
    assert_eq!(bytes[scalars_at..scalars_at + 32], r_minus_1().to_be_bytes());
    assert_eq!(bytes[scalars_at..scalars_at + 2], [0x30, 0x64]);
    let values_at = scalars_at + 32 * 13;
    assert_eq!(bytes[values_at..values_at + 32], FrBytes::from_u64(2000).to_be_bytes());
    assert_eq!(bytes[values_at + 32..values_at + 64], FrBytes::from_u64(3000).to_be_bytes());
    assert_eq!(bytes[values_at + 64..values_at + 96], FrBytes::from_u64(4000).to_be_bytes());
    assert_eq!(bytes[values_at + 96..values_at + 128], FrBytes::from_u64(7).to_be_bytes());
    assert_eq!(bytes[values_at + 128..], FrBytes::from_u64(8).to_be_bytes());

    // The wrong length, and a scalar that is not canonical, are refused.
    assert!(Proof::from_bytes(&bytes[1..], &names.shape()).is_err());
    let mut bad = bytes.clone();
    bad[scalars_at..scalars_at + 32].copy_from_slice(&[0xff; 32]);
    assert!(Proof::from_bytes(&bad, &names.shape()).is_err());
}

#[test]
fn the_json_view_is_snarkjs_style() {
    let names = ProofNames::new(&sample_global_info(), &[&sample_info()]).unwrap();
    let value: Value =
        serde_json::from_str(&sample_proof(&names).to_json(&names).unwrap().to_json_string().unwrap()).unwrap();
    assert_eq!(value["protocol"], "pilfflonk");
    assert_eq!(value["curve"], "bn128");
    assert_eq!(value["polynomials"]["f1"], json!(["1", "2", "1"]));
    assert_eq!(value["polynomials"]["W"], json!(["500", "501", "1"]));
    assert_eq!(value["polynomials"]["Wp"], json!(["600", "601", "1"]));
    assert!(value["polynomials"].get("f0").is_none(), "the fixed commitments are not in the proof");
    assert_eq!(value["evaluations"]["Sample.gsumw-1"], "1003");
    assert_eq!(value["evaluations"]["Sample.aw"], "1011");
    assert_eq!(value["evaluations"]["inv"], "7");
    assert_eq!(value["evaluations"]["invZh"], "8");
    assert_eq!(value["evaluations"]["pv"], "4000");
    assert_eq!(value["evaluations"].as_object().unwrap().len(), 13 + 3 + 2);
}

#[test]
fn a_json_view_must_have_exactly_the_names_of_the_proof() {
    let names = ProofNames::new(&sample_global_info(), &[&sample_info()]).unwrap();
    let json = sample_proof(&names).to_json(&names).unwrap();
    let mut missing = json.clone();
    missing.evaluations.remove("Sample.aw");
    assert!(Proof::from_json(&missing, &names).is_err());
    let mut extra = json.clone();
    extra.polynomials.insert("f0".into(), extra.polynomials["f1"]);
    assert!(Proof::from_json(&extra, &names).is_err());

    let mut infinity = sample_proof(&names);
    infinity.w = G1Affine::INFINITY;
    assert!(infinity.to_json(&names).is_err(), "[x, y, \"1\"] cannot be the point at infinity");
}

#[test]
fn a_proof_of_several_instances_prefixes_its_names() {
    let mut global_info = sample_global_info();
    global_info.airs[0].push(GlobalInfoAir { name: "Other".into(), num_rows: 16 });
    let mut other = sample_info();
    other.name = "Other".into();
    other.air_id = 1;
    let sample = sample_info();

    let names = ProofNames::new(&global_info, &[&sample, &sample, &other]).unwrap();
    // A fixed f per AIR, f0 and f1, then the four non-fixed f of each instance.
    assert_eq!(names.commitments().len(), 12);
    assert_eq!(names.commitments()[0], "f2");
    assert_eq!(names.commitments()[11], "f13");
    let evaluations = names.evaluations();
    assert_eq!(evaluations[0], "0.0:Sample.L1");
    assert_eq!(evaluations[3], "0.1:Sample.L1");
    assert_eq!(evaluations[6], "0.0.0:Sample.gsumw-1");
    assert_eq!(evaluations[6 + 10], "0.0.1:Sample.gsumw-1");
    assert_eq!(evaluations[6 + 20], "0.1.0:Sample.gsumw-1");
    assert_eq!(evaluations.len(), 3 * 2 + 10 * 3);
    assert_eq!(names.air_values(), ["0.0.0:Sample.av", "0.0.1:Sample.av", "0.1.0:Sample.av"]);
    assert_eq!(names.airgroup_values(), ["0:Sample.gsum_result"]);
    assert_eq!(names.proof_values(), ["pv"]);

    let proof = sample_proof(&names);
    let json = proof.to_json(&names).unwrap();
    assert_eq!(Proof::from_json(&json, &names).unwrap(), proof);

    // Not in canonical order, or not the globalInfo's AIRs.
    assert!(ProofNames::new(&global_info, &[&other, &sample]).is_err());
    assert!(ProofNames::new(&sample_global_info(), &[&sample, &other]).is_err());
    assert!(ProofNames::new(&global_info, &[]).is_err());
}

#[test]
fn names_that_collide_are_refused() {
    // A column "Sample.aw" at offset 0 is named as "Sample.a" at offset 1.
    let mut info = sample_info();
    info.cm_pols_map[1].name = "Sample.aw".into();
    info.layout.0[1].pols[1].name = "Sample.aw".into();
    assert!(info.validate().is_ok());
    let err = ProofNames::new(&sample_global_info(), &[&info]).unwrap_err();
    assert!(err.to_string().contains("\"Sample.aw\""), "{err}");

    // Two columns named alike, as the std names its `im_cluster`, without the index the setup gives
    // each (spec A.6, plan M34b); with it, `Sample.a[0]` and `Sample.a[1]`, they are named apart.
    let mut info = sample_info();
    info.cm_pols_map[1].name = "Sample.a".into();
    info.layout.0[1].pols[1].name = "Sample.a".into();
    assert!(info.validate().is_ok());
    let err = ProofNames::new(&sample_global_info(), &[&info]).unwrap_err();
    assert!(err.to_string().contains("\"Sample.a\""), "{err}");
    for (i, name) in ["Sample.a[0]", "Sample.a[1]"].into_iter().enumerate() {
        info.cm_pols_map[i].lengths = vec![i as u64];
        info.layout.0[1].pols[i].name = name.into();
    }
    assert!(info.validate().is_ok());
    let names = ProofNames::new(&sample_global_info(), &[&info]).unwrap();
    assert_eq!(names.evaluations()[5..7], ["Sample.a[0]", "Sample.a[1]"]);
    assert_eq!(names.evaluations()[11..], ["Sample.a[0]w", "Sample.a[1]w"]);
}

// ---------------------------------------------------------------------------------------------
// The Rust side of the Rust → file → C++ round trip.
// ---------------------------------------------------------------------------------------------

/// Read by `pil2-stark/test/pilfflonk/pilfflonk_info_test.cpp`, which checks every value of the
/// sample. Regenerate it with `PILFFLONK_UPDATE_FIXTURES=1 cargo test -p proofman-pilfflonk`, and
/// update the C++ test with it.
const PILFFLONKINFO_FIXTURE: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pilfflonkinfo/Sample.pilfflonkinfo.json");

#[test]
fn the_cpp_fixture_is_what_the_rust_types_write() {
    let text = sample_info().to_json_string().unwrap();
    let path = Path::new(PILFFLONKINFO_FIXTURE);
    if std::env::var_os("PILFFLONK_UPDATE_FIXTURES").is_some() {
        fs::write(path, &text).unwrap();
    }
    let fixture = fs::read_to_string(path).unwrap();
    assert!(fixture == text, "{PILFFLONKINFO_FIXTURE} is not what PilfflonkInfo writes: regenerate it (see above)");
    assert_eq!(PilfflonkInfo::read(path).unwrap(), sample_info());
}

// ---------------------------------------------------------------------------------------------
// The STARK loader refuses a pilfflonk globalInfo (spec §4.2.6, plan M12).
// ---------------------------------------------------------------------------------------------

#[test]
fn the_stark_loader_refuses_a_pilfflonk_global_info() {
    let dir = TempDir::new("stark_loader");
    sample_global_info().write(&dir.0.join(GLOBAL_INFO_FILE)).unwrap();

    let Err(err) = proofman_common::GlobalInfo::from_file(&dir.0.display().to_string()) else {
        panic!("common::GlobalInfo must not load a pilfflonk globalInfo");
    };
    // The first field it requires that the pilfflonk file does not have.
    let err = err.to_string();
    assert!(err.contains("missing field `curve`"), "{err}");

    // With the STARK's fields added, it would: what refuses the file is exactly their absence.
    let mut value: Value = serde_json::from_str(&fs::read_to_string(dir.0.join(GLOBAL_INFO_FILE)).unwrap()).unwrap();
    let object = value.as_object_mut().unwrap();
    object.insert("curve".into(), json!("None"));
    let err = stark_load(&dir.0, &value);
    assert!(err.contains("missing field `transcriptArity`"), "{err}");

    // And the pilfflonk runtime reads it with its own type.
    fs::write(dir.0.join(GLOBAL_INFO_FILE), sample_global_info().to_json_string().unwrap()).unwrap();
    assert_eq!(PilfflonkGlobalInfo::from_proving_key(&dir.0).unwrap(), sample_global_info());
}

/// The error `common::GlobalInfo::from_file` gives for this globalInfo.
fn stark_load(dir: &Path, value: &Value) -> String {
    fs::write(dir.join(GLOBAL_INFO_FILE), serde_json::to_string(value).unwrap()).unwrap();
    match proofman_common::GlobalInfo::from_file(&dir.display().to_string()) {
        Ok(_) => String::new(),
        Err(err) => err.to_string(),
    }
}
