//! The main's data checks on the generator's own output and on that output broken; no circom needed.

use proofman_common::StarkInfo;
use serde_json::{json, Value};

use super::syntax::data_functions;
use super::{data, deep, proof, Vadcop};
use crate::stark2circom::circom_verifier::{gen_stark_verifier_gl, key_dispatch, Pil2CircomOptions};
use crate::stark2circom::circuit_templates::get_sha256_inputs::sha256_defs;
use crate::stark2circom::circuit_templates::vadcop_inputs::vadcop_data;

/// The sample key (FRI with grinding, a custom commit, an airgroup value, an opening without evaluations).
fn sample() -> Value {
    crate::stark2circom::circom_verifier::lib_gl::tests::stark_info()
}

/// The sample over STIR: two folds of 3, queries 10 then 6.
fn sample_stir() -> Value {
    let mut si = sample();
    let ss = si["starkStruct"].as_object_mut().unwrap();
    for (k, v) in [
        ("lowDegreeTest", json!("STIR")),
        ("foldingFactors", json!([3, 3])),
        ("logDegrees", json!([16, 13, 10])),
        ("logDomainSizes", json!([17, 16, 15])),
        ("numQueries", json!([10, 6])),
        ("grindingBitsQueries", json!([8, 4])),
    ] {
        ss.insert(k.into(), v);
    }
    si
}

fn main_of(si: &Value) -> String {
    let vi = crate::stark2circom::circom_verifier::lib_gl::tests::verifier_info();
    let opts = Pil2CircomOptions { verkey_input: true, ..Default::default() };
    let v = gen_stark_verifier_gl(None, si, &vi, &opts).unwrap();
    key_dispatch(std::slice::from_ref(&v.data), true)
}

/// The offsets, DEEP and batch checks of key 0 of `main`.
fn check(si: &Value, main: &str) -> anyhow::Result<()> {
    let info = StarkInfo::try_from_value(si)?;
    let fns = data_functions(main, "key_")?;
    data::check_key_data(&fns, 0, &proof::expected_offsets(&info)?)?;
    deep::check_deep(&info, &fns, 0)?;
    data::check_batch_size(&fns, 0)
}

/// `main` with `key_<name>` returning `value`.
fn with(main: &str, name: &str, value: &str) -> String {
    let head = format!("function key_{name}(k) {{ return ");
    let line = main.lines().find(|l| l.starts_with(&head)).unwrap_or_else(|| panic!("no key_{name}"));
    main.replace(line, &format!("{head}{value}; }}"))
}

fn value(main: &str, name: &str) -> String {
    let head = format!("function key_{name}(k) {{ return ");
    let line = main.lines().find(|l| l.starts_with(&head)).unwrap();
    line[head.len()..line.len() - 3].to_string()
}

fn refused(r: anyhow::Result<()>, why: &str) {
    let e = format!("{:#}", r.expect_err(why));
    assert!(e.contains(why), "{e}");
}

#[test]
fn sample_data_passes() {
    for si in [sample(), sample_stir()] {
        let main = main_of(&si);
        check(&si, &main).unwrap_or_else(|e| panic!("{e:#}\n{main}"));
    }
}

#[test]
fn a_wrong_offset_is_refused() {
    let (fri, stir) = (sample(), sample_stir());
    let (main, main_stir) = (main_of(&fri), main_of(&stir));
    let evals: i64 = value(&main, "offEvals").parse().unwrap();
    refused(check(&fri, &with(&main, "offEvals", &(evals + 3).to_string())), "key_offEvals(0)");
    let mut fold = numbers(&main_stir, "OFF_FOLD_SIBLINGS");
    fold[1] += 1;
    refused(check(&stir, &with(&main_stir, "OFF_FOLD_SIBLINGS", &list(&fold))), "key_OFF_FOLD_SIBLINGS(0)");
    let size: i64 = value(&main_stir, "proofSize").parse().unwrap();
    refused(check(&stir, &with(&main_stir, "proofSize", &(size - 1).to_string())), "key_proofSize(0)");
}

fn numbers(main: &str, name: &str) -> Vec<i64> {
    value(main, name).trim_matches(['[', ']']).split(", ").map(|x| x.parse().unwrap()).collect()
}

fn list(v: &[i64]) -> String {
    format!("[{}]", v.iter().map(i64::to_string).collect::<Vec<_>>().join(", "))
}

#[test]
fn a_dropped_or_duplicated_deep_column_is_refused() {
    let si = sample();
    let main = main_of(&si);
    // the first group: its A source, its column count, then (column, offset, dim) per column
    let g = numbers(&main, "COLUMN_GROUPS");
    assert!(g[1] >= 2, "the sample's first group has two columns: {g:?}");
    let mut dropped = g.clone();
    dropped[1] -= 1;
    dropped.drain(2..5);
    refused(check(&si, &with(&main, "COLUMN_GROUPS", &list(&dropped))), "is in no DEEP column group");
    let mut twice = g.clone();
    twice[1] += 1;
    let first: Vec<i64> = g[2..5].to_vec();
    twice.splice(2..2, first);
    refused(check(&si, &with(&main, "COLUMN_GROUPS", &list(&twice))), "twice");
    let mut moved = g;
    moved[3] += 1;
    refused(check(&si, &with(&main, "COLUMN_GROUPS", &list(&moved))), "is in no DEEP column group");
}

#[test]
fn a_wrong_deep_opening_is_refused() {
    let si = sample();
    let main = main_of(&si);
    let exp = value(&main, "OPENING_EXPONENT");
    assert_eq!(exp, "[-1, 1]");
    refused(check(&si, &with(&main, "OPENING_EXPONENT", "[-1, 2]")), "has exponent 2");
    let roots = value(&main, "OPENING_ROOT");
    let mut swapped: Vec<&str> = roots.trim_matches(['[', ']']).split(", ").collect();
    swapped.reverse();
    refused(check(&si, &with(&main, "OPENING_ROOT", &format!("[{}]", swapped.join(", ")))), "has root");
    let cols = value(&main, "EVAL_COLUMN").replacen("[0", "[1", 1);
    refused(check(&si, &with(&main, "EVAL_COLUMN", &cols)), "the K sums bind");
    refused(check(&si, &with(&main, "OPENING_NUM_EVALS", "[4, 3]")), "the K sums bind");
}

#[test]
fn a_batch_size_outside_the_queries_is_refused() {
    let si = sample();
    let main = main_of(&si);
    refused(check(&si, &with(&main, "batchSize", "0")), "key_batchSize(0) is 0");
    refused(check(&si, &with(&main, "batchSize", "11")), "key_batchSize(0) is 11");
}

/// One airgroup of two airs and one of one.
fn global_info() -> Value {
    json!({
        "name": "g", "air_groups": ["a", "b"], "airs": [[{"name": "x"}, {"name": "y"}], [{"name": "z"}]],
        "aggTypes": [[], []], "curve": "None", "latticeSize": 368, "nPublics": 2, "numProofValues": [0],
        "proofValuesMap": [], "numChallenges": [0, 0]
    })
}

fn vk(x: u64) -> [u64; 4] {
    [x, x + 1, x + 2, x + 3]
}

fn strings(k: [u64; 4]) -> Vec<String> {
    k.iter().map(u64::to_string).collect()
}

fn check_vadcop(main: &str, role: &Vadcop, first_key: Option<&Value>) -> anyhow::Result<()> {
    let fns = data_functions(main, "vadcop_")?;
    data::check_vadcop_data(&fns, &data::expected_vadcop_data(&global_info(), role, first_key)?)
}

#[test]
fn vadcop_ids_and_child_keys() {
    let gi = global_info();
    let r1 = vec![vec![vk(1), vk(5)], vec![vk(9)]];
    let r2 = vec![vk(20), vk(30)];
    let s = |v: &Vec<[u64; 4]>| v.iter().map(|k| strings(*k)).collect::<Vec<_>>();
    let (basics, aggs) = (vec![s(&r1[0]), s(&r1[1])], vec![strings(r2[0]), strings(r2[1])]);

    let fin = vadcop_data(&gi, &basics, &aggs, 0, 0).unwrap() + "function vadcop_nGlobalChallenges() { return 0; }\n";
    let role = Vadcop::Final { recursive1: r1.clone(), recursive2: r2.clone() };
    check_vadcop(&fin, &role, None).unwrap();
    let wrong = Vadcop::Final { recursive1: vec![vec![vk(1), vk(6)], vec![vk(9)]], recursive2: r2.clone() };
    refused(check_vadcop(&fin, &wrong, None), "vadcop_basics(0)");
    let wrong = Vadcop::Final { recursive1: r1.clone(), recursive2: vec![vk(20), vk(31)] };
    refused(check_vadcop(&fin, &wrong, None), "vadcop_rootCAgg(1)");
    let missing = Vadcop::Final { recursive1: vec![vec![vk(1)], vec![vk(9)]], recursive2: r2 };
    refused(check_vadcop(&fin, &missing, None), "2 airs and 1 recursive1 keys");

    let rec2 = vadcop_data(&gi, &[vec![], basics[1].clone()], &[], 1, 0).unwrap();
    check_vadcop(&rec2, &Vadcop::Airgroup { airgroup: 1, recursive1: r1[1].clone() }, None).unwrap();
    refused(check_vadcop(&rec2, &Vadcop::Airgroup { airgroup: 0, recursive1: r1[0].clone() }, None), "vadcop_");

    let rec1 = vadcop_data(&gi, &[], &[], 0, 1).unwrap();
    check_vadcop(&rec1, &Vadcop::Air, Some(&json!({"airgroupId": 0, "airId": 1}))).unwrap();
    refused(check_vadcop(&rec1, &Vadcop::Air, Some(&json!({"airgroupId": 0, "airId": 0}))), "vadcop_airId()");
    refused(check_vadcop(&rec1, &Vadcop::Air, Some(&json!({"airgroupId": 1, "airId": 1}))), "not in globalInfo");
}

/// The final main's data for `publics_info`, as `gen_recursion_final` writes it.
fn final_main(publics_info: &Value) -> String {
    let s = sha256_defs(Some(publics_info)).unwrap();
    let vk = s.vk.map_or(-1, |v| v.pos as i64);
    format!("{}    component f = FinalSnark({}, {}, {vk});\n", s.defs, s.n_publics, s.n_defs)
}

fn check_defs(main: &str, publics_info: &Value) -> anyhow::Result<()> {
    data::check_publics_defs(&data_functions(main, "vadcop_")?, main, publics_info)
}

#[test]
fn publics_definitions() {
    let pi = json!({"nPublics": 8, "definitions": [
        {"name": "a", "initialPos": 0, "nValues": 4, "chunks": [1, 64]},
        {"name": "vk", "initialPos": 4, "nValues": 4, "chunks": [1, 64], "verificationKey": true}
    ]});
    let main = final_main(&pi);
    check_defs(&main, &pi).unwrap();
    assert!(data::unhashed_publics(&pi).is_empty());
    check_defs(&final_main(&Value::Null), &Value::Null).unwrap();

    let mut other = pi.clone();
    other["definitions"][0]["nValues"] = json!(3);
    refused(check_defs(&main, &other), "vadcop_DEFS()");
    assert_eq!(data::unhashed_publics(&other), [3]);
    let mut past = pi.clone();
    past["nPublics"] = json!(7);
    refused(check_defs(&final_main(&past), &past), "past the 7 publics");
    refused(check_defs(&main.replace("FinalSnark(8, 2, 4)", "FinalSnark(8, 2, -1)"), &pi), "FinalSnark(8, 2, 4)");
}
