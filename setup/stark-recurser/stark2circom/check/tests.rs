//! The checker on the generator's own output (sample keys, random DAGs) and on that output broken.
//! Each test compiles with circom and runs node (and g++ for the global constraints); without them it says so
//! and passes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use constraint_dag::{ColType, ConstraintDag, DagBoundary, DagNode, DagRoot};
use serde_json::{json, Value};

use super::witness::Circom;
use super::{check_circuit, check_global, Circuit, Field, Key};
use crate::stark2circom::circom_verifier::constraint_roots::global_constraints;
use crate::stark2circom::circom_verifier::{
    gen_stark_verifier_bn128, gen_stark_verifier_gl, key_dispatch, KeyVerifier, Pil2CircomOptions,
};

fn library() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("stark2circom/circom_verifier")
}

fn circom_exec() -> Option<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let exec = root.join("setup/circom").join(if cfg!(target_os = "macos") { "circom_mac" } else { "circom" });
    let have = |cmd: &str, arg: &str| Command::new(cmd).arg(arg).output().is_ok_and(|o| o.status.success());
    let missing = [("node", have("node", "--version")), ("g++", have("g++", "--version")), ("circom", exec.is_file())]
        .into_iter()
        .find(|(_, ok)| !ok);
    if let Some((tool, _)) = missing {
        eprintln!("skipped: {tool} not available");
        return None;
    }
    Some(exec.to_string_lossy().into_owned())
}

fn gl_libs() -> Vec<String> {
    ["helper_circuits", "circuits.gl"].map(|l| library().join(l).to_string_lossy().into_owned()).to_vec()
}

/// A directory of the test's own, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("check_templates_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// xorshift: the shape of the random DAGs is reproducible, their check inputs are not.
struct Shape(u64);

impl Shape {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// `n_ops` operations over `leaves`, mostly on recent nodes; the roots are the nodes nothing reads.
fn random_dag(leaves: Vec<DagNode>, n_ops: usize, seed: u64, boundary: DagBoundary) -> ConstraintDag {
    let mut shape = Shape(seed);
    let mut nodes = leaves;
    let mut seen: BTreeSet<DagNode> = nodes.iter().cloned().collect();
    let target = nodes.len() + n_ops;
    for _ in 0..100 * n_ops {
        if nodes.len() == target {
            break;
        }
        let i = nodes.len();
        let mut pick = || if shape.below(4) == 0 { shape.below(i) } else { i - 1 - shape.below(i.min(12)) } as u32;
        let (a, b) = (pick(), pick());
        let n = match shape.below(7) {
            0 | 1 => DagNode::Add { a: a.min(b), b: a.max(b) },
            2 | 3 => DagNode::Sub { a, b },
            4 => DagNode::Neg { a },
            _ => DagNode::Mul { a: a.min(b), b: a.max(b) },
        };
        if seen.insert(n.clone()) {
            nodes.push(n);
        }
    }
    let read: BTreeSet<u32> = nodes.iter().flat_map(DagNode::children).collect();
    let roots = (0..nodes.len() as u32)
        .filter(|i| !read.contains(i))
        .map(|node| DagRoot { node, boundary: boundary.clone(), label: format!("r{node}") })
        .collect();
    ConstraintDag { version: ConstraintDag::VERSION, field: "gl".into(), ext_dim: 3, nodes, roots }
}

fn col(col: ColType, id: u32, offset: i64) -> DagNode {
    let commit_id = (col == ColType::Custom).then_some(0);
    DagNode::Col { col, commit_id, id, offset }
}

fn constant(v: &str) -> DagNode {
    DagNode::Const { value: v.into() }
}

/// Every leaf the sample starkinfo (`lib_gl::tests::stark_info`) resolves, a zero and the largest constant.
fn sample_leaves() -> Vec<DagNode> {
    vec![
        col(ColType::Witness, 0, -1),
        col(ColType::Witness, 1, -1),
        col(ColType::Custom, 0, -1),
        col(ColType::Witness, 0, 1),
        col(ColType::Fixed, 0, 1),
        col(ColType::Custom, 0, 1),
        col(ColType::Witness, 2, 1),
        DagNode::Challenge { stage: 2, stage_id: 0 },
        DagNode::Challenge { stage: 3, stage_id: 0 },
        DagNode::Public { id: 0 },
        DagNode::Public { id: 3 },
        DagNode::AirgroupValue { airgroup_id: None, id: 0 },
        constant("0"),
        constant("3"),
        constant("18446744069414584320"),
    ]
}

fn verifier_info(dag: &ConstraintDag) -> Value {
    json!({ "constraintDag": serde_json::to_value(dag).unwrap() })
}

/// The generated key of the sample starkinfo over `vi`, its circom and its main (data and dispatch).
fn gl_key(vi: &Value) -> (Value, KeyVerifier, String) {
    let si = crate::stark2circom::circom_verifier::lib_gl::tests::stark_info();
    let opts = Pil2CircomOptions { verkey_input: true, ..Default::default() };
    let v = gen_stark_verifier_gl(None, &si, vi, &opts).unwrap();
    let main = key_dispatch(std::slice::from_ref(&v.data), true);
    (si, v, main)
}

/// Checks the GL key `circom` with main `main`, as the setup would.
fn check_gl(name: &str, si: &Value, vi: &Value, circom: &str, main: &str) -> Option<anyhow::Result<()>> {
    let exec = circom_exec()?;
    let dir = Scratch::new(name);
    let file = dir.0.join("key.verifier.circom");
    std::fs::write(&file, circom).unwrap();
    let key = Key {
        verifier_file: &file,
        stark_info: si,
        verifier_info: vi,
        const_root: None,
        enable_input: false,
        input_challenges: false,
    };
    let circuit = Circuit {
        field: Field::Goldilocks,
        main,
        keys: vec![key],
        global_info: &Value::Null,
        global_constraints: false,
    };
    let libs = gl_libs();
    Some(check_circuit(&Circom { exec: &exec, libs: &libs }, &dir.0.join("work"), &circuit).map(|_| ()))
}

/// `r` is a refusal whose message says `why`.
fn refused(r: Option<anyhow::Result<()>>, what: &str, why: &str) {
    if let Some(r) = r {
        let e = format!("{:#}", r.expect_err(what));
        assert!(e.contains(why), "{what}: {e}");
    }
}

#[test]
fn sample_key_passes() {
    let vi = crate::stark2circom::circom_verifier::lib_gl::tests::verifier_info();
    let (si, v, main) = gl_key(&vi);
    if let Some(Err(e)) = check_gl("sample", &si, &vi, &v.circom, &main) {
        panic!("{e:#}");
    }
}

/// ~600 operations: several parts, every leaf kind, a stage-1 airgroup value, zero and p - 1.
#[test]
fn random_dag_passes() {
    let vi = verifier_info(&random_dag(sample_leaves(), 600, 0x2545_f491_4f6c_dd1d, DagBoundary::EveryRow));
    let (si, v, main) = gl_key(&vi);
    assert!(v.circom.contains("Part2()"), "several parts");
    if let Some(Err(e)) = check_gl("random", &si, &vi, &v.circom, &main) {
        panic!("{e:#}");
    }
}

fn small_dag() -> Value {
    verifier_info(&random_dag(sample_leaves(), 60, 7, DagBoundary::EveryRow))
}

/// The first line of a part that `<==`-assigns `tmp_`: the one the negative tests break.
fn first_step(circom: &str) -> &str {
    circom.lines().find(|l| l.trim_start().starts_with("signal tmp_") && l.contains(" + ")).unwrap()
}

#[test]
fn a_flipped_sign_is_refused() {
    let vi = small_dag();
    let (si, v, main) = gl_key(&vi);
    let line = first_step(&v.circom);
    let broken = v.circom.replacen(line, &line.replacen(" + ", " - ", 1), 1);
    refused(check_gl("flipped", &si, &vi, &broken, &main), "a flipped sign", "the DAG gives");
}

#[test]
fn an_unconstrained_assignment_is_refused() {
    let vi = small_dag();
    let (si, v, main) = gl_key(&vi);
    let line = first_step(&v.circom);
    let broken = v.circom.replacen(line, &line.replacen(" <== ", " <-- ", 1), 1);
    refused(check_gl("unconstrained", &si, &vi, &broken, &main), "a <--", "an unconstrained assignment");
}

/// The last root's output left unassigned.
#[test]
fn a_dropped_root_is_refused() {
    let vi = small_dag();
    let (si, v, main) = gl_key(&vi);
    let last = v.circom.lines().rfind(|l| l.trim_start().starts_with("roots[")).unwrap();
    let broken = v.circom.replacen(&format!("{last}\n"), "", 1);
    refused(check_gl("dropped", &si, &vi, &broken, &main), "a dropped root", "the DAG gives");
}

#[test]
fn a_wrong_key_datum_is_refused() {
    let vi = small_dag();
    let (si, v, main) = gl_key(&vi);
    let datum = "function key_nEvals(k) { return 7; }";
    assert!(main.contains(datum), "{main}");
    let broken = main.replace(datum, "function key_nEvals(k) { return 8; }");
    refused(check_gl("datum", &si, &vi, &v.circom, &broken), "a wrong key_nEvals", "key_nEvals(0) is Int(8)");
}

#[test]
fn a_dispatch_to_another_template_is_refused() {
    let vi = small_dag();
    let (si, v, main) = gl_key(&vi);
    let broken = main.replace("roots <== ConstraintRoots()(", "roots <== ConstraintRootsX()(");
    refused(check_gl("dispatch", &si, &vi, &v.circom, &broken), "another template", "besides the constraint ones");
}

/// globalInfo of `constraint_roots`' tests: two airgroups (stages [1, 2] and [2]), 2 publics, proof values of
/// stages 1 and 2, challenges [1, 2].
fn global_info(dag: &ConstraintDag) -> Value {
    json!({
        "nPublics": 2, "numChallenges": [1, 2],
        "proofValuesMap": [{"stage": 1}, {"stage": 2}],
        "aggTypes": [[{"aggType": 0, "stage": 1}, {"aggType": 0, "stage": 2}], [{"aggType": 1, "stage": 2}]],
        "globalConstraintDag": serde_json::to_value(dag).unwrap(),
    })
}

fn random_global(n_ops: usize) -> Value {
    let leaves = vec![
        DagNode::Public { id: 0 },
        DagNode::Public { id: 1 },
        DagNode::ProofValue { id: 0 },
        DagNode::ProofValue { id: 1 },
        DagNode::AirgroupValue { airgroup_id: Some(0), id: 0 },
        DagNode::AirgroupValue { airgroup_id: Some(0), id: 1 },
        DagNode::AirgroupValue { airgroup_id: Some(1), id: 0 },
        DagNode::Challenge { stage: 2, stage_id: 0 },
        DagNode::Challenge { stage: 2, stage_id: 1 },
        constant("5"),
    ];
    global_info(&random_dag(leaves, n_ops, 11, DagBoundary::Global))
}

fn check_global_main(name: &str, main: &str, gi: &Value) -> Option<anyhow::Result<()>> {
    let exec = circom_exec()?;
    let dir = Scratch::new(name);
    let libs = gl_libs();
    Some(check_global(&Circom { exec: &exec, libs: &libs }, &dir.0, main, gi))
}

/// ~450 operations: three parts, every leaf kind, stage-1 values as their first limb.
#[test]
fn random_global_constraints_pass() {
    let gi = random_global(450);
    let main = global_constraints(&gi).unwrap();
    assert!(main.contains("GlobalConstraintsPart2("), "several parts");
    if let Some(Err(e)) = check_global_main("global", &main, &gi) {
        panic!("{e:#}");
    }
}

#[test]
fn a_dropped_global_root_is_refused() {
    let gi = random_global(40);
    let main = global_constraints(&gi).unwrap();
    let check = main.lines().find(|l| l.ends_with(" === 0;")).unwrap();
    refused(
        check_global_main("global_dropped", &main.replacen(&format!("{check}\n"), "", 1), &gi),
        "no zero check",
        "is not constrained to zero",
    );
}

#[test]
fn a_flipped_global_sign_is_refused() {
    let gi = random_global(40);
    let main = global_constraints(&gi).unwrap();
    let line = main.lines().find(|l| l.contains(" <== ") && l.contains(" - ")).unwrap();
    let broken = main.replacen(line, &line.replacen(" - ", " + ", 1), 1);
    refused(check_global_main("global_flipped", &broken, &gi), "a flipped global sign", "is not constrained to zero");
}

#[test]
fn a_global_check_against_another_value_is_refused() {
    let gi = random_global(40);
    let main = global_constraints(&gi).unwrap();
    let check = main.lines().find(|l| l.ends_with(" === 0;")).unwrap();
    let broken = main.replacen(check, &check.replace(" === 0;", " === 1;"), 1);
    refused(check_global_main("global_one", &broken, &gi), "a root checked against 1", "is not a zero check");
}

/// The BN128 verifier of `lib_bn128`'s sample starkinfo over a random DAG of its leaves.
fn bn128_key(seed: u64) -> (Value, Value, KeyVerifier, String) {
    let si = crate::stark2circom::circom_verifier::lib_bn128::tests::stark_info(false, 0);
    let leaves = vec![
        col(ColType::Fixed, 0, 0),
        col(ColType::Witness, 0, 0),
        DagNode::Challenge { stage: 3, stage_id: 0 },
        DagNode::Public { id: 0 },
        DagNode::Public { id: 1 },
        constant("0"),
        constant("7"),
    ];
    let vi = verifier_info(&random_dag(leaves, 80, seed, DagBoundary::EveryRow));
    let root = ["7".to_string(), "0".into(), "0".into(), "0".into()];
    let v = gen_stark_verifier_bn128(Some(&root), &si, &vi, &Pil2CircomOptions::default()).unwrap();
    let main = key_dispatch(std::slice::from_ref(&v.data), false);
    (si, vi, v, main)
}

fn check_bn128(name: &str, si: &Value, vi: &Value, circom: &str, main: &str) -> Option<anyhow::Result<()>> {
    let exec = circom_exec()?;
    let circomlib = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pil2-stark/node_modules/circomlib/circuits");
    if !circomlib.is_dir() {
        eprintln!("skipped: {} not present (npm install in setup/pil2-stark)", circomlib.display());
        return None;
    }
    let dir = Scratch::new(name);
    let file = dir.0.join("recursivef.verifier.circom");
    std::fs::write(&file, circom).unwrap();
    let root = ["7".to_string(), "0".into(), "0".into(), "0".into()];
    let key = Key {
        verifier_file: &file,
        stark_info: si,
        verifier_info: vi,
        const_root: Some(&root),
        enable_input: false,
        input_challenges: false,
    };
    let circuit =
        Circuit { field: Field::Bn128, main, keys: vec![key], global_info: &Value::Null, global_constraints: false };
    let libs: Vec<String> = ["helper_circuits", "circuits.bn128"]
        .map(|l| library().join(l).to_string_lossy().into_owned())
        .into_iter()
        .chain([circomlib.to_string_lossy().into_owned()])
        .collect();
    Some(check_circuit(&Circom { exec: &exec, libs: &libs }, &dir.0.join("work"), &circuit).map(|_| ()))
}

#[test]
fn bn128_constraint_eval_passes() {
    let (si, vi, v, main) = bn128_key(3);
    if let Some(Err(e)) = check_bn128("bn128", &si, &vi, &v.circom, &main) {
        panic!("{e:#}");
    }
}

#[test]
fn a_wrong_bn128_op_is_refused() {
    let (si, vi, v, main) = bn128_key(3);
    let op = ["GLCAdd()", "GLCSub()", "GLAdd()", "GLSub()"].into_iter().find(|op| v.circom.contains(op)).unwrap();
    let broken = v.circom.replacen(op, &op.replace("Add", "Mul").replace("Sub", "Mul"), 1);
    refused(check_bn128("bn128_wrong", &si, &vi, &broken, &main), "a wrong BN128 op", "the DAG's roots folded");
}
