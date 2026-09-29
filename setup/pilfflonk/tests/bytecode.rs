//! `<air>.bin` (M11, revision 2: the STARK's `.bin` with dimension 1): the encoder, the reader
//! and the fixture M17's C++ tests read.
//!
//! Every round trip checks two things. Encoding and decoding gives back the same [`Bytecode`]. And
//! the decoded code is the same code as `pil-info`'s: the same ops and operands, but for the
//! temporaries, whose ids are `pil-info`'s allocation, and for the STARK's order of the sources (a
//! sub may become a sub_swap, and a copy an add of 0). Both are also evaluated over `Fr` on the
//! same inputs, and every op must write the same value.
//!
//! The `#[ignore]` tests compile real PIL with the compiler `PIL2C_EXEC` names, as those of
//! `pil-info`'s `tests/bn254.rs` do:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pilfflonk-setup \
//!     --features proofman-starks-lib-c/cpu-only --test bytecode -- --ignored
//! ```

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use num_bigint::BigUint;
use pil2_pilout::pilout::{self as pb, constraint, expression, operand, SymbolType};
use pil2_pilout::pilout_proxy::PilOutProxy;
use pil_info::types::output::{CodeEntry, CodeRef};
use pil_info::{DegreePolicy, PilInfoCfg, PilInfoResult};
use pilfflonk_setup::bytecode::{
    write_air_bin, Bytecode, BytecodeError, Code, CodeContext, ConstraintBin, ExpressionBin, Op, Opcode, Operand,
    BIN_VERSION,
};
use proofman_pilfflonk::field::FrBytes;

#[path = "bytecode/interpreter.rs"]
mod interpreter;

const R: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";
const R_MINUS_ONE: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495616";
const R_MINUS_TWO: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495615";
/// 2^253 + 5: 254 bits.
const WIDE_254: &str = "14474011154664524427946373126085988481658748083205070504932198000989141204997";
/// 2^200 + 7.
const WIDE_200: &str = "1606938044258990275541962092341162602522202993782792835301383";

fn modulus() -> BigUint {
    BigUint::parse_bytes(R.as_bytes(), 10).unwrap()
}

fn fr(decimal: &str) -> FrBytes {
    FrBytes::from_decimal(decimal).unwrap()
}

/// A path under the target's temporary directory, fresh for `name`.
fn tmp_path(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("pilfflonk_setup_bytecode");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let _ = fs::remove_file(&path);
    path
}

fn write_and_read(bytecode: &Bytecode, name: &str) -> (Bytecode, Vec<u8>) {
    let path = tmp_path(name);
    bytecode.write(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    let read = Bytecode::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    (read, bytes)
}

// ---------------------------------------------------------------------------------------------
// pil-info's code, built in code
// ---------------------------------------------------------------------------------------------

fn code_ref(ref_type: &str, id: usize) -> CodeRef {
    CodeRef {
        ref_type: ref_type.to_string(),
        id,
        dim: 1,
        prime: None,
        value: None,
        stage: None,
        stage_id: None,
        commit_id: None,
        opening: None,
        boundary_id: None,
        airgroup_id: None,
        exp_id: None,
    }
}

fn tmp(id: usize) -> CodeRef {
    code_ref("tmp", id)
}

fn column(ref_type: &str, id: usize, prime: i64) -> CodeRef {
    CodeRef { prime: Some(prime), ..code_ref(ref_type, id) }
}

fn number(value: &str) -> CodeRef {
    CodeRef { value: Some(value.to_string()), ..code_ref("number", 0) }
}

fn zi(boundary: usize) -> CodeRef {
    CodeRef { boundary_id: Some(boundary), ..code_ref("Zi", 0) }
}

fn entry(op: &str, dest: CodeRef, src: Vec<CodeRef>) -> CodeEntry {
    CodeEntry { op: op.to_string(), dest, src }
}

/// An AIR of one stage, with committed columns 0, 1 and 2 at `stagePos` 0, 1 and 2 of stage 1, and
/// opening points −2 … 1.
fn small_context() -> CodeContext {
    CodeContext { n_stages: 1, cm_pols: vec![(1, 0), (1, 1), (1, 2)], opening_points: vec![-2, -1, 0, 1] }
}

/// An intermediate polynomial's code as the passes write it, sparse temporaries and a last
/// destination `cm` included, with every operand kind, a copy, sources in either order and numbers
/// of 254 bits.
fn small_code() -> Vec<CodeEntry> {
    vec![
        entry("mul", tmp(10), vec![column("cm", 0, 1), column("cm", 1, 0)]),
        entry("add", tmp(11), vec![tmp(10), column("const", 0, -1)]),
        entry("sub", tmp(12), vec![tmp(11), number(R_MINUS_ONE)]),
        entry("mul", tmp(13), vec![code_ref("challenge", 2), number(WIDE_254)]),
        entry("add", tmp(14), vec![tmp(12), tmp(13)]),
        entry("sub", tmp(15), vec![code_ref("airvalue", 0), code_ref("public", 1)]),
        entry("mul", tmp(16), vec![code_ref("airgroupvalue", 0), code_ref("proofvalue", 3)]),
        entry("add", tmp(17), vec![tmp(15), tmp(16)]),
        entry("mul", tmp(18), vec![tmp(14), tmp(17)]),
        entry("copy", tmp(19), vec![number(R_MINUS_ONE)]),
        entry("mul", tmp(20), vec![tmp(19), zi(1)]),
        entry("add", tmp(21), vec![tmp(18), tmp(20)]),
        entry("mul", tmp(22), vec![code_ref("eval", 4), column("cm", 2, -2)]),
        entry("sub", column("cm", 3, 0), vec![tmp(21), tmp(22)]),
    ]
}

/// `tmpUsed` of `small_code`: the id of the temporary its last op writes instead of `cm 3`.
const SMALL_CODE_TMP_USED: usize = 23;

// ---------------------------------------------------------------------------------------------
// The decoded code as pil-info's, and both evaluated
// ---------------------------------------------------------------------------------------------

/// A `Code` as `pil-info`'s code, its columns back to their ids and offsets: a sub_swap is the sub
/// of its sources the other way round, and its last destination is `destId`.
fn entries_of(code: &Code, context: &CodeContext) -> Vec<CodeEntry> {
    let prime = |opening: u32| context.opening_points[opening as usize];
    let operand = |t: &Operand| match *t {
        Operand::Const { id, opening } => column("const", id as usize, prime(opening)),
        Operand::Cm { stage, stage_pos, opening } => {
            let id = context.cm_pols.iter().position(|&p| p == (stage, stage_pos)).unwrap();
            column("cm", id, prime(opening))
        }
        Operand::Zi { boundary } => zi(boundary as usize),
        Operand::Tmp(t) => tmp(t as usize),
        Operand::Public(id) => code_ref("public", id as usize),
        Operand::Number(value) => number(&value.to_decimal()),
        Operand::AirValue(id) => code_ref("airvalue", id as usize),
        Operand::ProofValue(id) => code_ref("proofvalue", id as usize),
        Operand::AirgroupValue(id) => code_ref("airgroupvalue", id as usize),
        Operand::Challenge(id) => code_ref("challenge", id as usize),
        Operand::Eval(id) => code_ref("eval", id as usize),
    };
    code.ops
        .iter()
        .map(|op| {
            let (name, src) = match op.opcode {
                Opcode::Add => ("add", [op.a, op.b]),
                Opcode::Sub => ("sub", [op.a, op.b]),
                Opcode::Mul => ("mul", [op.a, op.b]),
                Opcode::SubSwap => ("sub", [op.b, op.a]),
            };
            entry(name, tmp(op.dest as usize), src.iter().map(operand).collect())
        })
        .collect()
}

type Leaf = (String, usize, Option<i64>, Option<String>, Option<usize>);

/// What the format keeps of an operand, a temporary being only a temporary.
fn leaf(r: &CodeRef) -> Leaf {
    if r.ref_type == "tmp" {
        return ("tmp".into(), 0, None, None, None);
    }
    let prime = if matches!(r.ref_type.as_str(), "cm" | "const") { r.prime } else { None };
    (r.ref_type.clone(), r.id, prime, r.value.clone(), r.boundary_id)
}

/// A value of `Fr` for each leaf, the same for the same leaf, and the constant for a number.
fn leaf_value(r: &CodeRef) -> BigUint {
    if r.ref_type == "number" {
        return BigUint::parse_bytes(r.value.as_deref().unwrap().as_bytes(), 10).unwrap();
    }
    let mut hasher = DefaultHasher::new();
    leaf(r).hash(&mut hasher);
    let h = BigUint::from(hasher.finish()) + 1u32;
    h.pow(5) % modulus()
}

/// The value each op writes, over `Fr`.
fn evaluate(code: &[CodeEntry]) -> Vec<BigUint> {
    let r = modulus();
    let mut tmps: HashMap<usize, BigUint> = HashMap::new();
    let mut written = Vec::with_capacity(code.len());
    for c in code {
        let value = |s: &CodeRef| {
            if s.ref_type == "tmp" {
                tmps.get(&s.id).cloned().unwrap_or_else(|| panic!("tmp {} read before it is written", s.id))
            } else {
                leaf_value(s)
            }
        };
        let a = value(&c.src[0]);
        let v = match c.op.as_str() {
            "copy" => a,
            "add" => (a + value(&c.src[1])) % &r,
            "sub" => (a + &r - value(&c.src[1])) % &r,
            "mul" => (a * value(&c.src[1])) % &r,
            other => panic!("op {other}"),
        };
        if c.dest.ref_type == "tmp" {
            tmps.insert(c.dest.id, v.clone());
        }
        written.push(v);
    }
    written
}

/// An op's name and sources as the format keeps them: a copy is an add of 0, and the sources of
/// an add or a mul are in no particular order.
fn shape(c: &CodeEntry) -> (String, Vec<Leaf>) {
    let (op, mut src): (&str, Vec<Leaf>) = match c.op.as_str() {
        "copy" => ("add", vec![leaf(&c.src[0]), leaf(&number("0"))]),
        op => (op, c.src.iter().map(leaf).collect()),
    };
    if op != "sub" {
        src.sort();
    }
    (op.to_string(), src)
}

/// `decoded` is `original` in the format: the same ops and operands, temporaries and the order of
/// the sources apart, and the same value written by every op.
fn assert_same_code(original: &[CodeEntry], decoded: &Code, context: &CodeContext) {
    let entries = entries_of(decoded, context);
    assert_eq!(entries.len(), original.len());
    for (i, (o, d)) in original.iter().zip(&entries).enumerate() {
        assert_eq!(shape(o), shape(d), "op {i}");
    }
    assert_eq!(entries.last().unwrap().dest.id, decoded.dest_id as usize);
    assert_eq!(evaluate(original), evaluate(&entries));
}

fn all_numbers(bytecode: &Bytecode) -> Vec<String> {
    let codes = bytecode.expressions.iter().map(|e| &e.code).chain(bytecode.constraints.iter().map(|c| &c.code));
    codes
        .flat_map(|c| c.ops.iter().flat_map(|op| [op.a, op.b]))
        .filter_map(|t| match t {
            Operand::Number(v) => Some(v.to_decimal()),
            _ => None,
        })
        .collect()
}

/// Every code block of `bytecode` is the code of the same entry in `result`.
fn assert_is_the_code_of(bytecode: &Bytecode, result: &PilInfoResult) {
    let context = CodeContext::from_pil_info(result).unwrap();
    assert_eq!(bytecode.n_stages as usize, result.setup.n_stages);
    let info = &result.pil_code.expressions_info;
    assert_eq!(bytecode.expressions.len(), info.expressions_code.len());
    for (bin, e) in bytecode.expressions.iter().zip(&info.expressions_code) {
        assert_eq!(bin.exp_id as usize, e.exp_id);
        assert_eq!(bin.stage as usize, e.stage);
        assert_eq!(bin.line, e.line);
        assert_same_code(&e.code, &bin.code, &context);
    }
    assert_eq!(bytecode.constraints.len(), info.constraints.len());
    let n = 1u32 << result.setup.pil_power;
    for (bin, c) in bytecode.constraints.iter().zip(&info.constraints) {
        let rows = match c.boundary.as_str() {
            "everyRow" => (0, n),
            "firstRow" => (0, 1),
            "lastRow" => (n - 1, n),
            _ => (c.offset_min.unwrap(), n - c.offset_max.unwrap()),
        };
        assert_eq!((bin.first_row, bin.last_row), rows, "{}", c.boundary);
        assert_eq!(bin.stage as usize, c.stage);
        assert_eq!(bin.im_pol, c.im_pol == 1);
        assert_eq!(bin.line, c.line.clone().unwrap_or_default());
        assert_same_code(&c.code, &bin.code, &context);
    }
}

// ---------------------------------------------------------------------------------------------
// Code built in code
// ---------------------------------------------------------------------------------------------

#[test]
fn small_code_round_trips() {
    let context = small_context();
    let original = small_code();
    let code = Code::from_entries(&original, &context, Some(SMALL_CODE_TMP_USED)).unwrap();
    assert_same_code(&original, &code, &context);
    // pil-info's allocation packs the 13 temporaries and the result into a few.
    assert!(code.n_temp <= 4, "{} temporaries", code.n_temp);

    let bytecode = Bytecode {
        n_stages: 1,
        expressions: vec![ExpressionBin { exp_id: 5, stage: 1, line: "Small.ImPol".into(), code: code.clone() }],
        constraints: vec![ConstraintBin {
            stage: 1,
            first_row: 0,
            last_row: 16,
            im_pol: true,
            line: String::new(),
            code,
        }],
    };
    let (read, bytes) = write_and_read(&bytecode, "small.bin");
    assert_eq!(read, bytecode);
    assert_same_code(&original, &read.expressions[0].code, &context);
    assert_same_code(&original, &read.constraints[0].code, &context);

    // The 254-bit numbers are whole, and each section has each once. The copy is an add of 0.
    let per_code = [R_MINUS_ONE, WIDE_254, R_MINUS_ONE, "0"];
    assert_eq!(all_numbers(&read), [per_code, per_code].concat());
    let r_minus_one = fr(R_MINUS_ONE).to_le_bytes();
    let occurrences = bytes.windows(32).filter(|w| *w == r_minus_one).count();
    assert_eq!(occurrences, 2, "r − 1 once in the expressions and once in the constraints");
}

/// The sources in the STARK's order, as `get_operation` of `io/parser_args.rs` sorts them.
#[test]
fn sources_go_in_the_starks_order() {
    let context = small_context();
    let code = vec![
        // number (3) − cm (0): swapped, a sub_swap.
        entry("sub", tmp(0), vec![number("5"), column("cm", 0, 0)]),
        // challenge (11) · tmp (1): swapped.
        entry("mul", tmp(1), vec![code_ref("challenge", 0), tmp(0)]),
        // tmp (1) − public (2): as it is.
        entry("sub", tmp(2), vec![tmp(1), code_ref("public", 0)]),
        // Zi (0) and const (0) tie: as they are.
        entry("add", tmp(3), vec![zi(0), column("const", 1, 1)]),
        entry("add", tmp(4), vec![tmp(2), tmp(3)]),
    ];
    let lowered = Code::from_entries(&code, &context, None).unwrap();
    let five = Operand::Number(FrBytes::from_u64(5));
    let cm0 = Operand::Cm { stage: 1, stage_pos: 0, opening: 2 };
    let ops: Vec<(Opcode, Operand, Operand)> = lowered.ops.iter().map(|op| (op.opcode, op.a, op.b)).collect();
    assert_eq!(ops[0], (Opcode::SubSwap, cm0, five));
    assert_eq!((ops[1].0, ops[1].2), (Opcode::Mul, Operand::Challenge(0)));
    assert_eq!((ops[2].0, ops[2].2), (Opcode::Sub, Operand::Public(0)));
    assert_eq!(ops[3], (Opcode::Add, Operand::Zi { boundary: 0 }, Operand::Const { id: 1, opening: 3 }));
    assert_same_code(&code, &lowered, &context);
}

#[test]
fn the_same_content_gives_the_same_bytes() {
    let bytecode = sample();
    let (_, first) = write_and_read(&bytecode, "deterministic_1.bin");
    let (_, second) = write_and_read(&bytecode.clone(), "deterministic_2.bin");
    assert_eq!(first, second);
}

/// Every field at its place, as the module's documentation lays the file out.
#[test]
fn the_layout_is_the_documented_one() {
    let bytecode = Bytecode {
        n_stages: 1,
        expressions: vec![ExpressionBin {
            exp_id: 7,
            stage: 2,
            line: "q".into(),
            code: Code {
                ops: vec![Op {
                    opcode: Opcode::Mul,
                    dest: 0,
                    a: Operand::Cm { stage: 1, stage_pos: 3, opening: 0 },
                    b: Operand::Number(fr(R_MINUS_ONE)),
                }],
                n_temp: 1,
                dest_id: 0,
            },
        }],
        constraints: vec![ConstraintBin {
            stage: 1,
            first_row: 255,
            last_row: 256,
            im_pol: false,
            line: String::new(),
            code: Code {
                ops: vec![Op {
                    opcode: Opcode::Add,
                    dest: 0,
                    a: Operand::Zi { boundary: 2 },
                    b: Operand::Number(FrBytes::ZERO),
                }],
                n_temp: 1,
                dest_id: 0,
            },
        }],
    };
    let (_, bytes) = write_and_read(&bytecode, "layout.bin");

    let u32s = |out: &mut Vec<u8>, words: &[u32]| {
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    };
    let section = |out: &mut Vec<u8>, id: u32, payload: &[u8]| {
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(payload);
    };
    let mut expected: Vec<u8> = Vec::new();
    expected.extend_from_slice(b"chps");
    u32s(&mut expected, &[0x7066_0002, 3]);

    // With one stage, bs = 5: Zi is type 3 and a number type 8.
    let mut expressions = Vec::new();
    u32s(&mut expressions, &[0x7066_0002, 32]); // version, n8
    expressions.extend_from_slice(&modulus().to_bytes_le()); // r
    u32s(&mut expressions, &[1]); // nStages
                                  // maxTmp, maxArgs, maxOps, nOps, nArgs, nNumbers, nExpressions.
    u32s(&mut expressions, &[1, 8, 1, 1, 8, 1, 1]);
    // expId, destId, stage, nTemp, nOps, opsOffset, nArgs, argsOffset, line.
    u32s(&mut expressions, &[7, 0, 2, 1, 1, 0, 8, 0]);
    expressions.extend_from_slice(b"q\0");
    expressions.push(0); // the op's dimensions: dim1 = dim1 ∘ dim1
                         // mul, into 0, cm of stage 1 at stagePos 3 and opening 0, number 0.
    u32s(&mut expressions, &[2, 0, 1, 3, 0, 8, 0, 0]);
    expressions.extend_from_slice(&fr(R_MINUS_ONE).to_le_bytes());
    section(&mut expected, 1, &expressions);

    let mut constraints = Vec::new();
    // nOps, nArgs, nNumbers, nConstraints.
    u32s(&mut constraints, &[1, 8, 1, 1]);
    // stage, destId, firstRow, lastRow, nTemp, nOps, opsOffset, nArgs, argsOffset, imPol, line.
    u32s(&mut constraints, &[1, 0, 255, 256, 1, 1, 0, 8, 0, 0]);
    constraints.push(0);
    constraints.push(0);
    // add, into 0, Zi of boundary 2 (arg1 3), number 0.
    u32s(&mut constraints, &[0, 0, 3, 3, 0, 8, 0, 0]);
    constraints.extend_from_slice(&[0; 32]);
    section(&mut expected, 2, &constraints);

    section(&mut expected, 3, &0u32.to_le_bytes());
    assert_eq!(bytes, expected);
    assert_eq!(BIN_VERSION, 0x7066_0002);
}

// ---------------------------------------------------------------------------------------------
// What the encoder refuses
// ---------------------------------------------------------------------------------------------

fn encode_error(code: &[CodeEntry], redirect: Option<usize>) -> String {
    match Code::from_entries(code, &small_context(), redirect) {
        Err(e @ BytecodeError::Encode(_)) => e.to_string(),
        other => panic!("not an encoding error: {other:?}"),
    }
}

#[test]
fn code_the_format_cannot_hold_is_refused() {
    let two = |a: CodeRef, b: CodeRef| vec![entry("add", tmp(0), vec![a, b])];
    let refused = |code: Vec<CodeEntry>, why: &str| {
        let err = encode_error(&code, None);
        assert!(err.contains(why), "{err}");
    };
    let cm = || column("cm", 0, 0);

    refused(two(CodeRef { dim: 3, ..code_ref("challenge", 0) }, cm()), "dimension 3");
    refused(two(number(R), cm()), "not a canonical Fr");
    refused(two(number("0x10"), cm()), "not a canonical Fr");
    refused(two(column("custom", 0, 0), cm()), "custom is not an operand");
    refused(two(code_ref("xDivXSubXi", 0), cm()), "xDivXSubXi is not an operand");
    refused(two(column("cm", 0, 2), cm()), "which is not an opening point");
    refused(two(column("cm", 9, 0), cm()), "cm 9 is not in cmPolsMap");
    refused(Vec::new(), "no ops");
    refused(vec![entry("neg", tmp(0), vec![cm()])], "neg with 1 operands");
    refused(vec![entry("copy", tmp(0), vec![cm(), cm()])], "copy with 2 operands");
    refused(two(cm(), cm()).into_iter().map(|e| CodeEntry { dest: column("cm", 1, 0), ..e }).collect(), "writes a cm");
    refused(two(tmp(5), cm()), "read before it is written");
    let err = encode_error(&two(tmp(5), cm()), Some(3));
    assert!(err.contains("tmpUsed (3)"), "{err}");
}

/// What the reader would refuse is not written either.
#[test]
fn a_bytecode_the_reader_would_refuse_is_not_written() {
    let cm = |stage| Operand::Cm { stage, stage_pos: 0, opening: 0 };
    let op = |opcode, dest, a, b| Op { opcode, dest, a, b };
    let one_op = |op: Op, n_temp: u32| Code { ops: vec![op], n_temp, dest_id: 0 };
    let refused = |bytecode: Bytecode, why: &str| {
        let path = tmp_path("refused_on_write.bin");
        match bytecode.write(&path) {
            Err(e @ BytecodeError::Encode(_)) => assert!(e.to_string().contains(why), "{e}"),
            other => panic!("written, or not an encoding error: {other:?}"),
        }
    };
    let expression = |code: Code| Bytecode {
        n_stages: 1,
        expressions: vec![ExpressionBin { exp_id: 0, stage: 1, line: String::new(), code }],
        constraints: Vec::new(),
    };

    let good = one_op(op(Opcode::Add, 0, cm(1), cm(1)), 1);
    let reversed_rows =
        ConstraintBin { stage: 1, first_row: 2, last_row: 1, im_pol: false, line: String::new(), code: good.clone() };
    refused(Bytecode { n_stages: 1, expressions: Vec::new(), constraints: vec![reversed_rows] }, "rows 2..1");
    refused(expression(one_op(op(Opcode::Add, 0, cm(1), Operand::Tmp(0)), 1)), "read before it is written");
    refused(expression(one_op(op(Opcode::Add, 0, Operand::Public(0), cm(1)), 1)), "not in the STARK's order");
    refused(expression(one_op(op(Opcode::Add, 0, cm(1), cm(1)), 2)), "2 temporaries for 1 ops");
    refused(expression(Code { dest_id: 1, ..good.clone() }), "destId 1 is not the last op's dest");
    // Stage 0, and stage 3 in an AIR of 1: types another operand has.
    refused(expression(one_op(op(Opcode::Add, 0, cm(0), cm(1)), 1)), "in an AIR of 1 stages");
    refused(expression(one_op(op(Opcode::Add, 0, cm(3), cm(1)), 1)), "in an AIR of 1 stages");
    refused(Bytecode { n_stages: u32::MAX, ..expression(good) }, "stages");
}

#[test]
fn the_starks_passes_are_refused() {
    let result =
        pil_info::run(&wide_constants_pilout(), 0, 0, &PilInfoCfg::goldilocks(1), &Default::default()).unwrap();
    let err = Bytecode::from_pil_info(&result).unwrap_err().to_string();
    assert!(err.contains("FRI polynomial"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// What the reader refuses
// ---------------------------------------------------------------------------------------------

fn format_error(bytes: &[u8]) -> String {
    match Bytecode::from_bytes(bytes) {
        Err(e @ BytecodeError::Format(_)) => e.to_string(),
        other => panic!("not a format error: {other:?}"),
    }
}

/// Where section 1's payload starts: after the file's header and the section's.
const SECTION_1: usize = 12 + 12;

/// The offsets in section 1 of its ops and its args: after the prefix (11 words, r among them),
/// the counts (7 words) and the entries (8 words and a line each).
fn ops_and_args(section: &[u8]) -> (usize, usize) {
    let word = |at: usize| u32::from_le_bytes(section[at..at + 4].try_into().unwrap()) as usize;
    let (n_ops, n_expressions) = (word(44 + 12), word(44 + 24));
    let mut pos = 44 + 28;
    for _ in 0..n_expressions {
        pos += 4 * 8;
        pos += section[pos..].iter().position(|&b| b == 0).unwrap() + 1;
    }
    (pos, pos + n_ops)
}

#[test]
fn a_file_that_is_not_this_bytecode_is_refused() {
    let (_, bytes) = write_and_read(&sample(), "refused.bin");
    let patched = |at: usize, with: &[u8]| {
        let mut b = bytes.clone();
        b[at..at + with.len()].copy_from_slice(with);
        b
    };

    // A STARK chps: version 1.
    let err = format_error(&patched(4, &1u32.to_le_bytes()));
    assert!(err.contains("version 0x1"), "{err}");
    assert!(format_error(&patched(0, b"zkey")).contains("not a \"chps\""));
    // The prefix of section 1: its copy of the version, and r.
    assert!(format_error(&patched(SECTION_1, &2u32.to_le_bytes())).contains("section 1 starts with"));
    assert!(format_error(&patched(SECTION_1 + 8, &[0])).contains("section 1 starts with"));
    let err = format_error(&bytes[..bytes.len() - 1]);
    assert!(err.contains("ends before"), "{err}");
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(format_error(&longer).contains("after its content"));

    // The first op: its dimensions, its opType and its first source's type.
    let (ops, args) = ops_and_args(&bytes[SECTION_1..]);
    let err = format_error(&patched(SECTION_1 + ops, &[1]));
    assert!(err.contains("is of dimensions 1"), "{err}");
    let err = format_error(&patched(SECTION_1 + args, &4u32.to_le_bytes()));
    assert!(err.contains("unknown opType 4"), "{err}");
    let err = format_error(&patched(SECTION_1 + args + 8, &1000u32.to_le_bytes()));
    assert!(err.contains("is no operand of an AIR of 2 stages"), "{err}");
    // maxTmp.
    let err = format_error(&patched(SECTION_1 + 44, &9u32.to_le_bytes()));
    assert!(err.contains("maxTmp, maxArgs and maxOps"), "{err}");

    // Hints: revision 2 has none.
    let err = format_error(&patched(bytes.len() - 4, &1u32.to_le_bytes()));
    assert!(err.contains("1 hints"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// The passes over a pilout built in code
// ---------------------------------------------------------------------------------------------

fn big_be(decimal: &str) -> Vec<u8> {
    BigUint::parse_bytes(decimal.as_bytes(), 10).unwrap().to_bytes_be()
}

fn op(operand: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(operand) })
}

fn witness(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage: 1, col_idx, row_offset }))
}

fn fixed(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset: 0 }))
}

fn constant(decimal: &str) -> Option<pb::Operand> {
    op(operand::Operand::Constant(operand::Constant { value: big_be(decimal) }))
}

fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}

fn binary(operation: expression::Operation) -> pb::Expression {
    pb::Expression { operation: Some(operation) }
}

fn every_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("constraint on expression {idx}")),
        })),
    }
}

fn column_symbol(name: &str, kind: SymbolType, stage: u32, id: u32) -> pb::Symbol {
    pb::Symbol {
        name: name.to_string(),
        air_group_id: Some(0),
        air_id: Some(0),
        r#type: kind as i32,
        id,
        stage: Some(stage),
        ..Default::default()
    }
}

/// One BN254 air of 16 rows, as pil2com emits it: `L1·(a − (2^200 + 7))`, `L1·(b − (r − 2))`,
/// `(−a)·b + b'` and `(−(a'·a) + b)·(a·a)`. The negations become products by `r − 1`, and the last
/// constraint, of degree 4, gets an intermediate polynomial.
fn wide_constants_pilout() -> pb::PilOut {
    use expression::{Add, Mul, Neg, Operation, Sub};
    let expressions = vec![
        binary(Operation::Sub(Sub { lhs: witness(0, 0), rhs: constant(WIDE_200) })), // 0
        binary(Operation::Mul(Mul { lhs: fixed(0), rhs: exp(0) })),                  // 1
        binary(Operation::Sub(Sub { lhs: witness(1, 0), rhs: constant(R_MINUS_TWO) })), // 2
        binary(Operation::Mul(Mul { lhs: fixed(0), rhs: exp(2) })),                  // 3
        binary(Operation::Neg(Neg { value: witness(0, 0) })),                        // 4
        binary(Operation::Mul(Mul { lhs: exp(4), rhs: witness(1, 0) })),             // 5
        binary(Operation::Add(Add { lhs: exp(5), rhs: witness(1, 1) })),             // 6
        binary(Operation::Mul(Mul { lhs: witness(0, 1), rhs: witness(0, 0) })),      // 7
        binary(Operation::Neg(Neg { value: exp(7) })),                               // 8
        binary(Operation::Add(Add { lhs: exp(8), rhs: witness(1, 0) })),             // 9
        binary(Operation::Mul(Mul { lhs: witness(0, 0), rhs: witness(0, 0) })),      // 10
        binary(Operation::Mul(Mul { lhs: exp(9), rhs: exp(10) })),                   // 11
    ];
    let mut symbols = vec![column_symbol("L1", SymbolType::FixedCol, 0, 0)];
    symbols.push(column_symbol("a", SymbolType::WitnessCol, 1, 0));
    symbols.push(column_symbol("b", SymbolType::WitnessCol, 1, 1));
    let air = pb::Air {
        name: Some("Synthetic".to_string()),
        num_rows: Some(16),
        fixed_cols: vec![pb::FixedCol { values: Vec::new() }],
        stage_widths: vec![2],
        expressions,
        constraints: vec![every_row(1), every_row(3), every_row(6), every_row(11)],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("synthetic".to_string()),
        base_field: big_be(R),
        air_groups: vec![pb::AirGroup { name: Some("Synthetic".to_string()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0],
        symbols,
        ..Default::default()
    }
}

fn run_bn254(pilout: &pb::PilOut) -> PilInfoResult {
    pil_info::run(pilout, 0, 0, &PilInfoCfg::bn254(), &Default::default()).unwrap()
}

/// The file `write_air_bin` writes for `result`, checked against `result`'s code, and its bytes,
/// the same when it is written again.
fn check_air_bin(result: &PilInfoResult, name: &str) -> (Bytecode, Vec<u8>) {
    let path = tmp_path(&format!("{name}.bin"));
    write_air_bin(result, &path).unwrap();
    let bytecode = Bytecode::read(&path).unwrap();
    assert_eq!(bytecode, Bytecode::from_pil_info(result).unwrap());
    assert_is_the_code_of(&bytecode, result);

    let again = tmp_path(&format!("{name}.again.bin"));
    write_air_bin(result, &again).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes, fs::read(&again).unwrap(), "the same result gives the same bytes");
    fs::remove_file(&path).unwrap();
    fs::remove_file(&again).unwrap();
    (bytecode, bytes)
}

/// The expressions of the intermediate polynomials (by the `expId` of their `cmPolsMap` entries)
/// and of `Q` (`cExpId`) are in the file, as the STARK's prover finds them.
fn assert_has_im_pols_and_q(bytecode: &Bytecode, result: &PilInfoResult) -> Vec<u32> {
    let exp_ids: Vec<u32> = bytecode.expressions.iter().map(|e| e.exp_id).collect();
    let im_pols: Vec<u32> =
        result.setup.cm_pols_map.iter().filter(|p| p.im_pol).map(|p| p.exp_id.unwrap() as u32).collect();
    for id in im_pols.iter().chain([&(result.c_exp_id as u32)]) {
        assert!(exp_ids.contains(id), "expression {id} not in {exp_ids:?}");
    }
    im_pols
}

#[test]
fn the_passes_code_round_trips() {
    let result = run_bn254(&wide_constants_pilout());
    let (bytecode, first) = check_air_bin(&result, "wide_constants_in_code");

    let numbers = all_numbers(&bytecode);
    for value in [WIDE_200, R_MINUS_TWO, R_MINUS_ONE] {
        assert!(numbers.iter().any(|n| n == value), "{value} not in {numbers:?}");
    }
    let im_pols = assert_has_im_pols_and_q(&bytecode, &result);
    assert!(!im_pols.is_empty(), "the degree-4 constraint gets an intermediate polynomial");

    // The passes run twice give the same file.
    let (_, second) = check_air_bin(&run_bn254(&wide_constants_pilout()), "wide_constants_in_code_2");
    assert_eq!(first, second);
}

/// One BN254 air of 16 rows: `a·a − b` on `firstRow` (a public-free stand-in for `x·x − p`) and
/// `b − a` on `everyRow`. With its `Zi`, the first has degree 3; with `--max-constraint-degree 2`
/// the search promotes its whole expression to an im pol, whose code `pil-info` leaves empty for
/// the constraint (it marks the im pols as computed before it generates the constraints' code).
fn im_pol_constraint_pilout() -> pb::PilOut {
    use expression::{Mul, Operation, Sub};
    let expressions = vec![
        binary(Operation::Mul(Mul { lhs: witness(0, 0), rhs: witness(0, 0) })), // 0
        binary(Operation::Sub(Sub { lhs: exp(0), rhs: witness(1, 0) })),        // 1: a·a − b
        binary(Operation::Sub(Sub { lhs: witness(1, 0), rhs: witness(0, 0) })), // 2: b − a
    ];
    let first_row = pb::Constraint {
        constraint: Some(constraint::Constraint::FirstRow(constraint::FirstRow {
            expression_idx: Some(operand::Expression { idx: 1 }),
            debug_line: Some("a*a - b".to_string()),
        })),
    };
    let air = pb::Air {
        name: Some("Synthetic".to_string()),
        num_rows: Some(16),
        stage_widths: vec![2],
        expressions,
        constraints: vec![first_row, every_row(2)],
        ..Default::default()
    };
    pb::PilOut {
        name: Some("synthetic".to_string()),
        base_field: big_be(R),
        air_groups: vec![pb::AirGroup { name: Some("Synthetic".to_string()), airs: vec![air], ..Default::default() }],
        num_challenges: vec![0],
        symbols: vec![
            column_symbol("a", SymbolType::WitnessCol, 1, 0),
            column_symbol("b", SymbolType::WitnessCol, 1, 1),
        ],
        ..Default::default()
    }
}

/// Regression (plan M24): a constraint whose whole expression is an im pol has the code of a copy
/// of the im pol's column at the row, not an empty one, which the reader refuses and the setup
/// failed on ("Cannot encode constraint 0: it has no ops").
#[test]
fn a_constraint_that_is_an_im_pol_is_a_copy_of_its_column() {
    let cfg = PilInfoCfg { degree_policy: DegreePolicy::Search { max: 2 }, ..PilInfoCfg::bn254() };
    let result = pil_info::run(&im_pol_constraint_pilout(), 0, 0, &cfg, &Default::default()).unwrap();
    let setup = &result.setup;
    let im = setup.cm_pols_map.iter().position(|p| p.im_pol).expect("the search chooses an im pol");
    assert_eq!(setup.cm_pols_map[im].exp_id, Some(setup.constraints[0].e), "the whole constraint is the im pol");
    assert!(result.pil_code.expressions_info.constraints[0].code.is_empty(), "pil-info leaves its code empty");

    let path = tmp_path("im_pol_constraint.bin");
    write_air_bin(&result, &path).unwrap();
    let bytecode = Bytecode::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    let context = CodeContext::from_pil_info(&result).unwrap();
    let stage_pos = context.cm_pols[im].1;
    let at_0 = setup.opening_points.iter().position(|&p| p == 0).unwrap() as u32;
    let c = &bytecode.constraints[0];
    assert_eq!((c.first_row, c.last_row, c.im_pol, c.line.as_str()), (0, 1, false, "a*a - b == 0"));
    let copy = bin_op(Opcode::Add, 0, cm(1, stage_pos, at_0), Operand::Number(FrBytes::ZERO));
    assert_eq!(c.code, Code { ops: vec![copy], n_temp: 1, dest_id: 0 });
    // The other constraints are pil-info's code, the im pol's too.
    for (bin, entry) in bytecode.constraints.iter().zip(&result.pil_code.expressions_info.constraints).skip(1) {
        assert_same_code(&entry.code, &bin.code, &context);
    }
}

// ---------------------------------------------------------------------------------------------
// The passes over compiled PIL (need PIL2C_EXEC)
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Compile `pil` (relative to the repository root) over BN254 with `PIL2C_EXEC`.
fn compile_bn254(pil: &str) -> pb::PilOut {
    let compiler = std::env::var("PIL2C_EXEC")
        .expect("PIL2C_EXEC must name a pil2com that honours `prime` (e.g. <pil2-compiler>/src/pil.js)");
    // A path of each call's own: the tests run in parallel, and compile the same PIL.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let stem = Path::new(pil).file_stem().unwrap().to_string_lossy().into_owned();
    let out = tmp_path(&format!("{stem}.{call}.bn254.pilout"));
    let status = Command::new(compiler)
        .current_dir(repo_root())
        .arg(pil)
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json", "-o"])
        .arg(&out)
        .status()
        .expect("PIL2C_EXEC runs");
    assert!(status.success(), "pil2com failed on {pil}");
    let pilout = PilOutProxy::new(out.to_str().unwrap()).unwrap().pilout;
    fs::remove_file(&out).unwrap();
    assert_eq!(pilout.base_field, big_be(R), "{pil} was not compiled over BN254: does PIL2C_EXEC honour `prime`?");
    pilout
}

/// The M13 Fibonacci fixture: its intermediate polynomial (`l1' − next`, `cmPolsMap[2]`), `Q` and
/// six constraints over its 256 rows, the one of the intermediate polynomial included.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn fibonacci_fixture_round_trips() {
    let result = run_bn254(&compile_bn254("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil"));
    let (bytecode, bytes) = check_air_bin(&result, "fibonacci");

    let exp_ids: Vec<u32> = bytecode.expressions.iter().map(|e| e.exp_id).collect();
    assert_eq!(exp_ids, [6, result.c_exp_id as u32]);
    assert_eq!(assert_has_im_pols_and_q(&bytecode, &result), [6]);
    assert_eq!(result.setup.cm_pols_map[2].exp_id, Some(6));
    assert_eq!(bytecode.n_stages, 1);
    assert_eq!(bytecode.constraints.len(), 6);
    assert!(bytecode.constraints.iter().all(|c| (c.first_row, c.last_row) == (0, 256)));
    assert_eq!(bytecode.constraints.iter().filter(|c| c.im_pol).count(), 1);
    assert!(result.pil_code.expressions_info.hints_info.is_empty());

    // Q reads no piece Q0 … of cmPolsMap, at stage nStages + 1: it is computed whole.
    let q = &bytecode.expressions[1];
    for op in &q.code.ops {
        for t in [op.a, op.b] {
            if let Operand::Cm { stage, .. } = t {
                assert_ne!(stage, bytecode.n_stages + 1, "Q reads a piece of Q");
            }
        }
    }
    // Q · Zi(everyRow), the zerofier first (the STARK's order).
    let last = q.code.ops.last().unwrap();
    assert_eq!((last.opcode, last.a), (Opcode::Mul, Operand::Zi { boundary: 0 }), "Q ends dividing by Z_H");
    println!(
        "fibonacci: {} expressions ({} and {} ops, {} and {} temporaries), {} constraints, {} bytes",
        bytecode.expressions.len(),
        bytecode.expressions[0].code.ops.len(),
        q.code.ops.len(),
        bytecode.expressions[0].code.n_temp,
        q.code.n_temp,
        bytecode.constraints.len(),
        bytes.len()
    );
}

/// `pil-info`'s `tests/fixtures/wide_constants.pil`: the compiler's constants of up to 254 bits,
/// and the `r − 1` of its negations, survive whole.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn wide_constants_fixture_round_trips() {
    let result = run_bn254(&compile_bn254("setup/pil-info/tests/fixtures/wide_constants.pil"));
    let (bytecode, _) = check_air_bin(&result, "wide_constants");
    let numbers = all_numbers(&bytecode);
    for value in [WIDE_200, R_MINUS_TWO, R_MINUS_ONE] {
        assert!(numbers.iter().any(|n| n == value), "{value} not in {numbers:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// The fixture of the Rust → file → C++ round trip (M17)
// ---------------------------------------------------------------------------------------------

/// For the C++ reader's tests (M17), which are to check every value of [`sample`]: the Rust half of
/// the Rust → file → C++ round trip. Regenerate it with
/// `PILFFLONK_UPDATE_FIXTURES=1 cargo test -p pilfflonk-setup --test bytecode`, and update the C++
/// tests with it.
const BYTECODE_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/bytecode/Sample.bin");

fn bin_op(opcode: Opcode, dest: u32, a: Operand, b: Operand) -> Op {
    Op { opcode, dest, a, b }
}

fn cm(stage: u32, stage_pos: u32, opening: u32) -> Operand {
    Operand::Cm { stage, stage_pos, opening }
}

/// An AIR of `N = 8` rows and 2 stages, with opening points −2, −1, 0, 1 and 2 (indices 0 to 4).
/// Its columns are `a`, `b` and the intermediate polynomial `im` of stage 1 (`stagePos` 0, 1 and
/// 2), and `c` of stage 2. It has constraints on the rows of everyRow, firstRow, lastRow and
/// `everyFrame{1, 2}`. The sample has every opcode, every operand type, every opening, temporaries
/// written by the op that reads them, a copy (an add of 0), and numbers of 254 bits, some twice in
/// a section.
fn sample() -> Bytecode {
    use Opcode::{Add, Mul, Sub, SubSwap};
    let (at_m2, at_m1, at_0, at_1, at_2) = (0, 1, 2, 3, 4);
    let r_minus_one = Operand::Number(fr(R_MINUS_ONE));
    let wide = Operand::Number(fr(WIDE_254));
    let zero = Operand::Number(FrBytes::ZERO);
    let (t0, t1, t2) = (Operand::Tmp(0), Operand::Tmp(1), Operand::Tmp(2));
    let code = |ops: Vec<Op>, n_temp: u32| {
        let dest_id = ops.last().unwrap().dest;
        Code { ops, n_temp, dest_id }
    };
    Bytecode {
        n_stages: 2,
        expressions: vec![
            // im = a' · b + const0[−1] − (r − 1)
            ExpressionBin {
                exp_id: 3,
                stage: 1,
                line: "Sample.ImPol".into(),
                code: code(
                    vec![
                        bin_op(Mul, 0, cm(1, 0, at_1), cm(1, 1, at_0)),
                        bin_op(Add, 1, Operand::Const { id: 0, opening: at_m1 }, t0),
                        bin_op(Sub, 0, t1, r_minus_one),
                    ],
                    2,
                ),
            },
            // Q = ((c · challenge0 + (public1 − airvalue0)) · ((2^253 + 5) + 0) + Zi1 · b[+2]) · Zi0
            ExpressionBin {
                exp_id: 7,
                stage: 3,
                line: String::new(),
                code: code(
                    vec![
                        bin_op(Mul, 0, cm(2, 0, at_0), Operand::Challenge(0)),
                        bin_op(Sub, 1, Operand::Public(1), Operand::AirValue(0)),
                        bin_op(Add, 0, t0, t1),
                        bin_op(Add, 1, wide, zero),
                        bin_op(Mul, 0, t0, t1),
                        bin_op(Mul, 2, Operand::Zi { boundary: 1 }, cm(1, 1, at_2)),
                        bin_op(Add, 0, t0, t2),
                        bin_op(Mul, 0, Operand::Zi { boundary: 0 }, t0),
                    ],
                    3,
                ),
            },
            // (airgroupvalue0 − proofvalue1) · public0 · (r − 1)
            ExpressionBin {
                exp_id: 9,
                stage: 1,
                line: "a hint's expression".into(),
                code: code(
                    vec![
                        bin_op(SubSwap, 0, Operand::ProofValue(1), Operand::AirgroupValue(0)),
                        bin_op(Mul, 1, t0, Operand::Public(0)),
                        bin_op(Mul, 0, t1, r_minus_one),
                    ],
                    2,
                ),
            },
            // Zi0 + eval0 · eval3: code evaluated at ξ
            ExpressionBin {
                exp_id: 11,
                stage: 0,
                line: "verifier form".into(),
                code: code(
                    vec![
                        bin_op(Mul, 0, Operand::Eval(0), Operand::Eval(3)),
                        bin_op(Add, 0, Operand::Zi { boundary: 0 }, t0),
                    ],
                    1,
                ),
            },
        ],
        constraints: vec![
            // a' − a · b, everyRow
            ConstraintBin {
                stage: 1,
                first_row: 0,
                last_row: 8,
                im_pol: false,
                line: "sample.pil:10 a' - a * b === 0".into(),
                code: code(vec![bin_op(Mul, 0, cm(1, 0, at_0), cm(1, 1, at_0)), bin_op(Sub, 0, cm(1, 0, at_1), t0)], 1),
            },
            // const0 · (a − public0), firstRow
            ConstraintBin {
                stage: 1,
                first_row: 0,
                last_row: 1,
                im_pol: false,
                line: "sample.pil:11 L1 * (a - in1) === 0".into(),
                code: code(
                    vec![
                        bin_op(Sub, 0, cm(1, 0, at_0), Operand::Public(0)),
                        bin_op(Mul, 0, Operand::Const { id: 0, opening: at_0 }, t0),
                    ],
                    1,
                ),
            },
            // b − (2^253 + 5), lastRow
            ConstraintBin {
                stage: 1,
                first_row: 7,
                last_row: 8,
                im_pol: false,
                line: "sample.pil:12 b - (2**253 + 5) === 0".into(),
                code: code(vec![bin_op(Sub, 0, cm(1, 1, at_0), wide)], 1),
            },
            // im − (a' · b + const0[−1] − (r − 1)) + (c[−2] + 0), everyFrame{1, 2}
            ConstraintBin {
                stage: 2,
                first_row: 1,
                last_row: 6,
                im_pol: true,
                line: "Sample.ImPol".into(),
                code: code(
                    vec![
                        bin_op(Mul, 0, cm(1, 0, at_1), cm(1, 1, at_0)),
                        bin_op(Add, 0, Operand::Const { id: 0, opening: at_m1 }, t0),
                        bin_op(Sub, 0, t0, r_minus_one),
                        bin_op(Sub, 0, cm(1, 2, at_0), t0),
                        bin_op(Add, 1, cm(2, 0, at_m2), zero),
                        bin_op(Add, 0, t0, t1),
                    ],
                    2,
                ),
            },
        ],
    }
}

#[test]
fn the_cpp_fixture_is_what_the_encoder_writes() {
    let path = Path::new(BYTECODE_FIXTURE);
    if std::env::var_os("PILFFLONK_UPDATE_FIXTURES").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        sample().write(path).unwrap();
    }
    let (_, written) = write_and_read(&sample(), "sample.bin");
    let fixture = fs::read(path).unwrap();
    assert!(fixture == written, "{BYTECODE_FIXTURE} is not what the encoder writes: regenerate it (see above)");
    assert_eq!(Bytecode::read(path).unwrap(), sample());
}
