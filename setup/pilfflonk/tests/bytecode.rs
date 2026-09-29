//! `<air>.bin` (M11): the encoder, the reader and the fixture M17's C++ tests read.
//!
//! Every round trip checks two things. Encoding and decoding gives back the same [`Bytecode`]. And
//! the decoded code is the same code as `pil-info`'s: the same ops, the same operands but for the
//! temporaries, whose slots are `pil-info`'s allocation, and the same value at every op when both
//! are evaluated over `Fr` on the same inputs.
//!
//! The `#[ignore]` tests compile real PIL with the compiler `PIL2C_EXEC` names, as those of
//! `pil-info`'s `tests/bn254.rs` do:
//!
//! ```text
//! PIL2C_EXEC=<pil2-compiler>/src/pil.js cargo test -p pilfflonk-setup --test bytecode -- --ignored
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
use pil_info::{PilInfoCfg, PilInfoResult};
use pilfflonk_setup::bytecode::{
    write_air_bin, Bytecode, BytecodeError, Code, ConstraintBin, ExpressionBin, ExpressionDest, Op, Opcode, Operand,
    BIN_VERSION,
};
use proofman_pilfflonk::field::FrBytes;

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
    (Bytecode::read(&path).unwrap(), bytes)
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

/// An intermediate polynomial's code as the passes write it, sparse temporaries included, with
/// every operand kind and constants of 254 bits.
fn small_code() -> Vec<CodeEntry> {
    vec![
        entry("mul", tmp(10), vec![column("cm", 0, 1), column("cm", 1, 0)]),
        entry("add", tmp(11), vec![tmp(10), column("const", 0, -1)]),
        entry("sub", tmp(12), vec![tmp(11), number(R_MINUS_ONE)]),
        entry("mul", tmp(13), vec![code_ref("challenge", 2), number(WIDE_254)]),
        entry("add", tmp(14), vec![tmp(12), tmp(13)]),
        entry("sub", tmp(15), vec![code_ref("public", 1), code_ref("airvalue", 0)]),
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

// ---------------------------------------------------------------------------------------------
// The decoded code as pil-info's, and both evaluated
// ---------------------------------------------------------------------------------------------

/// A `Code` as `pil-info`'s code: its last destination is the result's slot.
fn entries_of(code: &Code) -> Vec<CodeEntry> {
    let operand = |t: &Operand| match *t {
        Operand::Const { id, offset } => column("const", id as usize, offset.into()),
        Operand::Cm { id, offset } => column("cm", id as usize, offset.into()),
        Operand::Tmp(slot) => tmp(slot as usize),
        Operand::Public(id) => code_ref("public", id as usize),
        Operand::AirgroupValue(id) => code_ref("airgroupvalue", id as usize),
        Operand::Challenge(id) => code_ref("challenge", id as usize),
        Operand::Number(value) => number(&value.to_decimal()),
        Operand::AirValue(id) => code_ref("airvalue", id as usize),
        Operand::ProofValue(id) => code_ref("proofvalue", id as usize),
        Operand::Zi(boundary) => zi(boundary as usize),
        Operand::Eval(id) => code_ref("eval", id as usize),
    };
    code.ops
        .iter()
        .map(|op| {
            entry(op.opcode.name(), tmp(op.dest as usize), std::iter::once(&op.a).chain(&op.b).map(operand).collect())
        })
        .collect()
}

/// What the format keeps of an operand that is not a temporary.
fn leaf(r: &CodeRef) -> (String, usize, Option<i64>, Option<String>, Option<usize>) {
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

/// `decoded` is `original` in the format: the same ops and operands, temporaries apart, and the
/// same value written by every op.
fn assert_same_code(original: &[CodeEntry], decoded: &Code) {
    let entries = entries_of(decoded);
    assert_eq!(entries.len(), original.len());
    for (i, (o, d)) in original.iter().zip(&entries).enumerate() {
        assert_eq!(o.op, d.op, "op {i}");
        assert_eq!(o.src.len(), d.src.len(), "op {i}");
        for (os, ds) in o.src.iter().zip(&d.src) {
            if os.ref_type == "tmp" {
                assert_eq!(ds.ref_type, "tmp", "op {i}");
            } else {
                assert_eq!(leaf(os), leaf(ds), "op {i}");
            }
        }
    }
    assert_eq!(entries.last().unwrap().dest.id, decoded.result as usize);
    assert_eq!(evaluate(original), evaluate(&entries));
}

fn all_numbers(bytecode: &Bytecode) -> Vec<String> {
    let codes = bytecode.expressions.iter().map(|e| &e.code).chain(bytecode.constraints.iter().map(|c| &c.code));
    codes
        .flat_map(|c| c.ops.iter().flat_map(|op| std::iter::once(op.a).chain(op.b)))
        .filter_map(|t| match t {
            Operand::Number(v) => Some(v.to_decimal()),
            _ => None,
        })
        .collect()
}

/// Every code block of `bytecode` is the code of the same entry in `result`.
fn assert_is_the_code_of(bytecode: &Bytecode, result: &PilInfoResult) {
    let info = &result.pil_code.expressions_info;
    assert_eq!(bytecode.expressions.len(), info.expressions_code.len());
    for (bin, e) in bytecode.expressions.iter().zip(&info.expressions_code) {
        assert_eq!(bin.exp_id as usize, e.exp_id);
        assert_eq!(bin.stage as usize, e.stage);
        assert_eq!(bin.line, e.line);
        let dest = if e.exp_id == result.c_exp_id {
            ExpressionDest::Quotient
        } else if let Some(d) = &e.dest {
            ExpressionDest::ImPol { cm_id: d.id as u32 }
        } else {
            ExpressionDest::Value
        };
        assert_eq!(bin.dest, dest, "expression {}", e.exp_id);
        assert_same_code(&e.code, &bin.code);
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
        assert_same_code(&c.code, &bin.code);
    }
}

// ---------------------------------------------------------------------------------------------
// Code built in code
// ---------------------------------------------------------------------------------------------

#[test]
fn small_code_round_trips() {
    let original = small_code();
    let code = Code::from_entries(&original).unwrap();
    assert_same_code(&original, &code);
    // pil-info's allocation packs the 13 temporaries and the result into a few slots.
    assert!(code.n_temps <= 4, "{} slots", code.n_temps);

    let bytecode = Bytecode {
        expressions: vec![ExpressionBin {
            exp_id: 5,
            stage: 1,
            dest: ExpressionDest::ImPol { cm_id: 3 },
            line: "Small.ImPol".into(),
            code,
        }],
        constraints: vec![ConstraintBin {
            stage: 1,
            first_row: 0,
            last_row: 16,
            im_pol: true,
            line: String::new(),
            code: Code::from_entries(&original).unwrap(),
        }],
    };
    let (read, bytes) = write_and_read(&bytecode, "small.bin");
    assert_eq!(read, bytecode);
    assert_same_code(&original, &read.expressions[0].code);
    assert_same_code(&original, &read.constraints[0].code);

    // The 254-bit constants are whole, each once per section.
    assert_eq!(all_numbers(&read), [R_MINUS_ONE, WIDE_254, R_MINUS_ONE, R_MINUS_ONE, WIDE_254, R_MINUS_ONE]);
    let r_minus_one = fr(R_MINUS_ONE).to_le_bytes();
    let occurrences = bytes.windows(32).filter(|w| *w == r_minus_one).count();
    assert_eq!(occurrences, 2, "r − 1 once in the expressions and once in the constraints");
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
        expressions: vec![ExpressionBin {
            exp_id: 7,
            stage: 2,
            dest: ExpressionDest::Quotient,
            line: "q".into(),
            code: Code {
                ops: vec![Op {
                    opcode: Opcode::Mul,
                    dest: 0,
                    a: Operand::Cm { id: 3, offset: -1 },
                    b: Some(Operand::Number(fr(R_MINUS_ONE))),
                }],
                n_temps: 1,
                result: 0,
            },
        }],
        constraints: vec![ConstraintBin {
            stage: 1,
            first_row: 255,
            last_row: 256,
            im_pol: false,
            line: String::new(),
            code: Code {
                ops: vec![Op { opcode: Opcode::Copy, dest: 0, a: Operand::Zi(2), b: None }],
                n_temps: 1,
                result: 0,
            },
        }],
    };
    let (_, bytes) = write_and_read(&bytecode, "layout.bin");

    let mut expected: Vec<u8> = Vec::new();
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
    expected.extend_from_slice(b"chps");
    u32s(&mut expected, &[0x7066_0001, 4]);

    let mut header = Vec::new();
    u32s(&mut header, &[0x7066_0001, 32]);
    header.extend_from_slice(&modulus().to_bytes_le());
    section(&mut expected, 1, &header);

    let mut expressions = Vec::new();
    // nExpressions, nOps, nConstants, maxTemps; then expId, stage, destType (q), destId, nTemps,
    // result, opsOffset, nOps, line.
    u32s(&mut expressions, &[1, 1, 1, 1, 7, 2, 15, 0, 1, 0, 0, 1]);
    expressions.extend_from_slice(b"q\0");
    // mul, slot 0, cm 3 at −1, number 0.
    u32s(&mut expressions, &[2, 0, 1, 3, (-1i32) as u32, 6, 0, 0]);
    expressions.extend_from_slice(&fr(R_MINUS_ONE).to_le_bytes());
    section(&mut expected, 2, &expressions);

    let mut constraints = Vec::new();
    // nConstraints, nOps, nConstants, maxTemps; then stage, firstRow, lastRow, imPol, nTemps,
    // result, opsOffset, nOps, line.
    u32s(&mut constraints, &[1, 1, 0, 1, 1, 255, 256, 0, 1, 0, 0, 1]);
    constraints.push(0);
    // copy, slot 0, Zi of boundary 2, and three zeros.
    u32s(&mut constraints, &[4, 0, 12, 2, 0, 0, 0, 0]);
    section(&mut expected, 3, &constraints);

    section(&mut expected, 4, &0u32.to_le_bytes());
    assert_eq!(bytes, expected);
    assert_eq!(BIN_VERSION, 0x7066_0001);
}

// ---------------------------------------------------------------------------------------------
// What the encoder refuses
// ---------------------------------------------------------------------------------------------

fn encode_error(code: &[CodeEntry]) -> String {
    match Code::from_entries(code) {
        Err(e @ BytecodeError::Encode(_)) => e.to_string(),
        other => panic!("not an encoding error: {other:?}"),
    }
}

#[test]
fn code_the_format_cannot_hold_is_refused() {
    let two = |a: CodeRef, b: CodeRef| vec![entry("add", tmp(0), vec![a, b])];

    let err = encode_error(&two(CodeRef { dim: 3, ..code_ref("challenge", 0) }, tmp(1)));
    assert!(err.contains("dimension 3"), "{err}");
    let err = encode_error(&two(number(R), column("cm", 0, 0)));
    assert!(err.contains("not a canonical Fr"), "{err}");
    let err = encode_error(&two(number("0x10"), column("cm", 0, 0)));
    assert!(err.contains("not a canonical Fr"), "{err}");
    let err = encode_error(&two(column("custom", 0, 0), column("cm", 0, 0)));
    assert!(err.contains("custom is not an operand"), "{err}");
    let err = encode_error(&two(code_ref("xDivXSubXi", 0), column("cm", 0, 0)));
    assert!(err.contains("xDivXSubXi is not an operand"), "{err}");
    let err = encode_error(&two(column("cm", 0, 1 << 40), column("cm", 0, 0)));
    assert!(err.contains("row offset"), "{err}");
    let err = encode_error(&[]);
    assert!(err.contains("no ops"), "{err}");
    let err = encode_error(&[entry("neg", tmp(0), vec![tmp(1)])]);
    assert!(err.contains("unknown operation neg"), "{err}");
    let err = encode_error(&[entry("copy", tmp(0), vec![tmp(1), tmp(2)])]);
    assert!(err.contains("copy with 2 operands"), "{err}");
    let err = encode_error(&[
        entry("add", column("cm", 0, 0), vec![number("1"), number("2")]),
        entry("add", tmp(0), vec![number("1"), number("2")]),
    ]);
    assert!(err.contains("an op before the last writes a cm"), "{err}");
    let err = encode_error(&[entry("add", tmp(0), vec![tmp(5), number("2")])]);
    assert!(err.contains("read before it is written"), "{err}");
}

/// What the reader would refuse is not written either.
#[test]
fn a_bytecode_the_reader_would_refuse_is_not_written() {
    let one_op = |op: Op, n_temps: u32| Code { ops: vec![op], n_temps, result: 0 };
    let refused = |bytecode: Bytecode, why: &str| {
        let path = tmp_path("refused_on_write.bin");
        match bytecode.write(&path) {
            Err(e @ BytecodeError::Encode(_)) => assert!(e.to_string().contains(why), "{e}"),
            other => panic!("written, or not an encoding error: {other:?}"),
        }
    };
    let expression = |code: Code| Bytecode {
        expressions: vec![ExpressionBin {
            exp_id: 0,
            stage: 1,
            dest: ExpressionDest::Value,
            line: String::new(),
            code,
        }],
        constraints: Vec::new(),
    };

    let good = one_op(copy_op(0, cm(0, 0)), 1);
    refused(
        Bytecode {
            expressions: Vec::new(),
            constraints: vec![ConstraintBin {
                stage: 1,
                first_row: 2,
                last_row: 1,
                im_pol: false,
                line: String::new(),
                code: good.clone(),
            }],
        },
        "rows 2..1",
    );
    refused(expression(one_op(binary_op(Opcode::Add, 0, Operand::Tmp(0), cm(0, 0)), 1)), "read before it is written");
    refused(expression(one_op(Op { b: Some(cm(1, 0)), ..copy_op(0, cm(0, 0)) }, 1)), "copy with the wrong number");
    refused(expression(one_op(copy_op(0, cm(0, 0)), 2)), "2 slots for 1 ops");
    refused(expression(Code { result: 1, ..good }), "the result, slot 1");
}

#[test]
fn the_starks_passes_are_refused() {
    let result = pil_info::run(&wide_constants_pilout(), 0, 0, &PilInfoCfg::goldilocks(1), &Default::default());
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
    // The header's copy of the version, and its modulus.
    assert!(format_error(&patched(24, &2u32.to_le_bytes())).contains("header"));
    assert!(format_error(&patched(32, &[0])).contains("header"));
    let err = format_error(&bytes[..bytes.len() - 1]);
    assert!(err.contains("ends before"), "{err}");
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(format_error(&longer).contains("after its content"));

    // An op of the sample: the first of section 2, after its counts and entries.
    let expressions_start = 12 + (12 + 40) + 12;
    let first_op = find_first_op(&bytes[expressions_start..]) + expressions_start;
    let err = format_error(&patched(first_op, &3u32.to_le_bytes()));
    assert!(err.contains("unknown opcode 3"), "{err}");
    let err = format_error(&patched(first_op + 8, &10u32.to_le_bytes()));
    assert!(err.contains("kind 10 (custom) is not an operand"), "{err}");

    // Hints: revision 1 has none.
    let err = format_error(&patched(bytes.len() - 4, &1u32.to_le_bytes()));
    assert!(err.contains("1 hints"), "{err}");
}

/// The offset of the first op record in a code table: after the four counts and the entries,
/// each of eight words and a NUL-terminated line.
fn find_first_op(table: &[u8]) -> usize {
    let n_entries = u32::from_le_bytes(table[..4].try_into().unwrap()) as usize;
    let mut pos = 16;
    for _ in 0..n_entries {
        pos += 4 * 8;
        pos += table[pos..].iter().position(|&b| b == 0).unwrap() + 1;
    }
    pos
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
    pil_info::run(pilout, 0, 0, &PilInfoCfg::bn254(), &Default::default())
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
    (bytecode, bytes)
}

#[test]
fn the_passes_code_round_trips() {
    let result = run_bn254(&wide_constants_pilout());
    let (bytecode, _) = check_air_bin(&result, "wide_constants_in_code");

    let numbers = all_numbers(&bytecode);
    for value in [WIDE_200, R_MINUS_TWO, R_MINUS_ONE] {
        assert!(numbers.iter().any(|n| n == value), "{value} not in {numbers:?}");
    }
    let dests: Vec<ExpressionDest> = bytecode.expressions.iter().map(|e| e.dest).collect();
    let im_pols: Vec<u32> =
        result.setup.cm_pols_map.iter().enumerate().filter(|(_, p)| p.im_pol).map(|(i, _)| i as u32).collect();
    assert!(!im_pols.is_empty(), "the degree-4 constraint gets an intermediate polynomial");
    for cm_id in &im_pols {
        assert!(dests.contains(&ExpressionDest::ImPol { cm_id: *cm_id }), "{dests:?}");
    }
    assert_eq!(dests.iter().filter(|d| **d == ExpressionDest::Quotient).count(), 1);
    let q = bytecode.expressions.iter().find(|e| e.dest == ExpressionDest::Quotient).unwrap();
    assert_eq!(q.exp_id as usize, result.c_exp_id);

    // The passes run twice give the same file.
    let second = run_bn254(&wide_constants_pilout());
    let (a, b) = (tmp_path("passes_1.bin"), tmp_path("passes_2.bin"));
    write_air_bin(&result, &a).unwrap();
    write_air_bin(&second, &b).unwrap();
    assert_eq!(fs::read(a).unwrap(), fs::read(b).unwrap());
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
    let stem = Path::new(pil).file_stem().unwrap().to_string_lossy().into_owned();
    let out = tmp_path(&format!("{stem}.bn254.pilout"));
    let status = Command::new(compiler)
        .current_dir(repo_root())
        .arg(pil)
        .args(["-I", "pil2-components/lib/std/pil", "-P", "pilfflonk/tests/fixtures/fibonacci/bn254.json", "-o"])
        .arg(&out)
        .status()
        .expect("PIL2C_EXEC runs");
    assert!(status.success(), "pil2com failed on {pil}");
    let pilout = PilOutProxy::new(out.to_str().unwrap()).unwrap().pilout;
    assert_eq!(pilout.base_field, big_be(R), "{pil} was not compiled over BN254: does PIL2C_EXEC honour `prime`?");
    pilout
}

/// The M13 Fibonacci fixture: its intermediate polynomial (`l1' − next`, column 2), `Q` and six
/// constraints over its 256 rows, the one of the intermediate polynomial included.
#[test]
#[ignore = "needs PIL2C_EXEC"]
fn fibonacci_fixture_round_trips() {
    let result = run_bn254(&compile_bn254("pilfflonk/tests/fixtures/fibonacci/fibonacci.pil"));
    let (bytecode, bytes) = check_air_bin(&result, "fibonacci");

    let dests: Vec<(u32, ExpressionDest)> = bytecode.expressions.iter().map(|e| (e.exp_id, e.dest)).collect();
    assert_eq!(dests, [(6, ExpressionDest::ImPol { cm_id: 2 }), (result.c_exp_id as u32, ExpressionDest::Quotient)]);
    assert_eq!(result.setup.cm_pols_map[2].name, "Fibonacci.ImPol");
    assert_eq!(bytecode.constraints.len(), 6);
    assert!(bytecode.constraints.iter().all(|c| (c.first_row, c.last_row) == (0, 256)));
    assert_eq!(bytecode.constraints.iter().filter(|c| c.im_pol).count(), 1);
    assert!(result.pil_code.expressions_info.hints_info.is_empty());

    // Q reads no piece Q0 … of cmPolsMap: it is computed whole.
    let q_stage = result.setup.n_stages + 1;
    let q = &bytecode.expressions[1];
    for op in &q.code.ops {
        for t in std::iter::once(op.a).chain(op.b) {
            if let Operand::Cm { id, .. } = t {
                assert_ne!(result.setup.cm_pols_map[id as usize].stage, Some(q_stage), "Q reads a piece of Q");
            }
        }
    }
    assert!(matches!(q.code.ops.last().unwrap().b, Some(Operand::Zi(0))), "Q ends dividing by Z_H");
    println!(
        "fibonacci: {} expressions ({} and {} ops, {} and {} slots), {} constraints, {} bytes",
        bytecode.expressions.len(),
        bytecode.expressions[0].code.ops.len(),
        q.code.ops.len(),
        bytecode.expressions[0].code.n_temps,
        q.code.n_temps,
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

fn binary_op(opcode: Opcode, dest: u32, a: Operand, b: Operand) -> Op {
    Op { opcode, dest, a, b: Some(b) }
}

fn copy_op(dest: u32, a: Operand) -> Op {
    Op { opcode: Opcode::Copy, dest, a, b: None }
}

fn cm(id: u32, offset: i32) -> Operand {
    Operand::Cm { id, offset }
}

/// Every opcode, every operand kind, offsets −2 to 2, slots written by the op that reads them, and
/// constants of 254 bits, one of them twice in a section. As for an AIR of `N = 8` rows with
/// columns `cm0`, `cm1` and the intermediate polynomial `cm2`, and constraints on the rows of
/// everyRow, firstRow, lastRow and `everyFrame{1, 2}`.
fn sample() -> Bytecode {
    use Opcode::{Add, Mul, Sub};
    let r_minus_one = Operand::Number(fr(R_MINUS_ONE));
    let wide = Operand::Number(fr(WIDE_254));
    let code = |ops: Vec<Op>, n_temps: u32, result: u32| Code { ops, n_temps, result };
    Bytecode {
        expressions: vec![
            // cm2 = cm0' · cm1 + const0[−1] − (r − 1)
            ExpressionBin {
                exp_id: 3,
                stage: 1,
                dest: ExpressionDest::ImPol { cm_id: 2 },
                line: "Sample.ImPol".into(),
                code: code(
                    vec![
                        binary_op(Mul, 0, cm(0, 1), cm(1, 0)),
                        binary_op(Add, 1, Operand::Tmp(0), Operand::Const { id: 0, offset: -1 }),
                        binary_op(Sub, 0, Operand::Tmp(1), r_minus_one),
                    ],
                    2,
                    0,
                ),
            },
            // Q = ((challenge0 · cm2 + (public1 − airvalue0)) · (2^253 + 5) + cm1[+2] · Zi1) · Zi0
            ExpressionBin {
                exp_id: 7,
                stage: 2,
                dest: ExpressionDest::Quotient,
                line: String::new(),
                code: code(
                    vec![
                        binary_op(Mul, 0, Operand::Challenge(0), cm(2, 0)),
                        binary_op(Sub, 1, Operand::Public(1), Operand::AirValue(0)),
                        binary_op(Add, 0, Operand::Tmp(0), Operand::Tmp(1)),
                        copy_op(1, wide),
                        binary_op(Mul, 0, Operand::Tmp(0), Operand::Tmp(1)),
                        binary_op(Mul, 2, cm(1, 2), Operand::Zi(1)),
                        binary_op(Add, 0, Operand::Tmp(0), Operand::Tmp(2)),
                        binary_op(Mul, 0, Operand::Tmp(0), Operand::Zi(0)),
                    ],
                    3,
                    0,
                ),
            },
            // (airgroupvalue0 − proofvalue1) · public0 · (r − 1)
            ExpressionBin {
                exp_id: 9,
                stage: 1,
                dest: ExpressionDest::Value,
                line: "a hint's expression".into(),
                code: code(
                    vec![
                        binary_op(Sub, 0, Operand::AirgroupValue(0), Operand::ProofValue(1)),
                        copy_op(1, Operand::Public(0)),
                        binary_op(Mul, 1, Operand::Tmp(0), Operand::Tmp(1)),
                        binary_op(Mul, 0, Operand::Tmp(1), r_minus_one),
                    ],
                    2,
                    0,
                ),
            },
            // eval0 · eval3 + Zi0: code evaluated at ξ
            ExpressionBin {
                exp_id: 11,
                stage: 0,
                dest: ExpressionDest::Value,
                line: "verifier form".into(),
                code: code(
                    vec![
                        binary_op(Mul, 0, Operand::Eval(0), Operand::Eval(3)),
                        binary_op(Add, 0, Operand::Tmp(0), Operand::Zi(0)),
                    ],
                    1,
                    0,
                ),
            },
        ],
        constraints: vec![
            // cm0' − cm0 · cm1, everyRow
            ConstraintBin {
                stage: 1,
                first_row: 0,
                last_row: 8,
                im_pol: false,
                line: "sample.pil:10 a' - a * b === 0".into(),
                code: code(
                    vec![binary_op(Mul, 0, cm(0, 0), cm(1, 0)), binary_op(Sub, 0, cm(0, 1), Operand::Tmp(0))],
                    1,
                    0,
                ),
            },
            // const0 · (cm0 − public0), firstRow
            ConstraintBin {
                stage: 1,
                first_row: 0,
                last_row: 1,
                im_pol: false,
                line: "sample.pil:11 L1 * (a - in1) === 0".into(),
                code: code(
                    vec![
                        binary_op(Sub, 0, cm(0, 0), Operand::Public(0)),
                        binary_op(Mul, 0, Operand::Const { id: 0, offset: 0 }, Operand::Tmp(0)),
                    ],
                    1,
                    0,
                ),
            },
            // cm1 − (2^253 + 5), lastRow
            ConstraintBin {
                stage: 1,
                first_row: 7,
                last_row: 8,
                im_pol: false,
                line: "sample.pil:12 b - (2**253 + 5) === 0".into(),
                code: code(vec![binary_op(Sub, 0, cm(1, 0), wide)], 1, 0),
            },
            // cm2 − (cm0' · cm1 + const0[−1] − (r − 1)), everyFrame{1, 2}
            ConstraintBin {
                stage: 1,
                first_row: 1,
                last_row: 6,
                im_pol: true,
                line: "Sample.ImPol".into(),
                code: code(
                    vec![
                        binary_op(Mul, 0, cm(0, 1), cm(1, 0)),
                        binary_op(Add, 0, Operand::Tmp(0), Operand::Const { id: 0, offset: -1 }),
                        binary_op(Sub, 0, Operand::Tmp(0), r_minus_one),
                        binary_op(Sub, 0, cm(2, 0), Operand::Tmp(0)),
                        copy_op(1, cm(1, -2)),
                        binary_op(Mul, 1, Operand::Tmp(1), Operand::Number(fr("0"))),
                        binary_op(Add, 0, Operand::Tmp(0), Operand::Tmp(1)),
                    ],
                    2,
                    0,
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
