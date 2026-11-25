//! Pilouts built in code for the core tests.
#![allow(dead_code)]

use num_bigint::BigUint;
use pil_info::core::from_pilout::from_pilout;
use pil_info::core::im::{substitute, Checked, ImChoice};
use pil_info::core::ir::{degree, ConstraintSystem, Expr};
use pil_info::core::layout::{layout, Layout, LayoutParams};
use pil_info::core::openings::{openings, Openings};
use pil_info::emit::prover::{to_arena, ProverExprs};
use pil_info::{Opening, PilInfoCfg};
use pil2_pilout::pilout::{self as pb, constraint, expression, global_expression, global_operand, operand, SymbolType};

pub const R: &str = "21888242871839275222246405745257275088548364400416034343698204186575808495617";

pub fn op(o: operand::Operand) -> Option<pb::Operand> {
    Some(pb::Operand { operand: Some(o) })
}
pub fn witness_at(stage: u32, col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    op(operand::Operand::WitnessCol(operand::WitnessCol { stage, col_idx, row_offset }))
}
pub fn witness(col_idx: u32, row_offset: i32) -> Option<pb::Operand> {
    witness_at(1, col_idx, row_offset)
}
pub fn fixed(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::FixedCol(operand::FixedCol { idx, row_offset: 0 }))
}
pub fn big_be(decimal: &str) -> Vec<u8> {
    BigUint::parse_bytes(decimal.as_bytes(), 10).expect("a decimal constant").to_bytes_be()
}
pub fn constant(decimal: &str) -> Option<pb::Operand> {
    op(operand::Operand::Constant(operand::Constant { value: big_be(decimal) }))
}
pub fn exp(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Expression(operand::Expression { idx }))
}
pub fn challenge(stage: u32, idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::Challenge(operand::Challenge { stage, idx }))
}
pub fn public(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::PublicValue(operand::PublicValue { idx }))
}
pub fn air_value(idx: u32) -> Option<pb::Operand> {
    op(operand::Operand::AirValue(operand::AirValue { idx }))
}
pub fn add(l: Option<pb::Operand>, r: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Add(expression::Add { lhs: l, rhs: r })) }
}
pub fn sub(l: Option<pb::Operand>, r: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Sub(expression::Sub { lhs: l, rhs: r })) }
}
pub fn mul(l: Option<pb::Operand>, r: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Mul(expression::Mul { lhs: l, rhs: r })) }
}
pub fn neg(v: Option<pb::Operand>) -> pb::Expression {
    pb::Expression { operation: Some(expression::Operation::Neg(expression::Neg { value: v })) }
}

pub fn every_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("constraint on expression {idx}")),
        })),
    }
}
pub fn first_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::FirstRow(constraint::FirstRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("firstRow {idx}")),
        })),
    }
}
pub fn last_row(idx: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::LastRow(constraint::LastRow {
            expression_idx: Some(operand::Expression { idx }),
            debug_line: Some(format!("lastRow {idx}")),
        })),
    }
}
pub fn every_frame(idx: u32, min: u32, max: u32) -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryFrame(constraint::EveryFrame {
            expression_idx: Some(operand::Expression { idx }),
            offset_min: min,
            offset_max: max,
            debug_line: Some(format!("everyFrame {idx}")),
        })),
    }
}
pub fn every_row_no_idx() -> pb::Constraint {
    pb::Constraint {
        constraint: Some(constraint::Constraint::EveryRow(constraint::EveryRow {
            expression_idx: None,
            debug_line: Some("no idx".to_string()),
        })),
    }
}
pub fn empty_constraint() -> pb::Constraint {
    pb::Constraint { constraint: None }
}

pub fn symbol(name: &str, kind: SymbolType, stage: u32, id: u32) -> pb::Symbol {
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

/// One BN128-fielded air of 16 rows with one stage (`numChallenges = [0]`), as pil2com emits it.
pub fn pilout(
    witness_names: &[&str],
    fixed_names: &[&str],
    expressions: Vec<pb::Expression>,
    constraints: Vec<pb::Constraint>,
) -> pb::PilOut {
    let mut symbols: Vec<pb::Symbol> =
        fixed_names.iter().enumerate().map(|(i, n)| symbol(n, SymbolType::FixedCol, 0, i as u32)).collect();
    symbols.extend(witness_names.iter().enumerate().map(|(i, n)| symbol(n, SymbolType::WitnessCol, 1, i as u32)));
    let air = pb::Air {
        name: Some("Synthetic".to_string()),
        num_rows: Some(16),
        fixed_cols: fixed_names.iter().map(|_| pb::FixedCol { values: Vec::new() }).collect(),
        stage_widths: vec![witness_names.len() as u32],
        expressions,
        constraints,
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

// ---- global constraints ----
pub fn gop(o: global_operand::Operand) -> Option<pb::GlobalOperand> {
    Some(pb::GlobalOperand { operand: Some(o) })
}
pub fn gpublic(idx: u32) -> Option<pb::GlobalOperand> {
    gop(global_operand::Operand::PublicValue(global_operand::PublicValue { idx }))
}
pub fn gexp(idx: u32) -> Option<pb::GlobalOperand> {
    gop(global_operand::Operand::Expression(global_operand::Expression { idx }))
}
pub fn gconst(v: u64) -> Option<pb::GlobalOperand> {
    gop(global_operand::Operand::Constant(global_operand::Constant { value: BigUint::from(v).to_bytes_be() }))
}
pub fn gsub(l: Option<pb::GlobalOperand>, r: Option<pb::GlobalOperand>) -> pb::GlobalExpression {
    pb::GlobalExpression {
        operation: Some(global_expression::Operation::Sub(global_expression::Sub { lhs: l, rhs: r })),
    }
}
pub fn gmul(l: Option<pb::GlobalOperand>, r: Option<pb::GlobalOperand>) -> pb::GlobalExpression {
    pb::GlobalExpression {
        operation: Some(global_expression::Operation::Mul(global_expression::Mul { lhs: l, rhs: r })),
    }
}
pub fn gconstraint(idx: Option<u32>) -> pb::GlobalConstraint {
    pb::GlobalConstraint {
        expression_idx: idx.map(|idx| global_operand::Expression { idx }),
        debug_line: Some("global".to_string()),
    }
}

// ---- real pilouts ----
pub fn gl() -> pil_info::FieldCfg {
    pil_info::FieldCfg::goldilocks()
}

/// The example pilouts built under `pil2-components/test/*/build` (the loaded pilout and its path).
pub fn real_pilouts() -> Vec<(std::path::PathBuf, pb::PilOut)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pil2-components/test");
    let mut out = Vec::new();
    for d in std::fs::read_dir(&root).expect("pil2-components/test").flatten() {
        let Ok(files) = std::fs::read_dir(d.path().join("build")) else { continue };
        for f in files.flatten().map(|e| e.path()).filter(|f| f.extension().is_some_and(|e| e == "pilout")) {
            let Some(path) = f.to_str() else { continue };
            let Ok(proxy) = pil2_pilout::pilout_proxy::PilOutProxy::new(path) else { continue };
            out.push((f, proxy.pilout));
        }
    }
    assert!(!out.is_empty(), "no pilout under {root:?}");
    out
}

// ---- shared core helpers ----

pub fn cs_of(p: &pb::PilOut) -> ConstraintSystem {
    from_pilout(p, 0, 0, &gl(), Opening::Deep).expect("a lowerable pilout")
}

/// 4 witness columns a,b,c,d; one every-row constraint per entry of `roots`.
pub fn system(exprs: Vec<pb::Expression>, roots: &[u32]) -> ConstraintSystem {
    cs_of(&pilout(&["a", "b", "c", "d"], &[], exprs, roots.iter().map(|r| every_row(*r)).collect()))
}

/// 0: a*b, 1: e0*c, 2: e1*d (degree 4), 3: e2*a (degree 5).
pub fn chain() -> Vec<pb::Expression> {
    vec![
        mul(witness(0, 0), witness(1, 0)),
        mul(exp(0), witness(2, 0)),
        mul(exp(1), witness(3, 0)),
        mul(exp(2), witness(0, 0)),
    ]
}

/// The expression with every Ref inlined (small systems only).
pub fn inlined(cs: &ConstraintSystem, e: &Expr) -> Expr {
    substitute(cs, &ImChoice { exprs: Vec::new() }, e).unwrap()
}

/// The table index whose inlined body is `want` (lowering renumbers the input).
pub fn find(cs: &ConstraintSystem, want: &Expr) -> u32 {
    let hits: Vec<usize> = (0..cs.exprs.len()).filter(|&i| inlined(cs, &Expr::Ref(i as u32)) == *want).collect();
    assert_eq!(hits.len(), 1, "body {want:?} found {hits:?}");
    hits[0] as u32
}

pub fn choice(v: &[u32]) -> ImChoice {
    ImChoice { exprs: v.to_vec() }
}

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A seeded random DAG of <= 9 expressions over 4 columns, depth <= 6, every constraint of degree 1..=24.
pub fn random_pilout(seed: u64) -> (pb::PilOut, ConstraintSystem) {
    random_pilout_in(seed, Opening::Deep)
}

/// As `random_pilout`; under Shplonk the roots also take firstRow/lastRow/everyFrame boundaries.
pub fn random_pilout_in(seed: u64, opening: Opening) -> (pb::PilOut, ConstraintSystem) {
    let mut rng = Rng(seed);
    loop {
        let n = 3 + rng.below(7) as u32;
        let mut depth: Vec<u32> = Vec::new();
        let mut exprs = Vec::new();
        for i in 0..n {
            let operand = |rng: &mut Rng, depth: &Vec<u32>| -> (Option<pb::Operand>, u32) {
                match rng.below(20) {
                    0..=7 => (witness(rng.below(4) as u32, rng.below(2) as i32), 0),
                    8..=10 => (constant(&(1 + rng.below(9)).to_string()), 0),
                    _ if i > 0 => {
                        let j = rng.below(i as u64) as usize;
                        if depth[j] < 6 {
                            (exp(j as u32), depth[j])
                        } else {
                            (witness(rng.below(4) as u32, 0), 0)
                        }
                    }
                    _ => (witness(rng.below(4) as u32, 0), 0),
                }
            };
            let (l, dl) = operand(&mut rng, &depth);
            let (r, dr) = operand(&mut rng, &depth);
            let (e, d) = match rng.below(9) {
                0..=2 => (add(l, r), dl.max(dr) + 1),
                3..=4 => (sub(l, r), dl.max(dr) + 1),
                5..=7 => (mul(l, r), dl.max(dr) + 1),
                _ => (neg(l), dl + 1),
            };
            exprs.push(e);
            depth.push(d);
        }
        let mut roots = vec![n - 1];
        for _ in 0..rng.below(3) {
            roots.push(rng.below(n as u64) as u32);
        }
        let cons = roots
            .iter()
            .map(|r| match (opening, rng.below(4)) {
                (Opening::Deep, _) | (_, 0) => every_row(*r),
                (_, 1) => first_row(*r),
                (_, 2) => last_row(*r),
                _ => every_frame(*r, 0, 1),
            })
            .collect();
        let p = pilout(&["a", "b", "c", "d"], &[], exprs, cons);
        let Ok(cs) = from_pilout(&p, 0, 0, &gl(), opening) else { continue };
        let degs: Vec<u32> = cs.constraints.iter().map(|c| degree(&cs, &c.e).unwrap()).collect();
        if degs.iter().all(|d| (1..=24).contains(d)) {
            return (p, cs);
        }
    }
}

/// Two stages: witness a,b (stage 1) and x,y (stage 2, extension), challenges alpha,beta (stage 2).
pub fn pilout2(fixed_names: &[&str], exprs: Vec<pb::Expression>, constraints: Vec<pb::Constraint>) -> pb::PilOut {
    let mut p = pilout(&["a", "b"], fixed_names, exprs, constraints);
    p.num_challenges = vec![0, 2];
    for (i, n) in ["x", "y"].iter().enumerate() {
        p.symbols.push(symbol(n, SymbolType::WitnessCol, 2, i as u32));
    }
    for (i, n) in ["alpha", "beta"].iter().enumerate() {
        p.symbols.push(symbol(n, SymbolType::Challenge, 2, i as u32));
    }
    p.air_groups[0].airs[0].stage_widths = vec![2, 2];
    p
}

/// The core on one air: the system, the searched im choice and its check.
pub struct Run {
    pub cs: ConstraintSystem,
    pub im: ImChoice,
    pub chk: Checked,
    pub air_name: String,
    pub blowup_bits: usize,
    pub im_pols_stages: bool,
}

/// None when the search finds the air infeasible at this blowup.
pub fn run(p: &pb::PilOut, airgroup: usize, air: usize, blowup_bits: usize) -> Option<Run> {
    run_with(p, airgroup, air, blowup_bits, false)
}

/// `run`, given `--impols`.
pub fn run_with(p: &pb::PilOut, airgroup: usize, air: usize, blowup_bits: usize, im_pols_stages: bool) -> Option<Run> {
    let (cs, im, chk) = pil_info::pipeline::search(p, airgroup, air, &PilInfoCfg::goldilocks(blowup_bits)).ok()?;
    let air_name = p.air_groups[airgroup].airs[air].name.clone().unwrap_or_default();
    Some(Run { cs, im, chk, air_name, blowup_bits, im_pols_stages })
}

/// `run` at the first blowup of 1..=3 bits that is feasible.
pub fn run_any(p: &pb::PilOut, airgroup: usize, air: usize) -> Option<Run> {
    (1..=3).find_map(|b| run(p, airgroup, air, b))
}

/// The core chain on `run`, as Task 7 runs it: layout, the arena once, openings with the hints' points.
pub struct CoreOut {
    pub lay: Layout,
    pub exprs: ProverExprs,
    pub ops: Openings,
}

pub fn core_out(run: &Run) -> CoreOut {
    let params = LayoutParams {
        im_pols_stages: run.im_pols_stages,
        deep_challenges: true,
        q_deg: run.chk.q_deg,
        air_name: run.air_name.clone(),
    };
    let lay = layout(&run.cs, &run.im, &params).unwrap();
    let exprs = to_arena(&run.cs, &run.im, &lay).unwrap();
    let ops = openings(&run.cs, &run.im, &lay).unwrap();
    CoreOut { lay, exprs, ops }
}

// ---- compiled fixtures (PIL2C_EXEC) ----

pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("the repository root")
}

/// `pil` (relative to the repository root) compiled over BN128 by `$PIL2C_EXEC`; None when it does not compile.
pub fn compile_bn128(pil: &std::path::Path) -> Option<pb::PilOut> {
    let compiler = std::env::var("PIL2C_EXEC").expect("PIL2C_EXEC must name a pil2com that has `--field`");
    let stem = pil.file_stem()?.to_string_lossy().into_owned();
    // common/ is also compiled into pil2-stark's unit tests, which have no CARGO_TARGET_TMPDIR
    let tmp = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, std::path::PathBuf::from);
    let out = tmp.join(format!("{stem}.bn128.pilout"));
    let status = std::process::Command::new(compiler)
        .current_dir(repo_root())
        .arg(pil)
        .args(["-I", "pil2-components/lib/std/pil", "--field", "bn128", "-o"])
        .arg(&out)
        .status()
        .expect("PIL2C_EXEC runs");
    if !status.success() {
        eprintln!("skipped {}: does not compile", pil.display());
        return None;
    }
    let pilout = pil2_pilout::pilout_proxy::PilOutProxy::new(out.to_str()?).expect("a pilout").pilout;
    assert_eq!(
        pilout.base_field,
        big_be(R),
        "{} was not compiled over BN128: does PIL2C_EXEC have `--field`?",
        pil.display()
    );
    Some(pilout)
}
