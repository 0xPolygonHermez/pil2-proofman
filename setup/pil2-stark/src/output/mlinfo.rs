//! Compilation of the pilout constraint expressions into the multilinear
//! prover's IR (`<AIR>.mlinfo.bin`, a bincode-serialized
//! [`proofman_multilinear::AirIr`]).
//!
//! The setup expression arena is lowered into a flat instruction list; arena
//! references (`op == "exp"`) are memoized so shared subexpressions become
//! shared temps. AIRs using features the multilinear prover does not support
//! yet (`everyFrame` constraints, shifted columns inside boundary
//! constraints) are rejected with a descriptive error — the setup then simply
//! skips the mlinfo artifact.

use std::collections::{BTreeSet, HashMap};

use anyhow::{anyhow, bail, Result};
use fields::{Goldilocks, PrimeField64};
use proofman_multilinear::{
    AirIr, Boundary, BusIr, BusTerm, ConstraintIr, Instr, MlCustomCommit, MlParams, Op, Operand, SrcKind,
};

use crate::expr::expression::{ExprChild, Expression};
use crate::types::pilout_info::{HintFieldValue, HintInfo, SetupResult};

struct Compiler<'a> {
    setup: &'a SetupResult,
    exprs: &'a [Expression],
    instrs: Vec<Instr>,
    numbers: Vec<u64>,
    /// Memoized lowering of arena entries: arena id → (operand, degree, offsets used).
    memo: HashMap<usize, (Operand, u32, BTreeSet<i32>)>,
    offsets: BTreeSet<i32>,
}

impl Compiler<'_> {
    fn number(&mut self, value: &str) -> Result<Operand> {
        let v = value
            .parse::<u128>()
            .map(|v| (v % Goldilocks::ORDER_U64 as u128) as u64)
            .map_err(|_| anyhow!("cannot parse number '{value}'"))?;
        let idx = self.numbers.iter().position(|&n| n == v).unwrap_or_else(|| {
            self.numbers.push(v);
            self.numbers.len() - 1
        });
        Ok(Operand { kind: SrcKind::Number, idx: idx as u32, row_offset: 0, dim: 1 })
    }

    fn emit(&mut self, op: Op, a: Operand, b: Operand) -> Operand {
        let dst = self.instrs.len() as u32;
        self.instrs.push(Instr { op, dst, a, b });
        Operand { kind: SrcKind::Temp, idx: dst, row_offset: 0, dim: 1 }
    }

    /// Lower an arena entry (memoized).
    fn lower_id(&mut self, id: usize) -> Result<(Operand, u32, BTreeSet<i32>)> {
        if let Some(hit) = self.memo.get(&id) {
            return Ok(hit.clone());
        }
        let result = self.lower_expr(&self.exprs[id])?;
        self.memo.insert(id, result.clone());
        Ok(result)
    }

    fn lower_child(&mut self, child: &ExprChild) -> Result<(Operand, u32, BTreeSet<i32>)> {
        match child {
            ExprChild::Id(id) => self.lower_id(*id),
            ExprChild::Inline(expr) => self.lower_expr(expr),
        }
    }

    /// Lower one expression node. Returns (operand, degree, row offsets in its cone).
    fn lower_expr(&mut self, e: &Expression) -> Result<(Operand, u32, BTreeSet<i32>)> {
        let leaf_offset = |e: &Expression| e.row_offset.unwrap_or(0) as i32;
        match e.op.as_str() {
            "cm" => {
                // `id` is the global pol id: index into cm_pols_map, which
                // carries the stage, base-slot offset (stage_pos = sum of
                // earlier same-stage dims) and dimension.
                let pol_id = e.id.ok_or_else(|| anyhow!("cm operand without id"))?;
                let info =
                    self.setup.cm_pols_map.get(pol_id).ok_or_else(|| anyhow!("cm pol id {pol_id} out of range"))?;
                let stage = info.stage.ok_or_else(|| anyhow!("cm pol {pol_id} without stage"))? as u8;
                if stage as usize > self.setup.n_stages || stage == 0 {
                    bail!("witness column of stage {stage} (quotient stage is univariate-only)");
                }
                let base_slot = info.stage_pos.ok_or_else(|| anyhow!("cm pol {pol_id} without stage_pos"))? as u32;
                let dim = info.dim as u8;
                let s = leaf_offset(e);
                self.offsets.insert(s);
                Ok((
                    Operand { kind: SrcKind::Witness { stage }, idx: base_slot, row_offset: s, dim },
                    1,
                    BTreeSet::from([s]),
                ))
            }
            "const" => {
                let s = leaf_offset(e);
                self.offsets.insert(s);
                let col = e.id.ok_or_else(|| anyhow!("const operand without id"))? as u32;
                Ok((Operand { kind: SrcKind::Const, idx: col, row_offset: s, dim: 1 }, 1, BTreeSet::from([s])))
            }
            "public" => {
                let idx = e.id.ok_or_else(|| anyhow!("public operand without id"))? as u32;
                Ok((Operand { kind: SrcKind::Public, idx, row_offset: 0, dim: 1 }, 0, BTreeSet::new()))
            }
            "challenge" => {
                let idx = e.id.ok_or_else(|| anyhow!("challenge operand without id"))? as u32;
                Ok((Operand { kind: SrcKind::Challenge, idx, row_offset: 0, dim: 3 }, 0, BTreeSet::new()))
            }
            "airvalue" => {
                let idx = e.id.ok_or_else(|| anyhow!("airvalue operand without id"))?;
                let dim = self.setup.air_values_map.get(idx).map(|v| v.dim as u8).unwrap_or(e.dim.max(1) as u8);
                Ok((Operand { kind: SrcKind::AirValue, idx: idx as u32, row_offset: 0, dim }, 0, BTreeSet::new()))
            }
            "airgroupvalue" => {
                let idx = e.id.ok_or_else(|| anyhow!("airgroupvalue operand without id"))?;
                let dim = self.setup.airgroup_values_map.get(idx).map(|v| v.dim as u8).unwrap_or(e.dim.max(1) as u8);
                Ok((Operand { kind: SrcKind::AirGroupValue, idx: idx as u32, row_offset: 0, dim }, 0, BTreeSet::new()))
            }
            "proofvalue" => {
                let idx = e.id.ok_or_else(|| anyhow!("proofvalue operand without id"))?;
                let dim = self.setup.proof_values_map.get(idx).map(|v| v.dim as u8).unwrap_or(e.dim.max(1) as u8);
                Ok((Operand { kind: SrcKind::ProofValue, idx: idx as u32, row_offset: 0, dim }, 0, BTreeSet::new()))
            }
            "number" => {
                let value = e.value.as_deref().ok_or_else(|| anyhow!("number operand without value"))?;
                Ok((self.number(value)?, 0, BTreeSet::new()))
            }
            "custom" => {
                // Custom-commit column (e.g. a ROM). Only stage-0 (fixed)
                // custom commits are supported: they are committed like the
                // const columns, with their own root.
                let commit_id = e.commit_id.ok_or_else(|| anyhow!("custom operand without commit_id"))?;
                let pol_id = e.id.ok_or_else(|| anyhow!("custom operand without id"))?;
                let info = self
                    .setup
                    .custom_commits_map
                    .get(commit_id)
                    .and_then(|m| m.get(pol_id))
                    .ok_or_else(|| anyhow!("custom pol ({commit_id},{pol_id}) out of range"))?;
                let stage = info.stage.unwrap_or(0);
                if stage != 0 {
                    bail!("custom commit column of stage {stage} (only stage-0 custom commits supported)");
                }
                let base_slot =
                    info.stage_pos.ok_or_else(|| anyhow!("custom pol ({commit_id},{pol_id}) without stage_pos"))?
                        as u32;
                let s_off = leaf_offset(e);
                self.offsets.insert(s_off);
                Ok((
                    Operand {
                        kind: SrcKind::Custom { commit: commit_id as u8 },
                        idx: base_slot,
                        row_offset: s_off,
                        dim: info.dim.max(1) as u8,
                    },
                    1,
                    BTreeSet::from([s_off]),
                ))
            }
            "exp" => self.lower_id(e.id.ok_or_else(|| anyhow!("exp operand without id"))?),
            "add" | "sub" | "mul" => {
                if e.values.len() != 2 {
                    bail!("binary op '{}' with {} children", e.op, e.values.len());
                }
                let (a, da, mut offs_a) = self.lower_child(&e.values[0])?;
                let (b, db, offs_b) = self.lower_child(&e.values[1])?;
                offs_a.extend(offs_b);
                let (op, deg) = match e.op.as_str() {
                    "add" => (Op::Add, da.max(db)),
                    "sub" => (Op::Sub, da.max(db)),
                    _ => (Op::Mul, da + db),
                };
                Ok((self.emit(op, a, b), deg, offs_a))
            }
            "neg" => {
                if e.values.len() != 1 {
                    bail!("neg with {} children", e.values.len());
                }
                let (a, d, offs) = self.lower_child(&e.values[0])?;
                Ok((self.emit(Op::Neg, a, a), d, offs))
            }
            other => bail!("unsupported operand '{other}' (multilinear milestone 2)"),
        }
    }
}

/// The single-expression value of a hint field.
fn hint_field<'h>(hint: &'h HintInfo, name: &str) -> Result<&'h Expression> {
    let f = hint
        .fields
        .iter()
        .find(|f| f.name == name)
        .ok_or_else(|| anyhow!("hint '{}' missing field '{name}'", hint.name))?;
    match f.values.as_slice() {
        [HintFieldValue::Single(e)] => Ok(e),
        _ => bail!("hint '{}' field '{name}' is not a single expression", hint.name),
    }
}

/// Lower the LogUp-GKR bus hints (`@gkr_sum_bus` marker, one `@gkr_sum_term`
/// per row fraction, one `@gkr_sum_direct_term` per instance-level fraction)
/// emitted by the STD's GKR mode into a [`BusIr`]. Returns `None` when the
/// AIR has no GKR bus (plain-logup or bus-free AIRs).
fn lower_bus(compiler: &mut Compiler) -> Result<Option<BusIr>> {
    let hints = &compiler.setup.hints;
    let marker = {
        let markers: Vec<&HintInfo> = hints.iter().filter(|h| h.name == "gkr_sum_bus").collect();
        match markers.as_slice() {
            [] => return Ok(None),
            [one] => *one,
            _ => bail!("multiple @gkr_sum_bus hints in one AIR"),
        }
    };

    let mut terms = Vec::new();
    let mut max_term_degree = 1u32;
    for hint in hints.iter().filter(|h| h.name == "gkr_sum_term") {
        let (num, deg_num, _) = compiler.lower_expr(hint_field(hint, "numerator")?)?;
        let (den, deg_den, _) = compiler.lower_expr(hint_field(hint, "denominator")?)?;
        let degree = deg_num.max(deg_den);
        max_term_degree = max_term_degree.max(degree);
        terms.push(BusTerm { num, den, degree });
    }
    if terms.is_empty() {
        bail!("@gkr_sum_bus without @gkr_sum_term hints");
    }

    let mut scalar_terms = Vec::new();
    for hint in hints.iter().filter(|h| h.name == "gkr_sum_direct_term") {
        let (num, deg_num, offs_num) = compiler.lower_expr(hint_field(hint, "numerator")?)?;
        let (den, deg_den, offs_den) = compiler.lower_expr(hint_field(hint, "denominator")?)?;
        if deg_num.max(deg_den) > 0 || !offs_num.is_empty() || !offs_den.is_empty() {
            bail!("@gkr_sum_direct_term references columns (scalar terms must be scalar-only)");
        }
        scalar_terms.push(BusTerm { num, den, degree: 0 });
    }

    // `result` is the `gsum_result` airgroup value, or the literal 0 in
    // one-instance mode (where the instance's contribution must vanish).
    let (result_op, _, _) = compiler.lower_expr(hint_field(marker, "result")?)?;
    let result_airgroupvalue = match result_op.kind {
        SrcKind::AirGroupValue => Some(result_op.idx),
        SrcKind::Number if compiler.numbers[result_op.idx as usize] == 0 => None,
        _ => bail!("@gkr_sum_bus result must be an airgroup value or the literal 0"),
    };

    Ok(Some(BusIr { terms, scalar_terms, result_airgroupvalue, max_term_degree }))
}

/// Compile a `SetupResult` into the multilinear prover's `AirIr`.
pub fn build_air_ir(setup: &SetupResult, n_bits: u32, params: MlParams) -> Result<AirIr> {
    let n_stages = setup.n_stages;
    let mut cols_per_stage = Vec::with_capacity(n_stages);
    for stage in 1..=n_stages {
        let width = *setup
            .map_sections_n
            .get(&format!("cm{stage}"))
            .ok_or_else(|| anyhow!("missing cm{stage} section width"))?;
        cols_per_stage.push(width as u32);
    }
    // GKR-mode AIRs declare stage-2 challenges but commit no stage-2 columns;
    // drop trailing empty stages so the prover neither expects nor commits
    // empty matrices.
    while cols_per_stage.last() == Some(&0) {
        cols_per_stage.pop();
    }
    if cols_per_stage.is_empty() {
        bail!("AIR commits no witness columns");
    }

    let mut compiler = Compiler {
        setup,
        exprs: &setup.expressions,
        instrs: Vec::new(),
        numbers: Vec::new(),
        memo: HashMap::new(),
        offsets: BTreeSet::from([0]),
    };

    let mut constraints = Vec::with_capacity(setup.constraints.len());
    let mut max_degree = 1u32;
    for (idx, con) in setup.constraints.iter().enumerate() {
        let boundary = match con.boundary.as_str() {
            "everyRow" => Boundary::EveryRow,
            "firstRow" => Boundary::FirstRow,
            "lastRow" => Boundary::LastRow,
            other => bail!("constraint {idx}: unsupported boundary '{other}'"),
        };
        let (root, degree, offsets) = compiler.lower_id(con.e)?;
        if boundary != Boundary::EveryRow && offsets.iter().any(|&s| s != 0) {
            bail!("constraint {idx}: boundary constraints must not reference shifted columns");
        }
        if boundary == Boundary::EveryRow {
            max_degree = max_degree.max(degree);
        }
        constraints.push(ConstraintIr { boundary, root, degree });
    }

    // Full global challenge map: the multilinear protocol derives only the
    // challenges of stages 2..=n_stages (the later ones belong to the
    // univariate quotient/evals/FRI machinery and stay zero).
    let challenge_stages: Vec<u8> = setup.challenges_map.iter().map(|c| c.stage.unwrap_or(0) as u8).collect();

    // Custom commits: stage-0 (fixed) only.
    let mut custom_commits_ir = Vec::with_capacity(setup.custom_commits.len());
    for cc in &setup.custom_commits {
        if cc.stage_widths.iter().skip(1).any(|&w| w > 0) {
            bail!("custom commit '{}' has columns beyond stage 0 (unsupported)", cc.name);
        }
        custom_commits_ir
            .push(MlCustomCommit { name: cc.name.clone(), n_cols: cc.stage_widths.first().copied().unwrap_or(0) });
    }
    let airvalue_stages: Vec<u8> = setup.air_values_map.iter().map(|c| c.stage.unwrap_or(0) as u8).collect();
    let airgroupvalue_stages: Vec<u8> = setup.airgroup_values_map.iter().map(|c| c.stage.unwrap_or(0) as u8).collect();
    let proofvalue_stages: Vec<u8> = setup.proof_values_map.iter().map(|c| c.stage.unwrap_or(0) as u8).collect();

    // LogUp-GKR bus (present iff the STD compiled the AIR in GKR mode).
    let bus = lower_bus(&mut compiler)?;

    Ok(AirIr {
        name: setup.name.clone(),
        airgroup_id: setup.airgroup_id as u32,
        air_id: setup.air_id as u32,
        n_bits,
        cols_per_stage,
        n_const_cols: setup.n_constants as u32,
        custom_commits: custom_commits_ir,
        n_publics: setup.n_publics as u32,
        challenge_stages,
        airvalue_stages,
        airgroupvalue_stages,
        proofvalue_stages,
        numbers: compiler.numbers,
        n_temps: compiler.instrs.len() as u32,
        instrs: compiler.instrs,
        constraints,
        max_constraint_degree: max_degree,
        opening_offsets: compiler.offsets.into_iter().collect(),
        bus,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pil::prepare::PrepareOptions;
    use crate::types::stark_struct::{generate_stark_struct, StarkSettings};
    use pilout::pilout as pb;
    use prost::Message;
    use proofman_multilinear::SrcKind;

    /// Load a checked-in pilout and compile one AIR into its `AirIr`.
    /// Returns `None` (test skipped) when the fixture is absent.
    fn compile_fixture_air(path: &str, air_name: &str) -> Option<AirIr> {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: {path} not found");
                return None;
            }
        };
        let pilout = pb::PilOut::decode(data.as_slice()).expect("decode pilout");

        let (ag_idx, air_idx, n_bits) = pilout
            .air_groups
            .iter()
            .enumerate()
            .find_map(|(g, ag)| {
                ag.airs.iter().enumerate().find_map(|(a, air)| {
                    (air.name.as_deref() == Some(air_name))
                        .then(|| (g, a, (air.num_rows.unwrap() as usize).trailing_zeros()))
                })
            })
            .unwrap_or_else(|| panic!("{air_name} in pilout"));

        let stark_struct = generate_stark_struct(&StarkSettings::default(), n_bits as usize);
        let opts = PrepareOptions { debug: false, im_pols_stages: false };
        let pil_result = crate::pil::info::pil_info(&pilout, ag_idx, air_idx, &stark_struct, &opts);

        Some(build_air_ir(&pil_result.setup, n_bits, MlParams::default()).expect("build AirIr"))
    }

    /// Compile SimpleLeft (std lookups: stage-2 gsum + im-pols, challenges,
    /// airgroup values) from the checked-in pilout and pin the AirIr shape.
    #[test]
    fn simple_left_compiles_to_multistage_ir() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pil2-components/test/simple/build/simple.pilout");
        let Some(ir) = compile_fixture_air(path, "SimpleLeft") else { return };

        // Two stages: 16 base columns in stage 1, 21 (7 ext columns) in stage 2.
        assert_eq!(ir.cols_per_stage, vec![16, 21]);
        // std_alpha/std_gamma (stage 2) + the univariate-only vc (3) / xi (4);
        // the FRI folding challenges are not part of the setup challenge map.
        assert_eq!(ir.challenge_stages, vec![2, 2, 3, 4]);
        assert_eq!(ir.airgroupvalue_stages, vec![2]);
        assert!(ir.airvalue_stages.is_empty());

        // Constraints must reference ext-valued stage-2 columns and challenges.
        let has_ext_witness = ir
            .instrs
            .iter()
            .any(|i| [i.a, i.b].iter().any(|o| matches!(o.kind, SrcKind::Witness { stage: 2 }) && o.dim == 3));
        let has_challenge = ir.instrs.iter().any(|i| [i.a, i.b].iter().any(|o| o.kind == SrcKind::Challenge));
        let has_agv = ir.instrs.iter().any(|i| [i.a, i.b].iter().any(|o| o.kind == SrcKind::AirGroupValue));
        assert!(has_ext_witness, "expected dim-3 stage-2 witness operands");
        assert!(has_challenge, "expected challenge operands");
        assert!(has_agv, "expected airgroup-value operands");

        // Base-slot layout: stage-2 operands must address slots 0..21 in steps
        // compatible with dim 3 and never run past the section width.
        for instr in &ir.instrs {
            for o in [&instr.a, &instr.b] {
                if let SrcKind::Witness { stage } = o.kind {
                    let width = ir.cols_per_stage[stage as usize - 1];
                    assert!(o.idx + o.dim as u32 <= width, "operand past section: {o:?}");
                }
            }
        }
    }

    /// The same system compiled with the LogUp-GKR sum bus
    /// (`simple_gkr.pilout`): single stage, no gsum/im columns or bus
    /// constraints, and the `@gkr_*` hints lowered into a `BusIr`.
    #[test]
    fn simple_left_gkr_compiles_to_single_stage_bus_ir() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pil2-components/test/simple/build/simple_gkr.pilout");
        let Some(ir) = compile_fixture_air(path, "SimpleLeft") else { return };

        // Same 16 stage-1 columns as the plain system, no stage 2.
        assert_eq!(ir.cols_per_stage, vec![16], "GKR mode must commit no stage-2 columns");
        // std_alpha/std_gamma still declared at stage 2.
        assert!(ir.challenge_stages.contains(&2), "bus challenges must survive");
        // gsum_result is still an airgroup value.
        assert_eq!(ir.airgroupvalue_stages, vec![2]);

        let bus = ir.bus.as_ref().expect("GKR mode must lower a BusIr");
        // SimpleLeft feeds the bus with 3 permutation terms, 1 lookup term and
        // 7 range checks — at least 11 row terms, exactly one result value.
        assert!(bus.terms.len() >= 11, "expected >= 11 bus terms, got {}", bus.terms.len());
        assert!(bus.max_term_degree >= 1);
        assert!(bus.result_airgroupvalue.is_some(), "gsum_result index must be resolved");

        // No operand anywhere may reference a stage-2 witness column.
        for instr in &ir.instrs {
            for o in [&instr.a, &instr.b] {
                assert!(!matches!(o.kind, SrcKind::Witness { stage: 2 }), "stage-2 witness operand in GKR mode: {o:?}");
            }
        }

        // Every denominator must involve the bus challenge γ somewhere in its
        // cone (it is `compressed_exprs + std_gamma`): check at least one
        // challenge operand exists in the instruction list.
        assert!(
            ir.instrs.iter().any(|i| [i.a, i.b].iter().any(|o| o.kind == SrcKind::Challenge)),
            "expected challenge operands in the bus term cones"
        );

        // The plain and GKR systems must agree on the stage-1 layout.
        let plain_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pil2-components/test/simple/build/simple.pilout");
        if let Some(plain) = compile_fixture_air(plain_path, "SimpleLeft") {
            assert_eq!(plain.cols_per_stage[0], ir.cols_per_stage[0]);
            assert!(plain.bus.is_none(), "plain mode must not carry a bus");
        }
    }
}
