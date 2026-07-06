//! Compilation of the pilout constraint expressions into the multilinear
//! prover's IR (`<AIR>.mlinfo.bin`, a bincode-serialized
//! [`proofman_multilinear::AirIr`]).
//!
//! The setup expression arena is lowered into a flat instruction list; arena
//! references (`op == "exp"`) are memoized so shared subexpressions become
//! shared temps. AIRs using features the multilinear prover does not support
//! yet (air values, proof values, custom commits, `everyFrame` constraints,
//! shifted columns inside boundary constraints) are rejected with a
//! descriptive error — the setup then simply skips the mlinfo artifact.

use std::collections::{BTreeSet, HashMap};

use anyhow::{anyhow, bail, Result};
use fields::{Goldilocks, PrimeField64};
use proofman_multilinear::{AirIr, Boundary, ConstraintIr, Instr, MlCustomCommit, MlParams, Op, Operand, SrcKind};

use crate::expr::expression::{ExprChild, Expression};
use crate::types::pilout_info::SetupResult;

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
        numbers: compiler.numbers,
        n_temps: compiler.instrs.len() as u32,
        instrs: compiler.instrs,
        constraints,
        max_constraint_degree: max_degree,
        opening_offsets: compiler.offsets.into_iter().collect(),
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

    /// Compile SimpleLeft (std lookups: stage-2 gsum + im-pols, challenges,
    /// airgroup values) from the checked-in pilout and pin the AirIr shape.
    #[test]
    fn simple_left_compiles_to_multistage_ir() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pil2-components/test/simple/build/simple.pilout");
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping: {path} not found");
                return;
            }
        };
        let pilout = pb::PilOut::decode(data.as_slice()).expect("decode pilout");

        // Locate SimpleLeft.
        let (ag_idx, air_idx, n_bits) = pilout
            .air_groups
            .iter()
            .enumerate()
            .find_map(|(g, ag)| {
                ag.airs.iter().enumerate().find_map(|(a, air)| {
                    (air.name.as_deref() == Some("SimpleLeft"))
                        .then(|| (g, a, (air.num_rows.unwrap() as usize).trailing_zeros()))
                })
            })
            .expect("SimpleLeft in pilout");

        let stark_struct = generate_stark_struct(&StarkSettings::default(), n_bits as usize);
        let opts = PrepareOptions { debug: false, im_pols_stages: false };
        let pil_result = crate::pil::info::pil_info(&pilout, ag_idx, air_idx, &stark_struct, &opts);

        let ir = build_air_ir(&pil_result.setup, n_bits, MlParams::default()).expect("build AirIr");

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
}
