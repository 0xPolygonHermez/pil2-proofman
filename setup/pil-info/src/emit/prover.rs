//! Typed IR -> today's expression arena -> the existing prover code generator.

use std::cell::RefCell;

use pil2_pilout::pilout::{self as pb, expression as px, global_operand, hint_field, operand as po};

use crate::cfg::FieldCfg;
use crate::core::fold::fold_roots;
use crate::core::from_pilout::stage_base;
use crate::core::im::{im_column_id, Checked, ImChoice};
use crate::core::ir::{Boundary, ColType, ConstraintSystem, Expr, HintField, HintValue, NamedValue};
use crate::core::layout::{Layout, PolMap};
use crate::core::openings::{EvEntry, Openings};
use crate::error::{PilInfoError, Result};
use crate::expr::expression::{ExprChild, Expression};
use crate::expr::helpers::{add_info_expression_inline, add_info_expressions, get_exp_dim};
use crate::expr::print::{print_bounded, PrintCtx};
use crate::pil::gen_code::{generate_pil_code, CodeGenParams, PilCodeResult};
use crate::types::pilout_info::{format_expressions, format_hints, ConstraintInfo, HintInfo, SymbolInfo};

/// Arena ids of the roles; until Task 9 they equal today's ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Roles {
    /// Q = fold_roots of the substituted roots.
    pub c_exp: usize,
}

/// Today's arena and roles, with the formatted hints and constraints the generator reads; built once.
pub struct ProverExprs {
    pub arena: Vec<Expression>,
    pub roles: Roles,
    pub hints: Vec<HintInfo>,
    pub constraints: Vec<ConstraintInfo>,
}

/// What the generator needs that the typed IR does not hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmitParams {
    pub air_id: usize,
    pub airgroup_id: usize,
}

/// Today's push order: pilout expressions (old formatter), the cExp fold, the im roots, then Q.
pub fn to_arena(cs: &ConstraintSystem, im: &ImChoice, lay: &Layout) -> Result<ProverExprs> {
    let (arena, hints, constraints) = formatted(cs)?;
    fold_quotient(cs, im, lay, arena, hints, constraints)
}

/// The generator's output plus what the AIR summary counts.
pub struct ProverCode {
    pub code: PilCodeResult,
    /// The im expressions printed, (base field, extension), in column order.
    pub im_pols_info: (Vec<String>, Vec<String>),
    /// Stage of each constraint, im roots included.
    pub constraint_stages: Vec<usize>,
    pub n_expressions: usize,
}

/// Expressionsinfo, hints and qVerifier through the existing generator, from `to_arena`'s output.
pub fn emit_prover(
    cs: &ConstraintSystem,
    im: &ImChoice,
    lay: &Layout,
    ops: &Openings,
    chk: &Checked,
    exprs: ProverExprs,
    p: &EmitParams,
) -> Result<ProverCode> {
    let ProverExprs { mut arena, roles, hints, mut constraints } = exprs;
    let maps = Maps::of(cs, im, lay, p);
    let ctx = maps.print_ctx();
    // the old map's order: im-root lines, then the im expressions; printing caches lines in the arena
    for c in constraints.iter_mut().filter(|c| c.im_pol) {
        c.line = Some(print_bounded(&ctx, &mut arena, c.e, true, "im_pol constraint expression"));
    }
    constraints.iter_mut().for_each(|c| c.line.get_or_insert_with(String::new).push_str(" == 0"));
    let mut im_pols_info = (Vec::new(), Vec::new());
    for s in maps.cm.iter().filter(|s| s.im_pol) {
        let line = print_bounded(&ctx, &mut arena, s.exp_id.ok_or_else(fold_bug)?, false, "im_pol expression");
        if s.dim == 1 { &mut im_pols_info.0 } else { &mut im_pols_info.1 }.push(line);
    }
    let params = CodeGenParams {
        air_id: p.air_id,
        airgroup_id: p.airgroup_id,
        n_stages: cs.n_stages as usize,
        c_exp_id: roles.c_exp,
        q_deg: chk.q_deg as usize,
        q_dim: lay.q_dim as usize,
        opening_points: ops.opening_points.clone(),
        cm_pols_map: maps.cm.clone(),
        custom_commits_count: cs.custom_commits.len(),
        field: cs.field.clone(),
        opening: cs.opening,
    };
    let code = generate_pil_code(&params, &maps.symbols(), &constraints, &mut arena, &hints, Some(&ctx))?;
    same_ev_map(&code, ops)?;
    let constraint_stages = constraints.iter().map(|c| c.stage.unwrap_or(0)).collect();
    Ok(ProverCode { code, im_pols_info, constraint_stages, n_expressions: arena.len() })
}

/// TEMPORARY (piece 3): the generator's evMap indexes the qVerifier, so it must be the core's.
fn same_ev_map(code: &PilCodeResult, ops: &Openings) -> Result<()> {
    let ev = |e: &EvEntry| {
        let t = match e.pol.col {
            ColType::Witness => "cm",
            ColType::Fixed => "const",
            ColType::Custom => "custom",
        };
        (t.to_string(), e.pol.id as usize, e.pol.offset, e.opening_pos as usize, e.pol.commit_id.map(|c| c as usize))
    };
    let old_ev = code.ev_map.iter().map(|e| (e.entry_type.clone(), e.id, e.prime, e.opening_pos, e.commit_id));
    if !ops.ev_map.iter().map(ev).eq(old_ev) {
        return Err(PilInfoError::InvalidPilout("the generator's evMap differs from the core's".into()));
    }
    Ok(())
}

type Formatted = (Vec<Expression>, Vec<HintInfo>, Vec<ConstraintInfo>);

/// The pilout expressions and hints through the old formatters, then `prepare_pil`'s info passes.
fn formatted(cs: &ConstraintSystem) -> Result<Formatted> {
    let field = &cs.field;
    let t = Tables::of(cs)?;
    let mut arena = format_expressions(
        &t.exprs,
        &t.stage_widths,
        &t.num_challenges,
        &t.air_values,
        &t.airgroup_values,
        &t.custom_commits,
        field,
    )?;
    let raw = (cs.hints.iter())
        .map(|h| {
            Ok(pb::Hint { name: h.name.clone(), hint_fields: t.hint_fields(cs, &h.fields)?, ..Default::default() })
        })
        .collect::<Result<Vec<_>>>()?;
    let hints = format_hints(
        &raw,
        &t.exprs,
        &t.stage_widths,
        &t.num_challenges,
        &t.air_values,
        &t.airgroup_values,
        &t.custom_commits,
        &mut arena,
        field,
    )?;

    // prepare_pil: stage 1, info on the constraints, then on every expression
    arena.iter_mut().for_each(|e| e.stage = 1);
    let mut constraints = cs.constraints.iter().map(constraint_info).collect::<Result<Vec<_>>>()?;
    for c in constraints.iter_mut() {
        add_info_expressions(&mut arena, c.e, field)?;
        c.stage = Some(arena[c.e].stage);
    }
    for i in 0..arena.len() {
        add_info_expressions(&mut arena, i, field)?;
    }
    Ok((arena, hints, constraints))
}

/// The cExp fold of the constraint and im roots, Q = cExp * Zi, and the im-root constraints.
fn fold_quotient(
    cs: &ConstraintSystem,
    im: &ImChoice,
    lay: &Layout,
    arena: Vec<Expression>,
    hints: Vec<HintInfo>,
    mut constraints: Vec<ConstraintInfo>,
) -> Result<ProverExprs> {
    let field = &cs.field;
    let st = RefCell::new(FoldState {
        arena,
        boundaries: vec![Boundary::EveryRow],
        im_roots: Vec::new(),
        q_stage: cs.n_stages as usize + 1,
        vc_id: lay.vc_index(),
        field,
    });
    let mut roots: Vec<Root> = (0..cs.constraints.len()).map(Root::Constraint).collect();
    if roots.is_empty() {
        roots.push(Root::Zero);
    }
    roots.extend((0..im.exprs.len()).map(Root::Im));
    let ctx = Ctx { cs, im, lay, constraints: &constraints };
    let roots: Vec<Result<Node>> = roots.into_iter().map(|r| Ok(Node::Root(r))).collect();
    let folded = fold_roots(
        roots,
        |a| a.and_then(|a| Ok(Node::Scaled(st.borrow_mut().materialise(&ctx, a)?))),
        |a, b| st.borrow_mut().add(&ctx, a?, b?),
    );
    let mut st = st.into_inner();
    let folded = folded.ok_or_else(|| PilInfoError::InvalidPilout("no quotient root".into()))??;
    let c_exp = st.materialise(&ctx, folded)?;

    // Q = cExp * Zi(everyRow)
    let zi =
        Expression { op: "Zi".into(), boundary_id: Some(0), boundary: Some("everyRow".into()), ..Default::default() };
    let mut q = Expression {
        op: "mul".into(),
        values: vec![ExprChild::Inline(Box::new(st.arena[c_exp].clone())), ExprChild::Inline(Box::new(zi))],
        ..Default::default()
    };
    add_info_expression_inline(&mut st.arena, &mut q, field)?;
    let c_exp = st.push(q);
    st.arena[c_exp].dim = get_exp_dim(&st.arena, c_exp, field)?;

    for (e, stage) in st.im_roots {
        constraints.push(ConstraintInfo {
            e,
            boundary: "everyRow".into(),
            line: None,
            stage: Some(stage),
            offset_min: None,
            offset_max: None,
            im_pol: true,
        });
    }
    Ok(ProverExprs { arena: st.arena, roles: Roles { c_exp }, hints, constraints })
}

fn constraint_info(c: &crate::core::ir::Constraint) -> Result<ConstraintInfo> {
    let (offset_min, offset_max) = match c.boundary {
        Boundary::EveryFrame { min, max } => (Some(min), Some(max)),
        _ => (None, None),
    };
    Ok(ConstraintInfo {
        boundary: c.boundary.name().into(),
        e: ref_id(&c.e)?,
        line: Some(c.label.clone()),
        offset_min,
        offset_max,
        stage: None,
        im_pol: false,
    })
}

fn ref_id(e: &Expr) -> Result<usize> {
    match e {
        Expr::Ref(id) => Ok(*id as usize),
        _ => Err(PilInfoError::InvalidPilout("a constraint that is not an expression reference".into())),
    }
}

#[derive(Clone, Copy)]
enum Root {
    Constraint(usize),
    Im(usize),
    /// Today's placeholder cExp of an air without constraints.
    Zero,
}

/// A fold operand: a root not yet pushed, an arena id, or `vc * id` awaiting its sum.
enum Node {
    Root(Root),
    Id(usize),
    Scaled(usize),
}

struct Ctx<'a> {
    cs: &'a ConstraintSystem,
    im: &'a ImChoice,
    lay: &'a Layout,
    constraints: &'a [ConstraintInfo],
}

struct FoldState<'a> {
    arena: Vec<Expression>,
    boundaries: Vec<Boundary>,
    /// (sub id, stage) per im root, in fold order.
    im_roots: Vec<(usize, usize)>,
    q_stage: usize,
    vc_id: usize,
    field: &'a FieldCfg,
}

impl FoldState<'_> {
    fn push(&mut self, e: Expression) -> usize {
        self.arena.push(e);
        self.arena.len() - 1
    }

    fn exp(&self, id: usize) -> ExprChild {
        let e = Expression {
            op: "exp".into(),
            id: Some(id),
            row_offset: Some(0),
            stage: self.q_stage,
            ..Default::default()
        };
        ExprChild::Inline(Box::new(e))
    }

    /// `l op r` over two arena ids, info computed before the push.
    fn push_bin(&mut self, op: &str, l: ExprChild, r: ExprChild) -> Result<usize> {
        let mut e = Expression { op: op.into(), values: vec![l, r], ..Default::default() };
        add_info_expression_inline(&mut self.arena, &mut e, self.field)?;
        Ok(self.push(e))
    }

    /// Today's order: the root first, then `vc * acc`, then the sum.
    fn add(&mut self, ctx: &Ctx, a: Node, b: Node) -> Result<Node> {
        let Node::Scaled(acc) = a else { return Err(fold_bug()) };
        let Node::Root(root) = b else { return Err(fold_bug()) };
        let c = self.root(ctx, root)?;
        // the cExp builder names std_vc, the im builder does not
        let value = matches!(root, Root::Constraint(_)).then(|| "std_vc".to_string());
        let vc = Expression {
            op: "challenge".into(),
            stage: self.q_stage,
            dim: self.field.ext_dim(),
            stage_id: Some(0),
            id: Some(self.vc_id),
            value,
            ..Default::default()
        };
        let w = self.push_bin("mul", ExprChild::Inline(Box::new(vc)), self.exp(acc))?;
        Ok(Node::Id(self.push_bin("add", self.exp(w), self.exp(c))?))
    }

    fn materialise(&mut self, ctx: &Ctx, n: Node) -> Result<usize> {
        match n {
            Node::Root(r) => self.root(ctx, r),
            Node::Id(id) => Ok(id),
            Node::Scaled(_) => Err(fold_bug()),
        }
    }

    fn root(&mut self, ctx: &Ctx, r: Root) -> Result<usize> {
        match r {
            Root::Zero => {
                let e = Expression { op: "number".into(), value: Some("0".into()), dim: 1, ..Default::default() };
                Ok(self.push(e))
            }
            Root::Constraint(k) => {
                let e = ctx.constraints[k].e;
                let b = &ctx.cs.constraints[k].boundary;
                if *b == Boundary::EveryRow {
                    return Ok(e);
                }
                let boundary_id = match self.boundaries.iter().position(|x| x == b) {
                    Some(i) => i,
                    None => {
                        self.boundaries.push(b.clone());
                        self.boundaries.len() - 1
                    }
                };
                let zi = Expression {
                    op: "Zi".into(),
                    boundary_id: Some(boundary_id),
                    boundary: Some(ctx.constraints[k].boundary.clone()),
                    ..Default::default()
                };
                let mut ee = Expression { op: "exp".into(), id: Some(e), row_offset: Some(0), ..Default::default() };
                ee.stage = self.q_stage;
                let m = Expression {
                    op: "mul".into(),
                    values: vec![ExprChild::Inline(Box::new(ee)), ExprChild::Inline(Box::new(zi))],
                    ..Default::default()
                };
                Ok(self.push(m))
            }
            Root::Im(k) => {
                let exp_id = ctx.im.exprs[k] as usize;
                let pol_id = im_column_id(ctx.cs, k) as usize;
                let col = ctx.lay.cm_pols_map.get(pol_id).ok_or_else(fold_bug)?;
                let stage = col.stage as usize;
                let target = self.arena.get_mut(exp_id).ok_or_else(fold_bug)?;
                (target.im_pol, target.pol_id, target.stage) = (true, Some(pol_id), stage);
                let cm = Expression {
                    op: "cm".into(),
                    id: Some(pol_id),
                    row_offset: Some(0),
                    stage,
                    dim: col.dim as usize,
                    ..Default::default()
                };
                let copy = self.arena[exp_id].clone();
                let sub = self.push_bin("sub", ExprChild::Inline(Box::new(cm)), ExprChild::Inline(Box::new(copy)))?;
                self.im_roots.push((sub, stage));
                Ok(sub)
            }
        }
    }
}

fn fold_bug() -> PilInfoError {
    PilInfoError::InvalidPilout("the quotient fold met an unexpected operand".into())
}

/// The old formatter's inputs, rebuilt from the typed IR.
struct Tables {
    exprs: Vec<pb::Expression>,
    stage_widths: Vec<u32>,
    num_challenges: Vec<u32>,
    air_values: Vec<pb::AirValue>,
    airgroup_values: Vec<pb::AirGroupValue>,
    custom_commits: Vec<pb::CustomCommit>,
    /// Stage of each witness column, by id.
    witness_stage: Vec<Option<u32>>,
}

impl Tables {
    fn of(cs: &ConstraintSystem) -> Result<Self> {
        let per_stage = |stages: &mut dyn Iterator<Item = u32>| {
            let mut v = vec![0u32; cs.n_stages as usize];
            stages.for_each(|s| {
                if let Some(w) = (s as usize).checked_sub(1).and_then(|i| v.get_mut(i)) {
                    *w += 1;
                }
            });
            v
        };
        if cs.airgroup_values.len() != cs.airgroup_agg_types.len() {
            return Err(PilInfoError::InvalidPilout("airgroup values and their aggTypes differ in number".into()));
        }
        let mut witness_stage = vec![None; cs.witness.len()];
        for c in &cs.witness {
            *witness_stage.get_mut(c.id as usize).ok_or_else(not_pilout)? = Some(c.stage);
        }
        let mut t = Tables {
            exprs: Vec::with_capacity(cs.exprs.len()),
            stage_widths: per_stage(&mut cs.witness.iter().map(|c| c.stage)),
            num_challenges: per_stage(&mut cs.challenges.iter().map(|c| c.0)),
            air_values: cs.air_values.iter().map(|v| pb::AirValue { stage: v.stage }).collect(),
            airgroup_values: cs
                .airgroup_values
                .iter()
                .zip(&cs.airgroup_agg_types)
                .map(|(v, a)| pb::AirGroupValue { agg_type: *a, stage: v.stage })
                .collect(),
            custom_commits: cs
                .custom_commits
                .iter()
                .map(|c| pb::CustomCommit {
                    name: Some(c.name.clone()),
                    public_values: c.public_values.iter().map(|&idx| global_operand::PublicValue { idx }).collect(),
                    stage_widths: c.stage_widths.clone(),
                })
                .collect(),
            witness_stage,
        };
        for e in &cs.exprs {
            let ex = t.expression(cs, e)?;
            t.exprs.push(ex);
        }
        Ok(t)
    }

    /// The typed hint fields back in the pilout form the old formatter reads. TODO(piece 3): type format_hints.
    fn hint_fields(&self, cs: &ConstraintSystem, fields: &[HintField]) -> Result<Vec<pb::HintField>> {
        let field = |f: &HintField| {
            let value = match &f.value {
                HintValue::Str(s) => hint_field::Value::StringValue(s.clone()),
                HintValue::Expr(e) => hint_field::Value::Operand(pb::Operand { operand: Some(self.operand(cs, e)?) }),
                HintValue::Array(a) => {
                    hint_field::Value::HintFieldArray(pb::HintFieldArray { hint_fields: self.hint_fields(cs, a)? })
                }
            };
            Ok(pb::HintField { name: f.name.clone(), value: Some(value) })
        };
        fields.iter().map(field).collect()
    }

    fn expression(&self, cs: &ConstraintSystem, e: &Expr) -> Result<pb::Expression> {
        let o = |x: &Expr| self.operand(cs, x).map(|o| Some(pb::Operand { operand: Some(o) }));
        let op = match e {
            Expr::Add(a, b) => px::Operation::Add(px::Add { lhs: o(a)?, rhs: o(b)? }),
            Expr::Sub(a, b) => px::Operation::Sub(px::Sub { lhs: o(a)?, rhs: o(b)? }),
            Expr::Mul(a, b) => px::Operation::Mul(px::Mul { lhs: o(a)?, rhs: o(b)? }),
            Expr::Neg(a) => px::Operation::Neg(px::Neg { value: o(a)? }),
            _ => return Err(not_pilout()),
        };
        Ok(pb::Expression { operation: Some(op) })
    }

    fn operand(&self, cs: &ConstraintSystem, e: &Expr) -> Result<po::Operand> {
        Ok(match e {
            Expr::Ref(idx) => po::Operand::Expression(po::Expression { idx: *idx }),
            Expr::Const(c) => po::Operand::Constant(po::Constant { value: c.to_bytes_be() }),
            Expr::Col { col: ColType::Witness, id, offset, .. } => {
                let stage = self.witness_stage.get(*id as usize).copied().flatten().ok_or_else(not_pilout)?;
                let base =
                    stage_base(&self.stage_widths, stage).map_err(|e| PilInfoError::InvalidPilout(e.to_string()))?;
                let col_idx = id.checked_sub(base).ok_or_else(not_pilout)?;
                po::Operand::WitnessCol(po::WitnessCol { stage, col_idx, row_offset: *offset as i32 })
            }
            Expr::Col { col: ColType::Fixed, id, offset, .. } => {
                po::Operand::FixedCol(po::FixedCol { idx: *id, row_offset: *offset as i32 })
            }
            Expr::Col { col: ColType::Custom, commit_id, id, offset } => po::Operand::CustomCol(po::CustomCol {
                commit_id: commit_id.ok_or_else(not_pilout)?,
                stage: 0,
                col_idx: *id,
                row_offset: *offset as i32,
            }),
            Expr::Challenge { stage, stage_id } => {
                po::Operand::Challenge(po::Challenge { stage: *stage, idx: *stage_id })
            }
            Expr::Public(idx) => po::Operand::PublicValue(po::PublicValue { idx: *idx }),
            Expr::ProofValue(idx) => {
                let stage = cs.proof_values.get(*idx as usize).ok_or_else(not_pilout)?.stage;
                po::Operand::ProofValue(po::ProofValue { stage, idx: *idx })
            }
            Expr::AirValue(idx) => po::Operand::AirValue(po::AirValue { idx: *idx }),
            Expr::AirgroupValue { airgroup: None, id } => po::Operand::AirGroupValue(po::AirGroupValue { idx: *id }),
            _ => return Err(not_pilout()),
        })
    }
}

fn not_pilout() -> PilInfoError {
    PilInfoError::InvalidPilout("an expression the pilout format cannot hold".into())
}

/// The generator's symbol maps, projected from the layout.
struct Maps {
    cm: Vec<SymbolInfo>,
    fixed: Vec<SymbolInfo>,
    custom: Vec<Vec<SymbolInfo>>,
    challenges: Vec<SymbolInfo>,
    publics: Vec<SymbolInfo>,
    proof_values: Vec<SymbolInfo>,
    air_values: Vec<SymbolInfo>,
    airgroup_values: Vec<SymbolInfo>,
}

impl Maps {
    fn of(cs: &ConstraintSystem, im: &ImChoice, lay: &Layout, p: &EmitParams) -> Self {
        let ext = cs.field.ext_dim();
        let lengths = |l: &Option<Vec<u32>>| l.as_ref().map(|v| v.iter().map(|x| *x as usize).collect());
        let pol = |kind: &str, i: usize, c: &PolMap| SymbolInfo {
            name: c.name.clone(),
            sym_type: kind.into(),
            stage: Some(c.stage as usize),
            dim: c.dim as usize,
            id: None,
            pol_id: Some(i),
            stage_id: Some(c.stage_id as usize),
            air_id: Some(p.air_id),
            airgroup_id: Some(p.airgroup_id),
            commit_id: c.commit_id.map(|x| x as usize),
            lengths: lengths(&c.lengths),
            idx: None,
            stage_pos: Some(c.stage_pos as usize),
            im_pol: c.im_pol,
            exp_id: None,
        };
        let value = |kind: &str, i: usize, v: &NamedValue| SymbolInfo {
            name: v.name.clone(),
            sym_type: kind.into(),
            stage: Some(v.stage as usize),
            dim: if v.stage == 1 { 1 } else { ext },
            id: Some(i),
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: lengths(&v.lengths),
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        let n_witness = cs.witness.len();
        let cm = lay.cm_pols_map.iter().enumerate().map(|(i, c)| {
            let mut s = pol("witness", i, c);
            if let Some(e) = i.checked_sub(n_witness).and_then(|k| im.exprs.get(k)).filter(|_| c.im_pol) {
                (s.id, s.exp_id) = (Some(*e as usize), Some(*e as usize));
            }
            s
        });
        let values = |kind: &str, m: &[NamedValue]| m.iter().enumerate().map(|(i, v)| value(kind, i, v)).collect();
        // the DEEP challenges are not the generator's
        let q = cs.n_stages + 2;
        let challenges = lay.challenges_map.iter().enumerate().filter(|(_, c)| c.stage <= q).map(|(i, c)| {
            let mut s = value("challenge", i, &NamedValue { name: c.name.clone(), stage: c.stage, lengths: None });
            (s.dim, s.stage_id) = (c.dim as usize, Some(c.stage_id as usize));
            s
        });
        Maps {
            cm: cm.collect(),
            fixed: lay.const_pols_map.iter().enumerate().map(|(i, c)| pol("fixed", i, c)).collect(),
            custom: lay
                .custom_pols_map
                .iter()
                .map(|m| m.iter().enumerate().map(|(i, c)| pol("custom", i, c)).collect())
                .collect(),
            challenges: challenges.collect(),
            publics: values("public", &lay.publics_map),
            proof_values: values("proofvalue", &lay.proof_values_map),
            air_values: values("airvalue", &lay.air_values_map),
            airgroup_values: values("airgroupvalue", &lay.airgroup_values_map),
        }
    }

    /// Today's symbol list: the maps in the old order.
    fn symbols(&self) -> Vec<SymbolInfo> {
        let custom = self.custom.iter().flatten();
        let tail = [&self.challenges, &self.publics, &self.proof_values, &self.air_values, &self.airgroup_values];
        self.cm.iter().chain(&self.fixed).chain(custom).chain(tail.into_iter().flatten()).cloned().collect()
    }

    fn print_ctx(&self) -> PrintCtx<'_> {
        PrintCtx {
            cm_pols_map: &self.cm,
            const_pols_map: &self.fixed,
            custom_commits_map: &self.custom,
            publics_map: &self.publics,
            challenges_map: &self.challenges,
            air_values_map: &self.air_values,
            airgroup_values_map: &self.airgroup_values,
            proof_values_map: &self.proof_values,
        }
    }
}
