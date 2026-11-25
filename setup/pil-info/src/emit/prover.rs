//! Typed IR -> today's expression arena -> the existing prover code generator.

use std::cell::RefCell;

use crate::cfg::{FieldCfg, Opening};
use crate::core::fold::fold_roots;
use crate::core::im::{im_column_id, ImChoice};
use crate::core::ir::{value_dim, Boundary, ConstraintSystem, Expr, NamedValue};
use crate::core::layout::{Layout, PolMap};
use crate::core::openings::Openings;
use crate::error::{PilInfoError, Result};
use crate::expr::expression::{ExprChild, Expression};
use crate::expr::helpers::{add_info_expression_inline, add_info_expressions, get_exp_dim};
use crate::emit::arena;
use crate::emit::verifier::shplonk_q_verifier;
use crate::expr::print::{print_bounded, PrintCtx};
use crate::pil::gen_code::{generate_pil_code, CodeGenParams, ExpressionCodeEntry, PilCodeResult};
use crate::types::pilout_info::{ConstraintInfo, HintInfo, SymbolInfo};

/// Arena ids with a fixed meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Roles {
    /// Q = fold_roots of the substituted roots, always the last entry; the starkinfo's `cExpId`.
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

/// Arena order, a function of the typed input: expressions in index order (then the formatter's children),
/// hint operands in hint order, the fold of the roots (im roots in im order), then Q.
pub fn to_arena(cs: &ConstraintSystem, im: &ImChoice, lay: &Layout) -> Result<ProverExprs> {
    let (arena, hints, constraints) = formatted(cs, lay)?;
    fold_quotient(cs, im, lay, arena, hints, constraints)
}

/// The generator's output and what it read.
pub struct ProverCode {
    pub code: PilCodeResult,
    /// The im expressions printed, (base field, extension), in column order.
    pub im_pols_info: (Vec<String>, Vec<String>),
    pub maps: Maps,
    pub arena: Vec<Expression>,
    /// The constraints, im roots included.
    pub constraints: Vec<ConstraintInfo>,
    pub hints: Vec<HintInfo>,
}

/// Expressionsinfo and hints through the existing generator from `to_arena`'s output; pilfflonk's qVerifier from the DAG.
pub fn emit_prover(
    cs: &ConstraintSystem,
    im: &ImChoice,
    lay: &Layout,
    ops: &Openings,
    exprs: ProverExprs,
    p: &EmitParams,
) -> Result<ProverCode> {
    let ProverExprs { mut arena, roles, hints, mut constraints } = exprs;
    let maps = Maps::of(cs, lay, p);
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
        q_dim: lay.q_dim as usize,
        opening_points: ops.opening_points.clone(),
        field: cs.field.clone(),
    };
    let mut code = generate_pil_code(&params, &maps.symbols(), &constraints, &mut arena, &hints, Some(&ctx))?;
    if cs.opening == Opening::Shplonk {
        let q = shplonk_q_verifier(cs, im, lay, ops).map_err(|e| PilInfoError::ConstraintDag(format!("{e:?}")))?;
        code.verifier_info.q_verifier = Some(ExpressionCodeEntry { exp_id: roles.c_exp, ..q });
    }
    Ok(ProverCode { code, im_pols_info, maps, arena, constraints, hints })
}

type Formatted = (Vec<Expression>, Vec<HintInfo>, Vec<ConstraintInfo>);

/// The arena and hints from the IR, then the info passes.
fn formatted(cs: &ConstraintSystem, lay: &Layout) -> Result<Formatted> {
    let field = &cs.field;
    let leaf = arena::air_leaf(cs, lay);
    let mut arena = arena::from_ir(&cs.exprs, &leaf)?;
    let hints = arena::hints(&cs.hints, &cs.exprs, &leaf, &mut arena)?;

    // stage 1, info on the constraints, then on every expression
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
        im_roots: Vec::new(),
        q_stage: cs.n_stages as usize + 1,
        vc_id: lay.vc_index(),
        field,
    });
    let mut roots: Vec<Root> = (0..cs.constraints.len()).map(Root::Constraint).collect();
    roots.extend((0..im.exprs.len()).map(Root::Im));
    let ctx = Ctx { cs, im, lay, constraints: &constraints, boundaries: cs.boundaries() };
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
    /// In `boundaryId` order.
    boundaries: Vec<Boundary>,
}

struct FoldState<'a> {
    arena: Vec<Expression>,
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
            Root::Constraint(k) if ctx.cs.constraints[k].boundary == Boundary::EveryRow => Ok(ctx.constraints[k].e),
            // pilfflonk's (from_pilout rejects it for the STARK): `e * Zi`, pushed without info as the old cExp did
            Root::Constraint(k) => {
                let b = &ctx.cs.constraints[k].boundary;
                let id = ctx.boundaries.iter().position(|x| x == b).ok_or_else(fold_bug)?;
                let zi = Expression {
                    op: "Zi".into(),
                    boundary_id: Some(id),
                    boundary: Some(b.name().into()),
                    ..Default::default()
                };
                let e = self.exp(ctx.constraints[k].e);
                Ok(self.push(Expression {
                    op: "mul".into(),
                    values: vec![e, ExprChild::Inline(Box::new(zi))],
                    ..Default::default()
                }))
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

/// The generator's symbol maps, projected from the layout.
pub struct Maps {
    pub cm: Vec<SymbolInfo>,
    pub fixed: Vec<SymbolInfo>,
    pub custom: Vec<Vec<SymbolInfo>>,
    pub challenges: Vec<SymbolInfo>,
    pub publics: Vec<SymbolInfo>,
    pub proof_values: Vec<SymbolInfo>,
    pub air_values: Vec<SymbolInfo>,
    pub airgroup_values: Vec<SymbolInfo>,
}

impl Maps {
    fn of(cs: &ConstraintSystem, lay: &Layout, p: &EmitParams) -> Self {
        let ext = cs.field.ext_dim();
        let lengths = |l: &Option<Vec<u32>>| l.as_ref().map(|v| v.iter().map(|x| *x as usize).collect());
        let pol = |kind: &str, i: usize, c: &PolMap| SymbolInfo {
            name: c.name.clone(),
            sym_type: kind.into(),
            stage: Some(c.stage as usize),
            dim: c.dim as usize,
            // an im column's id is its expression's
            id: c.exp_id.map(|e| e as usize),
            pol_id: Some(i),
            stage_id: Some(c.stage_id as usize),
            air_id: Some(p.air_id),
            airgroup_id: Some(p.airgroup_id),
            commit_id: c.commit_id.map(|x| x as usize),
            lengths: lengths(&c.lengths),
            stage_pos: Some(c.stage_pos as usize),
            im_pol: c.exp_id.is_some(),
            exp_id: c.exp_id.map(|e| e as usize),
        };
        let value = |kind: &str, i: usize, v: &NamedValue| SymbolInfo {
            name: v.name.clone(),
            sym_type: kind.into(),
            stage: Some(v.stage as usize),
            dim: value_dim(v.stage, ext),
            id: Some(i),
            pol_id: None,
            stage_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: lengths(&v.lengths),
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        let cm = lay.cm_pols_map.iter().enumerate().map(|(i, c)| pol("witness", i, c));
        let values = |kind: &str, m: &[NamedValue]| m.iter().enumerate().map(|(i, v)| value(kind, i, v)).collect();
        // the DEEP challenges are not the generator's
        let q = cs.n_stages + 2;
        let challenges = lay.challenges_map.iter().enumerate().filter(|(_, c)| c.stage <= q).map(|(i, c)| {
            let mut s =
                value("challenge", i, &NamedValue { name: c.name.clone(), stage: c.stage, ..Default::default() });
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
    pub fn symbols(&self) -> Vec<SymbolInfo> {
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
