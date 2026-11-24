//! Orchestrates the symbolic passes for a single air: `run` from a pilout, or `pil_info` after
//! `prepare_pil`.
//! TODO(piece 3): pilfflonk's old path; delete once pilfflonk builds from the core.

use pil2_pilout::pilout as pb;

use crate::cfg::{Opening, PilInfoCfg};
use crate::core::dag::build::build_dag;
use crate::core::dag::format::{ConstraintDag, DagBoundary};
use crate::core::from_pilout::{from_pilout, pilout_expr_ids};
use crate::core::im::ImChoice;
use crate::error::{PilInfoError, Result};
use crate::pil::constraint_poly::Boundary;
use crate::pil::gen_code::{CodeGenParams, PilCodeResult};
use crate::pil::im_polynomials::{add_im_polynomials, calculate_intermediate_polynomials};
use crate::pil::map;
use crate::types::pilout_info::{SetupResult, SymbolInfo};
use crate::pil::prepare::{prepare_pil, PrepareOptions, PreparePilResult};
use crate::expr::print::PrintCtx;

/// The result of the passes, as returned by `pil_info`.
pub struct PilInfoResult {
    pub setup: SetupResult,
    pub pil_code: PilCodeResult,
    /// Intermediate polynomial info: (base_field, extended_field) expression strings.
    pub im_pols_info: (Vec<String>, Vec<String>),
    /// Constraint polynomial expression ID.
    pub c_exp_id: usize,
    /// Polynomial Q degree.
    pub q_deg: i64,
    /// Boundary definitions.
    pub boundaries: Vec<Boundary>,
}

/// Prepare the air and run every pass on it for `cfg`: `prepare_pil` then `pil_info`.
///
/// Fails on an air the pilout does not have, on a pilout that breaks its own format and on
/// constraints the passes cannot process; see [`crate::PilInfoError`].
pub fn run(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
    options: &PrepareOptions,
) -> Result<PilInfoResult> {
    // the passes are pilfflonk's (SHPLONK)
    if cfg.opening == Opening::Deep {
        return Err(PilInfoError::DeepPasses);
    }
    let prepared = prepare_pil(pilout, airgroup_id, air_id, &cfg.field)?;
    pil_info(prepared, pilout, airgroup_id, air_id, cfg, options)
}

/// The air's constraint DAG from the core, with the passes' im pols in their column order (the prover's
/// cExp folds them in that order): the qVerifier's source. `symbols` are the passes' (`r.setup.symbols`).
pub fn constraint_dag(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
    symbols: &[SymbolInfo],
) -> Result<ConstraintDag> {
    let err = |e: &dyn std::fmt::Debug| PilInfoError::ConstraintDag(format!("{e:?}"));
    let cs = from_pilout(pilout, airgroup_id, air_id, &cfg.field, cfg.opening).map_err(|e| err(&e))?;
    let ids = pilout_expr_ids(pilout, airgroup_id, air_id).map_err(|e| err(&e))?;
    let mut im: Vec<&SymbolInfo> = symbols.iter().filter(|s| s.im_pol && s.sym_type == "witness").collect();
    im.sort_by_key(|s| s.pol_id);
    // an im pol's exp_id is its pilout expression's index
    let exprs = im
        .iter()
        .map(|s| s.exp_id.and_then(|e| ids.get(e).copied()).ok_or_else(|| err(&format!("im pol {}", s.name))))
        .collect::<Result<Vec<_>>>()?;
    // canonical numbering may merge two equal im expressions: refuse, the passes kept them apart
    if exprs.iter().collect::<std::collections::BTreeSet<_>>().len() != exprs.len() {
        return Err(err(&"two im pols share one core expression"));
    }
    let dag = build_dag(&cs, &ImChoice { exprs }).map_err(|e| err(&e))?;
    if dag.roots.len() != cs.constraints.len() + im.len() {
        return Err(err(&format!(
            "{} roots for {} constraints and {} im pols",
            dag.roots.len(),
            cs.constraints.len(),
            im.len()
        )));
    }
    Ok(dag)
}

/// `boundaries` (in `boundaryId` order) as the DAG's.
pub fn dag_boundaries(boundaries: &[Boundary]) -> Result<Vec<DagBoundary>> {
    boundaries
        .iter()
        .map(|b| match (b.name.as_str(), b.offset_min, b.offset_max) {
            ("everyRow", ..) => Ok(DagBoundary::EveryRow),
            ("firstRow", ..) => Ok(DagBoundary::FirstRow),
            ("lastRow", ..) => Ok(DagBoundary::LastRow),
            ("everyFrame", Some(min), Some(max)) => Ok(DagBoundary::EveryFrame { min: min.into(), max: max.into() }),
            (name, ..) => Err(PilInfoError::UnsupportedBoundary(name.to_string())),
        })
        .collect()
}

/// Run the passes on the output of `prepare_pil`, which must have been prepared over `cfg.field`,
/// and of the pilout it came from (the qVerifier's constraint DAG); SHPLONK only, checked by `run`.
///
/// Steps:
/// 1. calculate_intermediate_polynomials, as `cfg.degree_policy` says
/// 2. add_intermediate_polynomials
/// 3. map
/// 4. generate_pil_code
fn pil_info(
    prepared: PreparePilResult,
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
    options: &PrepareOptions,
) -> Result<PilInfoResult> {
    let mut setup = prepared.setup;
    let mut expressions = prepared.expressions;
    let mut constraints = prepared.constraints;
    let mut symbols = prepared.symbols;
    let hints = prepared.hints;
    let boundaries = prepared.boundaries;
    let constraint_poly = prepared.constraint_poly;

    let mut c_exp_id = constraint_poly.c_exp_id;
    let q_dim = constraint_poly.q_dim;

    // Calculate intermediate polynomials
    let im_result = calculate_intermediate_polynomials(&expressions, c_exp_id, &cfg.degree_policy, q_dim, &symbols)?;
    let im_exps = im_result.im_exps;
    let q_deg = im_result.q_deg;

    // Build boundary tuples for add_im_polynomials
    let boundary_tuples: Vec<(String, Option<i64>, Option<i64>)> = boundaries
        .iter()
        .map(|b| (b.name.clone(), b.offset_min.map(|v| v as i64), b.offset_max.map(|v| v as i64)))
        .collect();

    // Add intermediate polynomials
    let mut n_commitments = setup.n_commitments;
    let q_dim_final = add_im_polynomials(
        &mut expressions,
        &mut constraints,
        &mut symbols,
        &setup.name,
        air_id,
        airgroup_id,
        setup.n_stages,
        &mut n_commitments,
        &mut c_exp_id,
        &im_exps,
        q_deg,
        options.im_pols_stages,
        &boundary_tuples,
        &cfg.field,
    )?;
    setup.n_commitments = n_commitments;

    // Store back into setup for mapping
    setup.expressions = expressions;
    setup.constraints = constraints;
    setup.symbols = symbols;

    // Map
    map::map(&mut setup, false);

    // Compute opening points from ALL expressions that will be code-generated:
    // constraints, kept expressions (from hints), and imPol expressions.
    // This mirrors the filter in generate_expressions_code which processes
    // expressions with keep=true, im_pol=true, or matching c_exp_id.
    let mut opening_points: Vec<i64> = vec![0];
    for c in &setup.constraints {
        let offsets = &setup.expressions[c.e].rows_offsets;
        for &offset in offsets {
            if !opening_points.contains(&offset) {
                opening_points.push(offset);
            }
        }
    }
    for expr in &setup.expressions {
        if expr.keep.unwrap_or(false) || expr.im_pol {
            for &offset in &expr.rows_offsets {
                if !opening_points.contains(&offset) {
                    opening_points.push(offset);
                }
            }
        }
    }
    opening_points.sort();

    // Build code-gen params
    let n_stages = setup.n_stages;
    let params = CodeGenParams {
        air_id,
        airgroup_id,
        n_stages,
        c_exp_id,
        q_deg: q_deg as usize,
        q_dim: q_dim_final,
        opening_points: opening_points.clone(),
        cm_pols_map: setup.cm_pols_map.clone(),
        custom_commits_count: setup.custom_commits.len(),
        field: cfg.field.clone(),
        opening: cfg.opening,
    };

    // Store hints back into setup for generate_pil_code
    setup.hints = hints;

    // Temporarily take out mutable fields to allow PrintCtx to borrow map fields
    let mut expressions = std::mem::take(&mut setup.expressions);
    let symbols = std::mem::take(&mut setup.symbols);

    let print_ctx = PrintCtx {
        cm_pols_map: &setup.cm_pols_map,
        const_pols_map: &setup.const_pols_map,
        custom_commits_map: &setup.custom_commits_map,
        publics_map: &setup.publics_map,
        challenges_map: &setup.challenges_map,
        air_values_map: &setup.air_values_map,
        airgroup_values_map: &setup.airgroup_values_map,
        proof_values_map: &setup.proof_values_map,
    };

    let pil_code = crate::pil::gen_code::generate_pil_code(
        &params,
        &symbols,
        &setup.constraints,
        &mut expressions,
        &setup.hints,
        Some(&print_ctx),
        Some((&constraint_dag(pilout, airgroup_id, air_id, cfg, &symbols)?, &dag_boundaries(&boundaries)?)),
    )?;

    // Put expressions and symbols back
    setup.expressions = expressions;
    setup.symbols = symbols;

    // Store the sorted opening points in the setup result so callers don't recompute them.
    setup.opening_points = opening_points;

    let im_pols_info = setup.im_pols_info.clone();

    Ok(PilInfoResult { setup, pil_code, im_pols_info, c_exp_id, q_deg, boundaries })
}
