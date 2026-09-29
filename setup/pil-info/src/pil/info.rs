//! Orchestrates the symbolic passes that follow `prepare_pil` for a single air.

use crate::pil::constraint_poly::Boundary;
use crate::pil::gen_code::{CodeGenParams, PilCodeResult};
use crate::pil::im_polynomials::{add_im_polynomials, calculate_intermediate_polynomials};
use crate::pil::map;
use crate::types::pilout_info::SetupResult;
use crate::pil::prepare::{PrepareOptions, PreparePilResult};
use crate::expr::print::PrintCtx;

/// The result of the passes, as returned by `pil_info`.
pub struct PilInfoResult {
    pub setup: SetupResult,
    pub pil_code: PilCodeResult,
    /// Intermediate polynomial info: (base_field, extended_field) expression strings.
    pub im_pols_info: (Vec<String>, Vec<String>),
    /// Constraint polynomial expression ID.
    pub c_exp_id: usize,
    /// FRI polynomial expression ID (distinct from c_exp_id).
    pub fri_exp_id: usize,
    /// Polynomial Q degree.
    pub q_deg: i64,
    /// Boundary definitions.
    pub boundaries: Vec<Boundary>,
}

/// Run the passes on the output of `prepare_pil`.
///
/// Steps:
/// 1. calculate_intermediate_polynomials, bounded by `max_deg`
/// 2. add_intermediate_polynomials
/// 3. map
/// 4. generate_pil_code
pub fn pil_info(
    prepared: PreparePilResult,
    airgroup_id: usize,
    air_id: usize,
    max_deg: usize,
    options: &PrepareOptions,
) -> PilInfoResult {
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
    let im_result = calculate_intermediate_polynomials(&expressions, c_exp_id, max_deg, q_dim, &symbols);
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
    );
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
    // expressions with keep=true, im_pol=true, or matching c_exp_id/fri_exp_id.
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
    // fri_exp_id will be updated by generate_pil_code after FRI polynomial generation
    let mut params = CodeGenParams {
        air_id,
        airgroup_id,
        n_stages,
        c_exp_id,
        fri_exp_id: c_exp_id, // placeholder; will be overwritten
        q_deg: q_deg as usize,
        q_dim: q_dim_final,
        opening_points: opening_points.clone(),
        cm_pols_map: setup.cm_pols_map.clone(),
        custom_commits_count: setup.custom_commits.len(),
    };

    // Store hints back into setup for generate_pil_code
    setup.hints = hints;

    // Temporarily take out mutable fields to allow PrintCtx to borrow map fields
    let mut expressions = std::mem::take(&mut setup.expressions);
    let mut symbols = std::mem::take(&mut setup.symbols);

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
        &mut params,
        &mut symbols,
        &setup.constraints,
        &mut expressions,
        &setup.hints,
        options.debug,
        Some(&print_ctx),
    );

    // Put expressions and symbols back
    setup.expressions = expressions;
    setup.symbols = symbols;

    // Store the sorted opening points in the setup result so callers don't recompute them.
    setup.opening_points = opening_points;

    let im_pols_info = setup.im_pols_info.clone();
    let fri_exp_id = pil_code.fri_exp_id;

    PilInfoResult { setup, pil_code, im_pols_info, c_exp_id, fri_exp_id, q_deg, boundaries }
}
