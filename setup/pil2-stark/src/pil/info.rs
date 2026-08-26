//! Top-level orchestrator for computing pil info for a single air.

use pil2_pilout::pilout as pb;

use crate::pil::constraint_poly::Boundary;
use crate::pil::gen_code::{CodeGenParams, PilCodeResult};
use crate::pil::im_polynomials::{add_im_polynomials, calculate_intermediate_polynomials};
use crate::pil::map;
use crate::types::pilout_info::{SetupResult, FIELD_EXTENSION};
use crate::pil::prepare::{prepare_pil, PrepareOptions};
use crate::expr::print::PrintCtx;
use crate::types::stark_struct::StarkStruct;

/// The assembled pil info result returned by `pil_info`.
pub struct PilInfoResult {
    pub setup: SetupResult,
    pub pil_code: PilCodeResult,
    /// Summary line for the AIR.
    pub summary: String,
    /// Prover memory estimate string (GB).
    pub prover_memory: String,
    /// Intermediate polynomial info: (base_field, extended_field) expression strings.
    pub im_pols_info: (Vec<String>, Vec<String>),
    /// Constraint polynomial expression ID.
    pub c_exp_id: usize,
    /// FRI polynomial expression ID (distinct from c_exp_id).
    pub fri_exp_id: usize,
    /// Polynomial Q degree.
    pub q_deg: i64,
}

/// Main entry point: assemble pil info for a single air.
///
/// Steps:
/// 1. prepare_pil
/// 2. calculate_intermediate_polynomials
/// 3. add_intermediate_polynomials
/// 4. map
/// 5. generate_pil_code
/// 6. compute prover memory estimate and print AIR info summary
pub fn pil_info(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    stark_struct: &StarkStruct,
    options: &PrepareOptions,
) -> PilInfoResult {
    let result = prepare_pil(pilout, airgroup_id, air_id, stark_struct, options);

    let mut setup = result.setup;
    let mut expressions = result.expressions;
    let mut constraints = result.constraints;
    let mut symbols = result.symbols;
    let hints = result.hints;
    let boundaries = result.boundaries;
    let constraint_poly = result.constraint_poly;

    let mut c_exp_id = constraint_poly.c_exp_id;
    let q_dim = constraint_poly.q_dim;

    let max_deg = (1usize << (stark_struct.n_bits_ext - stark_struct.n_bits)) + 1;

    // Calculate intermediate polynomials
    let im_result = calculate_intermediate_polynomials(&expressions, c_exp_id, max_deg, q_dim);
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

    // Print AIR info summary
    let mut summary = String::new();
    println!("------------------------- AIR INFO -------------------------");
    let mut n_columns_base_field: usize = 0;
    let mut n_columns: usize = 0;
    let n_const = *setup.map_sections_n.get("const").unwrap_or(&0);
    summary.push_str(&format!(
        "nBits: {} | blowUpFactor: {} | maxConstraintDegree: {} ",
        stark_struct.n_bits,
        stark_struct.n_bits_ext - stark_struct.n_bits,
        q_deg + 1
    ));

    // Helper: resolve an ev_map entry to its symbol's stage.
    // Mirrors the JS per-stage evMap filter in pil_info.js.
    let sym_stage_of = |entry_type: &str, id: usize, commit_id: Option<usize>| -> Option<usize> {
        match entry_type {
            "const" => {
                setup.symbols.iter().find(|s| s.pol_id == Some(id) && s.sym_type == "fixed").and_then(|s| s.stage)
            }
            "cm" => {
                setup.symbols.iter().find(|s| s.pol_id == Some(id) && s.sym_type == "witness").and_then(|s| s.stage)
            }
            "custom" => setup
                .symbols
                .iter()
                .find(|s| s.pol_id == Some(id) && s.sym_type == "custom" && s.commit_id == commit_id)
                .and_then(|s| s.stage),
            _ => None,
        }
    };

    // Fixed columns: count ev_map entries whose symbol is a fixed ("const") column.
    let mut fixed_opening_points = std::collections::HashSet::new();
    let mut fixed_evals: usize = 0;
    for e in &pil_code.ev_map {
        if e.entry_type == "const" && setup.symbols.iter().any(|s| s.pol_id == Some(e.id) && s.sym_type == "fixed") {
            fixed_opening_points.insert(e.opening_pos);
            fixed_evals += 1;
        }
    }
    println!(
        "Columns fixed: {} -> Columns in the basefield: {} | Openings: {} | Evals: {}",
        n_const,
        n_const,
        fixed_opening_points.len(),
        fixed_evals
    );
    summary.push_str(&format!("| Fixed: {} ", n_const));

    let mut previous_evals = fixed_evals;

    for i in 1..=(n_stages + 1) {
        let stage_debug = if i == n_stages + 1 { "Q".to_string() } else { i.to_string() };
        let stage_name = format!("cm{}", i);
        let n_cols_stage = setup.cm_pols_map.iter().filter(|p| p.stage == Some(i)).count();
        let n_cols_base_field = *setup.map_sections_n.get(&stage_name).unwrap_or(&0);
        let im_pols: Vec<_> = setup.cm_pols_map.iter().filter(|p| p.stage == Some(i) && p.im_pol).collect();

        // Constraint count for this stage (stage 1 absorbs stage-0 constraints).
        let stage_constraints_count = if i == 1 {
            setup.constraints.iter().filter(|c| c.stage == Some(0) || c.stage == Some(1)).count()
        } else {
            setup.constraints.iter().filter(|c| c.stage == Some(i)).count()
        };

        // Cumulative ev_map entries whose symbol's stage <= i.
        let mut stage_opening_points = std::collections::HashSet::new();
        let cumulative_evals = pil_code
            .ev_map
            .iter()
            .filter(|e| {
                if let Some(s) = sym_stage_of(&e.entry_type, e.id, e.commit_id) {
                    if s <= i {
                        stage_opening_points.insert(e.opening_pos);
                        return true;
                    }
                }
                false
            })
            .count();
        let new_evals = cumulative_evals - previous_evals;
        previous_evals = cumulative_evals;

        // Only print if there are columns in this stage (mirrors JS behaviour).
        if n_cols_stage > 0 {
            let constraint_info = if stage_constraints_count > 0 {
                format!(" | Constraints: {}", stage_constraints_count)
            } else {
                String::new()
            };
            let stats_suffix =
                format!("{} | Openings: {} | Evals: {}", constraint_info, stage_opening_points.len(), new_evals);

            if i == n_stages + 1 || (i < n_stages && !options.im_pols_stages) {
                println!(
                    "Columns stage {}: {} -> Columns in the basefield: {}{}",
                    stage_debug, n_cols_stage, n_cols_base_field, stats_suffix
                );
            } else {
                let im_dim_sum: usize = im_pols.iter().map(|p| p.dim).sum();
                let im_count_label = if im_pols.len() == 1 { "intermediate" } else { "intermediates" };
                let im_dim_label = if im_dim_sum == 1 { "intermediate" } else { "intermediates" };
                println!(
                    "Columns stage {}: {} ({} {}) -> Columns in the basefield: {} ({} from {}){}",
                    stage_debug,
                    n_cols_stage,
                    im_pols.len(),
                    im_count_label,
                    n_cols_base_field,
                    im_dim_sum,
                    im_dim_label,
                    stats_suffix
                );
            }
        }

        if i < n_stages + 1 {
            summary.push_str(&format!("| Stage{}: {} ", i, n_cols_base_field));
        } else {
            summary.push_str(&format!("| StageQ: {} ", n_cols_base_field));
        }
        n_columns += n_cols_stage;
        n_columns_base_field += n_cols_base_field;
    }

    let all_im_pols: Vec<_> = setup.cm_pols_map.iter().filter(|p| p.im_pol).collect();
    let im_dim_sum: usize = all_im_pols.iter().map(|p| p.dim).sum();
    let im_dim1_sum: usize = all_im_pols.iter().filter(|p| p.dim == 1).map(|p| p.dim).sum();
    let im_dim3_sum: usize = all_im_pols.iter().filter(|p| p.dim == FIELD_EXTENSION).map(|p| p.dim).sum();
    summary.push_str(&format!(
        "| ImPols: {} => {} = {} + {} ",
        all_im_pols.len(),
        im_dim_sum,
        im_dim1_sum,
        im_dim3_sum
    ));

    summary.push_str(&format!(
        "| Total: {} | nConstraints: {} | nExpressions: {}",
        n_columns_base_field,
        setup.constraints.len(),
        setup.expressions.len()
    ));
    if !options.debug {
        summary.push_str(&format!(" | nOpeningPoints: {}", opening_points.len()));
    }
    summary.push_str(&format!(" | nEvals: {}", pil_code.ev_map.len()));

    println!("Total Columns: {} -> Columns in the basefield: {}", n_columns, n_columns_base_field);
    println!("Total Constraints: {}", setup.constraints.len());
    println!("Total Expressions: {}", setup.expressions.len());
    if !options.debug {
        println!("Number of opening points: {}", opening_points.len());
        println!("Number of evaluations: {}", pil_code.ev_map.len());
    }

    let prover_memory_str = get_prover_memory(&setup, stark_struct, &opening_points, &boundaries);
    println!("Prover memory: {} GB", prover_memory_str);
    summary.push_str(&format!("| Prover memory: {} GB", prover_memory_str));

    println!("------------------------------------------------------------");
    println!("SUMMARY | {} | {}", setup.name, summary);
    println!("------------------------------------------------------------");

    // Store the sorted opening points in the setup result so callers don't recompute them.
    setup.opening_points = opening_points;

    let im_pols_info = setup.im_pols_info.clone();
    let fri_exp_id = pil_code.fri_exp_id;

    PilInfoResult {
        setup,
        pil_code,
        summary,
        prover_memory: prover_memory_str,
        im_pols_info,
        c_exp_id,
        fri_exp_id,
        q_deg,
    }
}

fn get_num_nodes_mt(height: u64, merkle_tree_arity: usize) -> u64 {
    let arity = merkle_tree_arity as u64;
    let mut num_nodes = height;
    let mut nodes_level = height;

    while nodes_level > 1 {
        let extra_zeros = (arity - (nodes_level % arity)) % arity;
        num_nodes += extra_zeros;
        let next_n = nodes_level.div_ceil(arity);
        num_nodes += next_n;
        nodes_level = next_n;
    }

    num_nodes * 4
}

fn get_prover_memory(
    setup: &SetupResult,
    stark_struct: &StarkStruct,
    _opening_points: &[i64],
    boundaries: &[Boundary],
) -> String {
    if stark_struct.n_bits_ext >= 64 || stark_struct.n_bits >= 64 {
        return "N/A".to_string();
    }
    let n_extended = 1u64 << stark_struct.n_bits_ext;
    let n = 1u64 << stark_struct.n_bits;
    let num_nodes = get_num_nodes_mt(n_extended, stark_struct.merkle_tree_arity);

    let mut prover_memory: u64 = 0;

    // Custom commits
    for cc in &setup.custom_commits {
        if !cc.stage_widths.is_empty() && cc.stage_widths[0] > 0 {
            prover_memory += cc.stage_widths[0] as u64 * (n + n_extended) + num_nodes;
        }
    }

    // Constants
    let n_constants = setup.n_constants as u64;
    prover_memory += 2 + n_extended * n_constants + num_nodes;

    if (n_constants * n * 8) / (1024 * 1024) < 512 {
        prover_memory += n * n_constants;
    }

    let mut offset_traces: u64 = 0;
    let n_stages = setup.n_stages;
    for i in 1..=(n_stages + 1) {
        if i == 2 {
            offset_traces = prover_memory;
        }
        let key = format!("cm{}", i);
        let section_n = *setup.map_sections_n.get(&key).unwrap_or(&0) as u64;
        prover_memory += section_n * (1u64 << stark_struct.n_bits_ext) + num_nodes;
    }

    for i in (1..=n_stages).rev() {
        let key = format!("cm{}", i);
        let section_n = *setup.map_sections_n.get(&key).unwrap_or(&0) as u64;
        offset_traces += section_n * n;
    }

    if offset_traces > prover_memory {
        prover_memory = offset_traces;
    }

    prover_memory += (FIELD_EXTENSION as u64 + FIELD_EXTENSION as u64 + boundaries.len() as u64) * n_extended;

    if stark_struct.steps.len() > 1 {
        for i in 0..stark_struct.steps.len() - 1 {
            let sb = stark_struct.steps[i + 1].n_bits;
            let sa = stark_struct.steps[i].n_bits;
            if sb >= 64 || sa >= 64 {
                continue;
            }
            let height = 1u64 << sb;
            let width = ((1u64 << sa) / height) * FIELD_EXTENSION as u64;
            prover_memory += height * width + get_num_nodes_mt(height, stark_struct.merkle_tree_arity);
        }
    }

    let gb = (prover_memory as f64 * 8.0) / (1024.0 * 1024.0 * 1024.0);
    format!("{:.2}", gb)
}
