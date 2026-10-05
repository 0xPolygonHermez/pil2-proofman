//! Top-level orchestrator for computing pil info for a single air.

use pil2_pilout::pilout as pb;

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

    let prover_memory_str = get_prover_memory(
        &setup,
        stark_struct,
        pil_code.ev_map.len(),
        opening_points.len(),
        boundaries.len(),
        q_deg as u64,
        options.inplace_stage_commit,
    );
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

/// GPU prover buffer, in GB: mirrors the GPU branch of `StarkInfo::setMapOffsets`
/// (pil2-stark/src/starkpil/stark_info.cpp), so it is the `mapTotalN` the prover sizes its
/// buffer with. Keep the two in step: a layout change there must be repeated here.
#[allow(clippy::too_many_arguments)]
fn get_prover_memory(
    setup: &SetupResult,
    stark_struct: &StarkStruct,
    n_evals: usize,
    n_opening_points: usize,
    n_boundaries: usize,
    q_deg: u64,
    inplace_stage_commit: bool,
) -> String {
    if stark_struct.n_bits_ext >= 64 || stark_struct.n_bits >= 64 {
        return "N/A".to_string();
    }
    const HASH_SIZE: u64 = 4;
    const GRIND_NONCE_BLOCKS_MAX: u64 = 1024;
    const EVALS_HELPER_CHUNKS: u64 = 16;
    let fe = FIELD_EXTENSION as u64;
    let align = |n: u64| (n + 31) & !31u64;

    let n = 1u64 << stark_struct.n_bits;
    let n_ext = 1u64 << stark_struct.n_bits_ext;
    let arity = stark_struct.merkle_tree_arity as u64;
    let is_gl = stark_struct.verification_hash_type == "GL";
    let is_bn128 = stark_struct.verification_hash_type == "BN128";
    let n_queries = stark_struct.n_queries as u64;
    let section = |name: &str| *setup.map_sections_n.get(name).unwrap_or(&0) as u64;

    // Commit trees drop their leaf level when the last levels and the build's scratch allow it
    let drop_leaf_level =
        is_gl && n_ext > arity.pow(stark_struct.last_level_verification as u32) && n_ext >= arity * arity * arity;
    let num_nodes_commit = |height: u64| {
        let full = get_num_nodes_mt(height, stark_struct.merkle_tree_arity);
        if drop_leaf_level {
            full - (height + (arity - height % arity) % arity) * HASH_SIZE
        } else {
            full
        }
    };
    let num_nodes = num_nodes_commit(n_ext);

    // Constant tree, then the constants on the small domain
    let n_constants = setup.n_constants as u64;
    let mut total = align(n_ext * n_constants + num_nodes);
    total += n * n_constants;

    let mut custom_fixed = 0u64;
    for cc in &setup.custom_commits {
        let width = cc.stage_widths.first().copied().unwrap_or(0) as u64;
        if width > 0 {
            custom_fixed += width * n + width * n_ext + get_num_nodes_mt(n_ext, stark_struct.merkle_tree_arity);
        }
    }
    total += custom_fixed;

    let values_size = |map: &[crate::types::pilout_info::SymbolInfo]| -> u64 {
        map.iter().map(|v| if v.stage == Some(1) { 1 } else { fe }).sum()
    };
    total += setup.n_publics as u64;
    total += values_size(&setup.proof_values_map);
    total += values_size(&setup.airgroup_values_map);
    total += values_size(&setup.air_values_map);
    total += HASH_SIZE + 1 + GRIND_NONCE_BLOCKS_MAX; // challenge, nonce, nonce blocks
    if is_bn128 {
        total = total.div_ceil(4) * 4 + 12;
    } else {
        total += HASH_SIZE;
    }
    total += n_evals as u64 * fe;
    total += setup.challenges_map.len() as u64 * fe;
    total += (n_evals + n_opening_points) as u64 * fe; // folded FRI constants
    total += n_queries;
    let perm_bits = stark_struct.steps.first().map_or(0, |s| s.n_bits as u64);
    total += (n_queries * perm_bits).div_ceil(63);

    // Query proofs
    let mut max_tree_width = setup.map_sections_n.values().copied().max().unwrap_or(0) as u64;
    for w in stark_struct.steps.windows(2) {
        let n_groups = 1u64 << w[1].n_bits;
        max_tree_width = max_tree_width.max(((1u64 << w[0].n_bits) / n_groups) * fe);
    }
    let levels = if stark_struct.n_bits_ext == 0 {
        0
    } else if is_bn128 {
        (stark_struct.n_bits_ext as u64 - 1) / (arity as f64).log2().ceil() as u64 + 1
    } else {
        (stark_struct.n_bits_ext as f64 / (arity as f64).log2()).ceil() as u64
    };
    let n_siblings = levels.saturating_sub(stark_struct.last_level_verification as u64);
    let siblings_per_level = if is_bn128 { arity * 4 } else { (arity - 1) * HASH_SIZE };
    let n_trees = 1 + (setup.n_stages as u64 + 1) + setup.custom_commits.len() as u64;
    let n_trees_fri = stark_struct.steps.len().saturating_sub(1) as u64;
    total += (n_trees + n_trees_fri) * (max_tree_width + n_siblings * siblings_per_level) * n_queries;
    if drop_leaf_level {
        total += n_queries * (arity - 1) * (max_tree_width + HASH_SIZE);
    }

    // Stage traces on the extended domain with their trees. In place, each stage is committed
    // where it lands; otherwise cm1/cm2 keep a small-domain copy that overlaps the next stage.
    total = align(total);
    total += n_ext * section("cm1") + align(num_nodes);
    if inplace_stage_commit {
        total += n_ext * section("cm2") + align(num_nodes);
        total += n_ext * section("cm3") + align(num_nodes);
    } else {
        let cm1_small = total;
        total += n_ext * section("cm2") + align(num_nodes);
        total = total.max(cm1_small + n * section("cm1"));
        let cm2_small = total;
        total += n_ext * section("cm3") + align(num_nodes);
        total = total.max(cm2_small + n * section("cm2"));
    }

    // Q, with the boundary helpers (zi) above it while the quotient is built
    let q_offset = total;
    total += n_ext * fe;
    let mut max_total = total + n_boundaries as u64 * n_ext;
    max_total = max_total.max(q_offset + n * fe + 2 * fe + n_evals as u64 * EVALS_HELPER_CHUNKS * fe);
    max_total = max_total.max(q_offset + n_ext * fe + n_ext * fe + q_deg);

    // FRI layers follow q: x, zi and the expression tmps are dead once folding starts
    for w in stark_struct.steps.windows(2) {
        let height = 1u64 << w[1].n_bits;
        let width = ((1u64 << w[0].n_bits) / height) * fe;
        total += height * width;
        if is_gl {
            total += align(get_num_nodes_mt(height, stark_struct.merkle_tree_arity));
        }
    }
    total = total.max(max_total);

    let gb = (total as f64 * 8.0) / (1024.0 * 1024.0 * 1024.0);
    format!("{:.2}", gb)
}
