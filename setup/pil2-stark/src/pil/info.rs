//! What is STARK-specific around `stark_setup`: the starkStruct validation, the prover memory
//! estimate and the printed AIR info summary.

use std::collections::HashSet;

use anyhow::{bail, Result};
use pil_info::core::ir::ColType;
use pil_info::core::project::VerifierView;

use crate::pil::gen_code::PilCodeResult;
use crate::pil::prepare::PrepareOptions;
use crate::types::output::StarkInfoOutput;
use crate::types::stark_struct::{LowDegreeTest, StarkStruct};
use crate::types::FIELD_EXTENSION;

/// The assembled pil info result returned by `stark_setup`.
pub struct PilInfoResult {
    /// The core's starkinfo, solved low-degree test included; no circuit type.
    pub starkinfo: StarkInfoOutput,
    pub pil_code: PilCodeResult,
    /// Summary line for the AIR.
    pub summary: String,
    /// Prover memory estimate string (GB).
    pub prover_memory: String,
    /// Intermediate polynomial info: (base_field, extended_field) expression strings.
    pub im_pols_info: (Vec<String>, Vec<String>),
}

/// The starkStruct must fit the air: its degree, and nBitsExt the low-degree test's first domain.
pub(crate) fn validate_stark_struct(
    stark_struct: &StarkStruct,
    pil_power: u32,
    airgroup_id: usize,
    air_id: usize,
) -> Result<()> {
    if stark_struct.n_bits != pil_power as usize {
        bail!(
            "starkStruct and pilfile have degree mismatch (airId: {} airgroupId: {} starkStruct:{} pilfile:{})",
            air_id,
            airgroup_id,
            stark_struct.n_bits,
            pil_power
        );
    }

    let initial_domain = stark_struct.low_degree_test.log_domain_sizes()[0];
    if stark_struct.n_bits_ext != initial_domain {
        bail!(
            "starkStruct.nBitsExt and the low-degree test's initial domain have a mismatch (nBitsExt:{} domain:{})",
            stark_struct.n_bits_ext,
            initial_domain
        );
    }
    Ok(())
}

/// Print the AIR info and return its summary line and prover memory estimate (GB).
///
/// `constraint_stages` holds the stage of each constraint, im roots included.
pub(crate) fn report(
    v: &VerifierView,
    air_name: &str,
    constraint_stages: &[usize],
    n_expressions: usize,
    stark_struct: &StarkStruct,
    options: &PrepareOptions,
) -> (String, String) {
    let n_stages = v.n_stages as usize;
    let section = |name: &str| v.map_sections_n.iter().find(|(k, _)| k == name).map_or(0, |(_, n)| *n as usize);

    let mut summary = String::new();
    println!("------------------------- AIR INFO -------------------------");
    let mut n_columns_base_field: usize = 0;
    let mut n_columns: usize = 0;
    let n_const = section("const");
    summary.push_str(&format!(
        "nBits: {} | blowUpFactor: {} | maxConstraintDegree: {} ",
        stark_struct.n_bits,
        stark_struct.n_bits_ext - stark_struct.n_bits,
        v.q_deg + 1
    ));

    // The stage of an evaluated column, as the JS per-stage evMap filter in pil_info.js.
    let stage_of = |col: ColType, commit_id: Option<u32>, id: u32| -> Option<usize> {
        let m = match col {
            ColType::Witness => &v.cm_pols_map,
            ColType::Fixed => &v.const_pols_map,
            ColType::Custom => v.custom_pols_map.get(commit_id? as usize)?,
        };
        m.get(id as usize).map(|p| p.stage as usize)
    };

    let fixed: Vec<_> = v
        .ev_map
        .iter()
        .filter(|e| e.pol.col == ColType::Fixed && stage_of(e.pol.col, None, e.pol.id).is_some())
        .collect();
    let fixed_evals = fixed.len();
    let fixed_points: HashSet<u32> = fixed.iter().map(|e| e.opening_pos).collect();
    println!(
        "Columns fixed: {} -> Columns in the basefield: {} | Openings: {} | Evals: {}",
        n_const,
        n_const,
        fixed_points.len(),
        fixed_evals
    );
    summary.push_str(&format!("| Fixed: {} ", n_const));

    let mut previous_evals = fixed_evals;

    for i in 1..=(n_stages + 1) {
        let stage_debug = if i == n_stages + 1 { "Q".to_string() } else { i.to_string() };
        let in_stage = || v.cm_pols_map.iter().filter(move |p| p.stage as usize == i);
        let n_cols_stage = in_stage().count();
        let n_cols_base_field = section(&format!("cm{}", i));
        let im_pols: Vec<_> = in_stage().filter(|p| p.im_pol).collect();

        // stage 1 absorbs the stage-0 constraints
        let stage_constraints_count = constraint_stages.iter().filter(|&&s| s == i || (i == 1 && s == 0)).count();

        // cumulative evaluations of the columns of stage <= i
        let mut stage_opening_points = HashSet::new();
        let cumulative_evals = v
            .ev_map
            .iter()
            .filter(|e| stage_of(e.pol.col, e.pol.commit_id, e.pol.id).is_some_and(|s| s <= i))
            .inspect(|e| {
                stage_opening_points.insert(e.opening_pos);
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
                let im_dim_sum: u32 = im_pols.iter().map(|p| p.dim).sum();
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

    let all_im_pols: Vec<_> = v.cm_pols_map.iter().filter(|p| p.im_pol).collect();
    let im_dim_sum: u32 = all_im_pols.iter().map(|p| p.dim).sum();
    let im_dim1_sum: u32 = all_im_pols.iter().filter(|p| p.dim == 1).map(|p| p.dim).sum();
    let im_dim3_sum: u32 = all_im_pols.iter().filter(|p| p.dim as usize == FIELD_EXTENSION).map(|p| p.dim).sum();
    summary.push_str(&format!(
        "| ImPols: {} => {} = {} + {} ",
        all_im_pols.len(),
        im_dim_sum,
        im_dim1_sum,
        im_dim3_sum
    ));

    let n_constraints = constraint_stages.len();
    let (n_points, n_evals) = (v.opening_points.len(), v.ev_map.len());
    summary.push_str(&format!(
        "| Total: {} | nConstraints: {} | nExpressions: {}",
        n_columns_base_field, n_constraints, n_expressions
    ));
    summary.push_str(&format!(" | nOpeningPoints: {}", n_points));
    summary.push_str(&format!(" | nEvals: {}", n_evals));

    println!("Total Columns: {} -> Columns in the basefield: {}", n_columns, n_columns_base_field);
    println!("Total Constraints: {}", n_constraints);
    println!("Total Expressions: {}", n_expressions);
    println!("Number of opening points: {}", n_points);
    println!("Number of evaluations: {}", n_evals);

    let prover_memory_str = get_prover_memory(v, stark_struct, options.inplace_stage_commit);
    println!("Prover memory: {} GB", prover_memory_str);
    summary.push_str(&format!("| Prover memory: {} GB", prover_memory_str));

    println!("------------------------------------------------------------");
    println!("SUMMARY | {} | {}", air_name, summary);
    println!("------------------------------------------------------------");

    (summary, prover_memory_str)
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
fn get_prover_memory(v: &VerifierView, stark_struct: &StarkStruct, inplace_stage_commit: bool) -> String {
    let (n_evals, n_opening_points) = (v.ev_map.len(), v.opening_points.len());
    let (n_boundaries, q_deg) = (v.boundaries.len(), v.q_deg as u64);
    if stark_struct.n_bits_ext >= 64 || stark_struct.n_bits >= 64 {
        return "N/A".to_string();
    }
    const HASH_SIZE: u64 = 4;
    const GRIND_NONCE_BLOCKS_MAX: u64 = 1024;
    const EVALS_HELPER_CHUNKS: u64 = 32;
    let fe = FIELD_EXTENSION as u64;
    let align = |n: u64| (n + 31) & !31u64;

    let n = 1u64 << stark_struct.n_bits;
    let n_ext = 1u64 << stark_struct.n_bits_ext;
    let arity = stark_struct.merkle_tree_arity as u64;
    let is_gl = stark_struct.verification_hash_type == "GL";
    let is_bn128 = stark_struct.verification_hash_type == "BN128";
    let ldt = &stark_struct.low_degree_test;
    // Queries into f0 (STIR's first round is the widest), and the low-degree test's committed oracles.
    let n_queries = match ldt {
        LowDegreeTest::Fri(fri) => fri.num_queries,
        LowDegreeTest::Stir(stir) => stir.num_queries.iter().copied().max().unwrap_or(0),
    } as u64;
    let ldt_trees = ldt.commitment_trees();
    let section = |name: &str| v.map_sections_n.iter().find(|(k, _)| k == name).map_or(0, |(_, n)| *n as u64);

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
    let n_constants = v.n_constants as u64;
    let mut total = align(n_ext * n_constants + num_nodes);
    total += n * n_constants;

    let mut custom_fixed = 0u64;
    for cc in &v.custom_commits {
        let width = cc.stage_widths.first().copied().unwrap_or(0) as u64;
        if width > 0 {
            custom_fixed += width * n + width * n_ext + get_num_nodes_mt(n_ext, stark_struct.merkle_tree_arity);
        }
    }
    total += custom_fixed;

    let values_size =
        |map: &[pil_info::core::ir::NamedValue]| -> u64 { map.iter().map(|x| if x.stage == 1 { 1 } else { fe }).sum() };
    total += v.n_publics as u64;
    total += values_size(&v.proof_values_map);
    total += values_size(&v.airgroup_values_map);
    total += values_size(&v.air_values_map);
    total += HASH_SIZE + 1 + GRIND_NONCE_BLOCKS_MAX; // challenge, nonce, nonce blocks
    if is_bn128 {
        total = total.div_ceil(4) * 4 + 12;
    } else {
        total += HASH_SIZE;
    }
    total += n_evals as u64 * fe;
    // as before the core: the DEEP challenges are not counted
    total += v.challenges_map.iter().filter(|c| c.stage <= v.n_stages + 2).count() as u64 * fe;
    total += (n_evals + 2 * n_opening_points) as u64 * fe; // FRI constants
    total += n_queries;
    let perm_bits = ldt.log_domain_sizes().first().map_or(0, |&b| b as u64);
    total += (n_queries * perm_bits).div_ceil(63);

    // Query proofs
    let mut max_tree_width = v.map_sections_n.iter().map(|(_, n)| *n).max().unwrap_or(0) as u64;
    for &(_, width) in &ldt_trees {
        max_tree_width = max_tree_width.max(width as u64);
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
    let n_trees = 1 + (v.n_stages as u64 + 1) + v.custom_commits.len() as u64;
    let n_trees_fri = ldt_trees.len() as u64;
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

    // The low-degree test's layers follow q: zi and the expression tmps are dead once folding starts
    for &(log_height, width) in &ldt_trees {
        let height = 1u64 << log_height;
        total += height * width as u64;
        if is_gl {
            total += align(get_num_nodes_mt(height, stark_struct.merkle_tree_arity));
        }
    }
    total = total.max(max_total);

    let gb = (total as f64 * 8.0) / (1024.0 * 1024.0 * 1024.0);
    format!("{:.2}", gb)
}
