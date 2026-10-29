use pil2_pilout::pilout as pb;

use crate::cfg::PilInfoCfg;
use crate::error::Result;
use crate::pil::constraint_poly::{
    generate_constraint_polynomial, normalize_constraint_domains, Boundary, ConstraintPolyResult,
};
use crate::expr::expression::Expression;
use crate::expr::helpers::add_info_expressions;
use crate::types::pilout_info::{ConstraintInfo, HintInfo, SetupResult, SymbolInfo};

/// Options controlling the preparePil flow.
#[derive(Debug, Clone, Default)]
pub struct PrepareOptions {
    /// When true, skip code generation here, and the starkStruct validation in the STARK setup.
    pub debug: bool,
    /// When true, enable intermediate polynomial batching by stage.
    pub im_pols_stages: bool,
    /// A basic air: the GPU prover commits its stages in place, which the prover memory
    /// estimate must know. False for recursion, final and SNARK-wrapper airs.
    pub inplace_stage_commit: bool,
}

/// Aggregate result of the preparePil pipeline.
#[derive(Debug)]
pub struct PreparePilResult {
    /// The setup result with populated maps and metadata.
    pub setup: SetupResult,
    /// The flat expression arena (may have been extended with constraint/FRI nodes).
    pub expressions: Vec<Expression>,
    /// Constraints with their stages populated.
    pub constraints: Vec<ConstraintInfo>,
    /// Symbols including injected challenge symbols.
    pub symbols: Vec<SymbolInfo>,
    /// Formatted hints.
    pub hints: Vec<HintInfo>,
    /// Boundary definitions.
    pub boundaries: Vec<Boundary>,
    /// Constraint polynomial info.
    pub constraint_poly: ConstraintPolyResult,
}

/// Prepare pilout info for a single air, assembling all derived data.
///
/// This function:
/// 1. Calls `get_pilout_info` to extract raw pilout data
/// 2. Sets up mapSectionsN for each stage
/// 3. Calls add_info_expressions on all constraints and remaining expressions
/// 4. Computes opening points
/// 5. Calls generate_constraint_polynomial
///
/// Every step computes over `cfg`'s field and opening; `pil_info` must then run with the same `cfg`.
/// Fails on what `get_pilout_info` refuses and on what the passes cannot process.
pub fn prepare_pil(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    cfg: &PilInfoCfg,
) -> Result<PreparePilResult> {
    let (field, opening) = (&cfg.field, cfg.opening);
    let mut setup = crate::types::pilout_info::get_pilout_info(pilout, airgroup_id, air_id, field)?;

    // Set all expression stages to 1 (mirrors JS: pil.expressions[i].stage = 1)
    for expr in setup.expressions.iter_mut() {
        if expr.op != "__placeholder__" {
            expr.stage = 1;
        }
    }

    // Initialize mapSectionsN for each stage
    for s in 1..=(setup.n_stages + 1) {
        setup.map_sections_n.insert(format!("cm{}", s), 0);
    }

    let mut expressions = std::mem::take(&mut setup.expressions);
    let mut constraints = std::mem::take(&mut setup.constraints);
    let mut symbols = std::mem::take(&mut setup.symbols);
    let hints = std::mem::take(&mut setup.hints);

    let n_bits = setup.pil_power;
    normalize_constraint_domains(&mut constraints, n_bits, opening, |i| domain_name(pilout, airgroup_id, air_id, i))?;

    // Run add_info_expressions on all constraints
    for i in 0..constraints.len() {
        add_info_expressions(&mut expressions, constraints[i].e, field)?;
        constraints[i].stage = Some(expressions[constraints[i].e].stage);
    }

    // Run add_info_expressions on remaining expressions that have not been processed.
    for i in 0..expressions.len() {
        if expressions[i].op != "__placeholder__" {
            add_info_expressions(&mut expressions, i, field)?;
        }
    }

    // Compute opening points
    let mut opening_points_set: Vec<i64> = vec![0];
    for c in &constraints {
        let offsets = &expressions[c.e].rows_offsets;
        for &offset in offsets {
            if !opening_points_set.contains(&offset) {
                opening_points_set.push(offset);
            }
        }
    }
    opening_points_set.sort();

    // Initialize boundaries
    let mut boundaries = vec![Boundary::EveryRow];

    // Generate constraint polynomial
    let constraint_poly = generate_constraint_polynomial(
        setup.n_stages,
        n_bits,
        &mut expressions,
        &mut symbols,
        &constraints,
        &mut boundaries,
        field,
    )?;

    Ok(PreparePilResult { setup, expressions, constraints, symbols, hints, boundaries, constraint_poly })
}

/// The name of the domain of constraint `i` of the air: its DOMAIN symbol, else `air:<name> domain#<idx>`.
fn domain_name(pilout: &pb::PilOut, airgroup_id: usize, air_id: usize, i: usize) -> String {
    use pb::constraint::Constraint as C;
    let air = &pilout.air_groups[airgroup_id].airs[air_id];
    let idx = match air.constraints.get(i).and_then(|c| c.constraint.as_ref()) {
        Some(C::DomainRows(r)) => r.domain_idx,
        Some(C::ComplementDomainRows(r)) => r.domain_idx,
        _ => return format!("air:{} constraint#{i}", air.name.as_deref().unwrap_or_default()),
    };
    let symbol = pilout.symbols.iter().find(|s| {
        s.r#type == pb::SymbolType::Domain as i32
            && s.air_group_id == Some(airgroup_id as u32)
            && s.air_id == Some(air_id as u32)
            && s.id == idx
    });
    match symbol {
        Some(s) => s.name.clone(),
        None => format!("air:{} domain#{idx}", air.name.as_deref().unwrap_or_default()),
    }
}
