use pilout::pilout as pb;

use crate::pil::constraint_poly::{generate_constraint_polynomial, Boundary, ConstraintPolyResult};
use crate::expr::expression::Expression;
use crate::expr::helpers::add_info_expressions;
use crate::types::pilout_info::{ConstraintInfo, HintInfo, SetupResult, SymbolInfo};
use crate::types::stark_struct::StarkStruct;

/// Options controlling the preparePil flow.
#[derive(Debug, Clone, Default)]
pub struct PrepareOptions {
    /// When true, skip starkStruct validation and code generation.
    pub debug: bool,
    /// When true, enable intermediate polynomial batching by stage.
    pub im_pols_stages: bool,
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
/// 3. Validates starkStruct.nBits matches pilPower (unless debug mode)
/// 4. Calls add_info_expressions on all constraints and remaining expressions
/// 5. Computes opening points
/// 6. Calls generate_constraint_polynomial
pub fn prepare_pil(
    pilout: &pb::PilOut,
    airgroup_id: usize,
    air_id: usize,
    stark_struct: &StarkStruct,
    options: &PrepareOptions,
) -> PreparePilResult {
    let mut setup = crate::types::pilout_info::get_pilout_info(pilout, airgroup_id, air_id);

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

    // Validate starkStruct
    if !options.debug {
        if stark_struct.n_bits != setup.pil_power as usize {
            panic!(
                "starkStruct and pilfile have degree mismatch (airId: {} airgroupId: {} starkStruct:{} pilfile:{})",
                air_id, airgroup_id, stark_struct.n_bits, setup.pil_power
            );
        }

        if stark_struct.n_bits_ext != stark_struct.steps[0].n_bits {
            panic!(
                "starkStruct.nBitsExt and first step of starkStruct have a mismatch (nBitsExt:{} step0:{})",
                stark_struct.n_bits_ext, stark_struct.steps[0].n_bits
            );
        }
    }

    let mut expressions = std::mem::take(&mut setup.expressions);
    let mut constraints = std::mem::take(&mut setup.constraints);
    let mut symbols = std::mem::take(&mut setup.symbols);
    let hints = std::mem::take(&mut setup.hints);

    // Run add_info_expressions on all constraints
    for i in 0..constraints.len() {
        add_info_expressions(&mut expressions, constraints[i].e);
        constraints[i].stage = Some(expressions[constraints[i].e].stage);
    }

    // Run add_info_expressions on remaining expressions that have not been processed.
    for i in 0..expressions.len() {
        if expressions[i].op != "__placeholder__" {
            add_info_expressions(&mut expressions, i);
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
    let mut boundaries = vec![Boundary { name: "everyRow".to_string(), offset_min: None, offset_max: None }];

    // Generate constraint polynomial
    let constraint_poly =
        generate_constraint_polynomial(setup.n_stages, &mut expressions, &mut symbols, &constraints, &mut boundaries);

    PreparePilResult { setup, expressions, constraints, symbols, hints, boundaries, constraint_poly }
}
