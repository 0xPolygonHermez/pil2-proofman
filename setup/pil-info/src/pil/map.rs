use crate::expr::expression::{ExprChild, Expression};
use crate::expr::print::PrintCtx;
use crate::types::pilout_info::{SetupResult, SymbolInfo};

const PRINT_NODE_BUDGET: usize = 50_000;

fn fits_print_budget(expressions: &[Expression], idx: usize, budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    let exp = &expressions[idx];
    if exp.op == "exp" {
        if let Some(rid) = exp.id {
            return fits_print_budget(expressions, rid, budget);
        }
        return true;
    }
    for child in &exp.values {
        let ok = match child {
            ExprChild::Id(id) => fits_print_budget(expressions, *id, budget),
            ExprChild::Inline(e) => fits_print_budget_inline(e, expressions, budget),
        };
        if !ok {
            return false;
        }
    }
    true
}

fn fits_print_budget_inline(expr: &Expression, expressions: &[Expression], budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    if expr.op == "exp" {
        if let Some(rid) = expr.id {
            return fits_print_budget(expressions, rid, budget);
        }
        return true;
    }
    for child in &expr.values {
        let ok = match child {
            ExprChild::Id(id) => fits_print_budget(expressions, *id, budget),
            ExprChild::Inline(e) => fits_print_budget_inline(e, expressions, budget),
        };
        if !ok {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// map_symbols
// ---------------------------------------------------------------------------

/// Populate the per-type maps (cmPolsMap, constPolsMap, customCommitsMap,
/// challengesMap, etc.) from the flat symbol list and accumulate
/// `mapSectionsN` counts.
///
/// Mirrors JS `mapSymbols` from `map.js`.
pub fn map_symbols(res: &mut SetupResult) {
    // Clone symbols to avoid borrow conflict: we iterate symbols while
    // mutating other fields of `res` via `add_pol`.
    let symbols = res.symbols.clone();
    for sym in &symbols {
        match sym.sym_type.as_str() {
            "witness" | "fixed" | "custom" => {
                add_pol(res, sym);
            }
            "challenge" => {
                if let Some(id) = sym.id {
                    ensure_vec_len(&mut res.challenges_map, id + 1);
                    res.challenges_map[id] = sym.clone();
                }
            }
            "public" => {
                if let Some(id) = sym.id {
                    ensure_vec_len(&mut res.publics_map, id + 1);
                    res.publics_map[id] = sym.clone();
                }
            }
            "airgroupvalue" => {
                if let Some(id) = sym.id {
                    ensure_vec_len(&mut res.airgroup_values_map, id + 1);
                    res.airgroup_values_map[id] = sym.clone();
                }
            }
            "airvalue" => {
                if let Some(id) = sym.id {
                    ensure_vec_len(&mut res.air_values_map, id + 1);
                    res.air_values_map[id] = sym.clone();
                }
            }
            "proofvalue" => {
                if let Some(id) = sym.id {
                    ensure_vec_len(&mut res.proof_values_map, id + 1);
                    res.proof_values_map[id] = sym.clone();
                }
            }
            _ => {}
        }
    }
}

/// Ensure a Vec<SymbolInfo> has at least `len` entries, padding with defaults.
fn ensure_vec_len(v: &mut Vec<SymbolInfo>, len: usize) {
    while v.len() < len {
        v.push(SymbolInfo::default());
    }
}

/// Insert a fixed/witness/custom polynomial into the appropriate map
/// and increment `mapSectionsN`. Mirrors JS `addPol`.
fn add_pol(res: &mut SetupResult, symbol: &SymbolInfo) {
    let pos = symbol.pol_id.unwrap_or(0);
    let stage = symbol.stage.unwrap_or(0);
    let dim = symbol.dim;

    // Build the map entry from the symbol
    let mut entry = symbol.clone();
    entry.stage = Some(stage);

    // Determine which map to insert into and which section key to increment
    match symbol.sym_type.as_str() {
        "fixed" => {
            ensure_vec_len(&mut res.const_pols_map, pos + 1);
            res.const_pols_map[pos] = entry;
            *res.map_sections_n.entry("const".to_string()).or_insert(0) += dim;
        }
        "witness" => {
            ensure_vec_len(&mut res.cm_pols_map, pos + 1);
            res.cm_pols_map[pos] = entry;
            let key = format!("cm{}", stage);
            *res.map_sections_n.entry(key).or_insert(0) += dim;
        }
        "custom" => {
            let commit_id = symbol.commit_id.unwrap_or(0);
            while res.custom_commits_map.len() <= commit_id {
                res.custom_commits_map.push(Vec::new());
            }
            let cc_map = &mut res.custom_commits_map[commit_id];
            ensure_vec_len(cc_map, pos + 1);
            cc_map[pos] = entry;
            let cc_name = if commit_id < res.custom_commits.len() {
                res.custom_commits[commit_id].name.clone()
            } else {
                format!("custom{}", commit_id)
            };
            let key = format!("{}{}", cc_name, stage);
            *res.map_sections_n.entry(key).or_insert(0) += dim;
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// set_stage_info_symbols
// ---------------------------------------------------------------------------

/// Compute `stagePos` (cumulative dim offset of same-stage earlier columns)
/// and `stageId` for witness/custom symbols.
///
/// Mirrors JS `setStageInfoSymbols`.
pub fn set_stage_info_symbols(res: &mut SetupResult) {
    let q_stage = res.n_stages + 1;

    // We iterate over symbols by index because we need to look up and mutate
    // both the symbol and the corresponding polsMap entry.
    for sym_idx in 0..res.symbols.len() {
        let sym_type = res.symbols[sym_idx].sym_type.clone();
        if sym_type != "witness" && sym_type != "custom" {
            continue;
        }

        let stage = res.symbols[sym_idx].stage.unwrap_or(0);
        let pol_id = res.symbols[sym_idx].pol_id.unwrap_or(0);
        let commit_id = res.symbols[sym_idx].commit_id.unwrap_or(0);

        // Get the relevant pols map to compute stagePos
        let pols_map: &Vec<SymbolInfo> = if sym_type == "witness" {
            &res.cm_pols_map
        } else if commit_id < res.custom_commits_map.len() {
            &res.custom_commits_map[commit_id]
        } else {
            continue;
        };

        // Sum dims of all earlier entries in the same stage
        let stage_pos: usize = pols_map
            .iter()
            .enumerate()
            .filter(|(idx, p)| p.stage == Some(stage) && *idx < pol_id)
            .map(|(_, p)| p.dim)
            .sum();

        // Compute stageId if not already set
        let current_stage_id = res.symbols[sym_idx].stage_id;
        let stage_id = if current_stage_id.is_none() || current_stage_id == Some(0) {
            if stage == q_stage {
                // For Q stage: count of same-stage earlier entries
                pols_map.iter().enumerate().filter(|(idx, p)| p.stage == Some(stage) && *idx < pol_id).count()
            } else {
                // For other stages: find position among same-stage entries
                let sym_name = &res.symbols[sym_idx].name;
                pols_map.iter().filter(|p| p.stage == Some(stage)).position(|p| p.name == *sym_name).unwrap_or(0)
            }
        } else {
            current_stage_id.unwrap_or(0)
        };

        // Update the symbol
        res.symbols[sym_idx].stage_pos = Some(stage_pos);
        res.symbols[sym_idx].stage_id = Some(stage_id);

        // Update the corresponding polsMap entry
        if sym_type == "witness" {
            if pol_id < res.cm_pols_map.len() {
                res.cm_pols_map[pol_id].stage_pos = Some(stage_pos);
                res.cm_pols_map[pol_id].stage_id = Some(stage_id);
            }
        } else if sym_type == "custom"
            && commit_id < res.custom_commits_map.len()
            && pol_id < res.custom_commits_map[commit_id].len()
        {
            res.custom_commits_map[commit_id][pol_id].stage_pos = Some(stage_pos);
            res.custom_commits_map[commit_id][pol_id].stage_id = Some(stage_id);
        }
    }
}

// ---------------------------------------------------------------------------
// map (main entry point)
// ---------------------------------------------------------------------------

/// Main column mapping function. Populates per-type maps, computes section
/// counts, marks imPol constraints, and builds imPolsInfo.
///
/// Mirrors the default export `map(res, symbols, expressions, constraints, options)`
/// from `map.js`.
pub fn map(res: &mut SetupResult, recursion: bool) {
    // Populate per-type maps and mapSectionsN
    map_symbols(res);
    set_stage_info_symbols(res);

    let im_pol_marker = format!("{}.ImPol", res.name);

    // Collect constraint indices that are ImPol, along with their expression IDs
    let im_pol_constraints: Vec<(usize, usize)> = res
        .constraints
        .iter()
        .enumerate()
        .filter(|(_, c)| c.line.as_deref() == Some(&im_pol_marker))
        .map(|(i, c)| (i, c.e))
        .collect();

    // Mark imPol constraints
    for &(ci, _) in &im_pol_constraints {
        res.constraints[ci].im_pol = true;
    }

    // Use printExpressions for ImPol constraint lines in non-recursion mode.
    // Skip prints for expressions that exceed PRINT_NODE_BUDGET — they're for
    // human-readable stats output only, and a runaway walk stalls setup.
    if !recursion && !im_pol_constraints.is_empty() && !res.expressions.is_empty() {
        let mut expressions = std::mem::take(&mut res.expressions);
        {
            let ctx = PrintCtx {
                cm_pols_map: &res.cm_pols_map,
                const_pols_map: &res.const_pols_map,
                custom_commits_map: &res.custom_commits_map,
                publics_map: &res.publics_map,
                challenges_map: &res.challenges_map,
                air_values_map: &res.air_values_map,
                airgroup_values_map: &res.airgroup_values_map,
                proof_values_map: &res.proof_values_map,
            };
            for &(ci, exp_id) in &im_pol_constraints {
                let mut budget = PRINT_NODE_BUDGET;
                let line = if fits_print_budget(&expressions, exp_id, &mut budget) {
                    crate::expr::print::print_expression_no_cache(&ctx, &mut expressions, exp_id, true)
                } else {
                    format!("<im_pol constraint expression too large to print: >{} nodes>", PRINT_NODE_BUDGET)
                };
                res.constraints[ci].line = Some(line);
            }
        }
        res.expressions = expressions;
    } else {
        // In recursion mode or no ImPol constraints, set empty lines
        for &(ci, _) in &im_pol_constraints {
            res.constraints[ci].line = Some(String::new());
        }
    }

    // Append " == 0" to ALL constraint lines
    for c in res.constraints.iter_mut() {
        if let Some(ref mut line) = c.line {
            line.push_str(" == 0");
        } else {
            c.line = Some(" == 0".to_string());
        }
    }

    // Build imPolsInfo: collect intermediate polynomial expression strings.
    // Use printExpressions for imPol expression strings.
    let mut base_field_info: Vec<String> = Vec::new();
    let mut extended_field_info: Vec<String> = Vec::new();

    let im_pols: Vec<(usize, usize)> =
        res.cm_pols_map.iter().filter(|p| p.im_pol).map(|p| (p.dim, p.exp_id.unwrap_or(0))).collect();

    if !recursion && !im_pols.is_empty() {
        let mut expressions = std::mem::take(&mut res.expressions);
        {
            let ctx = PrintCtx {
                cm_pols_map: &res.cm_pols_map,
                const_pols_map: &res.const_pols_map,
                custom_commits_map: &res.custom_commits_map,
                publics_map: &res.publics_map,
                challenges_map: &res.challenges_map,
                air_values_map: &res.air_values_map,
                airgroup_values_map: &res.airgroup_values_map,
                proof_values_map: &res.proof_values_map,
            };
            for &(dim, exp_id) in &im_pols {
                let mut budget = PRINT_NODE_BUDGET;
                let im_pol_expr = if fits_print_budget(&expressions, exp_id, &mut budget) {
                    crate::expr::print::print_expression_no_cache(&ctx, &mut expressions, exp_id, false)
                } else {
                    format!("<im_pol expression too large to print: >{} nodes>", PRINT_NODE_BUDGET)
                };
                if dim == 1 {
                    base_field_info.push(im_pol_expr);
                } else {
                    extended_field_info.push(im_pol_expr);
                }
            }
        }
        res.expressions = expressions;
    } else if !recursion {
        // No imPols but not recursion
        for &(dim, _) in &im_pols {
            if dim == 1 {
                base_field_info.push(String::new());
            } else {
                extended_field_info.push(String::new());
            }
        }
    }

    res.im_pols_info = (base_field_info, extended_field_info);

    // nCommitmentsStage1: count of stage-1 witness columns that are NOT imPols
    res.n_commitments_stage1 =
        res.cm_pols_map.iter().filter(|p| p.stage == Some(1) && !p.name.ends_with(".ImPol")).count();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pilout_info::{ConstraintInfo, CustomCommitInfo, SetupResult, SymbolInfo};
    use indexmap::IndexMap;

    /// Helper to create a minimal SetupResult for testing.
    fn make_setup_result(name: &str, n_stages: usize, symbols: Vec<SymbolInfo>) -> SetupResult {
        let mut map_sections_n = IndexMap::new();
        map_sections_n.insert("const".to_string(), 0);
        for s in 1..=n_stages + 1 {
            map_sections_n.insert(format!("cm{}", s), 0);
        }

        SetupResult {
            name: name.to_string(),
            air_id: 0,
            airgroup_id: 0,
            pil_power: 10,
            n_stages,
            n_constants: 0,
            n_publics: 0,
            n_commitments: 0,
            cm_pols_map: Vec::new(),
            const_pols_map: Vec::new(),
            challenges_map: Vec::new(),
            publics_map: Vec::new(),
            proof_values_map: Vec::new(),
            airgroup_values_map: Vec::new(),
            air_values_map: Vec::new(),
            map_sections_n,
            custom_commits: Vec::new(),
            custom_commits_map: Vec::new(),
            air_group_values: Vec::new(),
            expressions: Vec::new(),
            constraints: Vec::new(),
            symbols,
            hints: Vec::new(),
            n_commitments_stage1: 0,
            im_pols_info: (Vec::new(), Vec::new()),
            opening_points: Vec::new(),
        }
    }

    fn make_witness_symbol(name: &str, stage: usize, pol_id: usize, dim: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "witness".to_string(),
            stage: Some(stage),
            dim,
            pol_id: Some(pol_id),
            stage_id: Some(0),
            ..Default::default()
        }
    }

    fn make_fixed_symbol(name: &str, pol_id: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "fixed".to_string(),
            stage: Some(0),
            dim: 1,
            pol_id: Some(pol_id),
            stage_id: Some(pol_id),
            ..Default::default()
        }
    }

    fn make_challenge_symbol(name: &str, stage: usize, id: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "challenge".to_string(),
            stage: Some(stage),
            dim: 3,
            id: Some(id),
            stage_id: Some(0),
            ..Default::default()
        }
    }

    #[test]
    fn test_map_symbols_witness() {
        let symbols = vec![make_witness_symbol("a", 1, 0, 1), make_witness_symbol("b", 1, 1, 1)];
        let mut res = make_setup_result("test", 1, symbols);

        map_symbols(&mut res);

        assert_eq!(res.cm_pols_map.len(), 2);
        assert_eq!(res.cm_pols_map[0].name, "a");
        assert_eq!(res.cm_pols_map[1].name, "b");
        assert_eq!(res.map_sections_n["cm1"], 2);
    }

    #[test]
    fn test_map_symbols_fixed() {
        let symbols = vec![make_fixed_symbol("f0", 0), make_fixed_symbol("f1", 1)];
        let mut res = make_setup_result("test", 1, symbols);

        map_symbols(&mut res);

        assert_eq!(res.const_pols_map.len(), 2);
        assert_eq!(res.const_pols_map[0].name, "f0");
        assert_eq!(res.map_sections_n["const"], 2);
    }

    #[test]
    fn test_map_symbols_challenge() {
        let symbols = vec![make_challenge_symbol("vc", 2, 0)];
        let mut res = make_setup_result("test", 1, symbols);

        map_symbols(&mut res);

        assert_eq!(res.challenges_map.len(), 1);
        assert_eq!(res.challenges_map[0].name, "vc");
    }

    #[test]
    fn test_map_sections_n_multi_stage() {
        let symbols = vec![
            make_witness_symbol("a", 1, 0, 1),
            make_witness_symbol("b", 1, 1, 3), // dim=3 (extended)
            make_witness_symbol("c", 2, 2, 3),
        ];
        let mut res = make_setup_result("test", 2, symbols);

        map_symbols(&mut res);

        assert_eq!(res.map_sections_n["cm1"], 4); // 1 + 3
        assert_eq!(res.map_sections_n["cm2"], 3);
    }

    #[test]
    fn test_set_stage_info_symbols() {
        let symbols = vec![make_witness_symbol("a", 1, 0, 1), make_witness_symbol("b", 1, 1, 3)];
        let mut res = make_setup_result("test", 1, symbols);

        map_symbols(&mut res);
        set_stage_info_symbols(&mut res);

        // First symbol in stage 1 should have stagePos=0
        assert_eq!(res.cm_pols_map[0].stage_pos, Some(0));
        // Second symbol in stage 1 should have stagePos=1 (dim of first)
        assert_eq!(res.cm_pols_map[1].stage_pos, Some(1));
    }

    #[test]
    fn test_map_marks_im_pol_constraints() {
        let symbols = vec![make_witness_symbol("a", 1, 0, 1)];
        let mut res = make_setup_result("test", 1, symbols);
        res.constraints.push(ConstraintInfo {
            boundary: "everyRow".to_string(),
            e: 0,
            line: Some("test.ImPol".to_string()),
            offset_min: None,
            offset_max: None,
            stage: None,
            im_pol: false,
        });
        res.constraints.push(ConstraintInfo {
            boundary: "everyRow".to_string(),
            e: 1,
            line: Some("some other constraint".to_string()),
            offset_min: None,
            offset_max: None,
            stage: None,
            im_pol: false,
        });

        map(&mut res, false);

        assert!(res.constraints[0].im_pol);
        assert!(!res.constraints[1].im_pol);
        // Both should end with " == 0"
        assert!(res.constraints[0].line.as_ref().unwrap().ends_with(" == 0"));
        assert!(res.constraints[1].line.as_ref().unwrap().ends_with(" == 0"));
    }

    #[test]
    fn test_n_commitments_stage1() {
        let symbols = vec![
            make_witness_symbol("a", 1, 0, 1),
            make_witness_symbol("b", 1, 1, 1),
            SymbolInfo {
                name: "test.ImPol".to_string(),
                sym_type: "witness".to_string(),
                stage: Some(1),
                dim: 1,
                pol_id: Some(2),
                stage_id: Some(2),
                ..Default::default()
            },
        ];
        let mut res = make_setup_result("test", 1, symbols);

        map(&mut res, false);

        // Only "a" and "b" are non-imPol stage-1 witnesses
        assert_eq!(res.n_commitments_stage1, 2);
    }

    #[test]
    fn test_map_with_custom_commits() {
        let symbols = vec![SymbolInfo {
            name: "cc_col".to_string(),
            sym_type: "custom".to_string(),
            stage: Some(0),
            dim: 1,
            pol_id: Some(0),
            stage_id: Some(0),
            commit_id: Some(0),
            ..Default::default()
        }];
        let mut res = make_setup_result("test", 1, symbols);
        res.custom_commits.push(CustomCommitInfo {
            name: "MyCC".to_string(),
            stage_widths: vec![1],
            public_values: vec![],
        });
        res.custom_commits_map.push(Vec::new());
        res.map_sections_n.insert("MyCC0".to_string(), 0);

        map_symbols(&mut res);

        assert_eq!(res.custom_commits_map[0].len(), 1);
        assert_eq!(res.custom_commits_map[0][0].name, "cc_col");
        assert_eq!(res.map_sections_n["MyCC0"], 1);
    }

    #[test]
    fn test_map_empty_symbols() {
        let symbols = Vec::new();
        let mut res = make_setup_result("test", 1, symbols);

        map(&mut res, false);

        assert!(res.cm_pols_map.is_empty());
        assert!(res.const_pols_map.is_empty());
        assert_eq!(res.n_commitments_stage1, 0);
    }
}
