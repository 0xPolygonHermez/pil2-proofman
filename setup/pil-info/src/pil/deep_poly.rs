use std::collections::HashSet;

use crate::cfg::FieldCfg;
use crate::error::{PilInfoError, Result};
use crate::expr::helpers::EvMapItem;
use crate::types::pilout_info::SymbolInfo;

/// Register the DEEP polynomial's challenges `std_vf1`/`std_vf2` at stage `n_stages + 3` and return
/// their challengesMap entries, which go after every earlier challenge. The polynomial itself is
/// built natively from starkinfo by its consumers.
pub fn register_deep_challenges(
    n_stages: usize,
    symbols: &mut Vec<SymbolInfo>,
    field: &FieldCfg,
) -> Vec<ChallengeMapEntry> {
    let stage = n_stages + 3;
    let vf1_id = symbols.iter().filter(|s| s.sym_type == "challenge" && s.stage.is_some_and(|st| st < stage)).count();
    let mut entries = Vec::with_capacity(2);
    for (stage_id, name) in ["std_vf1", "std_vf2"].into_iter().enumerate() {
        let symbol = SymbolInfo {
            sym_type: "challenge".to_string(),
            name: name.to_string(),
            stage: Some(stage),
            dim: field.ext_dim(),
            stage_id: Some(stage_id),
            id: Some(vf1_id + stage_id),
            pol_id: None,
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        };
        entries.push(ChallengeMapEntry { name: name.to_string(), stage, dim: symbol.dim, stage_id });
        symbols.push(symbol);
    }
    entries
}

/// Check the evaluations the DEEP polynomial combines: each names a column with a symbol, no column
/// is evaluated twice at one opening, and there is at least one.
pub fn check_unique_evaluations(symbols: &[SymbolInfo], ev_map: &[EvMapItem]) -> Result<()> {
    if ev_map.is_empty() {
        return Err(PilInfoError::DeepWithoutEvaluations);
    }
    let mut seen = HashSet::new();
    for ev in ev_map {
        find_symbol_for_ev(symbols, ev)?;
        if !seen.insert((ev.entry_type.as_str(), ev.commit_id, ev.id, ev.prime)) {
            return Err(PilInfoError::DuplicateEvaluation {
                entry_type: ev.entry_type.clone(),
                id: ev.id,
                opening: ev.prime,
            });
        }
    }
    Ok(())
}

/// The DEEP polynomial's degree in (vf1, vf2), for the security calculator:
/// `max(evMap.len(), nOpenings + nDistinctPols - 1)`, counting only openings with evaluations.
pub fn fri_batch_size(ev_map: &[EvMapItem], opening_points: &[i64]) -> usize {
    let n_open = opening_points.iter().filter(|&&o| ev_map.iter().any(|ev| ev.prime == o)).count();
    let n_pols = ev_map.iter().map(|ev| (ev.entry_type.as_str(), ev.commit_id, ev.id)).collect::<HashSet<_>>().len();
    ev_map.len().max((n_open + n_pols).saturating_sub(1))
}

/// Entry in the challenges map (matches JSON output format).
#[derive(Debug, Clone)]
pub struct ChallengeMapEntry {
    pub name: String,
    pub stage: usize,
    pub dim: usize,
    pub stage_id: usize,
}

/// Find the symbol matching an evaluation map entry.
fn find_symbol_for_ev(symbols: &[SymbolInfo], ev: &EvMapItem) -> Result<SymbolInfo> {
    let sym_type_target = match ev.entry_type.as_str() {
        "const" => "fixed",
        "cm" => "witness",
        "custom" => "custom",
        other => return Err(PilInfoError::UnknownEvaluation(other.to_string())),
    };

    symbols
        .iter()
        .find(|s| {
            s.pol_id == Some(ev.id)
                && s.sym_type == sym_type_target
                && (ev.entry_type != "custom" || s.commit_id == ev.commit_id)
        })
        .cloned()
        .ok_or_else(|| PilInfoError::NoSymbolForEvaluation { entry_type: ev.entry_type.clone(), id: ev.id })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_witness_symbol(name: &str, pol_id: usize, stage: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "witness".to_string(),
            stage: Some(stage),
            dim: 1,
            pol_id: Some(pol_id),
            stage_id: Some(0),
            id: Some(pol_id),
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        }
    }

    fn make_fixed_symbol(name: &str, pol_id: usize) -> SymbolInfo {
        SymbolInfo {
            name: name.to_string(),
            sym_type: "fixed".to_string(),
            stage: Some(0),
            dim: 1,
            pol_id: Some(pol_id),
            stage_id: Some(0),
            id: Some(pol_id),
            air_id: None,
            airgroup_id: None,
            commit_id: None,
            lengths: None,
            idx: None,
            stage_pos: None,
            im_pol: false,
            exp_id: None,
        }
    }

    fn cm(id: usize, prime: i64) -> EvMapItem {
        EvMapItem { entry_type: "cm".to_string(), id, prime, commit_id: None }
    }

    #[test]
    fn deep_challenges_follow_the_earlier_ones() {
        let mut symbols = vec![make_witness_symbol("w0", 0, 1), make_fixed_symbol("f0", 0)];
        let challenges_map = register_deep_challenges(1, &mut symbols, &FieldCfg::goldilocks());
        let got: Vec<_> = challenges_map.iter().map(|c| (c.name.as_str(), c.stage, c.dim, c.stage_id)).collect();
        assert_eq!(got, [("std_vf1", 4, 3, 0), ("std_vf2", 4, 3, 1)]);
        assert_eq!(symbols.iter().filter(|s| s.sym_type == "challenge").count(), 2);
    }

    #[test]
    fn fri_batch_size_is_the_power_batching_degree() {
        let fixed = EvMapItem { entry_type: "const".to_string(), id: 0, prime: 0, commit_id: None };
        assert_eq!(fri_batch_size(&[cm(0, 0), fixed], &[0]), 2);
        // Disjoint openings: degree 1 + 1 exceeds the 2-entry ev_map's power-batching degree.
        assert_eq!(fri_batch_size(&[cm(0, 0), cm(1, 1)], &[0, 1]), 3);
        // An opening without evaluations does not count.
        assert_eq!(fri_batch_size(&[cm(0, 0), cm(1, 0)], &[-1, 0, 1]), 2);
    }

    fn check_err(ev_map: &[EvMapItem]) -> PilInfoError {
        check_unique_evaluations(&[make_witness_symbol("w0", 0, 1)], ev_map).expect_err("the evaluations were accepted")
    }

    #[test]
    fn a_duplicate_evaluation_is_an_error() {
        assert!(matches!(
            check_err(&[cm(0, 0), cm(0, 0)]),
            PilInfoError::DuplicateEvaluation { entry_type, id: 0, opening: 0 } if entry_type == "cm"
        ));
        check_unique_evaluations(&[make_witness_symbol("w0", 0, 1)], &[cm(0, 0), cm(0, 1)]).unwrap();
    }

    /// Were `expect("At least one opening point required")` and `panic!("Unknown ev type")`.
    #[test]
    fn a_deep_polynomial_without_evaluations_or_of_an_unknown_one_is_an_error() {
        assert!(matches!(check_err(&[]), PilInfoError::DeepWithoutEvaluations));
        let ev = EvMapItem { entry_type: "tmp".to_string(), id: 0, prime: 0, commit_id: None };
        assert!(matches!(check_err(&[ev]), PilInfoError::UnknownEvaluation(t) if t == "tmp"));
    }
}
