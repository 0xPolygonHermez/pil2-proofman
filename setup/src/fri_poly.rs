use crate::add_intermediate_pols::ExpressionOps;
use crate::helpers::get_exp_dim;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub symbol_type: String, // "challenge", "fixed", "witness", "custom"
    pub name: String,
    pub stage: usize,
    pub dim: usize,
    pub stage_id: usize,
    pub id: usize,
    pub pol_id: Option<usize>,
    pub commit_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Res {
    pub n_stages: usize,
    pub challenges_map: HashMap<usize, Symbol>,
    pub ev_map: Vec<Event>,
    pub opening_points: Vec<usize>,
    pub fri_exp_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub event_type: String, // "const", "cm", "custom"
    pub id: usize,
    pub commit_id: Option<usize>,
    pub prime: usize,
}

/// **Expressions are stored as JSON `Value` objects**
pub type Expression = Value;

pub fn generate_fri_polynomial(res: &mut Res, symbols: &mut Vec<Symbol>, expressions: &mut Vec<Expression>) {
    let e = ExpressionOps::new(res.n_stages + 3, 3);
    let stage = res.n_stages + 3;

    // Create challenge symbols
    let vf1_id = symbols.iter().filter(|s| s.symbol_type == "challenge" && s.stage < stage).count();
    let vf2_id = vf1_id + 1;

    let vf1_symbol = Symbol {
        symbol_type: "challenge".to_string(),
        name: "std_vf1".to_string(),
        stage,
        dim: 3,
        stage_id: 0,
        id: vf1_id,
        pol_id: None,
        commit_id: None,
    };

    let vf2_symbol = Symbol {
        symbol_type: "challenge".to_string(),
        name: "std_vf2".to_string(),
        stage,
        dim: 3,
        stage_id: 1,
        id: vf2_id,
        pol_id: None,
        commit_id: None,
    };

    symbols.push(vf1_symbol.clone());
    symbols.push(vf2_symbol.clone());

    res.challenges_map.insert(vf1_symbol.id, vf1_symbol.clone());
    res.challenges_map.insert(vf2_symbol.id, vf2_symbol.clone());

    let vf1 = e.challenge("std_vf1", stage, 3, 0, vf1_id);
    let vf2 = e.challenge("std_vf2", stage, 3, 1, vf2_id);

    let mut fri_exps: HashMap<usize, Expression> = HashMap::new();

    // Process events
    for ev in &res.ev_map {
        let symbol = match ev.event_type.as_str() {
            "const" => symbols.iter().find(|s| s.pol_id == Some(ev.id) && s.symbol_type == "fixed"),
            "cm" => symbols.iter().find(|s| s.pol_id == Some(ev.id) && s.symbol_type == "witness"),
            "custom" => symbols
                .iter()
                .find(|s| s.pol_id == Some(ev.id) && s.symbol_type == "custom" && s.commit_id == ev.commit_id),
            _ => None,
        };

        let symbol = symbol.expect("Symbol not found");

        // **FIX:** Use dynamic dispatch for `generate_expr`
        let expr = match ev.event_type.as_str() {
            "const" => e.const_(ev.id, 0, symbol.stage, symbol.dim),
            "cm" => e.cm(ev.id, 0, Some(symbol.stage), symbol.dim),
            "custom" => e.custom(ev.id, 0, Some(symbol.stage), symbol.dim, symbol.commit_id.unwrap_or(0)),
            _ => panic!("Invalid event type: {}", ev.event_type),
        };

        fri_exps
            .entry(ev.prime)
            .and_modify(|existing| {
                *existing = e.add(e.mul(existing.clone(), vf2.clone()), e.sub(expr.clone(), e.eval(ev.id, 3)))
            })
            .or_insert_with(|| e.sub(expr, e.eval(ev.id, 3)));
    }

    let mut fri_exp: Option<Expression> = None;

    for (opening, expr) in fri_exps.iter_mut() {
        let index = res.opening_points.iter().position(|&p| p == *opening).expect("Opening point not found");

        *expr = e.mul(expr.clone(), e.x_div_x_sub_xi(*opening, index));

        fri_exp = match fri_exp {
            Some(existing) => Some(e.add(e.mul(vf1.clone(), existing), expr.clone())),
            None => Some(expr.clone()),
        };
    }

    let fri_exp_id = expressions.len();
    res.fri_exp_id = fri_exp_id;
    expressions.push(fri_exp.expect("FRI Expression should not be None"));

    // **FIX:** Ensure expressions are `Vec<Value>` for `get_exp_dim`
    let dim = get_exp_dim(expressions, fri_exp_id);
    expressions[fri_exp_id]["dim"] = json!(dim);
    expressions[fri_exp_id]["stage"] = json!(res.n_stages + 2);
}
