use crate::{add_intermediate_pols::ExpressionOps, witness_calculator::Symbol};
use crate::helpers::get_exp_dim;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::convert::TryInto;

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
    let stage = (res.n_stages + 3) as u32;

    // Create challenge symbols
    let vf1_id = symbols.iter().filter(|s| s.r#type == 8 && s.stage.unwrap_or(0) < stage).count() as u32;
    let vf2_id = vf1_id + 1;

    let vf1_symbol = Symbol {
        name: "std_vf1".to_string(),
        id: vf1_id,
        stage: Some(stage),
        r#type: 8, // Challenge type
        dim: 3,
        air_group_id: Some(0),
        air_id: None,
        lengths: vec![],
        commit_id: None,
        debug_line: None,
    };

    let vf2_symbol = Symbol {
        name: "std_vf2".to_string(),
        id: vf2_id,
        stage: Some(stage),
        r#type: 8, // Challenge type
        dim: 3,
        air_group_id: Some(0),
        air_id: None,
        lengths: vec![],
        commit_id: None,
        debug_line: None,
    };

    symbols.push(vf1_symbol.clone());
    symbols.push(vf2_symbol.clone());

    res.challenges_map.insert(vf1_symbol.id as usize, vf1_symbol.clone());
    res.challenges_map.insert(vf2_symbol.id as usize, vf2_symbol.clone());

    let vf1 = e.challenge("std_vf1", stage as usize, 3, 0, vf1_id as usize);
    let vf2 = e.challenge("std_vf2", stage as usize, 3, 1, vf2_id as usize);

    let mut fri_exps: HashMap<usize, Expression> = HashMap::new();

    // Process events
    for ev in &res.ev_map {
        let symbol = match ev.event_type.as_str() {
            "const" => symbols.iter().find(|s| s.id == ev.id as u32 && s.r#type == 1),
            "cm" => symbols.iter().find(|s| s.id == ev.id as u32 && s.r#type == 3),
            "custom" => symbols
                .iter()
                .find(|s| s.id == ev.id as u32 && s.r#type == 10 && s.commit_id == ev.commit_id.map(|x| x as u32)),
            _ => None,
        };

        let symbol = symbol.expect("Symbol not found");

        let expr = match ev.event_type.as_str() {
            "const" => e.const_(ev.id, 0, symbol.stage.unwrap().try_into().unwrap(), symbol.dim.try_into().unwrap()),
            "cm" => e.cm(ev.id, 0, Some(symbol.stage.unwrap().try_into().unwrap()), symbol.dim.try_into().unwrap()),
            "custom" => e.custom(
                ev.id,
                0,
                Some(symbol.stage.unwrap().try_into().unwrap()),
                symbol.dim.try_into().unwrap(),
                symbol.commit_id.unwrap_or(0).try_into().unwrap(),
            ),
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

    let dim = get_exp_dim(expressions, fri_exp_id);
    expressions[fri_exp_id]["dim"] = json!(dim);
    expressions[fri_exp_id]["stage"] = json!(res.n_stages + 2);
}
