use crate::{add_intermediate_pols::ExpressionOps, helpers::get_exp_dim};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Initializes the standard challenges for Plookup.
pub fn init_challenges_plookup() -> Vec<Value> {
    let stage1 = 2;
    let stage2 = 3;
    let dim = 3;

    vec![
        json!({"name": "std_alpha", "stage": stage1, "dim": dim, "stageId": 0}),
        json!({"name": "std_beta", "stage": stage1, "dim": dim, "stageId": 1}),
        json!({"name": "std_gamma", "stage": stage2, "dim": dim, "stageId": 0}),
        json!({"name": "std_delta", "stage": stage2, "dim": dim, "stageId": 1}),
    ]
}

/// Implements the Grand Product Argument for Plookup.
pub fn grand_product_plookup(
    pil: &mut HashMap<String, Value>,
    symbols: &mut Vec<Value>,
    hints: &mut Vec<Value>,
    airgroup_id: u64,
    air_id: u64,
) {
    let stage1 = 2;
    let stage2 = 3;
    let dim = 3;
    let e = ExpressionOps::new(stage1, dim);

    let find_challenge = |name: &str| -> Value {
        let symbol = symbols.iter().find(|s| s["type"] == "challenge" && s["name"] == name).unwrap();
        e.challenge(
            name,
            symbol["stage"].as_u64().unwrap() as usize,
            symbol["dim"].as_u64().unwrap() as usize,
            symbol["stageId"].as_u64().unwrap() as usize,
            symbol["id"].as_u64().unwrap() as usize,
        )
    };

    let alpha = find_challenge("std_alpha");
    let beta = find_challenge("std_beta");
    let gamma = find_challenge("std_gamma");
    let delta = find_challenge("std_delta");

    let plookup_identities = pil.get("plookupIdentities").cloned().unwrap_or(json!([]));
    let expressions = pil.get("expressions").cloned().unwrap_or(json!([]));
    let mut expressions_array = expressions.as_array().unwrap().clone();
    let mut n_commitments = pil.get("nCommitments").and_then(|v| v.as_u64()).unwrap_or(0);

    if let Some(identity_list) = plookup_identities.as_array() {
        for (i, pi) in identity_list.iter().enumerate() {
            let mut pu_ctx = HashMap::new();
            let mut t_exp = None;

            if let Some(t) = pi.get("t").and_then(|t| t.as_array()) {
                for exp in t {
                    let e_exp = e.exp(exp.as_u64().unwrap() as usize, 0, stage1);
                    t_exp =
                        Some(if let Some(t_exp) = t_exp { e.add(e.mul(alpha.clone(), t_exp), e_exp) } else { e_exp });
                }
            }

            if let Some(sel_t) = pi.get("selT") {
                let sel_t = sel_t.as_u64().unwrap() as usize;
                t_exp = Some(e.mul(e.sub(t_exp.unwrap(), beta.clone()), e.exp(sel_t, 0, stage1)));
                t_exp = Some(e.add(t_exp.unwrap(), beta.clone()));
            }

            pu_ctx.insert("tExpId", json!(expressions_array.len()));

            let mut t_exp_val = json!(t_exp.unwrap());
            t_exp_val.as_object_mut().unwrap().insert("keep".to_string(), json!(true));
            t_exp_val.as_object_mut().unwrap().insert("stage".to_string(), json!(stage1));

            expressions_array.push(t_exp_val);
            let t_dim = get_exp_dim(&expressions_array, pu_ctx["tExpId"].as_u64().unwrap() as usize);

            if let Some(exp) = expressions_array.get_mut(pu_ctx["tExpId"].as_u64().unwrap() as usize) {
                exp.as_object_mut().unwrap().insert("deg".to_string(), json!(t_dim));
            }

            // ✅ Add fExp (same logic as tExp)
            let mut f_exp = None;
            if let Some(f) = pi.get("f").and_then(|f| f.as_array()) {
                for exp in f {
                    let e_exp = e.exp(exp.as_u64().unwrap() as usize, 0, stage1);
                    f_exp =
                        Some(if let Some(f_exp) = f_exp { e.add(e.mul(f_exp, alpha.clone()), e_exp) } else { e_exp });
                }
            }

            if let Some(sel_f) = pi.get("selF") {
                let sel_f = sel_f.as_u64().unwrap() as usize;
                f_exp = Some(e.mul(
                    e.sub(f_exp.unwrap(), e.exp(pu_ctx["tExpId"].as_u64().unwrap() as usize, 0, stage1)),
                    e.exp(sel_f, 0, stage1),
                ));
                f_exp = Some(e.add(f_exp.unwrap(), e.exp(pu_ctx["tExpId"].as_u64().unwrap() as usize, 0, stage1)));
            }

            pu_ctx.insert("fExpId", json!(expressions_array.len()));

            let mut f_exp_val = json!(f_exp.unwrap());
            f_exp_val.as_object_mut().unwrap().insert("keep".to_string(), json!(true));
            f_exp_val.as_object_mut().unwrap().insert("stage".to_string(), json!(stage1));

            expressions_array.push(f_exp_val);
            let f_dim = get_exp_dim(&expressions_array, pu_ctx["fExpId"].as_u64().unwrap() as usize);

            if let Some(exp) = expressions_array.get_mut(pu_ctx["fExpId"].as_u64().unwrap() as usize) {
                exp.as_object_mut().unwrap().insert("deg".to_string(), json!(f_dim));
            }

            // ✅ Add Witness Commitments
            pu_ctx.insert("h1Id", json!(n_commitments));
            n_commitments += 1;
            pu_ctx.insert("h2Id", json!(n_commitments));
            n_commitments += 1;
            pu_ctx.insert("zId", json!(n_commitments));
            n_commitments += 1;

            symbols.push(json!({"type": "witness", "name": format!("Plookup{}.h1", i), "polId": pu_ctx["h1Id"], "stage": stage1, "dim": t_dim.max(f_dim), "airId": air_id, "airgroupId": airgroup_id}));
            symbols.push(json!({"type": "witness", "name": format!("Plookup{}.h2", i), "polId": pu_ctx["h2Id"], "stage": stage1, "dim": t_dim.max(f_dim), "airId": air_id, "airgroupId": airgroup_id}));
            symbols.push(json!({"type": "witness", "name": format!("Plookup{}.z", i), "polId": pu_ctx["zId"], "stage": stage2, "dim": t_dim.max(f_dim), "airId": air_id, "airgroupId": airgroup_id}));
        }
    }

    // ✅ Update `pil`
    pil.insert("expressions".to_string(), json!(expressions_array));
    pil.insert("nCommitments".to_string(), json!(n_commitments));
}
