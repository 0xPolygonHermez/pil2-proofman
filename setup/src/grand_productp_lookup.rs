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
    let mut n_cm2 = pil.get("nCm2").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut n_cm3 = pil.get("nCm3").and_then(|v| v.as_u64()).unwrap_or(0);

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

            // ✅ Witness Commitments
            pu_ctx.insert("h1Id", json!(n_commitments));
            n_commitments += 1;
            pu_ctx.insert("h2Id", json!(n_commitments));
            n_commitments += 1;
            pu_ctx.insert("zId", json!(n_commitments));
            n_commitments += 1;

            let mut h1 = e.cm(pu_ctx["h1Id"].as_u64().unwrap() as usize, 0, Some(stage1), t_dim);
            let mut h2 = e.cm(pu_ctx["h2Id"].as_u64().unwrap() as usize, 0, Some(stage1), t_dim);
            let mut z = e.cm(pu_ctx["zId"].as_u64().unwrap() as usize, 0, Some(stage2), dim);

            h1.as_object_mut().unwrap().insert("stageId".to_string(), json!(n_cm2));
            h2.as_object_mut().unwrap().insert("stageId".to_string(), json!(n_cm2));
            z.as_object_mut().unwrap().insert("stageId".to_string(), json!(n_cm3));

            n_cm2 += 1;
            n_cm3 += 1;

            // ✅ Constraint `c1`
            let c1 = e.mul(
                e.const_(pil["references"]["Global.L1"]["id"].as_u64().unwrap() as usize, 0, 0, 1),
                e.sub(z.clone(), e.number(1.0)),
            );
            expressions_array.push(c1);
            let c1_id = expressions_array.len() - 1;
            expressions_array[c1_id].as_object_mut().unwrap().insert("deg".to_string(), json!(2));

            // ✅ Constraint `c2`
            let zp = e.cm(pu_ctx["zId"].as_u64().unwrap() as usize, 1, Some(stage2), dim);
            let c2 = e.sub(
                e.mul(zp, e.exp(pu_ctx["tExpId"].as_u64().unwrap() as usize, 0, stage2)),
                e.mul(z.clone(), e.exp(pu_ctx["tExpId"].as_u64().unwrap() as usize, 0, stage2)),
            );
            expressions_array.push(c2);
            let c2_id = expressions_array.len() - 1;
            expressions_array[c2_id].as_object_mut().unwrap().insert("deg".to_string(), json!(2));

            hints.push(json!({"name": "h1h2", "fields": [{"name": "referenceH1", "values": [h1]}, {"name": "referenceH2", "values": [h2]}]}));
            hints.push(json!({"name": "gprod", "fields": [{"name": "reference", "values": [z]}]}));
        }
    }

    pil.insert("expressions".to_string(), json!(expressions_array));
    pil.insert("nCommitments".to_string(), json!(n_commitments));
}
