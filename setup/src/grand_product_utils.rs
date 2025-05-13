use crate::{add_intermediate_pols::ExpressionOps, helpers::get_exp_dim, utils::get_ks};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Initializes the standard challenges for Connection.
pub fn init_challenges_connection() -> Vec<Value> {
    let dim = 3;
    vec![
        json!({"name": "std_gamma", "stage": 2, "dim": dim, "stageId": 0}),
        json!({"name": "std_delta", "stage": 2, "dim": dim, "stageId": 1}),
    ]
}

/// Implements the Grand Product Argument for Connection.
pub fn grand_product_connection(
    pil: &mut HashMap<String, Value>,
    symbols: &mut Vec<Value>,
    hints: &mut Vec<Value>,
    airgroup_id: u64,
    air_id: u64,
    fr_mul: fn(f64, f64) -> f64, // Accepts a function for field arithmetic
    k: f64,                      // Initial field element for `get_ks`
) {
    let stage = 2;
    let dim = 3;
    let e = ExpressionOps::new(stage, dim);

    let find_challenge = |stage_id: u64| -> Value {
        let symbol = symbols
            .iter()
            .find(|s| s["type"] == "challenge" && s["stage"] == stage && s["stageId"] == stage_id)
            .unwrap();
        e.challenge(
            symbol["name"].as_str().unwrap(),
            symbol["stage"].as_u64().unwrap() as usize,
            symbol["dim"].as_u64().unwrap() as usize,
            symbol["stageId"].as_u64().unwrap() as usize,
            symbol["id"].as_u64().unwrap() as usize,
        )
    };

    let gamma = find_challenge(0);
    let delta = find_challenge(1);

    let connection_identities = pil.get("connectionIdentities").cloned().unwrap_or(json!([]));
    let expressions = pil.get("expressions").cloned().unwrap_or(json!([]));
    let mut expressions_array = expressions.as_array().unwrap().clone();
    let mut n_commitments = pil.get("nCommitments").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut n_cm2 = pil.get("nCm2").and_then(|v| v.as_u64()).unwrap_or(0);

    if let Some(identity_list) = connection_identities.as_array() {
        for (i, ci) in identity_list.iter().enumerate() {
            let mut ci_ctx = HashMap::new();

            // Allocate `zId`
            ci_ctx.insert("zId", json!(n_commitments));
            n_commitments += 1;

            // Compute numerator expression
            let num_exp = e.add(
                e.add(e.exp(ci["pols"][0].as_u64().unwrap() as usize, 0, stage), e.mul(delta.clone(), e.x())),
                gamma.clone(),
            );

            // Compute denominator expression
            let den_exp = e.add(
                e.add(
                    e.exp(ci["pols"][0].as_u64().unwrap() as usize, 0, stage),
                    e.mul(delta.clone(), e.exp(ci["connections"][0].as_u64().unwrap() as usize, 0, stage)),
                ),
                gamma.clone(),
            );

            // Store numerator in expressions
            ci_ctx.insert("numId", json!(expressions_array.len()));
            let mut num_exp_val = json!(num_exp);
            num_exp_val.as_object_mut().unwrap().insert("stage".to_string(), json!(stage));
            expressions_array.push(num_exp_val);
            let n_dim = get_exp_dim(&mut expressions_array, ci_ctx["numId"].as_u64().unwrap() as usize);

            // Store denominator in expressions
            ci_ctx.insert("denId", json!(expressions_array.len()));
            let mut den_exp_val = json!(den_exp);
            den_exp_val.as_object_mut().unwrap().insert("stage".to_string(), json!(stage));
            expressions_array.push(den_exp_val);
            let d_dim = get_exp_dim(&mut expressions_array, ci_ctx["denId"].as_u64().unwrap() as usize);

            // Compute ks values
            let ks = get_ks(fr_mul, ci["pols"].as_array().unwrap().len() - 1, k);
            for (j, pol) in ci["pols"].as_array().unwrap().iter().enumerate().skip(1) {
                let num_exp = e.mul(
                    e.exp(ci_ctx["numId"].as_u64().unwrap() as usize, 0, stage),
                    e.add(
                        e.add(
                            e.exp(pol.as_u64().unwrap() as usize, 0, stage),
                            e.mul(e.mul(delta.clone(), e.number(ks[j - 1])), e.x()),
                        ),
                        gamma.clone(),
                    ),
                );

                let den_exp = e.mul(
                    e.exp(ci_ctx["denId"].as_u64().unwrap() as usize, 0, stage),
                    e.add(
                        e.add(
                            e.exp(pol.as_u64().unwrap() as usize, 0, stage),
                            e.mul(delta.clone(), e.exp(ci["connections"][j].as_u64().unwrap() as usize, 0, stage)),
                        ),
                        gamma.clone(),
                    ),
                );

                ci_ctx.insert("numId", json!(expressions_array.len()));
                expressions_array.push(json!(num_exp));

                ci_ctx.insert("denId", json!(expressions_array.len()));
                expressions_array.push(json!(den_exp));
            }

            // Compute z and zp
            let mut z = e.cm(ci_ctx["zId"].as_u64().unwrap() as usize, 0, Some(stage), dim);
            let zp = e.cm(ci_ctx["zId"].as_u64().unwrap() as usize, 1, Some(stage), dim);
            z.as_object_mut().unwrap().insert("stageId".to_string(), json!(n_cm2));
            n_cm2 += 1;

            // Compute constraint c1
            let l1 = e.const_(pil["references"]["Global.L1"]["id"].as_u64().unwrap() as usize, 0, 0, 1);
            let c1 = e.mul(l1, e.sub(z.clone(), e.number(1.0)));
            expressions_array.push(json!(c1));

            // Compute constraint c2
            let c2 = e.sub(
                e.mul(zp.clone(), e.exp(ci_ctx["denId"].as_u64().unwrap() as usize, 0, stage)),
                e.mul(z.clone(), e.exp(ci_ctx["numId"].as_u64().unwrap() as usize, 0, stage)),
            );
            expressions_array.push(json!(c2));

            // Store witness commitment
            symbols.push(json!({
                "type": "witness",
                "name": format!("Connection{}.z", i),
                "polId": ci_ctx["zId"],
                "stage": stage,
                "dim": std::cmp::max(n_dim, d_dim),
                "airId": air_id,
                "airGroupId": airgroup_id,
            }));

            // Append to hints
            hints.push(json!({
                "name": "gprod",
                "fields": [
                    {"name": "reference", "values": [z]},
                    {"name": "numerator", "values": [e.exp(ci_ctx["numId"].as_u64().unwrap() as usize, 0, stage)]},
                    {"name": "denominator", "values": [e.exp(ci_ctx["denId"].as_u64().unwrap() as usize, 0, stage)]},
                ]
            }));
        }
    }

    pil.insert("expressions".to_string(), json!(expressions_array));
    pil.insert("nCommitments".to_string(), json!(n_commitments));
}
