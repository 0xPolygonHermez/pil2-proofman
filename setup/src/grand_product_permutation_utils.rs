use crate::{add_intermediate_pols::ExpressionOps, helpers::get_exp_dim};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Initializes the standard challenges for Permutation.
pub fn init_challenges_permutation() -> Vec<Value> {
    let dim = 3;
    vec![
        json!({"name": "std_alpha", "stage": 2, "dim": dim, "stageId": 0}),
        json!({"name": "std_beta", "stage": 2, "dim": dim, "stageId": 1}),
        json!({"name": "std_gamma", "stage": 2, "dim": dim, "stageId": 2}),
    ]
}

/// Implements the Grand Product Argument for Permutation.
pub fn grand_product_permutation(
    pil: &mut HashMap<String, Value>,
    symbols: &mut Vec<Value>,
    hints: &mut Vec<Value>,
    airgroup_id: u64,
    air_id: u64,
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

    let alpha = find_challenge(0);
    let beta = find_challenge(1);
    let gamma = find_challenge(2);

    let permutation_identities = pil.get("permutationIdentities").cloned().unwrap_or(json!([]));
    let expressions = pil.get("expressions").cloned().unwrap_or(json!([]));
    let mut expressions_array = expressions.as_array().unwrap().clone();
    let mut n_commitments = pil.get("nCommitments").and_then(|v| v.as_u64()).unwrap_or(0);
    let mut n_cm2 = pil.get("nCm2").and_then(|v| v.as_u64()).unwrap_or(0);

    if let Some(identity_list) = permutation_identities.as_array() {
        for (i, pi) in identity_list.iter().enumerate() {
            let mut pe_ctx = HashMap::new();
            let mut t_exp = None;

            if let Some(t) = pi.get("t").and_then(|t| t.as_array()) {
                for exp in t {
                    let e_exp = e.exp(exp.as_u64().unwrap() as usize, 0, stage);
                    t_exp =
                        Some(if let Some(t_exp) = t_exp { e.add(e.mul(alpha.clone(), t_exp), e_exp) } else { e_exp });
                }
            }

            if let Some(sel_t) = pi.get("selT") {
                let sel_t = sel_t.as_u64().unwrap() as usize;
                t_exp = Some(e.mul(e.sub(t_exp.unwrap(), beta.clone()), e.exp(sel_t, 0, stage)));
                t_exp = Some(e.add(t_exp.unwrap(), beta.clone()));
            }

            pe_ctx.insert("tExpId", json!(expressions_array.len()));
            let mut t_exp_val = json!(t_exp.unwrap());
            t_exp_val.as_object_mut().unwrap().insert("stage".to_string(), json!(stage));
            expressions_array.push(t_exp_val);
            let t_dim = get_exp_dim(&expressions_array, pe_ctx["tExpId"].as_u64().unwrap() as usize);
            expressions_array[pe_ctx["tExpId"].as_u64().unwrap() as usize]
                .as_object_mut()
                .unwrap()
                .insert("deg".to_string(), json!(t_dim));

            let mut f_exp = None;
            if let Some(f) = pi.get("f").and_then(|f| f.as_array()) {
                for exp in f {
                    let e_exp = e.exp(exp.as_u64().unwrap() as usize, 0, stage);
                    f_exp =
                        Some(if let Some(f_exp) = f_exp { e.add(e.mul(f_exp, alpha.clone()), e_exp) } else { e_exp });
                }
            }

            if let Some(sel_f) = pi.get("selF") {
                let sel_f = sel_f.as_u64().unwrap() as usize;
                f_exp = Some(e.mul(e.sub(f_exp.unwrap(), beta.clone()), e.exp(sel_f, 0, stage)));
                f_exp = Some(e.add(f_exp.unwrap(), beta.clone()));
            }

            pe_ctx.insert("fExpId", json!(expressions_array.len()));
            let mut f_exp_val = json!(f_exp.unwrap());
            f_exp_val.as_object_mut().unwrap().insert("stage".to_string(), json!(stage));
            expressions_array.push(f_exp_val);
            let f_dim = get_exp_dim(&expressions_array, pe_ctx["fExpId"].as_u64().unwrap() as usize);
            expressions_array[pe_ctx["fExpId"].as_u64().unwrap() as usize]
                .as_object_mut()
                .unwrap()
                .insert("deg".to_string(), json!(f_dim));

            pe_ctx.insert("zId", json!(n_commitments));
            n_commitments += 1;

            let f = e.exp(pe_ctx["fExpId"].as_u64().unwrap() as usize, 0, stage);
            let t = e.exp(pe_ctx["tExpId"].as_u64().unwrap() as usize, 0, stage);
            let mut z = e.cm(pe_ctx["zId"].as_u64().unwrap() as usize, 0, Some(stage), dim);
            let zp = e.cm(pe_ctx["zId"].as_u64().unwrap() as usize, 1, Some(stage), dim);
            z.as_object_mut().unwrap().insert("stageId".to_string(), json!(n_cm2));
            n_cm2 += 1;

            let l1 = e.const_(pil["references"]["Global.L1"]["id"].as_u64().unwrap() as usize, 0, 0, 1);
            let c1 = e.mul(l1, e.sub(z.clone(), e.number(1.0)));
            expressions_array.push(json!(c1));

            let num_exp = e.add(f, gamma.clone());
            pe_ctx.insert("numId", json!(expressions_array.len()));
            expressions_array.push(json!(num_exp));
            let num_dim = get_exp_dim(&expressions_array, pe_ctx["numId"].as_u64().unwrap() as usize);

            let den_exp = e.add(t, gamma.clone());
            pe_ctx.insert("denId", json!(expressions_array.len()));
            expressions_array.push(json!(den_exp));
            let den_dim = get_exp_dim(&expressions_array, pe_ctx["denId"].as_u64().unwrap() as usize);

            let c2 = e.sub(
                e.mul(zp.clone(), e.exp(pe_ctx["denId"].as_u64().unwrap() as usize, 0, stage)),
                e.mul(z.clone(), e.exp(pe_ctx["numId"].as_u64().unwrap() as usize, 0, stage)),
            );
            expressions_array.push(json!(c2));

            hints.push(json!({
                "name": "gprod",
                "fields": [
                    {"name": "reference", "values": [z]},
                    {"name": "numerator", "values": [e.exp(pe_ctx["numId"].as_u64().unwrap() as usize, 0, stage)]},
                    {"name": "denominator", "values": [e.exp(pe_ctx["denId"].as_u64().unwrap() as usize, 0, stage)]},
                ]
            }));

            symbols.push(json!({
                "type": "witness",
                "name": format!("Permutation{}.z", i),
                "polId": pe_ctx["zId"],
                "stage": stage,
                "dim": std::cmp::max(num_dim, den_dim),
                "airId": air_id,
                "airGroupId": airgroup_id,
            }));
        }
    }

    pil.insert("expressions".to_string(), json!(expressions_array));
    pil.insert("nCommitments".to_string(), json!(n_commitments));
}
