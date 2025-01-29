// use serde_json::{json, Value};
// use std::collections::HashMap;

// use crate::add_intermediate_pols::ExpressionOps;

// /// Generates PIL1 polynomials, modifying `res` and returning processed `symbols`, `hints`, `expressions`, and `constraints`.
// pub fn generate_pil1_polynomials(
//     f: &str,
//     res: &mut HashMap<String, Value>,
//     pil: &Value,
//     options: Option<&HashMap<String, Value>>,
// ) -> HashMap<String, Value> {
//     let e = ExpressionOps::new();
//     let mut pil = pil.clone(); // Clone as we are modifying it

//     res.insert("airgroupId".to_string(), json!(0));
//     res.insert("airId".to_string(), json!(0));

//     if let Some(opts) = options {
//         if let Some(airgroup_id) = opts.get("airgroupId") {
//             res.insert("airgroupId".to_string(), airgroup_id.clone());
//         }
//         if let Some(air_id) = opts.get("airId") {
//             res.insert("airId".to_string(), air_id.clone());
//         }
//     }

//     res.insert("nPublics".to_string(), json!(pil["publics"].as_array().unwrap().len()));
//     res.insert("nConstants".to_string(), pil["nConstants"].clone());

//     res.insert("customCommits".to_string(), json!([]));

//     let n_stages = if pil["plookupIdentities"].as_array().unwrap().is_empty() { 2 } else { 3 };
//     res.insert("nStages".to_string(), json!(n_stages));

//     let mut symbols = vec![];
//     let mut hints = vec![];

//     if let Some(references) = pil.get("references").and_then(|r| r.as_object()) {
//         for (pol_ref, pol_info) in references {
//             let pol_info = pol_info.as_object().expect("Expected reference to be an object");

//             if pol_info["type"] == "imP" {
//                 continue;
//             }

//             let type_str = if pol_info["type"] == "constP" { "fixed" } else { "witness" };

//             let stage = if type_str == "witness" { 1 } else { 0 };

//             if pol_info.contains_key("isArray") {
//                 let len = pol_info["len"].as_u64().unwrap_or(1);
//                 for i in 0..len {
//                     let name_pol = format!("{}{}", pol_ref, i);
//                     let pol_id = pol_info["id"].as_u64().unwrap() + i;
//                     symbols.push(json!({
//                         "type": type_str,
//                         "name": name_pol,
//                         "polId": pol_id,
//                         "stage": stage,
//                         "dim": 1,
//                         "airgroupId": res["airgroupId"],
//                         "airId": res["airId"]
//                     }));

//                     if type_str == "witness" {
//                         e.cm(pol_id, 0, stage, 1);
//                     }
//                 }
//             } else {
//                 let pol_id = pol_info["id"].as_u64().unwrap();
//                 symbols.push(json!({
//                     "type": type_str,
//                     "name": pol_ref,
//                     "polId": pol_id,
//                     "stage": stage,
//                     "dim": 1,
//                     "airgroupId": res["airgroupId"],
//                     "airId": res["airId"]
//                 }));

//                 if type_str == "witness" {
//                     e.cm(pol_id, 0, stage, 1);
//                 }
//             }
//         }
//     }

//     generate_libs_polynomials(f, res, &pil, &mut symbols, &mut hints);

//     res.insert("nCommitments".to_string(), pil["nCommitments"].clone());
//     res.insert(
//         "pilPower".to_string(),
//         json!(log2(pil["references"].as_object().unwrap().values().next().unwrap()["polDeg"].as_u64().unwrap())),
//     );

//     let mut expressions = pil["expressions"].as_array().unwrap().clone();
//     let mut constraints = pil["polIdentities"].as_array().unwrap().clone();

//     for constraint in &mut constraints {
//         if !constraint.as_object().unwrap().contains_key("boundary") {
//             constraint.as_object_mut().unwrap().insert("boundary".to_string(), json!("everyRow"));
//         }
//     }

//     let n_publics = res["nPublics"].as_u64().unwrap();
//     for i in 0..n_publics {
//         symbols.push(json!({
//             "type": "public",
//             "name": format!("public{}", i),
//             "stage": 1,
//             "id": i
//         }));
//     }

//     json!({
//         "symbols": symbols,
//         "hints": hints,
//         "expressions": expressions,
//         "constraints": constraints
//     })
//     .as_object()
//     .unwrap()
//     .clone()
// }
