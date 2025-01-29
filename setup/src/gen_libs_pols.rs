// use crate::{
//     add_intermediate_pols::ExpressionOps,
//     helpers::get_exp_dim,
//     grand_productp_lookup::{grand_product_plookup, init_challenges_plookup},
// };
// use serde_json::{json, Value};
// use std::collections::HashMap;

// /// Generates library polynomials for PIL.
// pub fn generate_libs_polynomials(
//     f: &str,
//     res: &mut HashMap<String, Value>,
//     pil: &mut HashMap<String, Value>,
//     symbols: &mut Vec<Value>,
//     hints: &mut Vec<Value>,
// ) {
//     let mut pil_libs: Vec<Box<dyn FnMut()>> = Vec::new();

//     // Initialize `nCm2` and `nCm3`
//     pil.insert("nCm2".to_string(), json!(0));
//     pil.insert("nCm3".to_string(), json!(0));

//     if pil.get("plookupIdentities").map_or(false, |v| !v.as_array().unwrap().is_empty()) {
//         pil_libs.push(Box::new(|| {
//             grand_product_plookup(
//                 pil,
//                 symbols,
//                 hints,
//                 res["airgroupId"].as_u64().unwrap(),
//                 res["airId"].as_u64().unwrap(),
//             )
//         }));
//         let challenges = init_challenges_plookup();
//         calculate_challenges(symbols, &challenges);
//     }

//     if pil.get("permutationIdentities").map_or(false, |v| !v.as_array().unwrap().is_empty()) {
//         pil_libs.push(Box::new(|| {
//             grand_product_permutation(
//                 pil,
//                 symbols,
//                 hints,
//                 res["airgroupId"].as_u64().unwrap(),
//                 res["airId"].as_u64().unwrap(),
//             )
//         }));
//         let challenges = init_challenges_permutation();
//         calculate_challenges(symbols, &challenges);
//     }

//     if pil.get("connectionIdentities").map_or(false, |v| !v.as_array().unwrap().is_empty()) {
//         pil_libs.push(Box::new(|| {
//             grand_product_connection(
//                 pil,
//                 symbols,
//                 hints,
//                 res["airgroupId"].as_u64().unwrap(),
//                 res["airId"].as_u64().unwrap(),
//                 f,
//             )
//         }));
//         let challenges = init_challenges_connection();
//         calculate_challenges(symbols, &challenges);
//     }

//     // Execute all library functions
//     for lib in pil_libs.iter_mut() {
//         lib();
//     }
// }

// /// Calculates challenge values and assigns proper IDs.
// fn calculate_challenges(symbols: &mut Vec<Value>, challenges: &[Value]) {
//     for challenge in challenges {
//         if !symbols.iter().any(|c| {
//             c["type"] == "challenge" && c["stage"] == challenge["stage"] && c["stageId"] == challenge["stageId"]
//         }) {
//             symbols.push(json!({
//                 "type": "challenge",
//                 ..challenge.clone()
//             }));
//         }
//     }

//     let mut symbols_challenges: Vec<&mut Value> = symbols.iter_mut().filter(|s| s["type"] == "challenge").collect();

//     for ch in symbols_challenges.iter_mut() {
//         let id = symbols_challenges
//             .iter()
//             .filter(|c| {
//                 c["stage"].as_u64().unwrap() < ch["stage"].as_u64().unwrap()
//                     || (c["stage"] == ch["stage"] && c["stageId"].as_u64().unwrap() < ch["stageId"].as_u64().unwrap())
//             })
//             .count();
//         ch.as_object_mut().unwrap().insert("id".to_string(), json!(id));
//     }
// }
