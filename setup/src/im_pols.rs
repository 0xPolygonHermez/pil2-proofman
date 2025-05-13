use std::hash::{Hash, DefaultHasher, Hasher};

use serde_json::{json, Map, Value};

pub trait HashCode: Hash {
    fn hash_code(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn hash_code_string(&self) -> String {
        format!("{}", self.hash_code())
    }
}

impl<T: Hash> HashCode for T {}

// MATCHES JS
pub fn calculate_im_pols(expressions: &mut Vec<Value>, exp: &mut Value, max_deg: usize) -> (Vec<usize>, isize) {
    let mut im_pols: Option<Vec<Value>> = Some(Vec::new());
    let absolute_max = max_deg;
    let mut abs_max_d = 0;

    tracing::info!(
        "expressions.len(): {}, max_deg: {}, absolute_max: {}, abs_max_d: {}",
        expressions.len(),
        max_deg,
        absolute_max,
        abs_max_d
    );
    let (re, rd) = __calculate_im_pols(expressions, exp, &mut im_pols, max_deg, absolute_max, &mut abs_max_d);
    tracing::info!(
        "expressions.len(): {}, max_deg: {}, absolute_max: {}, abs_max_d: {}",
        expressions.len(),
        max_deg,
        absolute_max,
        abs_max_d
    );

    #[cfg(feature = "debug-fibonacci")]
    assert_eq!((re.clone(), rd.clone()), (Some(vec![2.into()]), 2));

    (
        re.unwrap_or_else(Vec::new).into_iter().map(|v| v.as_number().unwrap().as_u64().unwrap() as usize).collect(),
        rd.max(abs_max_d) - 1,
    )
}

// MATCHES JS
fn __calculate_im_pols(
    expressions: &mut Vec<Value>,
    exp: &mut Value,
    im_pols: &mut Option<Vec<Value>>,
    max_deg: usize,
    absolute_max: usize,
    abs_max_d: &mut isize,
) -> (Option<Vec<Value>>, isize) {
    if im_pols.is_none() {
        return (None, -1);
    }

    match exp["op"].as_str().unwrap() {
        // --------------------------------------------------
        // MATCHES JS: sum or difference -> track the MAX d
        // --------------------------------------------------
        "add" | "sub" => {
            let mut md = 0;
            let values = exp["values"].as_array_mut().unwrap();
            for value in values {
                let d;
                (*im_pols, d) = __calculate_im_pols(expressions, value, im_pols, max_deg, absolute_max, abs_max_d);
                if d > md {
                    md = d;
                }
            }
            (im_pols.clone(), md)
        }

        // --------------------------------------------------
        // MATCHES JS: multiply -> try all splits (l + r = max_deg)
        // --------------------------------------------------
        "mul" => {
            let max_deg_here = exp["expDeg"].as_i64().unwrap_or(0) as isize;
            let values = exp["values"].as_array_mut().unwrap();

            // If either side is a constant polynomial, just simplify to the other side
            if !["add", "mul", "sub", "exp"].contains(&values[0]["op"].as_str().unwrap())
                && values[0]["expDeg"].as_i64().unwrap_or(0) == 0
            {
                return __calculate_im_pols(expressions, &mut values[1], im_pols, max_deg, absolute_max, abs_max_d);
            }
            if !["add", "mul", "sub", "exp"].contains(&values[1]["op"].as_str().unwrap())
                && values[1]["expDeg"].as_i64().unwrap_or(0) == 0
            {
                return __calculate_im_pols(expressions, &mut values[0], im_pols, max_deg, absolute_max, abs_max_d);
            }

            // If the expression degree is within the limit, return immediately
            if max_deg_here <= max_deg as isize {
                return (im_pols.clone(), max_deg_here);
            }

            // Now try all splits: l + r = max_deg
            let mut eb: Option<Vec<Value>> = None;
            let mut ed: isize = -1;
            for l in 0..=max_deg {
                let r = max_deg - l;

                // Make e1 mutable so we can pass it again
                let (mut e1, d1) =
                    __calculate_im_pols(expressions, &mut values[0], im_pols, l, absolute_max, abs_max_d);
                let (e2, d2) = __calculate_im_pols(expressions, &mut values[1], &mut e1, r, absolute_max, abs_max_d);

                // if e2 != false and (eb == false || e2.len() < eb.len()) => update
                if let Some(e2v) = e2 {
                    match eb {
                        None => {
                            eb = Some(e2v);
                            ed = d1 + d2;
                        }
                        Some(ref current_eb) => {
                            if e2v.len() < current_eb.len() {
                                eb = Some(e2v);
                                ed = d1 + d2;
                            }
                        }
                    }

                    // if(eb != false && eb.len() == imPols.len()) => we are done
                    if let (Some(ref this_eb), Some(ref im_p)) = (eb.as_ref(), im_pols.as_ref()) {
                        if this_eb.len() == im_p.len() {
                            return (Some((*this_eb).clone()), ed);
                        }
                    }
                }
            }
            (eb, ed)
        }

        // --------------------------------------------------
        // MATCHES JS: exponent -> check if already in imPols or cache
        // --------------------------------------------------
        "exp" => {
            if max_deg < 1 {
                return (None, -1);
            }

            let exp_id = exp["id"].as_i64().unwrap();
            // If we've already recorded exp_id in im_pols, no further expansions needed
            if let Some(ref pols) = im_pols {
                if pols.iter().any(|im| im.as_i64().unwrap() == exp_id) {
                    return (im_pols.clone(), 1);
                }
            }

            // Check cache: exp["res"][absoluteMax][hashCode]
            let mut e = None;
            let mut d = -1;
            let mut found_in_cache = false;

            if let Some(exp_res) = exp["res"].as_array() {
                if let Some(obj_slot) = exp_res.get(absolute_max) {
                    if let Some(arr_at_hash) = obj_slot[im_pols.hash_code_string()].as_array() {
                        e = arr_at_hash[0].as_array().cloned();
                        d = arr_at_hash[1].as_number().map(|num| num.as_i64().unwrap() as isize).unwrap_or(-1);
                        found_in_cache = true;
                    }
                }
            }

            // If not found, recursively compute
            if !found_in_cache {
                use std::mem;
                let mut exps_clone = expressions.clone();

                // Temporarily remove the sub-expression to avoid double borrowing
                let mut subexp_value = mem::take(&mut exps_clone[exp_id as usize]);
                (e, d) = __calculate_im_pols(
                    &mut exps_clone,
                    &mut subexp_value,
                    im_pols,
                    absolute_max,
                    absolute_max,
                    abs_max_d,
                );
                exps_clone[exp_id as usize] = subexp_value;

                // Write back the updated subexp
                expressions[exp_id as usize] = exps_clone[exp_id as usize].clone();
            }

            // If (e == false)
            if e.is_none() {
                return (None, -1);
            }

            // If (d > maxDeg), record in imPols (with exp_id) but keep deg=1
            if d > max_deg as isize {
                if d > *abs_max_d {
                    *abs_max_d = d;
                }
                let mut combined = e.unwrap();
                combined.push(exp_id.into());
                return (Some(combined), 1);
            } else {
                // else store in cache
                if exp["res"].as_array().is_none() {
                    exp["res"] = Value::Array(Vec::new());
                }
                let exp_res = exp["res"].as_array_mut().unwrap();

                while exp_res.len() <= absolute_max {
                    exp_res.push(Value::Null);
                }
                if exp_res[absolute_max].as_object().is_none() {
                    exp_res[absolute_max] = Value::Object(Map::new());
                }
                let obj = exp_res[absolute_max].as_object_mut().unwrap();

                let hashcode = im_pols.hash_code_string();
                obj.insert(hashcode, Value::Array(vec![e.clone().into(), d.into()]));

                return (e, d);
            }
        }

        // --------------------------------------------------
        // Default: same as JS
        // --------------------------------------------------
        _ => {
            if let Some(exp_deg) = exp["expDeg"].as_number() {
                if *exp_deg == 0.into() {
                    // deg=0 => return [imPols, 0]
                    (im_pols.clone(), 0)
                } else if max_deg < 1 {
                    (None, -1)
                } else {
                    (im_pols.clone(), 1)
                }
            } else if max_deg < 1 {
                (None, -1)
            } else {
                (im_pols.clone(), 1)
            }
        }
    }
}

// MATCHES JS
/// Computes the number of added columns in the base field.
pub fn calculate_added_cols(
    max_deg: usize,
    expressions: &[Value],
    im_exps: &[usize],
    q_deg: usize,
    q_dim: usize,
) -> usize {
    let q_cols = q_deg * q_dim;
    let mut im_cols = 0;

    for &index in im_exps {
        if let Some(dim) =
            expressions.get(index).and_then(|exp| exp.get("dim")).and_then(|d| Some(d.as_u64().unwrap() as usize))
        {
            im_cols += dim;
        }
    }

    let added_cols = q_cols + im_cols;

    println!("Max constraint degree: {}", max_deg);
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(max_deg, 2);
    println!("Number of intermediate polynomials: {}", im_exps.len());
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(im_exps.len(), 2);
    println!("Polynomial Q degree: {}", q_deg);
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(q_deg, 1);
    println!(
        "Number of columns added in the basefield: {} (Polynomial Q columns: {} + Intermediate polynomials columns: {})",
        added_cols, q_cols, im_cols
    );
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(added_cols, 9);
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(q_cols, 3);
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(im_cols, 6);

    added_cols
}

// MATCHES JS
/// Computes intermediate polynomials for a given expression and returns a JSON object.
pub fn calculate_intermediate_polynomials(
    expressions: &mut Vec<Value>,
    c_exp_id: usize,
    max_q_deg: usize,
    q_dim: usize,
) -> Value {
    let mut d: usize = 2;

    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(max_q_deg, 3);

    println!("-------------------- POSSIBLE DEGREES ----------------------");
    println!(
        "** Considering degrees between 2 and {} (blowup factor: {}) **",
        max_q_deg,
        (max_q_deg as f64 - 1.0).log2()
    );
    println!("------------------------------------------------------------");

    // Clone the expression to avoid borrowing conflicts
    let mut c_exp = expressions[c_exp_id].clone();

    // First calculation
    let (mut im_exps, mut q_deg) = calculate_im_pols(expressions, &mut c_exp, d);
    #[cfg(feature = "debug-fibonacci")]
    assert_eq!(q_deg, 1);
    panic!("hey!");
    let mut added_basefield_cols = calculate_added_cols(d, expressions, &im_exps, q_deg as usize, q_dim);
    d += 1;

    while !im_exps.is_empty() && d <= max_q_deg {
        println!("------------------------------------------------------------");

        // Clone c_exp again for the new calculation
        let mut c_exp_clone = c_exp.clone();

        let (im_exps_p, q_deg_p) = calculate_im_pols(expressions, &mut c_exp_clone, d);
        let new_added_basefield_cols =
            calculate_added_cols(d, expressions, &im_exps_p, q_deg_p.try_into().unwrap(), q_dim);
        d += 1;

        if (max_q_deg > 0 && new_added_basefield_cols < added_basefield_cols)
            || (max_q_deg == 0 && im_exps_p.is_empty())
        {
            added_basefield_cols = new_added_basefield_cols;
            im_exps = im_exps_p.clone();
            q_deg = q_deg_p.try_into().unwrap();

            // Update the main c_exp with the latest changes
            c_exp = c_exp_clone;
        }

        if im_exps_p.is_empty() {
            break;
        }
    }

    // Ensure all keys are included: newExpressions, imExps, qDeg
    json!({
        "newExpressions": expressions,
        "imExps": im_exps,
        "qDeg": q_deg
    })
}
