use serde_json::{json, Value};

/// Computes the intermediate polynomials and their degrees.
pub fn calculate_im_pols(expressions: &[Value], exp: &Value, max_deg: i64) -> (Vec<usize>, i64) {
    let absolute_max = max_deg;
    let mut abs_max_d = 0;

    let (im_pols, degree) = _calculate_im_pols(expressions, exp, vec![], max_deg, absolute_max, &mut abs_max_d);

    // Adjust the polynomial degree by subtracting 1
    (im_pols, std::cmp::max(degree, abs_max_d) - 1)
}

/// Internal recursive function to compute intermediate polynomials.
fn _calculate_im_pols(
    expressions: &[Value],
    exp: &Value,
    im_pols: Vec<usize>,
    max_deg: i64,
    absolute_max: i64,
    abs_max_d: &mut i64,
) -> (Vec<usize>, i64) {
    if exp.get("op").is_none() {
        return (im_pols, -1);
    }

    match exp["op"].as_str().unwrap() {
        "add" | "sub" => {
            let mut max_d = 0;
            let mut updated_pols = im_pols.clone();

            if let Some(values) = exp["values"].as_array() {
                for e in values {
                    let (new_pols, d) =
                        _calculate_im_pols(expressions, e, updated_pols, max_deg, absolute_max, abs_max_d);
                    updated_pols = new_pols;
                    max_d = max_d.max(d);
                }
            }
            (updated_pols, max_d)
        }
        "mul" => {
            if let Some(values) = exp["values"].as_array() {
                let left = &values[0];
                let right = &values[1];

                if !["add", "mul", "sub", "exp"].contains(&left["op"].as_str().unwrap_or(""))
                    && left["expDeg"].as_i64().unwrap_or(0) == 0
                {
                    return _calculate_im_pols(expressions, right, im_pols, max_deg, absolute_max, abs_max_d);
                }

                if !["add", "mul", "sub", "exp"].contains(&right["op"].as_str().unwrap_or(""))
                    && right["expDeg"].as_i64().unwrap_or(0) == 0
                {
                    return _calculate_im_pols(expressions, left, im_pols, max_deg, absolute_max, abs_max_d);
                }

                let max_deg_here = exp["expDeg"].as_i64().unwrap_or(0);
                if max_deg_here <= max_deg {
                    return (im_pols, max_deg_here);
                }

                let mut best_pols: Vec<usize> = Vec::new();
                let mut best_degree = -1;

                for l in 0..=max_deg {
                    let r = max_deg - l;
                    let (e1, d1) = _calculate_im_pols(expressions, left, im_pols.clone(), l, absolute_max, abs_max_d);
                    let (e2, d2) = _calculate_im_pols(expressions, right, e1.clone(), r, absolute_max, abs_max_d);

                    if best_pols.is_empty() || e2.len() < best_pols.len() {
                        best_pols = e2.clone();
                        best_degree = d1 + d2;
                    }

                    if best_pols.len() == im_pols.len() {
                        return (best_pols.clone(), best_degree);
                    }
                }

                (best_pols, best_degree)
            } else {
                (vec![], -1)
            }
        }

        "exp" => {
            if max_deg < 1 {
                return (vec![], -1);
            }

            let exp_id = exp["id"].as_u64().unwrap() as usize;
            if im_pols.contains(&exp_id) {
                return (im_pols, 1);
            }

            let (e, d) = if let Some(res) = exp
                .get("res")
                .and_then(|res| res.get(absolute_max.to_string()))
                .and_then(|sub| sub.get(&serde_json::to_string(&im_pols).unwrap()))
            {
                let e = res[0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let d = res[1].as_i64().unwrap();
                (e, d)
            } else {
                _calculate_im_pols(
                    expressions,
                    &expressions[exp_id],
                    im_pols.clone(),
                    absolute_max,
                    absolute_max,
                    abs_max_d,
                )
            };

            if e.is_empty() {
                return (vec![], -1);
            }

            if d > max_deg {
                *abs_max_d = (*abs_max_d).max(d);
                let mut new_pols = e.clone();
                new_pols.push(exp_id);
                return (new_pols, 1);
            } else {
                let res_entry = exp.get("res").unwrap_or(&json!({})).clone();
                let mut updated_res = res_entry.as_object().unwrap().clone();
                updated_res.insert(
                    absolute_max.to_string(),
                    json!({ serde_json::to_string(&im_pols).unwrap(): [e.clone(), d] }),
                );
                (e, d)
            }
        }
        _ => {
            let exp_deg = exp["expDeg"].as_i64().unwrap_or(0);
            if exp_deg == 0 {
                return (im_pols, 0);
            } else if max_deg < 1 {
                return (vec![], -1);
            } else {
                return (im_pols, 1);
            }
        }
    }
}

/// Computes the number of added columns in the base field.
pub fn calculate_added_cols(max_deg: i64, expressions: &[Value], im_exps: &[usize], q_deg: i64, q_dim: i64) -> i64 {
    let q_cols = q_deg * q_dim;
    let mut im_cols = 0;

    for &index in im_exps {
        if let Some(dim) = expressions.get(index).and_then(|exp| exp.get("dim")).and_then(|d| d.as_i64()) {
            im_cols += dim;
        }
    }

    let added_cols = q_cols + im_cols;

    println!("Max constraint degree: {}", max_deg);
    println!("Number of intermediate polynomials: {}", im_exps.len());
    println!("Polynomial Q degree: {}", q_deg);
    println!(
        "Number of columns added in the basefield: {} (Polynomial Q columns: {} + Intermediate polynomials columns: {})",
        added_cols, q_cols, im_cols
    );

    added_cols
}

/// Computes intermediate polynomials for a given expression.
pub fn calculate_intermediate_polynomials(
    expressions: &[Value],
    c_exp_id: usize,
    max_q_deg: i64,
    q_dim: i64,
) -> (Vec<Value>, Vec<usize>, i64) {
    let mut d = 2;

    println!("-------------------- POSSIBLE DEGREES ----------------------");
    println!(
        "** Considering degrees between 2 and {} (blowup factor: {}) **",
        max_q_deg,
        (max_q_deg as f64 - 1.0).log2()
    );
    println!("------------------------------------------------------------");

    let c_exp = &expressions[c_exp_id];

    let (mut im_exps, mut q_deg) = calculate_im_pols(expressions, c_exp, d);
    let mut added_basefield_cols = calculate_added_cols(d, expressions, &im_exps, q_deg, q_dim);
    d += 1;

    while !im_exps.is_empty() && d <= max_q_deg {
        println!("------------------------------------------------------------");

        let (im_exps_p, q_deg_p) = calculate_im_pols(expressions, c_exp, d);
        let new_added_basefield_cols = calculate_added_cols(d, expressions, &im_exps_p, q_deg_p, q_dim);
        d += 1;

        if (max_q_deg > 0 && new_added_basefield_cols < added_basefield_cols)
            || (max_q_deg == 0 && im_exps_p.is_empty())
        {
            added_basefield_cols = new_added_basefield_cols;
            im_exps = im_exps_p.clone(); // Clone here to avoid move
            q_deg = q_deg_p;
        }

        if im_exps_p.is_empty() {
            break;
        }
    }

    (expressions.to_vec(), im_exps, q_deg)
}
