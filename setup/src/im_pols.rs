use serde_json::{json, Value};

/// Computes the intermediate polynomials and their degrees.
pub fn calculate_im_pols(expressions: &mut [Value], exp: &mut Value, max_deg: usize) -> (Vec<usize>, usize) {
    let absolute_max = max_deg;
    println!("Absolute max degree: {}", absolute_max);

    let mut im_pols = vec![];

    // Remove abs_max_d to match JS behavior
    let (im_pols_opt, degree) = _calculate_im_pols(expressions, exp, &mut im_pols, max_deg, absolute_max);

    let im_pols = im_pols_opt.unwrap_or_else(|| vec![]);

    // Adjust the polynomial degree by subtracting 1, matching JS behavior
    let adjusted_degree = std::cmp::max(degree, 0) - 1;

    (im_pols, adjusted_degree)
}

/// Internal recursive function to compute intermediate polynomials.
fn _calculate_im_pols(
    expressions: &mut [Value],
    exp: &mut Value,
    im_pols: &mut Vec<usize>,
    max_deg: usize,
    absolute_max: usize,
) -> (Option<Vec<usize>>, usize) {
    if exp.get("op").is_none() {
        return (Some(im_pols.clone()), 0); // Constants/leaves are degree 0
    }

    match exp["op"].as_str().unwrap() {
        "add" | "sub" => {
            let mut max_d = 0;

            if let Some(values) = exp["values"].as_array_mut() {
                for e in values {
                    let (new_pols_opt, d) = _calculate_im_pols(expressions, e, im_pols, max_deg, absolute_max);
                    if let Some(new_pols) = new_pols_opt {
                        im_pols.extend(new_pols);
                        max_d = max_d.max(d);
                    } else {
                        return (None, 0); // Propagate failure
                    }
                }
            }
            (Some(im_pols.clone()), max_d)
        }

        "mul" => {
            let max_deg_here = exp["expDeg"].as_u64().unwrap_or(0) as usize;
            if let Some(values) = exp["values"].as_array_mut() {
                if values.len() < 2 {
                    return (None, 0); // Invalid structure
                }
                let (left, right) = {
                    let (left, rest) = values.split_at_mut(1);
                    (&mut left[0], &mut rest[0])
                };

                // Handle constants in multiplication
                if !["add", "mul", "sub", "exp"].contains(&left["op"].as_str().unwrap_or(""))
                    && left["expDeg"].as_u64().unwrap_or(0) == 0
                {
                    return _calculate_im_pols(expressions, right, im_pols, max_deg, absolute_max);
                }

                if !["add", "mul", "sub", "exp"].contains(&right["op"].as_str().unwrap_or(""))
                    && right["expDeg"].as_u64().unwrap_or(0) == 0
                {
                    return _calculate_im_pols(expressions, left, im_pols, max_deg, absolute_max);
                }

                if max_deg_here <= max_deg {
                    return (Some(im_pols.clone()), max_deg_here);
                }

                let mut best_pols: Option<Vec<usize>> = None;
                let mut best_degree = None;

                for l in 0..=max_deg {
                    let r = max_deg - l;
                    let (e1_opt, d1) = _calculate_im_pols(expressions, left, im_pols, l, absolute_max);
                    if let Some(mut e1) = e1_opt {
                        let (e2_opt, d2) = _calculate_im_pols(expressions, right, &mut e1, r, absolute_max);
                        if let Some(e2) = e2_opt {
                            let total_degree = d1 + d2;
                            if best_pols.is_none() || e2.len() < best_pols.as_ref().unwrap().len() {
                                best_pols = Some(e2.clone());
                                best_degree = Some(total_degree);
                            }
                            if e2.len() == im_pols.len() {
                                return (Some(e2), best_degree.unwrap_or(0)); // Can't do better
                            }
                        }
                    }
                }

                (best_pols, best_degree.unwrap_or(0))
            } else {
                (None, 0)
            }
        }

        "exp" => {
            if max_deg < 1 {
                return (None, 0); // Can't proceed if degree is less than 1
            }

            let exp_id = exp["id"].as_u64().unwrap() as usize;
            if im_pols.contains(&exp_id) {
                return (Some(im_pols.clone()), 1); // Already processed
            }

            // Check cached results
            let (e_opt, d) = if let Some(res) = exp
                .get("res")
                .and_then(|res| res.get(absolute_max.to_string()))
                .and_then(|sub| sub.get(&serde_json::to_string(&im_pols).unwrap()))
            {
                let e = res[0].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
                let d = res[1].as_u64().unwrap() as usize;
                (Some(e), d)
            } else {
                let e_result = {
                    let mut exp_at_id = expressions[exp_id].clone();
                    let exp_ref = &mut exp_at_id;
                    let res = _calculate_im_pols(expressions, exp_ref, im_pols, absolute_max, absolute_max);

                    expressions[exp_id] = exp_at_id;
                    res
                };
                e_result
            };

            if let Some(e) = e_opt {
                if d >= max_deg {
                    let mut new_pols = e.clone();
                    new_pols.push(exp_id); // Add current exp_id as intermediate poly
                    return (Some(new_pols), 1); // Force degree to 1
                } else {
                    // Caching logic
                    if exp.get("res").is_none() {
                        exp.as_object_mut().unwrap().insert("res".to_string(), serde_json::json!({}));
                    }
                    let res_map = exp["res"].as_object_mut().unwrap();
                    res_map
                        .entry(absolute_max.to_string())
                        .or_insert_with(|| serde_json::json!({}))
                        .as_object_mut()
                        .unwrap()
                        .insert(serde_json::to_string(&im_pols).unwrap(), serde_json::json!([e, d]));

                    return (Some(e), d);
                }
            }

            (None, 0)
        }

        _ => {
            if let Some(exp_deg) = exp["expDeg"].as_u64() {
                if exp_deg == 0 {
                    return (Some(im_pols.clone()), 0);
                }
            }
            if max_deg < 1 {
                return (None, 0);
            } else {
                return (Some(im_pols.clone()), 1);
            }
        }
    }
}

/// Computes the number of added columns in the base field.
pub fn calculate_added_cols(
    max_deg: usize,
    expressions: &[Value],
    im_exps: &[usize],
    q_deg: usize,
    q_dim: usize,
) -> usize {
    let q_cols = std::cmp::max(q_deg, 0) * q_dim; // Clamp q_deg to zero if negative
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
    println!("Number of intermediate polynomials: {}", im_exps.len());
    println!("Polynomial Q degree: {}", q_deg);
    println!(
        "Number of columns added in the basefield: {} (Polynomial Q columns: {} + Intermediate polynomials columns: {})",
        added_cols, q_cols, im_cols
    );

    added_cols
}

/// Computes intermediate polynomials for a given expression and returns a JSON object.
pub fn calculate_intermediate_polynomials(
    expressions: &mut [Value],
    c_exp_id: usize,
    max_q_deg: usize,
    q_dim: usize,
) -> Value {
    let mut d: usize = 2;

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
    let mut added_basefield_cols = calculate_added_cols(d, expressions, &im_exps, q_deg, q_dim);
    d += 1;

    while !im_exps.is_empty() && d <= max_q_deg {
        println!("------------------------------------------------------------");

        // Clone c_exp again for the new calculation
        let mut c_exp_clone = c_exp.clone();

        let (im_exps_p, q_deg_p) = calculate_im_pols(expressions, &mut c_exp_clone, d);
        let new_added_basefield_cols = calculate_added_cols(d, expressions, &im_exps_p, q_deg_p, q_dim);
        d += 1;

        if (max_q_deg > 0 && new_added_basefield_cols < added_basefield_cols)
            || (max_q_deg == 0 && im_exps_p.is_empty())
        {
            added_basefield_cols = new_added_basefield_cols;
            im_exps = im_exps_p.clone();
            q_deg = q_deg_p;

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
