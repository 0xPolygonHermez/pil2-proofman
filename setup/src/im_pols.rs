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

/// Top‐level: prints “N 0 D” before and after, then returns
/// (intermediate IDs, q_degree).
pub fn calculate_im_pols(expressions: &mut [Value], exp: &mut Value, max_deg: usize) -> (Vec<usize>, usize) {
    let absolute_max = max_deg;
    let mut abs_max_d = 0isize;

    // JS: console.log(expressions.length, imPols.length, maxDeg);
    println!("{} {} {}", expressions.len(), 0, max_deg);

    // call the recursive helper
    let (maybe_pols, rd) =
        __calculate_im_pols(expressions, exp, Some(Vec::new()), max_deg, absolute_max, &mut abs_max_d);

    // unwrap into a Vec<usize>
    let pols_vec = maybe_pols.unwrap_or_default().into_iter().map(|v| v.as_u64().unwrap() as usize).collect::<Vec<_>>();

    // JS: console.log(expressions.length, imPols.length, maxDeg);
    println!("{} {} {}", expressions.len(), pols_vec.len(), max_deg);

    // JS: console.log("!!!!!!!!! re: ", re);
    println!("!!!!!!!!! re: {:?}", pols_vec);

    // JS: console.log("!!!!!!!!! rd: ", rd);
    println!("!!!!!!!!! rd: {}", rd);

    // finally compute q_degree = Math.max(rd, absMaxD) - 1, clamped ≥ 0
    let raw_q = rd.max(abs_max_d) - 1;
    let q_deg = if raw_q < 0 { 0 } else { raw_q as usize };

    (pols_vec, q_deg)
}

fn __calculate_im_pols(
    expressions: &mut [Value],
    exp: &mut Value,
    im_pols_opt: Option<Vec<Value>>,
    max_deg: usize,
    absolute_max: usize,
    abs_max_d: &mut isize,
) -> (Option<Vec<Value>>, isize) {
    let mut im_pols = match im_pols_opt {
        None => return (None, -1),
        Some(v) => v,
    };

    match exp["op"].as_str().unwrap() {
        // ==== add / sub ====
        "add" | "sub" => {
            let mut md = 0isize;
            for child in exp["values"].as_array_mut().unwrap() {
                let (next_pols_opt, d) =
                    __calculate_im_pols(expressions, child, Some(im_pols.clone()), max_deg, absolute_max, abs_max_d);
                // explicit match instead of unwrap_or_else
                let next_pols = match next_pols_opt {
                    Some(v) => v,
                    None => return (None, -1),
                };
                im_pols = next_pols;
                md = md.max(d);
            }
            (Some(im_pols), md)
        }

        // ==== mul ====
        "mul" => {
            let deg_here = exp["expDeg"].as_i64().unwrap_or(0) as usize;
            let vals = exp["values"].as_array_mut().unwrap();

            for i in 0..2 {
                let side = &vals[i];
                let sop = side["op"].as_str().unwrap();
                let sdeg = side["expDeg"].as_i64().unwrap_or(0);
                if !["add", "sub", "mul", "exp"].contains(&sop) && sdeg == 0 {
                    return __calculate_im_pols(
                        expressions,
                        &mut vals[1 - i],
                        Some(im_pols),
                        max_deg,
                        absolute_max,
                        abs_max_d,
                    );
                }
            }

            if deg_here <= max_deg {
                return (Some(im_pols), deg_here as isize);
            }

            let mut best: Option<(Vec<Value>, isize)> = None;
            for l in 0..=max_deg {
                let r = max_deg - l;

                let (p1_opt, d1) =
                    __calculate_im_pols(expressions, &mut vals[0], Some(im_pols.clone()), l, absolute_max, abs_max_d);
                let p1 = match p1_opt {
                    Some(v) => v,
                    None => continue,
                };

                let (p2_opt, d2) =
                    __calculate_im_pols(expressions, &mut vals[1], Some(p1.clone()), r, absolute_max, abs_max_d);
                let p2 = match p2_opt {
                    Some(v) => v,
                    None => continue,
                };

                let combined_d = d1 + d2;
                if best.as_ref().map_or(true, |(bp, _)| p2.len() < bp.len()) {
                    best = Some((p2.clone(), combined_d));
                }

                if let Some((bp, _)) = &best {
                    if bp.len() == im_pols.len() {
                        return (Some(bp.clone()), combined_d);
                    }
                }
            }

            best.map(|(pols, d)| (Some(pols), d)).unwrap_or((None, -1))
        }

        // ==== exp (no cache) ====
        "exp" => {
            if max_deg < 1 {
                return (None, -1);
            }
            let exp_id = exp["id"].as_u64().unwrap() as usize;

            if im_pols.iter().any(|v| v.as_u64().unwrap() as usize == exp_id) {
                return (Some(im_pols), 1);
            }

            let mut sub = expressions[exp_id].clone();
            let (maybe_e, d) = __calculate_im_pols(
                expressions,
                &mut sub,
                Some(im_pols.clone()),
                absolute_max,
                absolute_max,
                abs_max_d,
            );
            // write back
            expressions[exp_id] = sub;

            let mut e = match maybe_e {
                Some(v) => v,
                None => return (None, -1),
            };

            if (d as usize) > max_deg {
                *abs_max_d = (*abs_max_d).max(d);
                e.push(Value::from(exp_id as u64));
                return (Some(e), 1);
            }

            (Some(e), d)
        }

        // ==== default leaf ====
        _ => {
            let ed = exp["expDeg"].as_i64().unwrap_or(0);
            if ed == 0 {
                (Some(im_pols), 0)
            } else if max_deg < 1 {
                (None, -1)
            } else {
                (Some(im_pols), 1)
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
