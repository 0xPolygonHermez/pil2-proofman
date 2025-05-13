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

/// Top-level: mirrors your JS `calculateImPols`
pub fn calculate_im_pols(expressions: &mut Vec<Value>, exp: &mut Value, max_deg: usize) -> (Vec<usize>, isize) {
    let absolute_max = max_deg;
    let mut abs_max_d: isize = 0;

    // JS: console.log(expressions.length, imPols.length, maxDeg);
    println!("{} {} {}", expressions.len(), 0, max_deg);

    // recurse
    let (im_pols, rd) = __calculate_im_pols(
        expressions,
        exp,
        Vec::new(), // start with empty Vec<usize>
        max_deg,
        absolute_max,
        &mut abs_max_d,
    );

    // JS: console.log(expressions.length, imPols.length, maxDeg);
    println!("{} {} {}", expressions.len(), im_pols.len(), max_deg);
    // JS: console.log("!!!!!!!!! re: ", re);
    println!("!!!!!!!!! re: {:?}", im_pols);
    // JS: console.log("!!!!!!!!! rd: ", rd);
    println!("!!!!!!!!! rd: {}", rd);

    // return as JS: [ re, Math.max(rd, absMaxD) - 1 ]
    let raw_q = rd.max(abs_max_d) - 1;
    let q_deg = if raw_q < 0 { -1 } else { raw_q };
    (im_pols, q_deg)
}

fn __calculate_im_pols(
    expressions: &mut Vec<Value>,
    exp: &mut Value,
    mut im_pols: Vec<usize>,
    max_deg: usize,
    absolute_max: usize,
    abs_max_d: &mut isize,
) -> (Vec<usize>, isize) {
    let op = exp["op"].as_str().unwrap();

    // ---- false pathway ----
    // if JS had imPols===false it returns [false, -1]
    // we never actually pass a `false` Vec; instead we signal by returning d<0
    match op {
        // ---- add / sub ----
        "add" | "sub" => {
            let mut md = 0isize;
            for child in exp["values"].as_array_mut().unwrap() {
                let (next_pol, d) =
                    __calculate_im_pols(expressions, child, im_pols.clone(), max_deg, absolute_max, abs_max_d);
                if d < 0 {
                    return (Vec::new(), -1);
                }
                im_pols = next_pol;
                md = md.max(d);
            }
            (im_pols, md)
        }

        // ---- mul ----
        "mul" => {
            // 1) read expDeg before borrowing .values
            let deg_here = exp["expDeg"].as_i64().unwrap_or(0) as usize;
            let vals = exp["values"].as_array_mut().unwrap();

            // 2) constant-fold a deg-0 leaf
            for i in 0..2 {
                let sop = vals[i]["op"].as_str().unwrap();
                let sdeg = vals[i]["expDeg"].as_i64().unwrap_or(0);
                if !["add", "sub", "mul", "exp"].contains(&sop) && sdeg == 0 {
                    return __calculate_im_pols(
                        expressions,
                        &mut vals[1 - i],
                        im_pols,
                        max_deg,
                        absolute_max,
                        abs_max_d,
                    );
                }
            }

            // 3) if within max_deg, shortcut
            if deg_here <= max_deg {
                return (im_pols, deg_here as isize);
            }

            // 4) try all splits l+r=max_deg
            let mut best: Option<(Vec<usize>, isize)> = None;
            for l in 0..=max_deg {
                let r = max_deg - l;
                // left
                let (left_pol, d1) =
                    __calculate_im_pols(expressions, &mut vals[0], im_pols.clone(), l, absolute_max, abs_max_d);
                if d1 < 0 {
                    continue;
                }
                // right
                let (right_pol, d2) =
                    __calculate_im_pols(expressions, &mut vals[1], left_pol.clone(), r, absolute_max, abs_max_d);
                if d2 < 0 {
                    continue;
                }

                let combined = d1 + d2;
                let candidate = right_pol.clone();
                let pick = best.as_ref().map_or(true, |(bp, _)| candidate.len() < bp.len());
                if pick {
                    best = Some((candidate.clone(), combined));
                }
                if let Some((bp, _)) = &best {
                    if bp.len() == im_pols.len() {
                        return (bp.clone(), combined);
                    }
                }
            }
            if let Some((bp, d)) = best {
                (bp, d)
            } else {
                (Vec::new(), -1)
            }
        }

        // ---- exp ----
        "exp" => {
            if max_deg < 1 {
                return (Vec::new(), -1);
            }
            let exp_id = exp["id"].as_u64().unwrap() as usize;

            // already in im_pols?
            if im_pols.contains(&exp_id) {
                return (im_pols, 1);
            }

            // recurse at absolute_max on the referenced sub-expr
            let mut sub = expressions[exp_id].clone();
            let (mut e, d) =
                __calculate_im_pols(expressions, &mut sub, im_pols.clone(), absolute_max, absolute_max, abs_max_d);
            // write back any nested mutations
            expressions[exp_id] = sub;

            if d < 0 {
                return (Vec::new(), -1);
            }
            if (d as usize) > max_deg {
                *abs_max_d = (*abs_max_d).max(d);
                e.push(exp_id);
                return (e, 1);
            }
            (e, d)
        }

        // ---- leaf/default ----
        _ => {
            let ed = exp["expDeg"].as_i64().unwrap_or(0);
            if ed == 0 {
                (im_pols, 0)
            } else if max_deg < 1 {
                (Vec::new(), -1)
            } else {
                (im_pols, 1)
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
