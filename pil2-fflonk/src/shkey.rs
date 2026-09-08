//! Deriving the SHPLONK `f_i` packing.
//!
//! pil-fflonk read this from a pil1 shkey produced by pil-stark. pil2 has no
//! such file, so the packing has to be derived. The rules below were recovered
//! from a real shkey rather than invented, and each is checked against it in
//! the tests.
//!
//! **1. Group by (stage, opening-point set).** Two polynomials share an `f_i`
//! only if they are committed in the same stage *and* opened at the same
//! points. In the reference shkey `f0` and `f1` are both stage 0 and are
//! separate precisely because one is opened at `[0]` and the other at `[0, 1]`.
//!
//! **2. Bucket sizes must be an available root order.** `ShPlonkProver::
//! calculateRoots` looks up `omegas["w" + nPols]`, so a group can only be
//! packed into sizes for which the setup provides a root of unity. The
//! reference setup provides orders 1, 2, 3, 4 and 6, and every `f_i` in it has
//! one of those sizes.
//!
//! Which size to use is itself determined: it is the **largest divisor of the
//! group size that is an available root order**. That yields equal-sized
//! buckets, so a single `w{n}` opens every bucket in the group -- 6 + 3 would
//! need two different roots. It explains the reference exactly, including why
//! nine same-opening polynomials become 3 + 3 + 3 rather than 6 + 3 even though
//! the latter uses fewer commitments.
//!
//! **3. Degree follows the interleaving.** A combined polynomial holds its
//! components at stride `nPols`, so component `j` of degree `d` reaches index
//! `d * nPols + j` and the combined degree is the largest such index. This is
//! the same formula `CPolynomial::getDegree` uses.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::setup::{ShPlonkPol, ShPlonkStage, ShPlonkStagePol};

/// A polynomial awaiting placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub name: String,
    pub stage: u32,
    pub degree: u64,
    /// Opening points, as the shkey spells them (unsigned indices).
    pub opening_points: Vec<u32>,
}

/// Root orders the setup can open, read from the omega names it provides.
///
/// `w1`, `w2`, ... give the plain roots; `w2_1d2` and friends are their shifted
/// variants for non-zero opening points and are not bucket sizes themselves.
pub fn available_bucket_sizes(omegas: &BTreeMap<String, String>) -> Vec<u32> {
    let mut sizes: Vec<u32> = omegas
        .keys()
        .filter_map(|k| k.strip_prefix('w'))
        .filter(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
        .filter_map(|rest| rest.parse::<u32>().ok())
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The combined degree of `n_pols` interleaved components.
///
/// Component `j` of degree `d` occupies index `d * n_pols + j`.
pub fn combined_degree(degrees: &[u64]) -> u64 {
    let n = degrees.len() as u64;
    degrees.iter().enumerate().map(|(j, d)| d * n + j as u64).max().unwrap_or(0)
}

/// The bucket size for a group of `count` polynomials.
///
/// The largest divisor of `count` that has a root of unity. Dividing exactly
/// matters: every bucket in a group is then the same size, so one `w{n}` opens
/// all of them.
pub fn bucket_size_for(count: usize, sizes: &[u32]) -> Result<u32> {
    if count == 0 {
        bail!("an empty group has no bucket size");
    }
    sizes
        .iter()
        .copied()
        .filter(|&s| s > 0 && count % s as usize == 0)
        .max()
        .with_context(|| {
            format!(
                "cannot pack {count} polynomial(s): none of the available root orders {sizes:?} divides it, \
                 and unequal buckets would need more than one root"
            )
        })
}

/// Split `count` items into equal buckets of the derived size.
pub fn plan_buckets(count: usize, preferred: u32, sizes: &[u32]) -> Result<Vec<u32>> {
    if count == 0 {
        return Ok(vec![]);
    }
    if sizes.is_empty() {
        bail!("no root orders available, so no bucket size can be opened");
    }
    if preferred == 0 {
        bail!("preferred bucket size must be non-zero");
    }
    if !sizes.contains(&preferred) {
        bail!(
            "preferred bucket size {preferred} has no root of unity; available orders are {sizes:?}"
        );
    }

    let mut remaining = count;
    let mut plan = Vec::new();
    while remaining > 0 {
        let take = std::cmp::min(preferred as usize, remaining);
        // A short final bucket still has to be openable.
        let size = take as u32;
        if !sizes.contains(&size) {
            bail!(
                "a final bucket of {size} polynomial(s) is left over, and {size} has no root of unity; \
                 available orders are {sizes:?}"
            );
        }
        plan.push(size);
        remaining -= take;
    }
    Ok(plan)
}

/// Map a pil2 opening point onto pil-fflonk's convention.
///
/// pil2's StarkInfo carries signed row offsets (`-1` for the previous row);
/// every opening point in a pil-fflonk shkey is non-negative, and
/// `shplonk.cpp` reads them into a `u_int32_t`, so a negative value would wrap
/// rather than name a root.
///
/// An opening point is an exponent of the domain generator, and the domain has
/// order `domain_size`, so `-1` is the same point as `domain_size - 1`. The
/// convention is kept as pil-fflonk had it and the offset is reduced into it.
pub fn normalize_opening_point(point: i64, domain_size: u64) -> Result<u32> {
    if domain_size == 0 {
        bail!("domain size must be non-zero to reduce an opening point");
    }
    let n = domain_size as i128;
    let reduced = ((point as i128 % n) + n) % n;
    u32::try_from(reduced).map_err(|_| {
        anyhow::anyhow!("opening point {point} reduces to {reduced}, which does not fit the shkey's u32")
    })
}

/// The degree a polynomial occupies in its combined polynomial.
///
/// Recovered from the reference shkey and verified against every entry:
///
/// * **Constants** (stage 0) have degree `n`. They are public, so they carry no
///   blinding.
/// * **Committed** polynomials have degree `n + k + 1`, where `k` is the number
///   of points they are opened at: each opening leaks an evaluation, so a
///   blinding coefficient is added per opening plus one.
///
/// The quotient is not covered here -- see [`quotient_degree`].
pub fn committed_degree(domain_size: u64, stage: u32, n_opening_points: usize) -> u64 {
    if stage == 0 { domain_size } else { domain_size + n_opening_points as u64 + 1 }
}

/// The quotient polynomial's degree.
///
/// Follows the prover's own sizing in stage 4 of `pilfflonk_prover.cpp`:
/// `qDeg * N + maxPolsOpenings * (qDeg + 1)`. Both inputs come from the AIR
/// rather than from the packing, which is why this is separate from
/// [`committed_degree`].
pub fn quotient_degree(domain_size: u64, q_deg: u64, max_pols_openings: u64) -> u64 {
    q_deg * domain_size + max_pols_openings * (q_deg + 1)
}

/// The name under which SHPLONK looks up the root for a bucket of `n_pols`
/// opened at `point`.
///
/// From `ShPlonkProver::calculateRoots`: the plain root is `w{n}`, and a
/// non-zero opening point needs the shifted root `w{n}_{p}d{n}`. Opening point
/// zero needs no shifted root -- the code substitutes one directly.
///
/// NOTE ON SIGN: pil2's StarkInfo carries *signed* opening points (`-1` for the
/// previous row), while every opening point in pil-fflonk's shkey is
/// non-negative and `shplonk.cpp` reads them into a `u_int32_t`. A negative
/// point would therefore wrap rather than name a root. The name is spelled here
/// for completeness, but a setup that emits one has to generate the matching
/// omega, which pil-stark's generator never had to do.
pub fn omega_name(n_pols: u32, point: i64) -> String {
    if point == 0 { format!("w{n_pols}") } else { format!("w{n_pols}_{point}d{n_pols}") }
}

/// Every root of unity a packing needs the setup to provide.
///
/// Derivable, unlike the bucket size: it follows from the sizes and opening
/// points actually used. A setup missing one of these cannot open the
/// corresponding `f_i`.
pub fn required_omegas(f: &[ShPlonkPol]) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    for fi in f {
        let n = fi.pols.len() as u32;
        names.insert(omega_name(n, 0));
        for &p in &fi.opening_points {
            if p != 0 {
                names.insert(omega_name(n, p as i64));
            }
        }
    }
    names
}

/// Derive the `f_i` packing for a set of candidates.
///
/// Groups are ordered by (stage, opening points) so the result is
/// deterministic; `f_i` indices are assigned in that order. Bucket sizes follow
/// [`bucket_size_for`].
pub fn derive_f(candidates: &[Candidate], sizes: &[u32]) -> Result<Vec<ShPlonkPol>> {
    let mut groups: BTreeMap<(u32, Vec<u32>), Vec<&Candidate>> = BTreeMap::new();
    for c in candidates {
        let mut points = c.opening_points.clone();
        points.sort_unstable();
        points.dedup();
        groups.entry((c.stage, points)).or_default().push(c);
    }

    let mut out = Vec::new();
    let mut index = 0u32;

    for ((stage, points), members) in groups {
        let plan = plan_buckets(members.len(), bucket_size_for(members.len(), sizes)?, sizes)?;

        let mut taken = 0usize;
        for bucket in plan {
            let slice = &members[taken..taken + bucket as usize];
            taken += bucket as usize;

            let names: Vec<String> = slice.iter().map(|c| c.name.clone()).collect();
            let degrees: Vec<u64> = slice.iter().map(|c| c.degree).collect();

            out.push(ShPlonkPol {
                index,
                degree: combined_degree(&degrees),
                opening_points: points.clone(),
                pols: names.clone(),
                stages: vec![ShPlonkStage {
                    stage,
                    pols: slice
                        .iter()
                        .map(|c| ShPlonkStagePol { name: c.name.clone(), degree: c.degree })
                        .collect(),
                }],
            });
            index += 1;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::ShPlonkSetup;
    use serde_json::Value;

    const SHKEY: &str = include_str!("../tests/fixtures/pilfflonk.shkey.json");
    const VKEY: &str = include_str!("../tests/fixtures/pilfflonk.vkey");

    fn reference_f() -> Vec<ShPlonkPol> {
        let v: Value = serde_json::from_str(SHKEY).unwrap();
        serde_json::from_value(v["f"].clone()).unwrap()
    }

    fn reference_sizes() -> Vec<u32> {
        let setup = ShPlonkSetup::from_vkey_json(&serde_json::from_str(VKEY).unwrap()).unwrap();
        available_bucket_sizes(&setup.omegas)
    }

    /// The reference setup provides roots of order 1, 2, 3, 4 and 6 -- and only
    /// those are usable as bucket sizes.
    #[test]
    fn bucket_sizes_come_from_the_provided_roots() {
        assert_eq!(reference_sizes(), vec![1, 2, 3, 4, 6]);
    }

    /// Every f_i in the reference shkey has a size the setup can open. This is
    /// the constraint that makes stage 1's nine polynomials three buckets of
    /// three rather than one of nine.
    #[test]
    fn every_reference_bucket_size_is_openable() {
        let sizes = reference_sizes();
        for f in reference_f() {
            let n = f.pols.len() as u32;
            assert!(sizes.contains(&n), "f{} has {} polynomials, which has no root of unity", f.index, n);
        }
    }

    /// The degree formula must reproduce every declared degree.
    #[test]
    fn degree_formula_reproduces_the_reference() {
        for f in reference_f() {
            let degrees: Vec<u64> = f.stages.iter().flat_map(|s| s.pols.iter().map(|p| p.degree)).collect();
            assert_eq!(combined_degree(&degrees), f.degree, "f{} degree mismatch", f.index);
        }
    }

    /// Polynomials sharing a stage but opened differently must not share an
    /// f_i -- f0 and f1 in the reference are exactly that case.
    #[test]
    fn opening_points_split_a_stage() {
        let f = reference_f();
        let f0 = &f[0];
        let f1 = &f[1];
        assert_eq!(f0.stages[0].stage, f1.stages[0].stage, "both are stage 0");
        assert_ne!(f0.opening_points, f1.opening_points, "and they differ only by opening set");
    }

    /// Deriving from the reference's own polynomials must reproduce its shape:
    /// the same number of f_i, the same sizes, the same degrees.
    #[test]
    fn derivation_reproduces_the_reference_shape() {
        let reference = reference_f();
        let sizes = reference_sizes();

        let candidates: Vec<Candidate> = reference
            .iter()
            .flat_map(|f| {
                f.stages.iter().flat_map(move |s| {
                    s.pols.iter().map(move |p| Candidate {
                        name: p.name.clone(),
                        stage: s.stage,
                        degree: p.degree,
                        opening_points: f.opening_points.clone(),
                    })
                })
            })
            .collect();

        let derived = derive_f(&candidates, &sizes).unwrap();

        assert_eq!(derived.len(), reference.len(), "different number of combined polynomials");

        // Compare as multisets of (stage, opening points, size, degree): the
        // grouping is determined, the order within a group is not.
        let shape = |v: &[ShPlonkPol]| {
            let mut s: Vec<(u32, Vec<u32>, usize, u64)> = v
                .iter()
                .map(|f| (f.stages[0].stage, f.opening_points.clone(), f.pols.len(), f.degree))
                .collect();
            s.sort();
            s
        };
        assert_eq!(shape(&derived), shape(&reference));
    }

    /// The packing policy is an input. Given the reference's own choice, the
    /// derivation reproduces its layout; given a different one it does not,
    /// which is precisely why it cannot be inferred.
    #[test]
    fn bucket_plan_follows_the_requested_size() {
        assert_eq!(plan_buckets(9, 3, &[1, 2, 3, 4, 6]).unwrap(), vec![3, 3, 3]);
        assert_eq!(plan_buckets(9, 6, &[1, 2, 3, 4, 6]).unwrap(), vec![6, 3]);
        assert_eq!(plan_buckets(6, 6, &[1, 2, 3, 4, 6]).unwrap(), vec![6]);
        assert_eq!(plan_buckets(0, 3, &[1, 2, 3, 4, 6]).unwrap(), Vec::<u32>::new());
    }

    /// A size with no root of unity cannot be opened, whichever way it arises.
    #[test]
    fn refuses_a_size_without_a_root() {
        assert!(plan_buckets(10, 5, &[1, 2, 3, 4, 6]).is_err(), "5 has no root");
        // A short final bucket must be openable too: 7 items at 4 leaves 3, fine;
        // at 4 with only {2,4} available it leaves 3, which is not.
        assert!(plan_buckets(7, 4, &[2, 4]).is_err(), "leftover 3 has no root");
        assert!(plan_buckets(7, 4, &[1, 2, 3, 4]).is_ok());
    }

    /// The omega requirement follows from the packing, so it can be checked
    /// against what the reference setup actually provides.
    #[test]
    fn required_omegas_match_the_reference_setup() {
        let required = required_omegas(&reference_f());
        let provided: std::collections::BTreeSet<String> = {
            let setup = ShPlonkSetup::from_vkey_json(&serde_json::from_str(VKEY).unwrap()).unwrap();
            setup.omegas.keys().cloned().collect()
        };

        let missing: Vec<&String> = required.difference(&provided).collect();
        assert!(missing.is_empty(), "setup cannot open these buckets: {missing:?}");

        // The reference provides w1_1d1 which nothing uses; a superset is fine.
        assert!(required.contains("w2_1d2"));
        assert!(required.contains("w4_1d4"));
        assert!(required.contains("w6"), "f0 packs six polynomials");
    }

    /// The degree rule must reproduce every committed and constant degree in
    /// the reference.
    #[test]
    fn degree_rule_reproduces_the_reference() {
        let v: Value = serde_json::from_str(SHKEY).unwrap();
        let n: u64 = 1 << v["power"].as_u64().unwrap();

        for f in reference_f() {
            let k = f.opening_points.len();
            let stage = f.stages[0].stage;
            if stage == 4 {
                continue; // the quotient, sized separately
            }
            for pol in &f.stages[0].pols {
                assert_eq!(
                    committed_degree(n, stage, k),
                    pol.degree,
                    "stage {stage}, {k} opening point(s), polynomial {:?}",
                    pol.name
                );
            }
        }
    }

    /// Constants are unblinded and committed polynomials are not -- the
    /// difference is the whole point of the rule.
    #[test]
    fn constants_are_unblinded_and_committed_are_not() {
        let n = 256;
        assert_eq!(committed_degree(n, 0, 1), n, "a constant carries no blinding");
        assert_eq!(committed_degree(n, 0, 2), n, "even opened twice");
        assert_eq!(committed_degree(n, 1, 1), n + 2);
        assert_eq!(committed_degree(n, 1, 2), n + 3, "an extra opening costs an extra blinding factor");
    }

    /// The reference quotient is 780 with N = 256; the prover's formula gives
    /// that for qDeg = 3 and maxPolsOpenings = 3.
    #[test]
    fn quotient_degree_matches_the_reference() {
        assert_eq!(quotient_degree(256, 3, 3), 780);

        let q = reference_f().into_iter().find(|f| f.stages[0].stage == 4).unwrap();
        assert_eq!(q.degree, 780);
        assert_eq!(q.pols, vec!["Q"], "the quotient is packed alone");
    }

    /// The bucket size is not a free choice: it is the largest divisor of the
    /// group size with a root of unity, which is what makes every bucket in a
    /// group the same size and openable by one root.
    #[test]
    fn bucket_size_is_the_largest_dividing_root_order() {
        let sizes = [1, 2, 3, 4, 6];
        assert_eq!(bucket_size_for(6, &sizes).unwrap(), 6);
        assert_eq!(bucket_size_for(3, &sizes).unwrap(), 3);
        assert_eq!(bucket_size_for(4, &sizes).unwrap(), 4);
        assert_eq!(bucket_size_for(2, &sizes).unwrap(), 2);
        assert_eq!(bucket_size_for(1, &sizes).unwrap(), 1);
        // The case that distinguishes the rule: 9 takes 3, not 6, because 6
        // does not divide it and 6 + 3 would need two roots.
        assert_eq!(bucket_size_for(9, &sizes).unwrap(), 3);
    }

    /// Every group in the reference takes the size this rule predicts.
    #[test]
    fn bucket_rule_explains_every_reference_group() {
        let sizes = reference_sizes();
        let mut groups: BTreeMap<(u32, Vec<u32>), usize> = BTreeMap::new();
        for f in reference_f() {
            *groups.entry((f.stages[0].stage, f.opening_points.clone())).or_default() += f.pols.len();
        }
        for ((stage, points), total) in groups {
            let predicted = bucket_size_for(total, &sizes).unwrap();
            let actual = reference_f()
                .into_iter()
                .find(|f| f.stages[0].stage == stage && f.opening_points == points)
                .map(|f| f.pols.len() as u32)
                .unwrap();
            assert_eq!(predicted, actual, "stage {stage}, points {points:?}, {total} polynomials");
        }
    }

    #[test]
    fn refuses_a_group_no_root_order_divides() {
        // 5 with only {2,4,6} available: no equal split exists.
        let err = bucket_size_for(5, &[2, 4, 6]).unwrap_err().to_string();
        assert!(err.contains("cannot pack"), "unexpected error: {err}");
    }

    /// Signed pil2 offsets reduce into pil-fflonk's non-negative convention.
    #[test]
    fn signed_opening_points_reduce_into_the_shkey_convention() {
        let n = 256;
        assert_eq!(normalize_opening_point(0, n).unwrap(), 0);
        assert_eq!(normalize_opening_point(1, n).unwrap(), 1);
        // The previous row is the same point as the last one.
        assert_eq!(normalize_opening_point(-1, n).unwrap(), 255);
        assert_eq!(normalize_opening_point(-2, n).unwrap(), 254);
        // And it is idempotent on values already in range.
        assert_eq!(normalize_opening_point(255, n).unwrap(), 255);
    }

    /// The reference's own opening points are unchanged by the reduction,
    /// so adopting it cannot alter existing setups.
    #[test]
    fn reduction_leaves_the_reference_untouched() {
        let v: Value = serde_json::from_str(SHKEY).unwrap();
        let n: u64 = 1 << v["power"].as_u64().unwrap();
        for f in reference_f() {
            for p in f.opening_points {
                assert_eq!(normalize_opening_point(p as i64, n).unwrap(), p);
            }
        }
    }

    #[test]
    fn omega_names_follow_calculate_roots() {
        assert_eq!(omega_name(3, 0), "w3");
        assert_eq!(omega_name(3, 1), "w3_1d3");
        assert_eq!(omega_name(4, 1), "w4_1d4");
        // pil2 can ask for a previous-row opening; the name exists, but the
        // reference generator never produced such an omega.
        assert_eq!(omega_name(2, -1), "w2_-1d2");
    }

    #[test]
    fn refuses_when_no_roots_are_available() {
        assert!(plan_buckets(3, 3, &[]).is_err());
    }
}
