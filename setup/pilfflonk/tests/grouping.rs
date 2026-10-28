//! The grouping (pilfflonk/docs/protocol.md#grouping-rules): the golden of pil-fflonk's example
//! `all`, the rules on small cases, the errors, and properties of random inputs checked against a
//! brute force. The old system's files are cited at the versions of
//! pilfflonk/docs/README.md#references.
//!
//! # The golden
//!
//! `fixtures/grouping/all.json` holds the input and the output of the old system's grouping for the
//! example `all`, the one in pil-fflonk's `config/` (pilfflonk/docs/README.md#fixtures;
//! `pil-fflonk/pil/README.md`: `sm_all/all_main.pil`, pil-stark's
//! `test/cfiles/fflonk_gen_all_files.js`). It was extracted, without running any JS, from
//! `pil-fflonk/config/pilfflonk.fflonkinfo.json` and `pilfflonk.shkey.json`, the way
//! `fflonk_shkey.js` builds the input of shplonkjs's `setup`:
//!
//! - **The polynomials and their order** are the `setPolDefs` calls (`fflonk_shkey.js:34-153`),
//!   which the shkey's `polsNamesStage` lists by stage: stages 0 to 3 one after the other, then `Q`
//!   (stage 4, `:164`).
//! - **Their ids**: in stages 0 and 1, the position in `polsNamesStage` (pilcom numbers constants
//!   and commits in declaration order, and `all_main.pil` and its includes declare no arrays); in
//!   stages 2 and 3, `puCtx[0].h1Id`, `.h2Id`, `.zId`, `peCtx[0].zId`, `ciCtx[0].zId` and
//!   `imExp2cm["28"]`. `Q`, which has none, takes the next cm id, 21.
//! - **Their offsets**: `0` if the fflonkinfo's `evMap` has the column with `prime: false`, `1` if
//!   with `prime: true` (`:219-220`); `const` in stage 0, `cm` otherwise. `Permutation.a` and
//!   `Permutation.b` have no entry, and the old system drops them (`:239-241`;
//!   pilfflonk/docs/protocol.md#layout): they are listed as `unopened` and not grouped.
//! - **Their bounds** (`:216-226`): `N = 2^pilPower = 256` in stage 0, `N + 1 + |O|` in stages 1
//!   to 3, and for `Q` `qDeg·N + maxPolsOpenings·(qDeg + 1) = 3·256 + 3·4 = 780` (`:159-164`), the
//!   old system's degree of `Q`, one less than its number of coefficients.
//! - **`extraMuls = 2`**, from pil-stark's test (`README.md`); 9 `f_i` for 7 groups agree.
//! - **The expected result** is the shkey's `f`, in order: for each `f_i` its stage (the only one
//!   in `stages`), `openingPoints` as offsets, `degree`, and its polynomials in order with their
//!   bounds after the fusion (`stages[0].pols`). `powerW` is 12, a number in this shkey
//!   (shplonkjs writes a string when there is one `k`; the test reads either). The roots are the
//!   shkey's `w*`, which `shkey_roots` derives from the layout.

use std::collections::{BTreeMap, BTreeSet};

use num_bigint::BigUint;
use pilfflonk_setup::grouping::{fuse, group, GroupingError, GroupingParams, MAX_SEARCH_STEPS, MIN_POLS};
use pilfflonk_setup::layout::CommittedPol;
use proofman_pilfflonk::layout::is_valid_k;
use proofman_pilfflonk::{Layout, BN128_R};
use serde_json::Value;

// --- Helpers ----------------------------------------------------------------------------------

fn pol(name: &str, stage: u64, id: u64, offsets: &[i64], coefficients: u64) -> CommittedPol {
    CommittedPol { stage, id, name: name.to_string(), offsets: offsets.to_vec(), coefficients }
}

/// A column of stage `stage` whose bound is its own `O`'s, `N + |O| + 1`, or `N` if fixed
/// (pilfflonk/docs/protocol.md#degrees).
fn column(name: &str, stage: u64, id: u64, offsets: &[i64], n_bits: u64) -> CommittedPol {
    let n = 1u64 << n_bits;
    let coefficients = if stage == 0 { n } else { n + offsets.len() as u64 + 1 };
    pol(name, stage, id, offsets, coefficients)
}

fn params(n_bits: u64, extra_muls: u64, q_stage: u64) -> GroupingParams {
    GroupingParams { n_bits, extra_muls, q_stage }
}

/// Each `f_i` as `(stage, names, offsets, degree)`.
fn shape(layout: &Layout) -> Vec<(u64, Vec<&str>, Vec<i64>, u64)> {
    layout
        .0
        .iter()
        .map(|f| (f.stage, f.pols.iter().map(|p| p.name.as_str()).collect(), f.offsets.clone(), f.degree))
        .collect()
}

fn names_of(pols: &[CommittedPol]) -> Vec<(&str, Vec<i64>, u64)> {
    pols.iter().map(|p| (p.name.as_str(), p.offsets.clone(), p.coefficients)).collect()
}

// --- The golden -------------------------------------------------------------------------------

struct Golden {
    pols: Vec<CommittedPol>,
    params: GroupingParams,
    expected: Value,
}

fn golden() -> Golden {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/grouping/all.json")).unwrap();
    let u = |v: &Value| v.as_u64().unwrap();
    let offsets = |v: &Value| v.as_array().unwrap().iter().map(|o| o.as_i64().unwrap()).collect::<Vec<_>>();
    let pols = fixture["pols"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| CommittedPol {
            stage: u(&p["stage"]),
            id: u(&p["id"]),
            name: p["name"].as_str().unwrap().to_string(),
            offsets: offsets(&p["offsets"]),
            coefficients: u(&p["coefficients"]),
        })
        .collect();
    let params = params(u(&fixture["nBits"]), u(&fixture["extraMuls"]), u(&fixture["qStage"]));
    Golden { pols, params, expected: fixture["expected"].clone() }
}

fn modulus() -> BigUint {
    BigUint::parse_bytes(BN128_R.as_bytes(), 10).unwrap()
}

/// `5^((r−1)/d)`: a primitive `d`-th root of unity, for `d | r − 1`
/// (pilfflonk/docs/protocol.md#roots).
fn root_of_unity(d: &BigUint) -> BigUint {
    let r = modulus();
    let r_minus_1 = &r - 1u32;
    assert_eq!(&r_minus_1 % d, BigUint::ZERO);
    BigUint::from(5u32).modpow(&(r_minus_1 / d), &r)
}

/// The roots the old system's shkey holds for a layout (`shplonk.js:42-53`), derived from it:
/// `w{k}` = `w_k = 5^((r−1)/k)` for each `k` of the layout, `w{k}_{s}d{k}` =
/// `ω_{kN}^s` for each offset `s > 0` of an `f_i` of that `k`, and `w1_1d1` = `ω_N`, which
/// shplonkjs adds whatever the layout (`shplonk.js:53`).
fn shkey_roots(layout: &Layout, n_bits: u64) -> BTreeMap<String, BigUint> {
    let r = modulus();
    let n = BigUint::from(1u64 << n_bits);
    let mut roots = BTreeMap::new();
    for f in &layout.0 {
        let k = BigUint::from(f.k);
        roots.insert(format!("w{}", f.k), root_of_unity(&k));
        for &s in f.offsets.iter().filter(|&&s| s > 0) {
            let omega = root_of_unity(&(&k * &n));
            roots.insert(format!("w{}_{s}d{}", f.k, f.k), omega.modpow(&BigUint::from(s as u64), &r));
        }
    }
    roots.insert("w1_1d1".to_string(), root_of_unity(&n));
    roots
}

#[test]
fn example_all_is_grouped_as_the_old_system_did() {
    let Golden { pols, params, expected } = golden();
    let layout = group(&pols, &params).unwrap();
    let fused = fuse(&pols, &params).unwrap();
    let fused: BTreeMap<&str, &CommittedPol> = fused.iter().map(|p| (p.name.as_str(), p)).collect();
    let ids: BTreeMap<&str, u64> = pols.iter().map(|p| (p.name.as_str(), p.id)).collect();

    let expected_f = expected["f"].as_array().unwrap();
    assert_eq!(layout.0.len(), expected_f.len(), "the number of f_i");
    for (i, (f, e)) in layout.0.iter().zip(expected_f).enumerate() {
        let names: Vec<&str> = f.pols.iter().map(|p| p.name.as_str()).collect();
        let expected_names: Vec<&str> =
            e["pols"].as_array().unwrap().iter().map(|p| p["name"].as_str().unwrap()).collect();
        let expected_offsets: Vec<i64> = e["offsets"].as_array().unwrap().iter().map(|o| o.as_i64().unwrap()).collect();
        // The classes, the partitions and the order of the composition.
        assert_eq!(f.stage, e["stage"].as_u64().unwrap(), "f{i}'s stage");
        assert_eq!(names, expected_names, "f{i}'s polynomials, in order");
        assert_eq!(f.k, names.len() as u64, "f{i}'s k");
        assert_eq!(f.offsets, expected_offsets, "f{i}'s offsets");
        assert_eq!(f.degree, e["degree"].as_u64().unwrap(), "f{i}'s degree");
        for (p, e) in f.pols.iter().zip(e["pols"].as_array().unwrap()) {
            assert_eq!(p.id, ids[p.name.as_str()], "{}'s id", p.name);
            // The bounds after the fusion, for the O of the f.
            let fused = fused[p.name.as_str()];
            assert_eq!(fused.coefficients, e["coefficients"].as_u64().unwrap(), "{}'s bound", p.name);
            assert_eq!(fused.offsets, f.offsets, "{}'s offsets", p.name);
        }
    }

    // powerW, as a number whatever the file holds.
    let power_w = match &expected["powerW"] {
        Value::String(s) => s.parse::<u64>().unwrap(),
        v => v.as_u64().unwrap(),
    };
    assert_eq!(layout.power_w().unwrap(), power_w);

    // The roots: derived from the layout, the same ones and the same values as the shkey's.
    let roots = shkey_roots(&layout, params.n_bits);
    let expected_roots: BTreeMap<String, BigUint> = expected["roots"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), BigUint::parse_bytes(v.as_str().unwrap().as_bytes(), 10).unwrap()))
        .collect();
    assert_eq!(roots, expected_roots);
}

/// The roots (pilfflonk/docs/protocol.md#roots) on the golden's layout:
/// `x_j = xiSeed^(powerW/k)·ω_{kN}^s·w_k^j` are the `k` distinct roots of `x^k = ξ·ω_N^s`,
/// `ξ = xiSeed^powerW`.
#[test]
fn the_roots_of_example_all_are_those_of_xi_times_omega_to_the_offset() {
    let Golden { pols, params, .. } = golden();
    let layout = group(&pols, &params).unwrap();
    let r = modulus();
    let n = BigUint::from(1u64 << params.n_bits);
    let power_w = layout.power_w().unwrap();
    let xi_seed = BigUint::from(0x5eedu32);
    let xi = xi_seed.modpow(&BigUint::from(power_w), &r);
    let omega_n = root_of_unity(&n);
    for f in &layout.0 {
        let k = BigUint::from(f.k);
        let w_k = root_of_unity(&k);
        let omega_kn = root_of_unity(&(&k * &n));
        for &s in &f.offsets {
            let s = BigUint::from(u64::try_from(s).unwrap());
            let target = &xi * omega_n.modpow(&s, &r) % &r;
            let base = xi_seed.modpow(&BigUint::from(power_w / f.k), &r) * omega_kn.modpow(&s, &r) % &r;
            let roots: BTreeSet<BigUint> = (0..f.k).map(|j| &base * w_k.modpow(&BigUint::from(j), &r) % &r).collect();
            assert_eq!(roots.len() as u64, f.k);
            for x in &roots {
                assert_eq!(x.modpow(&k, &r), target);
            }
        }
    }
}

// --- Rule 1: classes and fusion ---------------------------------------------------------------

#[test]
fn the_documented_fusion_examples() {
    // {0}:4, {0,1}:2 stays two classes.
    let mut pols: Vec<CommittedPol> = (0..4).map(|i| column(&format!("a{i}"), 1, i, &[0], 3)).collect();
    pols.extend((4..6).map(|i| column(&format!("b{i}"), 1, i, &[0, 1], 3)));
    pols.push(pol("Q", 2, 6, &[0], 20));
    let fused = fuse(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(fused, pols);

    // {0}:5, {1}:1 gives {0}×5 and {0,1}×1, which gains a coefficient.
    let mut pols: Vec<CommittedPol> = (0..5).map(|i| column(&format!("a{i}"), 1, i, &[0], 3)).collect();
    pols.push(column("b", 1, 5, &[1], 3));
    pols.push(pol("Q", 2, 6, &[0], 20));
    let fused = fuse(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(fused[..5], pols[..5]);
    assert_eq!(names_of(&fused[5..6]), [("b", vec![0, 1], 8 + 3)]);
    assert_eq!(fused[6], pols[6], "Q does not move");
}

#[test]
fn a_class_moves_only_below_min_pols() {
    assert_eq!(MIN_POLS, 3);
    let with = |n_small: u64| {
        let mut pols: Vec<CommittedPol> = (0..n_small).map(|i| column(&format!("a{i}"), 1, i, &[0], 3)).collect();
        pols.push(column("u", 1, n_small, &[0, 1], 3));
        pols.push(pol("Q", 2, n_small + 1, &[0], 20));
        fuse(&pols, &params(3, 0, 2)).unwrap()
    };
    assert!(with(3).iter().filter(|p| p.stage == 1).any(|p| p.offsets == [0]), "3 polynomials stay");
    assert!(with(2).iter().filter(|p| p.stage == 1).all(|p| p.offsets == [0, 1]), "2 move");
    assert_eq!(with(2)[0].coefficients, 8 + 3);
}

#[test]
fn two_small_classes_move_to_the_union_and_merge() {
    let pols = vec![
        column("a", 1, 0, &[0], 3),
        column("b", 1, 1, &[1], 3),
        column("c", 1, 2, &[1], 3),
        column("d", 1, 3, &[0], 3),
        pol("Q", 2, 4, &[0], 20),
    ];
    let layout = group(&pols, &params(3, 0, 2)).unwrap();
    // One class {0,1}: a and d first (list of 0), b and c after (appended to it); reversed.
    assert_eq!(shape(&layout), [(1, vec!["c", "b", "d", "a"], vec![0, 1], 11 * 4 + 3), (2, vec!["Q"], vec![0], 20)]);
}

#[test]
fn a_fixed_column_that_moves_keeps_n_coefficients() {
    let pols = vec![
        column("A", 0, 0, &[0], 3),
        column("B", 0, 1, &[0, 1], 3),
        column("C", 0, 2, &[0, 1], 3),
        column("D", 0, 3, &[0, 1], 3),
        column("a", 1, 0, &[0], 3),
        pol("Q", 2, 1, &[0], 20),
    ];
    let fused = fuse(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(names_of(&fused[..1]), [("A", vec![0, 1], 8)]);
    assert_eq!(fused[4], pols[4], "alone in its stage, a stays: its O is the union");
}

#[test]
fn q_takes_no_part_in_the_classes() {
    // Stage 1 has {1} only: its union is {1}, whatever Q's offsets.
    let pols = vec![column("a", 1, 0, &[1], 3), pol("Q", 2, 1, &[0], 20)];
    assert_eq!(fuse(&pols, &params(3, 0, 2)).unwrap(), pols);
    // Q's pieces are one group of their own, in reverse order (the old system: the same fi).
    let pols = vec![column("a", 1, 0, &[0], 3), pol("Q0", 2, 1, &[0], 20), pol("Q1", 2, 2, &[0], 20)];
    let layout = group(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(shape(&layout), [(1, vec!["a"], vec![0], 10), (2, vec!["Q1", "Q0"], vec![0], 41)]);
}

#[test]
fn signed_offsets_move_to_the_union_too() {
    let mut pols = vec![column("m", 1, 0, &[-1], 3)];
    pols.extend((1..6).map(|i| column(&format!("z{i}"), 1, i, &[0], 3)));
    pols.push(column("p", 1, 6, &[1], 3));
    pols.push(pol("Q", 2, 7, &[0], 20));
    let fused = fuse(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(names_of(&fused[..1]), [("m", vec![-1, 0, 1], 8 + 4)]);
    assert_eq!(fused[1..6], pols[1..6]);
    assert_eq!(names_of(&fused[6..7]), [("p", vec![-1, 0, 1], 8 + 4)]);

    // The lists go by offset, -1 first: its class, met there first, is the first group. m and p
    // are inserted in it from the list of -1, m its own and p appended ({-1} moves before {1}).
    // The z are five, which no chunk is: the extra mul splits them.
    let layout = group(&pols, &params(3, 1, 2)).unwrap();
    assert_eq!(
        shape(&layout),
        [
            (1, vec!["p", "m"], vec![-1, 0, 1], 12 * 2 + 1),
            (1, vec!["z5", "z4"], vec![0], 10 * 2 + 1),
            (1, vec!["z3", "z2", "z1"], vec![0], 10 * 3 + 2),
            (2, vec!["Q"], vec![0], 20),
        ]
    );
}

// --- The groups, their order and the composition order ----------------------------------------

#[test]
fn a_polynomial_that_gains_the_first_offset_goes_after_its_class() {
    // b is opened at 1 only and moves to {0,1}: it is appended to the list of 0, after c.
    let pols = vec![
        column("a", 1, 0, &[0, 1], 3),
        column("b", 1, 1, &[1], 3),
        column("c", 1, 2, &[0, 1], 3),
        pol("Q", 2, 3, &[0], 20),
    ];
    let layout = group(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(shape(&layout)[0], (1, vec!["b", "c", "a"], vec![0, 1], 11 * 3 + 2));

    // b is opened at 0 only: it is in the list of 0 from the start, between a and c.
    let pols = vec![
        column("a", 1, 0, &[0, 1], 3),
        column("b", 1, 1, &[0], 3),
        column("c", 1, 2, &[0, 1], 3),
        pol("Q", 2, 3, &[0], 20),
    ];
    let layout = group(&pols, &params(3, 0, 2)).unwrap();
    assert_eq!(shape(&layout)[0], (1, vec!["c", "b", "a"], vec![0, 1], 11 * 3 + 2));
}

#[test]
fn the_groups_of_a_stage_go_in_the_order_they_are_met() {
    let group_of = |order: &[usize]| {
        let all = [
            column("x1", 1, 0, &[0], 3),
            column("y1", 1, 1, &[0, 1], 3),
            column("x2", 1, 2, &[0], 3),
            column("x3", 1, 3, &[0], 3),
        ];
        let mut pols: Vec<CommittedPol> = order.iter().map(|&i| all[i].clone()).collect();
        pols.push(pol("Q", 2, 4, &[0], 20));
        let layout = group(&pols, &params(3, 0, 2)).unwrap();
        layout.0.iter().map(|f| f.pols.iter().map(|p| p.name.clone()).collect::<Vec<_>>()).collect::<Vec<_>>()
    };
    assert_eq!(group_of(&[0, 1, 2, 3]), [vec!["x3", "x2", "x1"], vec!["y1"], vec!["Q"]]);
    assert_eq!(group_of(&[1, 0, 2, 3]), [vec!["y1"], vec!["x3", "x2", "x1"], vec!["Q"]]);
}

#[test]
fn the_layout_goes_by_stage_and_the_split_by_the_old_order() {
    // The old system numbers the stage-1 group first, from the list of 0, and the fixed one,
    // opened at 1 only, second. Both cost the same: the extra mul goes to the second in the old
    // order, the fixed one, and the layout puts it first (pilfflonk/docs/protocol.md#layout).
    let mut pols: Vec<CommittedPol> = (0..4).map(|i| pol(&format!("c{i}"), 0, i, &[1], 10)).collect();
    pols.extend((0..4).map(|i| pol(&format!("a{i}"), 1, i, &[0], 10)));
    pols.push(pol("Q", 2, 4, &[0], 5));
    let layout = group(&pols, &params(3, 1, 2)).unwrap();
    assert_eq!(
        shape(&layout),
        [
            (0, vec!["c3", "c2"], vec![1], 21),
            (0, vec!["c1", "c0"], vec![1], 21),
            (1, vec!["a3", "a2", "a1", "a0"], vec![0], 43),
            (2, vec!["Q"], vec![0], 5),
        ]
    );
}

// --- Rule 3: the split in f_i -----------------------------------------------------------------

#[test]
fn between_equal_groups_the_first_combination_wins() {
    // (c_1, c_2, c_Q) = (0, 1, 0) comes before (1, 0, 0), and (1, 0, 0) is only as good.
    let mut pols: Vec<CommittedPol> = (0..4).map(|i| pol(&format!("a{i}"), 1, i, &[0], 10)).collect();
    pols.extend((0..4).map(|i| pol(&format!("b{i}"), 2, 4 + i, &[0], 10)));
    pols.push(pol("Q", 3, 8, &[0], 5));
    let layout = group(&pols, &params(3, 1, 3)).unwrap();
    assert_eq!(
        shape(&layout),
        [
            (1, vec!["a3", "a2", "a1", "a0"], vec![0], 43),
            (2, vec!["b3", "b2"], vec![0], 21),
            (2, vec!["b1", "b0"], vec![0], 21),
            (3, vec!["Q"], vec![0], 5),
        ]
    );
}

#[test]
fn the_costs_are_compared_from_the_greatest() {
    // Splitting a costs [21, 103, 5] and splitting b [43, 51, 5]: in decreasing order, b's is
    // better (51 < 103), though a's is better in the groups' order (21 < 43), and in increasing.
    let mut pols: Vec<CommittedPol> = (0..4).map(|i| pol(&format!("a{i}"), 1, i, &[0], 10)).collect();
    pols.extend((0..4).map(|i| pol(&format!("b{i}"), 2, 4 + i, &[0], 25)));
    pols.push(pol("Q", 3, 8, &[0], 5));
    let layout = group(&pols, &params(3, 1, 3)).unwrap();
    let degrees: Vec<(u64, u64, u64)> = layout.0.iter().map(|f| (f.stage, f.k, f.degree)).collect();
    assert_eq!(degrees, [(1, 4, 43), (2, 2, 51), (2, 2, 51), (3, 1, 5)]);
}

#[test]
fn a_chunk_has_k_with_k_times_n_dividing_r_minus_1() {
    let four = |n_bits: u64, extra_muls: u64| {
        let mut pols: Vec<CommittedPol> = (0..4).map(|i| column(&format!("a{i}"), 1, i, &[0], n_bits)).collect();
        pols.push(pol("Q", 2, 4, &[0], 5));
        group(&pols, &params(n_bits, extra_muls, 2)).map(|l| l.0.iter().map(|f| f.k).collect::<Vec<_>>())
    };
    // v₂(4) + 26 ≤ 28, but not + 27: then 4 = 2 + 2.
    assert_eq!(four(26, 0).unwrap(), [4, 1]);
    assert_eq!(
        four(27, 0),
        Err(GroupingError::NoValidPartition { n_groups: 2, extra_muls: 0, n_bits: 27, max_extra_muls: 3 })
    );
    assert_eq!(four(27, 1).unwrap(), [2, 2, 1]);
    // With N = 2^28 only odd k: 4 = 1 + 3.
    assert_eq!(four(28, 1).unwrap(), [1, 3, 1]);
    // Five in one chunk: 5 does not divide r - 1.
    let mut pols: Vec<CommittedPol> = (0..5).map(|i| column(&format!("a{i}"), 1, i, &[0], 3)).collect();
    pols.push(pol("Q", 2, 5, &[0], 5));
    assert!(matches!(group(&pols, &params(3, 0, 2)), Err(GroupingError::NoValidPartition { .. })));
    assert_eq!(group(&pols, &params(3, 1, 2)).unwrap().0.iter().map(|f| f.k).collect::<Vec<_>>(), [2, 3, 1]);
}

// --- Errors ------------------------------------------------------------------------------------

#[test]
fn what_cannot_be_grouped_is_refused() {
    let q = || pol("Q", 2, 9, &[0], 5);
    let a = || column("a", 1, 0, &[0], 3);
    let refuse = |pols: Vec<CommittedPol>, params: GroupingParams| {
        let error = group(&pols, &params).unwrap_err();
        let fused = fuse(&pols, &params);
        let of_rule_3 = matches!(error, GroupingError::TooManyExtraMuls { .. } | GroupingError::SearchTooLarge { .. });
        if !matches!(error, GroupingError::NoQ { .. }) && !of_rule_3 {
            assert_eq!(fused, Err(error.clone()), "fuse refuses it too");
        }
        error
    };
    let ok = params(3, 0, 2);

    assert_eq!(refuse(vec![a(), q()], params(29, 0, 2)), GroupingError::NBits { n_bits: 29 });
    assert_eq!(refuse(vec![a(), q()], params(3, 0, 0)), GroupingError::QStage);
    assert!(matches!(refuse(vec![a(), column("b", 3, 1, &[0], 3), q()], ok), GroupingError::Stage { stage: 3, .. }));
    assert!(matches!(refuse(vec![pol("a", 1, 0, &[], 3), q()], ok), GroupingError::NotOpened { .. }));
    assert!(matches!(refuse(vec![pol("a", 1, 0, &[1, 0], 3), q()], ok), GroupingError::Offsets { .. }));
    assert!(matches!(refuse(vec![pol("a", 1, 0, &[0, 0], 3), q()], ok), GroupingError::Offsets { .. }));
    assert!(matches!(refuse(vec![a(), pol("Q", 2, 9, &[0, 1], 5)], ok), GroupingError::QOffsets { .. }));
    assert!(matches!(refuse(vec![pol("a", 1, 0, &[0], 0), q()], ok), GroupingError::NoCoefficients { .. }));
    assert!(matches!(
        refuse(vec![a(), column("b", 1, 0, &[0], 3), q()], ok),
        GroupingError::DuplicateId { kind: "cm", id: 0, .. }
    ));
    // A fixed column and a committed one of the same index are two polynomials.
    assert!(group(&[column("A", 0, 0, &[0], 3), a(), q()], &ok).is_ok());
    assert!(matches!(refuse(vec![a(), column("a", 1, 1, &[0], 3), q()], ok), GroupingError::DuplicateName { .. }));
    assert_eq!(refuse(vec![a()], ok), GroupingError::NoQ { q_stage: 2 });
    assert!(fuse(&[a()], &ok).is_ok(), "fuse does not need Q");

    // Three polynomials in two groups: one extra mul at most (setup.js:231-232).
    let pols = vec![a(), column("b", 1, 1, &[0], 3), q()];
    assert_eq!(group(&pols, &params(3, 1, 2)).unwrap().0.len(), 3);
    assert_eq!(
        refuse(pols, params(3, 2, 2)),
        GroupingError::TooManyExtraMuls { extra_muls: 2, n_pols: 3, n_groups: 2 }
    );

    // A search too large (the bound itself is in no_search_is_larger_than_the_bound).
    let mut pols: Vec<CommittedPol> =
        (0..40u64).flat_map(|s| (0..8).map(move |i| column(&format!("p{s}_{i}"), 1 + s, 8 * s + i, &[0], 3))).collect();
    pols.push(pol("Q", 41, 320, &[0], 5));
    assert!(matches!(refuse(pols, params(3, 100, 41)), GroupingError::SearchTooLarge { extra_muls: 100, .. }));

    // A bound that overflows: 2·(2^64 − 1).
    let pols = vec![pol("a", 1, 0, &[0], u64::MAX), pol("b", 1, 1, &[0], u64::MAX), q()];
    assert_eq!(group(&pols, &params(3, 0, 2)), Err(GroupingError::Overflow));
    // Or when the fusion adds its coefficient.
    let pols = vec![pol("a", 1, 0, &[0], u64::MAX), column("b", 1, 1, &[0, 1], 3), q()];
    assert_eq!(fuse(&pols, &params(3, 0, 2)), Err(GroupingError::Overflow));
}

/// A class of 5, 7, 10, 11, … columns cannot be one chunk
/// (pilfflonk/docs/protocol.md#grouping-errors): with too few extra muls there is no valid
/// partition, and the message says that more `--extra-muls` allow one.
#[test]
fn no_valid_partition_says_more_extra_muls_help() {
    // Three classes of five (stages 1 to 3), and Q: each needs one extra chunk at least.
    let mut pols: Vec<CommittedPol> =
        (0..3u64).flat_map(|s| (0..5).map(move |i| column(&format!("p{s}_{i}"), 1 + s, 5 * s + i, &[0], 8))).collect();
    pols.push(pol("Q", 4, 15, &[0], 300));
    for extra_muls in [0, 1, 2] {
        let error = group(&pols, &params(8, extra_muls, 4)).unwrap_err();
        assert_eq!(
            error,
            GroupingError::NoValidPartition { n_groups: 4, extra_muls, n_bits: 8, max_extra_muls: 12 },
            "{extra_muls} extra muls"
        );
        let message = error.to_string();
        assert!(message.contains("a larger --extra-muls, up to 12, allows smaller chunks"), "{message}");
    }
    // Three: [2,3] in each class.
    let layout = group(&pols, &params(8, 3, 4)).unwrap();
    assert_eq!(layout.0.iter().map(|f| f.k).collect::<Vec<_>>(), [2, 3, 2, 3, 2, 3, 1]);
    // And too many is refused with its own message.
    let error = group(&pols, &params(8, 13, 4)).unwrap_err();
    assert_eq!(error, GroupingError::TooManyExtraMuls { extra_muls: 13, n_pols: 16, n_groups: 4 });
    assert!(
        error
            .to_string()
            .contains("so at most 12 extra muls (pilfflonk/docs/protocol.md#grouping-errors): lower --extra-muls"),
        "{error}"
    );
}

/// The exhaustive search of rule 3 is bounded (`MAX_SEARCH_STEPS`): an `extraMuls` that would
/// make it too large is refused before anything is enumerated, with a message that says to lower
/// it, and the defaults stay far below the bound, also for large AIRs.
#[test]
fn no_search_is_larger_than_the_bound() {
    // Forty stages of eight columns: 280 extra muls are possible, and the combinations of a
    // hundred of them (compositions of 100 in 40 parts of at most 7) are astronomically many.
    let mut pols: Vec<CommittedPol> =
        (0..40u64).flat_map(|s| (0..8).map(move |i| column(&format!("p{s}_{i}"), 1 + s, 8 * s + i, &[0], 3))).collect();
    pols.push(pol("Q", 41, 320, &[0], 5));
    let started = std::time::Instant::now();
    let error = group(&pols, &params(3, 100, 41)).unwrap_err();
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "refused in {:?}", started.elapsed());
    match &error {
        GroupingError::SearchTooLarge { extra_muls: 100, steps, limit } => {
            assert_eq!(*limit, MAX_SEARCH_STEPS);
            assert!(*steps > MAX_SEARCH_STEPS);
        }
        other => panic!("{other:?}"),
    }
    assert!(error.to_string().contains("lower --extra-muls"), "{error}");
    // With the default, the same AIR is grouped: two of its classes are split once.
    let layout = group(&pols, &params(3, 2, 41)).unwrap();
    assert_eq!(layout.0.len(), 41 + 2);

    // A class of 500 columns (a large AIR's stage) with the default extra muls: within the bound.
    let mut pols: Vec<CommittedPol> = (0..500).map(|i| column(&format!("a{i}"), 1, i, &[0, 1], 10)).collect();
    pols.push(pol("Q", 2, 500, &[0], 3000));
    let layout = group(&pols, &params(10, 2, 2)).unwrap();
    assert_eq!(layout.0.iter().map(|f| f.pols.len()).sum::<usize>(), 501);
    assert_eq!(layout.0.len(), 1 + 3);
}

// --- Properties ---------------------------------------------------------------------------------

/// splitmix64: a deterministic generator, so that a failure can be replayed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len() as u64) as usize]
    }
}

/// A random AIR: up to 3 stages, up to 14 columns whose offsets come from a small palette (so
/// that classes have several polynomials and some are small), `Q` in 1 to 3 pieces, and bounds
/// either [`column`]'s or small random ones (which make ties likely).
fn random_input(rng: &mut Rng) -> (Vec<CommittedPol>, GroupingParams) {
    let n_stages = 1 + rng.below(3);
    let q_stage = n_stages + 1;
    let n_bits = *rng.pick(&[3, 4, 8, 26, 27, 28]);
    let subsets: [&[i64]; 9] = [&[0], &[1], &[0, 1], &[-1, 0], &[-1, 0, 1], &[2], &[0, 2], &[-2, 1], &[-1]];
    let palette: Vec<&[i64]> = (0..1 + rng.below(3)).map(|_| *rng.pick(&subsets)).collect();
    let small_bounds = rng.below(2) == 0;

    let mut pols = Vec::new();
    let mut ids = [0u64; 2];
    for i in 0..1 + rng.below(14) {
        let stage = rng.below(n_stages + 1);
        let offsets = *rng.pick(&palette);
        let id = &mut ids[usize::from(stage != 0)];
        let mut p = column(&format!("p{i}"), stage, *id, offsets, n_bits);
        *id += 1;
        if small_bounds {
            p.coefficients = 1 + rng.below(12);
        }
        pols.push(p);
    }
    for j in 0..1 + rng.below(3) {
        let coefficients = if small_bounds { 1 + rng.below(12) } else { (1u64 << n_bits) * 3 + 7 };
        pols.push(pol(&format!("Q{j}"), q_stage, ids[1], &[0], coefficients));
        ids[1] += 1;
    }
    // Q first sometimes: the input order is the caller's.
    if rng.below(4) == 0 {
        pols.rotate_right(1);
    }
    (pols, params(n_bits, rng.below(5), q_stage))
}

/// Rule 1, written again from its statement (pilfflonk/docs/protocol.md#grouping-rules): each
/// polynomial's `O` and bound after the fusion.
fn expected_fusion(pols: &[CommittedPol], q_stage: u64) -> Vec<(Vec<i64>, u64)> {
    pols.iter()
        .map(|p| {
            if p.stage == q_stage {
                return (p.offsets.clone(), p.coefficients);
            }
            let stage: Vec<&CommittedPol> = pols.iter().filter(|o| o.stage == p.stage).collect();
            let union: BTreeSet<i64> = stage.iter().flat_map(|o| o.offsets.iter().copied()).collect();
            let union: Vec<i64> = union.into_iter().collect();
            let class = stage.iter().filter(|o| o.offsets == p.offsets).count();
            if class < MIN_POLS && p.offsets != union {
                let gained = if p.stage == 0 { 0 } else { (union.len() - p.offsets.len()) as u64 };
                (union, p.coefficients + gained)
            } else {
                (p.offsets.clone(), p.coefficients)
            }
        })
        .collect()
}

fn cost(bounds: &[u64]) -> u64 {
    let k = bounds.len() as u64;
    bounds.iter().enumerate().map(|(j, &d)| d * k + j as u64).max().unwrap()
}

/// For each number of chunks, the best split of a group of bounds `bounds`, by brute force over
/// the positions of the cuts: the least cost, and the smallest sizes in lexicographic order that
/// reach it, among the non-decreasing ones of sizes `k` with `valid[k]` (`k·N | r − 1`).
fn brute_splits(bounds: &[u64], valid: &[bool]) -> BTreeMap<usize, (u64, Vec<usize>)> {
    let len = bounds.len();
    let mut best: BTreeMap<usize, (u64, Vec<usize>)> = BTreeMap::new();
    for cuts in 0u64..1 << (len - 1) {
        let mut sizes = Vec::new();
        let mut start = 0;
        for end in 1..=len {
            if end == len || cuts >> (end - 1) & 1 == 1 {
                sizes.push(end - start);
                start = end;
            }
        }
        if sizes.windows(2).any(|w| w[0] > w[1]) || !sizes.iter().all(|&k| valid[k]) {
            continue;
        }
        let mut start = 0;
        let mut split_cost = 0;
        for &k in &sizes {
            split_cost = split_cost.max(cost(&bounds[start..start + k]));
            start += k;
        }
        let entry = best.entry(sizes.len()).or_insert((split_cost, sizes.clone()));
        if (split_cost, &sizes) < (entry.0, &entry.1) {
            *entry = (split_cost, sizes);
        }
    }
    best
}

/// The least vector of the groups' costs in decreasing order over every combination of chunks
/// with `extra_muls` more than groups, by brute force.
fn brute_best_costs(options: &[BTreeMap<usize, (u64, Vec<usize>)>], extra_muls: usize) -> Option<Vec<u64>> {
    fn walk(
        options: &[BTreeMap<usize, (u64, Vec<usize>)>],
        left: usize,
        costs: &mut Vec<u64>,
        best: &mut Option<Vec<u64>>,
    ) {
        let Some((group, rest)) = options.split_first() else {
            if left == 0 {
                let mut sorted = costs.clone();
                sorted.sort_by(|a, b| b.cmp(a));
                if best.as_ref().is_none_or(|b| sorted < *b) {
                    *best = Some(sorted);
                }
            }
            return;
        };
        for (&chunks, (cost, _)) in group {
            if chunks - 1 <= left {
                costs.push(*cost);
                walk(rest, left - (chunks - 1), costs, best);
                costs.pop();
            }
        }
    }
    let mut best = None;
    walk(options, extra_muls, &mut Vec::new(), &mut best);
    best
}

#[test]
fn random_inputs_are_grouped_by_the_rules() {
    let mut rng = Rng(0x00c0_ffee_f1f0_2026);
    let (mut grouped, mut refused, mut with_moves, mut with_splits) = (0, 0, 0, 0);
    for case in 0..3000 {
        let (pols, params) = random_input(&mut rng);
        let context = format!("case {case}: {params:?}, {pols:?}");
        let result = group(&pols, &params);
        assert_eq!(result, group(&pols, &params), "{context}: deterministic");

        // Rule 1.
        let fused = fuse(&pols, &params).unwrap();
        let expected = expected_fusion(&pols, params.q_stage);
        for (f, (offsets, coefficients)) in fused.iter().zip(&expected) {
            assert_eq!((&f.offsets, f.coefficients), (offsets, *coefficients), "{context}: {}'s fusion", f.name);
        }
        if pols.iter().zip(&fused).any(|(p, f)| p.offsets != f.offsets) {
            with_moves += 1;
        }
        let classes: BTreeSet<(u64, &[i64])> = fused.iter().map(|p| (p.stage, p.offsets.as_slice())).collect();
        let n_groups = classes.len();
        let valid: Vec<bool> = (0..=pols.len() as u64).map(|k| is_valid_k(k, params.n_bits)).collect();
        let max_extra_muls = pols.len() - n_groups;

        let layout = match result {
            Ok(layout) => layout,
            Err(GroupingError::TooManyExtraMuls { extra_muls, n_pols, n_groups: n }) => {
                assert!(params.extra_muls as usize > max_extra_muls, "{context}");
                assert_eq!((extra_muls, n_pols, n as usize), (params.extra_muls, pols.len() as u64, n_groups));
                refused += 1;
                continue;
            }
            Err(GroupingError::NoValidPartition { .. }) => {
                assert!(params.extra_muls as usize <= max_extra_muls, "{context}");
                // No combination of numbers of chunks with valid sizes, whatever the order of each
                // group's polynomials, which only changes the costs.
                let options: Vec<_> = classes
                    .iter()
                    .map(|&(stage, offsets)| {
                        let in_group =
                            |p: &&CommittedPol| p.stage == stage && (stage == params.q_stage || p.offsets == offsets);
                        let bounds: Vec<u64> = fused.iter().filter(in_group).map(|p| p.coefficients).collect();
                        brute_splits(&bounds, &valid)
                    })
                    .collect();
                assert_eq!(brute_best_costs(&options, params.extra_muls as usize), None, "{context}");
                refused += 1;
                continue;
            }
            Err(e) => panic!("{context}: {e}"),
        };
        grouped += 1;
        if params.extra_muls > 0 {
            with_splits += 1;
        }

        // Every f: k polynomials, kN | r - 1, of one stage and one class, by stage, Q's last.
        let by_name: BTreeMap<&str, (&CommittedPol, &CommittedPol)> =
            pols.iter().zip(&fused).map(|(p, f)| (p.name.as_str(), (p, f))).collect();
        let mut covered = BTreeSet::new();
        for (i, f) in layout.0.iter().enumerate() {
            assert!(f.k >= 1 && f.k == f.pols.len() as u64, "{context}: f{i}");
            assert!(is_valid_k(f.k, params.n_bits), "{context}: f{i} has k = {}", f.k);
            let mut bounds = Vec::new();
            for lp in &f.pols {
                let (p, fp) = by_name[lp.name.as_str()];
                assert_eq!((p.stage, p.id), (f.stage, lp.id), "{context}: {} in f{i}", p.name);
                assert_eq!(fp.offsets, f.offsets, "{context}: {} in f{i}", p.name);
                assert!(covered.insert(p.name.as_str()), "{context}: {} in two f", p.name);
                bounds.push(fp.coefficients);
            }
            assert_eq!(f.degree, cost(&bounds), "{context}: f{i}'s degree");
        }
        assert_eq!(covered.len(), pols.len(), "{context}: every polynomial is in an f");
        for p in &pols {
            let f = layout.0.iter().find(|f| f.pols.iter().any(|lp| lp.name == p.name)).unwrap();
            assert!(p.offsets.iter().all(|s| f.offsets.contains(s)), "{context}: {} at each of its offsets", p.name);
        }
        assert!(layout.0.windows(2).all(|w| w[0].stage <= w[1].stage), "{context}: by stage");
        assert_eq!(layout.0.last().unwrap().stage, params.q_stage, "{context}");
        assert_eq!(layout.0.len(), n_groups + params.extra_muls as usize, "{context}: #groups + extraMuls");

        // Rule 3 against the brute force: each group's chunks, one after the other in the
        // layout, are the best split for their number; the combination is the best one.
        let mut groups: Vec<(Vec<u64>, Vec<usize>)> = Vec::new();
        for (i, f) in layout.0.iter().enumerate() {
            let bounds: Vec<u64> = f.pols.iter().map(|lp| by_name[lp.name.as_str()].1.coefficients).collect();
            let same = i > 0 && layout.0[i - 1].stage == f.stage && layout.0[i - 1].offsets == f.offsets;
            match groups.last_mut() {
                Some((b, sizes)) if same => {
                    b.extend(bounds);
                    sizes.push(f.pols.len());
                }
                _ => groups.push((bounds, vec![f.pols.len()])),
            }
        }
        assert_eq!(groups.len(), n_groups, "{context}: a group's chunks are consecutive");
        let options: Vec<_> = groups.iter().map(|(bounds, _)| brute_splits(bounds, &valid)).collect();
        let mut costs = Vec::new();
        for ((_, sizes), options) in groups.iter().zip(&options) {
            let (best_cost, best_sizes) = &options[&sizes.len()];
            assert_eq!(sizes, best_sizes, "{context}: the first best split");
            costs.push(*best_cost);
        }
        costs.sort_by(|a, b| b.cmp(a));
        assert_eq!(
            Some(costs),
            brute_best_costs(&options, params.extra_muls as usize),
            "{context}: the best combination"
        );
    }
    // The generator reaches both outcomes, fusions and splits.
    assert!(
        grouped > 1000 && refused > 100 && with_moves > 300 && with_splits > 300,
        "{grouped} grouped, {refused} refused, {with_moves} with fusions, {with_splits} with extra muls"
    );
}
