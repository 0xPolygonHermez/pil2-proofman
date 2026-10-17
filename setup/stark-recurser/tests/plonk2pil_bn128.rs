//! plonk2pil over BN128: a circuit circom compiles for BN128 is read and converted to PLONK gates and
//! additions by the same generic code the Goldilocks recursion runs, and the result is checked on a
//! real witness of the circuit.
//!
//! The circuit is `fixtures/bn128/arith.circom`, compiled here with the committed circom
//! (`setup/circom`), so the r1cs cannot go stale. Its witness is computed from
//! `fixtures/bn128/input.json` by the circom-generated wasm and snarkjs (`common`). Without Node.js
//! or that snarkjs the test says why and passes, as the circom tests of `stark2circom` do.

mod common;

use pil2_stark_recurser::plonk2pil::field::R1csPrime;
use pil2_stark_recurser::plonk2pil::merge_copies::r1cs2plonk_merged;
use pil2_stark_recurser::plonk2pil::r1cs::to_plonk::{r1cs2plonk, PlonkAddition, PlonkConstraint};
use pil2_stark_recurser::plonk2pil::r1cs::types::{
    r1cs_prime, read_r1cs_from_bytes, read_r1cs_header, LinearCombination, PlonkOptions, R1csFile,
};
use pil2_stark_recurser::plonk2pil::plonk2pil;
use proofman_fields::{Bn128, Field, PrimeField};

use common::{compile_and_witness, manifest, missing_prerequisite, read_wtns, Scratch};

fn eval<F: Field>(lc: &LinearCombination<F>, w: &[F]) -> F {
    lc.iter().fold(F::ZERO, |acc, (&wire, &q)| acc + q * w[wire as usize])
}

fn r1cs_holds<F: Field>(r1cs: &R1csFile<F>, w: &[F]) -> bool {
    r1cs.constraints.iter().all(|c| eval(&c.a, w) * eval(&c.b, w) == eval(&c.c, w))
}

/// The witness with the wires the additions introduce appended: the `i`-th is wire `n_vars + i`,
/// and reads only wires defined before it.
fn with_additions<F: Field>(witness: &[F], adds: &[PlonkAddition<F>]) -> Vec<F> {
    let mut w = witness.to_vec();
    for (i, a) in adds.iter().enumerate() {
        let wire = w.len();
        assert!(
            a.wires.iter().all(|&x| (x as usize) < wire),
            "addition {i}, wire {wire}, reads {:?}: a wire defined after it",
            a.wires
        );
        w.push(a.coeffs[0] * w[a.wires[0] as usize] + a.coeffs[1] * w[a.wires[1] as usize]);
    }
    w
}

/// The gates `qM·l·r + qL·l + qR·r + qO·o + qC` that are not zero on `w`.
fn failing_gates<F: Field>(cs: &[PlonkConstraint<F>], w: &[F]) -> Vec<usize> {
    let fails = |c: &PlonkConstraint<F>| {
        let [l, r, o] = c.wires.map(|x| w[x as usize]);
        let [q_m, q_l, q_r, q_o, q_c] = c.coeffs;
        !(q_m * l * r + q_l * l + q_r * r + q_o * o + q_c).is_zero()
    };
    cs.iter().enumerate().filter(|(_, c)| fails(c)).map(|(i, _)| i).collect()
}

fn plonk_holds<F: Field>(cs: &[PlonkConstraint<F>], adds: &[PlonkAddition<F>], witness: &[F]) -> bool {
    failing_gates(cs, &with_additions(witness, adds)).is_empty()
}

#[test]
fn a_bn128_circuit_converts_to_plonk_gates_that_hold_on_its_witness() {
    if let Some(why) = missing_prerequisite() {
        eprintln!("skipping the BN128 plonk2pil test: {why}");
        return;
    }
    let scratch = Scratch::new("plonk2pil_bn128");
    let fixtures = manifest().join("tests/fixtures/bn128");
    let (r1cs_bytes, wtns_bytes) =
        compile_and_witness(&scratch.0, &fixtures.join("arith.circom"), &fixtures.join("input.json"));

    assert_eq!(r1cs_prime(&read_r1cs_header(&r1cs_bytes).unwrap()).unwrap(), R1csPrime::Bn128);
    let r1cs = read_r1cs_from_bytes::<Bn128>(&r1cs_bytes).expect("a BN128 r1cs reads into Bn128");
    let witness = read_wtns::<Bn128>(&wtns_bytes);
    assert_eq!(r1cs.header.n8, 32);
    assert_eq!(witness.len(), r1cs.header.n_vars as usize);
    assert_eq!(witness[0], Bn128::ONE, "wire 0 is the constant one");
    assert!(r1cs_holds(&r1cs, &witness), "the r1cs as read must hold on snarkjs's witness");

    // What the fixture is for: coefficients no u64 holds, sums wide enough to need additions, and
    // both kinds of gate.
    let wide = r1cs.constraints.iter().flat_map(|c| [&c.a, &c.b, &c.c]).flat_map(|lc| lc.values());
    assert!(wide.into_iter().any(|q| q.as_canonical_biguint().bits() > 64), "no coefficient is wider than 64 bits");
    let (cs, adds) = r1cs2plonk(&r1cs);
    assert!(!adds.is_empty(), "the wide sum must introduce additions");
    assert!(cs.iter().any(|c| c.coeffs[0].is_zero()) && cs.iter().any(|c| !c.coeffs[0].is_zero()));

    // Every gate holds once the additions are applied, in order.
    let extended = with_additions(&witness, &adds);
    assert_eq!(extended.len(), r1cs.header.n_vars as usize + adds.len());
    assert_eq!(failing_gates(&cs, &extended), Vec::<usize>::new(), "gates that fail on the witness");

    // And the gates say what the r1cs says: change any one signal and the two agree on whether the
    // assignment still holds, so no constraint was lost on the way.
    let mut checked = 0;
    for wire in 1..witness.len() {
        let mut wrong = witness.clone();
        wrong[wire] += Bn128::from_decimal("340282366920938463463374607431768211457").unwrap();
        let (by_r1cs, by_plonk) = (r1cs_holds(&r1cs, &wrong), plonk_holds(&cs, &adds, &wrong));
        assert_eq!(by_r1cs, by_plonk, "wire {wire}: the r1cs says {by_r1cs}, the PLONK gates {by_plonk}");
        checked += usize::from(!by_r1cs);
    }
    assert!(checked > 0, "no single-wire change broke the r1cs, so this checked nothing");

    // The copy-merged conversion the setups run holds on the same witness.
    let (merged_cs, merged_adds, _) = r1cs2plonk_merged(&r1cs, true);
    assert!(plonk_holds(&merged_cs, &merged_adds, &witness), "the copy-merged gates fail on the witness");

    // The STARK recursion's families still refuse it: they are over Goldilocks.
    let err = plonk2pil::<u64>(&r1cs_bytes, "aggregation", &PlonkOptions::default()).unwrap_err().to_string();
    assert!(err.contains("an r1cs over BN128 is set up by the PoseidonBN128 family"), "{err}");
}
