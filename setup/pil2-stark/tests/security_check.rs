use pil2_stark_setup::output::stark_info::solve_low_degree_test;
use pil2_stark_setup::proving_key::recursive::{recursive_stark_settings, RecursiveTemplate};
use pil2_stark_setup::types::security::check::exact::{inv_sqrt_rate_upper, sqrt_rate_upper, ExactRegime, Rational};
use pil2_stark_setup::types::security::check::{check_low_degree_test, report_low_degree_test, SecurityError};
use pil2_stark_setup::types::security::pcs::{LowDegreeTest as Solved, Pcs};
use pil2_stark_setup::types::security::regimes::DecodingRegime;
use pil2_stark_setup::types::stark_struct::{
    generate_stark_struct, LowDegreeTest, LowDegreeTestKind, StarkSettings, StarkStruct,
};

fn settings(kind: LowDegreeTestKind, blowup: usize, fold: usize) -> StarkSettings {
    StarkSettings {
        low_degree_test: Some(kind),
        initial_blowup_factor: Some(blowup),
        initial_folding_factor: Some(fold),
        grinding_bits: Some(20),
        ..Default::default()
    }
}

/// The geometries the repo generates: basic airs (FRI and STIR, blowups 1..4) and recursion layers.
fn todays_configs() -> Vec<(String, StarkStruct, u64)> {
    let mut v = Vec::new();
    for (kind, name) in [(LowDegreeTestKind::Fri, "FRI"), (LowDegreeTestKind::Stir, "STIR")] {
        for blowup in [1usize, 2, 3] {
            for n_bits in [17usize, 20, 22] {
                for fold in [3usize, 4] {
                    let ss = generate_stark_struct(&settings(kind, blowup, fold), n_bits, "blake3");
                    v.push((format!("{name} basic 2^{n_bits} blowup {blowup} fold {fold}"), ss, 200));
                }
            }
        }
        for (t, tn) in [(RecursiveTemplate::Recursive2, "recursive2"), (RecursiveTemplate::Compressor, "compressor")] {
            for n_bits in [17usize, 19] {
                let user = StarkSettings { low_degree_test: Some(kind), ..Default::default() };
                let s = recursive_stark_settings(t, "blake3", &user);
                v.push((format!("{name} {tn} 2^{n_bits}"), generate_stark_struct(&s, n_bits, "blake3"), 135));
            }
        }
    }
    v
}

#[test]
fn exact_agrees_with_float_on_todays_configs() {
    let mut n = 0;
    for (name, ss, batch) in todays_configs() {
        let ldt = std::panic::catch_unwind(|| solve_low_degree_test(&ss, batch));
        let Ok(ldt) = ldt else { continue }; // a config the solver itself rejects is not a config
        let report = check_low_degree_test(&ss, &ldt, batch).unwrap_or_else(|e| panic!("{name}: {e}"));
        let float = ldt.total_security_bits() as i64;
        let exact = report.min_bits_floor as i64;
        assert!((exact - float).abs() <= 2, "{name}: exact {exact} vs float {float}");
        assert!(exact >= 128, "{name}: {exact}");
        n += 1;
    }
    assert!(n >= 20, "only {n} configs were solvable");
}

fn solved(kind: LowDegreeTestKind) -> (StarkStruct, Solved) {
    let ss = generate_stark_struct(&settings(kind, 2, 3), 20, "blake3");
    let ldt = solve_low_degree_test(&ss, 200);
    (ss, ldt)
}

#[test]
fn broken_schedule_rejected() {
    for kind in [LowDegreeTestKind::Fri, LowDegreeTestKind::Stir] {
        let (ss, ldt) = solved(kind);
        assert!(check_low_degree_test(&ss, &ldt, 200).is_ok());
        let mutate = |f: &dyn Fn(&mut StarkStruct)| {
            let mut bad = ss.clone();
            f(&mut bad);
            matches!(check_low_degree_test(&bad, &ldt, 200), Err(SecurityError::Schedule(_)))
        };
        let degrees = |s: &mut StarkStruct| match &mut s.low_degree_test {
            LowDegreeTest::Fri(f) => f.log_degrees.clone(),
            LowDegreeTest::Stir(f) => f.log_degrees.clone(),
        };
        // wrong folding step: a degree that does not follow from its folding factor
        assert!(mutate(&|s| match &mut s.low_degree_test {
            LowDegreeTest::Fri(f) => f.log_degrees[1] += 1,
            LowDegreeTest::Stir(f) => f.log_degrees[1] += 1,
        }));
        // wrong blowup: the extended domain is not degree + blowup
        assert!(mutate(&|s| match &mut s.low_degree_test {
            LowDegreeTest::Fri(f) => f.log_domain_sizes[0] += 1,
            LowDegreeTest::Stir(f) => f.log_domain_sizes[0] += 1,
        }));
        // wrong final degree
        assert!(mutate(&|s| match &mut s.low_degree_test {
            LowDegreeTest::Fri(f) => *f.log_degrees.last_mut().unwrap() += 1,
            LowDegreeTest::Stir(f) => *f.log_degrees.last_mut().unwrap() += 1,
        }));
        // a folding factor the degrees do not follow
        assert!(mutate(&|s| match &mut s.low_degree_test {
            LowDegreeTest::Fri(f) => f.folding_factors[0] += 1,
            LowDegreeTest::Stir(f) => f.folding_factors[0] += 1,
        }));
        let mut probe = ss.clone();
        assert!(!degrees(&mut probe).is_empty());
    }
}

#[test]
fn just_below_128_rejected() {
    use pil2_stark_setup::types::security::pcs::{Fri, Stir};
    let (ss, ldt) = solved(LowDegreeTestKind::Fri);
    let Solved::Fri(fri) = &ldt else { unreachable!() };
    let mut sp = fri.security_params().clone();
    assert_eq!(fri.alpha(), 0.0, "the one-query-fewer comparison assumes the unwidened gap");
    sp.n_queries -= 1;
    let weaker = Solved::Fri(Fri::with_security_params(fri.config().clone(), sp));
    assert!(matches!(check_low_degree_test(&ss, &weaker, 200), Err(SecurityError::BelowTarget { .. })));

    let (ss, ldt) = solved(LowDegreeTestKind::Stir);
    let Solved::Stir(stir) = &ldt else { unreachable!() };
    assert_eq!(stir.alpha(), 0.0);
    let last = stir.security_params().num_queries.len() - 1;
    let mut sp = stir.security_params().clone();
    sp.num_queries[last] -= 1;
    let weaker = Solved::Stir(Stir::with_security_params(stir.config().clone(), sp));
    assert!(matches!(check_low_degree_test(&ss, &weaker, 200), Err(SecurityError::BelowTarget { .. })));
}

#[test]
fn rounding_is_against_the_prover() {
    // √ρ and 1/√ρ for odd exponents are rounded up, never down.
    for mu in 1..=9u32 {
        let up = sqrt_rate_upper(mu);
        assert!(up.clone() * up >= Rational::pow2(-(mu as i64)), "sqrt ρ at μ={mu}");
        let inv = inv_sqrt_rate_upper(mu);
        assert!(inv.clone() * inv >= Rational::pow2(mu as i64), "1/sqrt ρ at μ={mu}");
    }
    // Bits are rounded down: 2^-10 is exactly 10 bits, a hair more error is under 10.
    assert_eq!(Rational::pow2(-10).security_bits(), Rational::int(10));
    assert!((Rational::pow2(-10) * Rational::new(1_000_001, 1_000_000)).security_bits() < Rational::int(10));
    // Hand case: ρ = 1/4 (√ρ = 1/2 exactly), γ = 1/300, n = 4·2^10: m = ⌈75⌉ = 75, so
    // error = ((2·75.5⁵ + 3·75.5·γ·ρ)·n/(3ρ√ρ) + 75.5/√ρ)/|F|, compared to the float solver.
    let q = pil2_stark_setup::types::security::check::exact::goldilocks_cubic_field_size();
    let eta = 1.0 / 300.0;
    let exact = ExactRegime::Jbr { eta: Rational::from_f64(eta) }.error_linear(2, 1 << 10, &q);
    let float = DecodingRegime::Jbr.instantiate(q.to_f64(), 0.0).error_linear(&0.25, 1 << 10);
    // never below the float value (up to float noise), and within it for an exact-√ρ rate.
    let (e, f) = (exact.clone() * q.clone(), float * q.to_f64());
    assert!((e.to_f64() / f - 1.0).abs() < 1e-9, "exact {} vs float {}", e.to_f64(), f);
    // Odd μ: exact is on the pessimistic side of the float value.
    let exact3 = ExactRegime::Jbr { eta: Rational::from_f64(eta) }.error_linear(3, 1 << 10, &q);
    let float3 = DecodingRegime::Jbr.instantiate(q.to_f64(), 0.0).error_linear(&0.125, 1 << 10);
    assert!((exact3 * q.clone()).to_f64() >= float3 * q.to_f64() * (1.0 - 1e-9));
}

#[test]
fn report_lists_both_bounds_fields() {
    let (ss, ldt) = solved(LowDegreeTestKind::Fri);
    let r = report_low_degree_test(&ss, &ldt, 200).unwrap();
    assert!(r.summary().starts_with("security: BCHKS25 "));
    assert!(r.min_bits_floor >= 128 && !r.weakest_phase.is_empty());
}
