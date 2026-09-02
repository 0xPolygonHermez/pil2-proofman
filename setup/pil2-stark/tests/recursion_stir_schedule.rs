// The recursion tree's low-degree test is STIR: check that the settings every recursion circuit
// is built from (`recursive_stark_settings`) produce a STIR schedule the security solver accepts
// at the sizes the layers actually have — recursive1/recursive2 at 2^17, the compressor at the
// larger trace sizes a big air's verifier packs into. `Stir::new` asserts the |Gᵢ| < dᵢ invariant,
// so constructing the solved test *is* the validity check.
use pil2_stark_setup::output::stark_info::solve_low_degree_test;
use pil2_stark_setup::proving_key::recursive::{check_stir_t0_fits, recursive_stark_settings, RecursiveTemplate};
use pil2_stark_setup::types::security::pcs::LowDegreeTest as Solved;
use pil2_stark_setup::types::stark_struct::{generate_stark_struct, LowDegreeTest};

fn solved_stir(template: RecursiveTemplate, n_bits: usize) -> (Vec<usize>, Vec<u64>) {
    let stark_struct = generate_stark_struct(&recursive_stark_settings(template), n_bits, "Poseidon2");
    let LowDegreeTest::Stir(stir) = &stark_struct.low_degree_test else {
        panic!("{template:?} must select STIR, got {:?}", stark_struct.low_degree_test.kind());
    };
    // A recursion circuit's DEEP batch is ~130 evaluations; the count only moves tᵢ slightly.
    let Solved::Stir(solved) = solve_low_degree_test(&stark_struct, 135) else { unreachable!() };
    (stir.log_domain_sizes.clone(), solved.security_params().num_queries.clone())
}

#[test]
fn recursive1_and_recursive2_are_stir_at_2_17() {
    let (domains, t) = solved_stir(RecursiveTemplate::Recursive2, 17);
    // Blowup 3, fold by 8, final degree 2^5: fresh half-size domains 2^20 … 2^16, four rounds.
    assert_eq!(domains, vec![20, 19, 18, 17, 16]);
    assert_eq!(t.len(), 4);
    assert!(t.windows(2).all(|w| w[0] >= w[1]), "query counts should not grow: {t:?}");
    println!("recursive1/2 STIR at 2^17: domains {domains:?}, t = {t:?}");
}

#[test]
fn compressor_is_stir_across_its_trace_sizes() {
    for n_bits in 18..=22 {
        let (domains, t) = solved_stir(RecursiveTemplate::Compressor, n_bits);
        assert_eq!(domains[0], n_bits + 2, "compressor blowup is 2");
        println!("compressor STIR at 2^{n_bits}: domains {domains:?}, t = {t:?}");
    }
}

#[test]
fn raising_t0_respects_the_first_quotient_round() {
    // A single fold has no quotient round: any t₀ goes (SpecifiedRanges at 2^8, t₀ 228 → 511).
    let one_fold = serde_json::json!({"lowDegreeTest": "STIR", "logDegrees": [8, 5], "numQueries": [228]});
    assert!(check_stir_t0_fits(&one_fold, 511, "test").is_ok());

    // Two folds: iteration 1 quotients over |G₁| = t₀ + 1 points and needs |G₁| < d₁ = 2^9.
    let two_folds = serde_json::json!({"lowDegreeTest": "STIR", "logDegrees": [12, 9, 6], "numQueries": [228, 60]});
    assert!(check_stir_t0_fits(&two_folds, 510, "test").is_ok());
    let err = check_stir_t0_fits(&two_folds, 511, "test").unwrap_err().to_string();
    assert!(err.contains("|G₁| = t₀ + 1 = 512 ≥ d₁ = 2^9"), "{err}");

    // FRI structs are not concerned.
    let fri = serde_json::json!({"logDegrees": [12, 9, 6], "numQueries": 511});
    assert!(check_stir_t0_fits(&fri, 100_000, "test").is_ok());
}
