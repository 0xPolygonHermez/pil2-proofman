// With the recursion entry set to STIR, check that the settings every recursion circuit is built
// from (`recursive_stark_settings`) produce a STIR schedule the security solver accepts
// at the sizes the layers actually have — recursive1/recursive2 at 2^17, the compressor at the
// larger trace sizes a big air's verifier packs into. `Stir::new` asserts the |Gᵢ| < dᵢ invariant,
// so constructing the solved test *is* the validity check.
use pil2_stark_setup::output::stark_info::solve_low_degree_test;
use pil2_stark_setup::proving_key::recursive::{check_stir_t0_fits, recursive_stark_settings, RecursiveTemplate};
use pil2_stark_setup::types::security::pcs::LowDegreeTest as Solved;
use pil2_stark_setup::types::stark_struct::{
    generate_stark_struct, LowDegreeTest, LowDegreeTestKind, StarkSettings, StarkStructsConfig,
};

fn stir_entry() -> StarkSettings {
    StarkSettings { low_degree_test: Some(LowDegreeTestKind::Stir), ..Default::default() }
}

fn solved_stir(template: RecursiveTemplate, n_bits: usize) -> (Vec<usize>, Vec<u64>) {
    let settings = recursive_stark_settings(template, "Poseidon2", &stir_entry());
    let stark_struct = generate_stark_struct(&settings, n_bits, "Poseidon2");
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

#[test]
fn the_recursion_entry_switches_the_tree_s_low_degree_test() {
    // No entry: FRI, with the family's own grinding and terminal degree (7 − blowup 2 = 5).
    let dflt = recursive_stark_settings(RecursiveTemplate::Recursive1, "blake3", &StarkSettings::default());
    assert_eq!(dflt.low_degree_test, Some(LowDegreeTestKind::Fri));
    assert_eq!(dflt.grinding_bits, Some(24));
    assert_eq!(dflt.final_degree, Some(5));

    // `{"recursion": {"lowDegreeTest": "STIR"}}` opts the tree into STIR; the family constant is
    // then the degree bound d_M itself.
    let stir = recursive_stark_settings(RecursiveTemplate::Recursive1, "blake3", &stir_entry());
    assert_eq!(stir.low_degree_test, Some(LowDegreeTestKind::Stir));
    assert_eq!(stir.grinding_bits, Some(24));
    assert_eq!(stir.final_degree, Some(7));

    // `{"recursion": {"lowDegreeTest": "FRI"}}` flips the test and nothing else, so the blake3
    // tree comes out on the solved FRI schedule it was originally sized on. For FRI the family's
    // terminal is a domain size, so the degree bound is that minus the blowup (7 − 2 = 5).
    let cfg = StarkStructsConfig::from_json_str(r#"{ "recursion": { "lowDegreeTest": "FRI" } }"#).unwrap();
    let user = cfg.recursion_settings().unwrap();
    let fri = recursive_stark_settings(RecursiveTemplate::Recursive1, "blake3", &user);
    assert_eq!(fri.low_degree_test, Some(LowDegreeTestKind::Fri));
    assert_eq!(fri.initial_blowup_factor, stir.initial_blowup_factor);
    assert_eq!(fri.grinding_bits, stir.grinding_bits);
    assert_eq!(fri.final_degree, Some(5));
    assert_eq!(fri.last_level_verification, stir.last_level_verification);
    let ss = generate_stark_struct(&fri, 19, "blake3");
    assert_eq!(ss.low_degree_test.kind(), LowDegreeTestKind::Fri);
    assert_eq!(ss.n_bits_ext, 21);
    // The measured optimum the blake3 recursion was sized on: 21 > 17 > 13 > 10 > 7, a 2^7
    // terminal domain. A degree bound of 7 would stop at 2^9 and overflow recursive2's 2^19.
    let sched = ss.low_degree_test.expect_fri("test");
    assert_eq!(sched.log_domain_sizes, vec![21, 17, 13, 10, 7]);
    assert_eq!(sched.folding_factors, vec![4, 4, 3, 3]);
    assert_eq!(*sched.log_degrees.last().unwrap(), 5);

    // The compressor folds the same terminal domain at its own blowup (7 − 1 = 6).
    let comp = recursive_stark_settings(RecursiveTemplate::Compressor, "blake3", &user);
    assert_eq!(comp.final_degree, Some(6));
    let comp_ss = generate_stark_struct(&comp, 19, "blake3");
    assert_eq!(*comp_ss.low_degree_test.expect_fri("test").log_domain_sizes.last().unwrap(), 7);

    // Poseidon: a 2^5 terminal domain at blowup 3 is a degree bound of 2.
    let pos = recursive_stark_settings(RecursiveTemplate::Recursive2, "Poseidon2", &user);
    assert_eq!(pos.final_degree, Some(2));
    let pos_ss = generate_stark_struct(&pos, 17, "Poseidon2");
    assert_eq!(*pos_ss.low_degree_test.expect_fri("test").log_domain_sizes.last().unwrap(), 5);

    // The STIR-only knobs ride along when asked for.
    let cfg = StarkStructsConfig::from_json_str(
        r#"{ "recursion": { "lowDegreeTest": "STIR", "initialFoldingFactor": 2, "grindingBitsQueries": [28, 26, 24, 22, 20, 18] } }"#,
    )
    .unwrap();
    let tuned = recursive_stark_settings(RecursiveTemplate::Recursive2, "blake3", &cfg.recursion_settings().unwrap());
    assert_eq!(tuned.initial_folding_factor, Some(2));
    let ss = generate_stark_struct(&tuned, 19, "blake3");
    let LowDegreeTest::Stir(s) = &ss.low_degree_test else { panic!("still STIR") };
    assert_eq!(s.grinding_bits_queries, vec![28, 26, 24, 22, 20, 18]);
}
