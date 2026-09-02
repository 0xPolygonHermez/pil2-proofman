// The recursion tree's low-degree test is STIR: check that the settings every recursion circuit
// is built from (`recursive_stark_settings`) produce a STIR schedule the security solver accepts
// at the sizes the layers actually have — recursive1/recursive2 at 2^17, the compressor at the
// larger trace sizes a big air's verifier packs into. `Stir::new` asserts the |Gᵢ| < dᵢ invariant,
// so constructing the solved test *is* the validity check.
use pil2_stark_setup::output::stark_info::solve_low_degree_test;
use pil2_stark_setup::proving_key::recursive::{recursive_stark_settings, RecursiveTemplate};
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
