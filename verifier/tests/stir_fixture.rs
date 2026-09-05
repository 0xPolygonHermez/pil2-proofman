// Cross-implementation test of the STIR verifier: the fixtures under tests/data/ are produced by
// the C++ prover (pil2-stark/src/starkpil/stir/tests/test_stir_prover.cpp, the dump_rust_fixture_*
// tests), each holding the schedule, the transcript seed, the f_0 claims a C++ verify run records,
// and the proof section serialized exactly as `Proofs::proof2pointer` lays it out on the wire.
// Regenerate with
//   cd pil2-stark && STIR_DUMP_DIR=$(realpath ../verifier/tests/data) make stir_test
//
// Hash geometry matches the C++ tests: Poseidon2 family, Merkle/transcript arity 4 (width-16
// hashes), width-8 grinding permutation.

use proofman_fields::{CubicExtensionField, Goldilocks, Poseidon2_16, Poseidon2_8, Transcript};
use proofman_verifier::{parse_stir_section, stir_section_size_words, stir_verify, StirParams};

struct Fixture {
    params: StirParams,
    seed: [Goldilocks; 4],
    /// (index of L_0, claimed f_0 value) per round-1 query, in query order.
    claims: Vec<(u64, [u64; 3])>,
    section: Vec<u64>,
}

fn take(words: &[u64], p: &mut usize, n: usize) -> Vec<u64> {
    let out = words[*p..*p + n].to_vec();
    *p += n;
    out
}

fn load(name: &str) -> Fixture {
    let path = format!("{}/tests/data/{}", env!("CARGO_MANIFEST_DIR"), name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert_eq!(bytes.len() % 8, 0, "{path} is not a u64 stream");
    let words: Vec<u64> = bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();

    let mut p = 0usize;
    let m = words[p] as usize;
    p += 1;
    let folding_factors = take(&words, &mut p, m);
    let log_degrees = take(&words, &mut p, m + 1);
    let log_domain_sizes = take(&words, &mut p, m + 1);
    let num_queries = take(&words, &mut p, m);
    let grinding_bits_queries = take(&words, &mut p, m);
    let arity = words[p];
    let last_level_verification = words[p + 1];
    let hash_commits = words[p + 2] != 0;
    p += 3;
    let seed_words = take(&words, &mut p, 4);
    let seed = [
        Goldilocks::new(seed_words[0]),
        Goldilocks::new(seed_words[1]),
        Goldilocks::new(seed_words[2]),
        Goldilocks::new(seed_words[3]),
    ];
    let mut claims = Vec::with_capacity(num_queries[0] as usize);
    for _ in 0..num_queries[0] {
        let c = take(&words, &mut p, 4);
        claims.push((c[0], [c[1], c[2], c[3]]));
    }
    let section_len = words[p] as usize;
    p += 1;
    let section = take(&words, &mut p, section_len);
    assert_eq!(p, words.len(), "{path} has trailing data");

    let params = StirParams {
        folding_factors,
        log_degrees,
        log_domain_sizes,
        num_queries,
        grinding_bits_queries,
        arity,
        last_level_verification,
        hash_commits,
    };
    assert_eq!(section.len(), stir_section_size_words(&params), "{path} section size mismatch");
    Fixture { params, seed, claims, section }
}

/// Replay the fixture's transcript and verify `section`, cross-checking every recorded f_0 claim
/// the way the STARK verifier's DEEP comparison would.
fn run(fx: &Fixture, section: &[u64]) -> bool {
    let mut p = 0usize;
    let parsed = parse_stir_section(section, &mut p, &fx.params);
    assert_eq!(p, section.len());

    let mut transcript: Transcript<Goldilocks, Poseidon2_16> = Transcript::new();
    transcript.put(&fx.seed);

    let mut claims_ok = true;
    let ok = stir_verify::<Poseidon2_16, Poseidon2_16, Poseidon2_16, Poseidon2_8>(
        &mut transcript,
        &parsed,
        &fx.params,
        &mut |q: usize, raw: u64, claim: CubicExtensionField<Goldilocks>| {
            let (idx, vals) = fx.claims[q];
            if raw != idx
                || claim.value[0] != Goldilocks::new(vals[0])
                || claim.value[1] != Goldilocks::new(vals[1])
                || claim.value[2] != Goldilocks::new(vals[2])
            {
                claims_ok = false;
            }
            true
        },
    );
    ok && claims_ok
}

/// Word offsets of the section's tail parts, mirroring `stir_section_size_words`.
fn offsets(params: &StirParams) -> (usize, usize, usize, usize) {
    let m = params.m();
    let n_sibs_per_level = ((params.arity - 1) * 4) as usize;
    let num_nodes_level = if params.last_level_verification == 0 {
        0
    } else {
        params.arity.pow(params.last_level_verification as u32) as usize
    };
    let mut betas = m * 4;
    for i in 0..m {
        let k = 1usize << params.folding_factors[i];
        let log_leaves = params.log_domain_sizes[i] - params.folding_factors[i];
        let levels = ((log_leaves as f64) / (params.arity as f64).log2()).ceil() as u64;
        let n_sibs = levels.saturating_sub(params.last_level_verification) as usize;
        betas += params.num_queries[i] as usize * (k * 3 + n_sibs * n_sibs_per_level);
        betas += num_nodes_level * 4;
    }
    let final_pol = betas + (m - 1) * 3;
    let nonces = final_pol + (1usize << params.log_degrees[m]) * 3;
    let ans_coeffs = nonces + m;
    (betas, final_pol, nonces, ans_coeffs)
}

fn check_fixture(name: &str) {
    let fx = load(name);
    assert!(run(&fx, &fx.section), "{name}: honest proof must verify");

    let (betas_off, final_pol_off, nonces_off, ans_off) = offsets(&fx.params);

    // A flipped bit anywhere that matters must be rejected.
    let tampers: &[(&str, usize)] = &[
        ("root of T_0", 0),
        ("opened coset value", fx.params.m() * 4),
        ("out-of-domain answer β", betas_off),
        ("final polynomial coefficient", final_pol_off),
        ("grinding nonce", nonces_off),
    ];
    for &(what, off) in tampers {
        let mut section = fx.section.clone();
        section[off] = section[off].wrapping_add(1) % 0xFFFFFFFF00000001;
        assert!(!run(&fx, &section), "{name}: tampered {what} must be rejected");
    }

    // The Âns coefficient hints are recursion-circuit material: the native verifier recomputes
    // Âns itself and ignores them, exactly like the C++ verifier.
    if fx.params.m() > 1 {
        let mut section = fx.section.clone();
        section[ans_off] = section[ans_off].wrapping_add(1) % 0xFFFFFFFF00000001;
        assert!(run(&fx, &section), "{name}: the Âns hints are not read by the native verifier");
    }
}

#[test]
fn cpp_fixture_plain() {
    check_fixture("stir_plain.bin");
}

#[test]
fn cpp_fixture_hashed_final_pol_and_last_levels() {
    check_fixture("stir_hashed_llv.bin");
}
