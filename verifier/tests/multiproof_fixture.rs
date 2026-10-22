// Real fibonacci-square blake3 compressed proofs; the digests pin the flat proofs both verifiers
// accepted. Regenerate after a layout or transcript change: setup-compressed-final, prove -a -f.

use proofman_fields::{Blake3Transcript, Blake3_8, Goldilocks, Hash, PrimeField64};
use proofman_verifier::{
    expand_proof, prune_proof, LowDegreeLayout, ProofLayout, StirParams, VadcopFinalProof, MULTIPROOF_TAG,
};

type T = Blake3Transcript<Goldilocks>;

const VK: [u64; 4] = [18340197221422479417, 8938428392300945921, 411837121511453426, 14159028078049047241];

fn layout(ldt: LowDegreeLayout) -> ProofLayout {
    ProofLayout {
        n_stages: 2,
        n_constants: 43,
        n_evals: 646,
        n_bits_ext: 23,
        num_vals: vec![128, 36, 21],
        arity: 2,
        last_level_verification: 0,
        hash_commits: true,
        ldt,
    }
}

fn stir() -> ProofLayout {
    layout(LowDegreeLayout::Stir(StirParams {
        folding_factors: vec![4, 3, 3],
        log_degrees: vec![18, 14, 11, 8],
        log_domain_sizes: vec![23, 22, 21, 20],
        num_queries: vec![41, 26, 22],
        grinding_bits_queries: vec![27, 26, 23],
        arity: 2,
        last_level_verification: 0,
        hash_commits: true,
    }))
}

fn fri() -> ProofLayout {
    layout(LowDegreeLayout::Fri { steps: vec![23, 19, 15, 11, 8], n_queries: 41 })
}

fn wire(name: &str) -> Vec<u64> {
    VadcopFinalProof::load(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap().proof_with_publics()
}

fn digest(words: &[u64]) -> [u64; 4] {
    let gl: Vec<Goldilocks> = words.iter().map(|&w| Goldilocks::new(w)).collect();
    <Blake3_8 as Hash<Goldilocks>>::linear_hash(&gl).map(|x| x.as_canonical_u64())
}

fn expand(w: &[u64], layout: &ProofLayout) -> Option<Vec<u64>> {
    expand_proof::<Blake3_8, Blake3_8, T>(w, &VK, layout)
}

fn check(name: &str, layout: ProofLayout, flat_len: usize, flat_digest: [u64; 4]) {
    let w = wire(name);
    let tag_at = 1 + w[0] as usize;
    assert_eq!(w[tag_at], MULTIPROOF_TAG);

    let flat = expand(&w, &layout).expect("the fixture expands");
    assert_eq!((flat.len(), digest(&flat)), (flat_len, flat_digest), "expand rebuilds the verified flat proof");
    assert_eq!(prune_proof::<T>(&flat, &VK, &layout).as_deref(), Some(&w[..]), "the wire form is prune's");

    // Untagged, short, padded, aliased, flat, or read as the other ldt: refused.
    let mut untagged = w.clone();
    untagged.remove(tag_at);
    assert!(expand(&untagged, &layout).is_none());
    assert!(expand(&w[..w.len() - 1], &layout).is_none());
    let mut padded = w.clone();
    padded.push(0);
    assert!(expand(&padded, &layout).is_none());
    let small = (tag_at + 1..w.len()).find(|&i| w[i] < (1 << 32) - 1).expect("a small word to alias");
    let mut aliased = w.clone();
    aliased[small] += Goldilocks::ORDER_U64;
    assert!(expand(&aliased, &layout).is_none());
    assert!(expand(&flat, &layout).is_none());
    let other = if matches!(layout.ldt, LowDegreeLayout::Stir(_)) { fri() } else { stir() };
    assert!(expand(&w, &other).is_none());
}

/// The expected wire size is within a few percent of a real proof's, and below the flat one's.
fn check_expected_size(name: &str, layout: ProofLayout) {
    let w = wire(name);
    let actual = (w.len() - 1 - w[0] as usize) as f64;
    let expected = layout.expected_multiproof_words();
    assert!((expected - actual).abs() / actual < 0.02, "{name}: expected {expected:.0} words, got {actual}");
    assert!(expected < layout.flat_proof_words() as f64);
}

#[test]
fn the_expected_multiproof_size_matches_the_fixtures() {
    check_expected_size("multiproof_stir.bin", stir());
    check_expected_size("multiproof_fri.bin", fri());
}

#[test]
fn stir_fixture_round_trips() {
    check(
        "multiproof_stir.bin",
        stir(),
        37394,
        [6694226346582352319, 9008143886349686640, 13700033906418492139, 8411194751221972452],
    );
}

#[test]
fn fri_fixture_round_trips() {
    check(
        "multiproof_fri.bin",
        fri(),
        42760,
        [13779139022878427377, 6598195400157273254, 13048093804633654918, 3035684251374134388],
    );
}
