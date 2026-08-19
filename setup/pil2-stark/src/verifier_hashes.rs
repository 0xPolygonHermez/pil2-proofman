//! How many hash invocations a verifier performs for one air.
//!
//! The count is a property of the proof's geometry, not of who checks it: the native verifier and
//! the in-circuit (recursive) one run the same algorithm, so they perform the same hashes. What
//! differs is the price of one — a Blake3 compression is nearly free on a CPU and expensive in a
//! circuit, a Poseidon permutation the other way round.
//!
//! Every rule here mirrors a specific piece of the verifier, and the tests name which:
//! `stark_verify.hpp` for the sequence, `merkleTreeGL.cpp` for the trees, `transcriptGL.cpp` for
//! Fiat-Shamir, `poseidon_goldilocks.cpp` for the sponge.
//!
//! These counts were checked against the native verifier itself, temporarily instrumented to count
//! its own hashes, over the three airs of the hashes example under all three families -- nine cases,
//! all exact (2026-08-19). The measured totals are pinned in the tests below, so if the verifier ever
//! changes shape they are what to re-derive. Re-instrumenting is a small, purely additive patch:
//! atomic counters bumped in `MerkleTreeGL::verifyGroupProof`, `calculateRootFromProof` and
//! `TranscriptGL::_updateState` -- atomic because the query loops in `starkVerify` are OpenMP
//! parallel -- summed and logged at the end of `starkVerify`.

use proofman_common::hash_family::{sponge_rate, transcript_out_size, transcript_pending_size, DIGEST_SIZE};

/// Hash invocations, split by what the verifier is doing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HashCounts {
    /// Linear hashes of the opened leaf values, one per query per tree.
    pub leaf: u64,
    /// Merkle path node hashes for the committed and constant trees.
    pub merkle: u64,
    /// Node and leaf hashes of the FRI folding trees.
    pub fri: u64,
    /// Fiat-Shamir transcript permutations.
    pub transcript: u64,
    /// The proof-of-work check.
    pub grinding: u64,
}

impl HashCounts {
    pub fn total(&self) -> u64 {
        self.leaf + self.merkle + self.fri + self.transcript + self.grinding
    }
}

/// The geometry the count depends on, all of it known at setup time.
#[derive(Debug, Clone, Default)]
pub struct VerifierGeometry {
    pub n_bits_ext: u64,
    pub arity: u64,
    pub last_level_verification: u64,
    pub n_queries: u64,
    pub pow_bits: u64,
    pub hash_commits: bool,
    /// Width of each committed tree, `cm1..cm{nStages+1}`.
    pub stage_widths: Vec<u64>,
    /// Width of the constant tree.
    pub n_constants: u64,
    /// Width of each custom-commit tree.
    pub custom_commit_widths: Vec<u64>,
    /// `nBits` of each FRI step; `steps[0]` is the committed domain and has no tree of its own.
    pub step_n_bits: Vec<u64>,
    pub n_publics: u64,
    /// Evaluations put into the transcript, each a field-extension element.
    pub n_evals: u64,
    /// Challenges squeezed before each stage root, for stages `2..=n_stages + 1`. Order matters:
    /// the verifier interleaves them with the roots, and a squeeze after an absorb costs a hash.
    pub stage_challenges: Vec<u64>,
    /// Extension-field air values absorbed after each of those same stage roots.
    pub stage_air_values: Vec<u64>,
    /// Coefficients of the final FRI polynomial, each a field-extension element.
    pub final_pol_size: u64,
}

/// Blake3 compressions over `bytes`: one per 64-byte block, plus one parent per extra 1024-byte
/// chunk. A 64-byte input — one Merkle node — is a single compression.
pub fn blake3_compressions(bytes: u64) -> u64 {
    if bytes == 0 {
        return 0;
    }
    let blocks = bytes.div_ceil(64);
    let chunks = bytes.div_ceil(1024);
    blocks + (chunks - 1)
}

/// Permutations to hash `width` field elements into one digest. Poseidon absorbs `sponge_rate`
/// per permutation (`linear_hash_seq`); Blake3 hashes the bytes in one call.
pub fn leaf_hashes(family: &str, arity: u64, width: u64) -> u64 {
    match family {
        "blake3" => blake3_compressions(width * 8),
        _ => width.div_ceil(sponge_rate(arity)),
    }
}

/// Node hashes on one Merkle path: `ceil(log_arity(height)) - last_level_verification`, matching
/// `MerkleTreeGL::getMerkleProofLength`. Each level is one permutation, since the sponge width is
/// exactly `arity * DIGEST_SIZE`.
pub fn merkle_path_hashes(n_bits_height: u64, arity: u64, last_level_verification: u64) -> u64 {
    if n_bits_height == 0 {
        return 0;
    }
    // The C++ divides the logs in floating point; mirror it so the two never disagree on an arity
    // that is not a power of two.
    let levels = (n_bits_height as f64 / (arity as f64).log2()).ceil() as u64;
    levels.saturating_sub(last_level_verification)
}

/// Hashes to reduce a stored bottom level to the root, done once per tree rather than per query
/// (`MerkleTreeGL::verifyMerkleRoot` -> `merkletreeReduce`). Zero unless the tree keeps levels.
///
/// The kept level is not `arity^llv` nodes wide: the verifier folds the height down until it is *at
/// most* that, which overshoots whenever the arity does not divide the height evenly. Arity 4 over an
/// odd `n_bits` stops at 8 rather than 16, and reduces in three hashes instead of five.
pub fn root_reduction_hashes(arity: u64, last_level_verification: u64, n_bits_height: u64) -> u64 {
    let stop = arity.pow(last_level_verification as u32);
    let mut pending = 1u64 << n_bits_height;
    while pending > stop {
        pending = pending.div_ceil(arity);
    }

    let mut hashes = 0;
    while pending > 1 {
        pending = pending.div_ceil(arity);
        hashes += pending;
    }
    hashes
}

/// The Fiat-Shamir transcript, replayed rather than modelled: `put` buffers and permutes when the
/// buffer fills, and invalidates the output FIFO, so a squeeze after any absorb costs a fresh
/// permutation. Mirrors `TranscriptGL`.
#[derive(Debug)]
pub struct TranscriptSim {
    pending: u64,
    out: u64,
    pending_size: u64,
    out_size: u64,
    pub hashes: u64,
}

impl TranscriptSim {
    pub fn new(arity: u64) -> Self {
        Self {
            pending: 0,
            out: 0,
            pending_size: transcript_pending_size(arity),
            out_size: transcript_out_size(arity),
            hashes: 0,
        }
    }

    fn update_state(&mut self) {
        self.hashes += 1;
        self.pending = 0;
        self.out = self.out_size;
    }

    pub fn put(&mut self, n: u64) {
        for _ in 0..n {
            self.pending += 1;
            // `_add1` resets the output cursor: anything absorbed invalidates what was squeezable.
            self.out = 0;
            if self.pending == self.pending_size {
                self.update_state();
            }
        }
    }

    fn get_fields1(&mut self) {
        if self.out == 0 {
            self.update_state();
        }
        self.out -= 1;
    }

    /// One field-extension challenge: three squeezed elements.
    pub fn get_field(&mut self) {
        for _ in 0..3 {
            self.get_fields1();
        }
    }

    pub fn get_state(&mut self) {
        if self.pending > 0 {
            self.update_state();
        }
    }

    /// Query indices: `ceil((n * n_bits - 1) / 63) + 1` squeezed elements, bit-sliced.
    pub fn get_permutations(&mut self, n: u64, n_bits: u64) {
        let total_bits = n * n_bits;
        let n_fields = if total_bits == 0 { 0 } else { (total_bits - 1) / 63 + 1 };
        for _ in 0..n_fields {
            self.get_fields1();
        }
    }
}

/// Build the geometry for `family` from what the stats pipeline already has. The widths do not
/// depend on the family -- only the tree arity does, and through it the FRI query count, so each
/// family gets its own security analysis.
pub fn geometry_for_family(
    family: &str,
    stark_struct: &crate::types::stark_struct::StarkStruct,
    setup: &crate::types::pilout_info::SetupResult,
    n_evals: usize,
) -> VerifierGeometry {
    use crate::types::security;
    use crate::types::security::pcs::{Batching, Fri, FriConfig};
    use crate::types::security::regimes::DecodingRegime;

    let arity = proofman_common::hash_family::merkle_tree_arity(family);

    // Same configuration `build_starkinfo_output` uses, with this family's arity: the query count
    // and grinding bits come out of the security analysis, not out of the settings.
    let fri = Fri::new(FriConfig {
        field_size: security::goldilocks_safe_extension_field_size(),
        trace_length: 1u32 << stark_struct.n_bits,
        rate: 1.0 / (1u64 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64,
        batch_size: n_evals.max(1) as u64,
        batching: Batching::Powers,
        log_folding_factors: crate::output::stark_info::compute_log_folding_factors(stark_struct),
        max_grinding_bits_query: stark_struct.pow_bits as u64,
        use_max_grinding_bits_query: true,
        tree_arity: arity,
        hash_size_bits: 256,
        target_security_bits: 128,
        regime: DecodingRegime::Jbr,
    });
    let security = fri.security_params();

    let width = |section: &str| setup.map_sections_n.get(section).copied().unwrap_or(0) as u64;

    VerifierGeometry {
        n_bits_ext: stark_struct.n_bits_ext as u64,
        arity,
        last_level_verification: stark_struct.last_level_verification as u64,
        n_queries: security.n_queries as u64,
        pow_bits: security.grinding_bits_query as u64,
        hash_commits: stark_struct.hash_commits,
        // One committed tree per stage, plus the quotient stage.
        stage_widths: (1..=setup.n_stages + 1).map(|s| width(&format!("cm{s}"))).collect(),
        n_constants: setup.n_constants as u64,
        custom_commit_widths: setup.custom_commits.iter().map(|c| width(&format!("{}0", c.name))).collect(),
        step_n_bits: stark_struct.steps.iter().map(|s| s.n_bits as u64).collect(),
        n_publics: setup.n_publics as u64,
        n_evals: n_evals as u64,
        stage_challenges: (2..=setup.n_stages + 1)
            .map(|s| setup.challenges_map.iter().filter(|c| c.stage == Some(s)).count() as u64)
            .collect(),
        stage_air_values: (2..=setup.n_stages + 1)
            .map(|s| setup.air_values_map.iter().filter(|v| v.stage == Some(s)).count() as u64)
            .collect(),
        final_pol_size: 1u64 << stark_struct.steps.last().map_or(0, |s| s.n_bits),
    }
}

/// Hashes the verifier performs for one air, under `family`.
pub fn verifier_hashes(geom: &VerifierGeometry, family: &str) -> HashCounts {
    let mut counts = HashCounts::default();

    // ── Query phase: every committed tree is opened at every query ──
    let open = |width: u64, n_bits_height: u64, leaf: &mut u64, merkle: &mut u64| {
        *leaf += geom.n_queries * leaf_hashes(family, geom.arity, width);
        *merkle += geom.n_queries * merkle_path_hashes(n_bits_height, geom.arity, geom.last_level_verification);
        // Reducing the kept level to the root is done once for the tree, not once per query.
        *merkle += root_reduction_hashes(geom.arity, geom.last_level_verification, n_bits_height);
    };

    for &width in &geom.stage_widths {
        open(width, geom.n_bits_ext, &mut counts.leaf, &mut counts.merkle);
    }
    open(geom.n_constants, geom.n_bits_ext, &mut counts.leaf, &mut counts.merkle);
    for &width in &geom.custom_commit_widths {
        open(width, geom.n_bits_ext, &mut counts.leaf, &mut counts.merkle);
    }

    // ── FRI folding trees: one per step past the committed domain ──
    for step in 1..geom.step_n_bits.len() {
        let n_bits = geom.step_n_bits[step];
        let group_size = 1u64 << (geom.step_n_bits[step - 1] - n_bits);
        let (mut leaf, mut merkle) = (0, 0);
        open(group_size * FIELD_EXTENSION, n_bits, &mut leaf, &mut merkle);
        counts.fri += leaf + merkle;
    }

    counts.transcript = transcript_hashes(geom);
    counts.grinding = u64::from(geom.pow_bits > 0);
    counts
}

/// Field-extension degree: every challenge, evaluation and FRI value is this many elements.
const FIELD_EXTENSION: u64 = 3;

/// Replays the absorb/squeeze sequence of `starkVerify`, statement by statement. The order is what
/// makes this exact rather than approximate: a squeeze right after an absorb costs a permutation
/// that the same calls in another order would not.
///
/// This is the standalone path (`challengesVadcop == false`), which a proof of a single air takes.
fn transcript_hashes(geom: &VerifierGeometry) -> u64 {
    let mut t = TranscriptSim::new(geom.arity);
    // A hashed commit absorbs a digest of the values rather than the values, and computing that
    // digest costs a sponge of its own (`transcriptHash.getState`).
    let put_values = |t: &mut TranscriptSim, n: u64| {
        if geom.hash_commits {
            let mut inner = TranscriptSim::new(geom.arity);
            inner.put(n);
            inner.get_state();
            t.hashes += inner.hashes;
            t.put(DIGEST_SIZE);
        } else {
            t.put(n);
        }
    };

    t.put(DIGEST_SIZE); // verkey
    if geom.n_publics > 0 {
        put_values(&mut t, geom.n_publics);
    }
    t.put(DIGEST_SIZE); // stage 1 root

    // Stages 2..=n_stages+1: the stage's challenges are squeezed first, then its root goes in,
    // then whatever air values it carries.
    for (i, &n_challenges) in geom.stage_challenges.iter().enumerate() {
        for _ in 0..n_challenges {
            t.get_field();
        }
        t.put(DIGEST_SIZE);
        t.put(geom.stage_air_values.get(i).copied().unwrap_or(0) * FIELD_EXTENSION);
    }

    t.get_field(); // evals challenge
    put_values(&mut t, geom.n_evals * FIELD_EXTENSION);
    t.get_field(); // the two FRI challenges
    t.get_field();

    // One folding challenge per step past the first, and a root for every step but the last, which
    // sends the final polynomial instead.
    for step in 0..geom.step_n_bits.len() {
        if step > 0 {
            t.get_field();
        }
        if step + 1 < geom.step_n_bits.len() {
            t.put(DIGEST_SIZE);
        } else {
            put_values(&mut t, geom.final_pol_size * FIELD_EXTENSION);
        }
    }
    t.get_field();

    // Query indices come from a transcript of their own, seeded with the last challenge and a nonce.
    let mut queries = TranscriptSim::new(geom.arity);
    queries.put(FIELD_EXTENSION);
    queries.put(1);
    queries.get_permutations(geom.n_queries, geom.step_n_bits[0]);

    t.hashes + queries.hashes
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_common::hash_family::merkle_tree_arity;

    /// One Merkle node is 64 bytes, which is exactly one Blake3 block.
    #[test]
    fn a_blake3_node_is_one_compression() {
        assert_eq!(blake3_compressions(8 * DIGEST_SIZE * 2), 1);
    }

    /// A chunk is 1024 bytes of 64-byte blocks; past that, each new chunk costs a parent too.
    #[test]
    fn blake3_counts_blocks_plus_chunk_parents() {
        assert_eq!(blake3_compressions(0), 0);
        assert_eq!(blake3_compressions(1), 1);
        assert_eq!(blake3_compressions(1024), 16);
        assert_eq!(blake3_compressions(1025), 18); // 17 blocks + 1 parent
    }

    /// Poseidon absorbs `rate` elements per permutation and nothing for an empty leaf.
    #[test]
    fn a_poseidon_leaf_costs_one_permutation_per_rate() {
        assert_eq!(sponge_rate(4), 12);
        assert_eq!(leaf_hashes("Poseidon1", 4, 0), 0);
        assert_eq!(leaf_hashes("Poseidon1", 4, 1), 1);
        assert_eq!(leaf_hashes("Poseidon1", 4, 12), 1);
        assert_eq!(leaf_hashes("Poseidon1", 4, 13), 2);
        assert_eq!(leaf_hashes("Poseidon2", 4, 100), 9);
    }

    /// A Blake3 leaf hashes `width * 8` bytes in one call.
    #[test]
    fn a_blake3_leaf_hashes_the_whole_row() {
        assert_eq!(leaf_hashes("blake3", 2, 8), 1); // 64 bytes
        assert_eq!(leaf_hashes("blake3", 2, 128), 16); // 1024 bytes
    }

    /// `ceil(log_arity(height))`, less the levels the verifier reads from the proof instead.
    #[test]
    fn a_merkle_path_is_one_hash_per_level() {
        assert_eq!(merkle_path_hashes(20, 4, 0), 10);
        assert_eq!(merkle_path_hashes(20, 2, 0), 20);
        assert_eq!(merkle_path_hashes(20, 4, 2), 8);
        assert_eq!(merkle_path_hashes(0, 4, 0), 0, "a height-1 tree has no path");
    }

    /// The kept bottom level is reduced to the root once, not once per query.
    #[test]
    fn the_root_reduction_is_paid_once_per_tree() {
        assert_eq!(root_reduction_hashes(4, 0, 20), 0);
        assert_eq!(root_reduction_hashes(4, 1, 20), 1, "4 nodes -> 1");
        assert_eq!(root_reduction_hashes(4, 2, 20), 5, "16 -> 4 -> 1");
        assert_eq!(root_reduction_hashes(2, 3, 20), 7, "8 -> 4 -> 2 -> 1");
    }

    /// Folding an odd `n_bits` by 4 cannot land on 16, so the kept level is 8 wide and costs two
    /// hashes less. Missing this made every Poseidon count too high; blake3's arity 2 always divides
    /// a power-of-two height exactly, so it hid the bug.
    #[test]
    fn an_odd_height_keeps_a_narrower_level() {
        assert_eq!(root_reduction_hashes(4, 2, 22), 5, "2^22 folds to exactly 16");
        assert_eq!(root_reduction_hashes(4, 2, 19), 3, "2^19 folds to 8, not 16");
        assert_eq!(root_reduction_hashes(2, 2, 19), 3, "arity 2 always lands on 4");
    }

    /// Absorbing exactly one buffer's worth permutes once; a partial buffer costs nothing until
    /// something is squeezed out of it.
    #[test]
    fn the_transcript_permutes_when_its_buffer_fills() {
        let mut t = TranscriptSim::new(4);
        t.put(12);
        assert_eq!(t.hashes, 1);

        let mut t = TranscriptSim::new(4);
        t.put(5);
        assert_eq!(t.hashes, 0, "a partial buffer has not permuted yet");
        t.get_state();
        assert_eq!(t.hashes, 1, "getState flushes it");
    }

    /// `_add1` clears the output cursor, so a squeeze right after an absorb must re-permute even
    /// though the previous permutation left a full FIFO.
    #[test]
    fn an_absorb_invalidates_the_squeezed_output() {
        let mut t = TranscriptSim::new(4);
        t.put(12); // permutes, FIFO now full
        assert_eq!(t.hashes, 1);
        t.get_field(); // 3 of 16 outputs, no new permutation needed
        assert_eq!(t.hashes, 1);

        let mut t = TranscriptSim::new(4);
        t.put(12);
        t.put(1); // invalidates the FIFO
        t.get_field();
        assert_eq!(t.hashes, 2, "the squeeze had to permute again");
    }

    /// A full FIFO serves 16 squeezed elements, so five extension-field challenges need two
    /// permutations, not one.
    #[test]
    fn squeezing_past_the_fifo_permutes_again() {
        let mut t = TranscriptSim::new(4);
        t.put(12);
        for _ in 0..5 {
            t.get_field(); // 15 elements
        }
        assert_eq!(t.hashes, 1);
        t.get_field(); // 16th, 17th ...
        assert_eq!(t.hashes, 2);
    }

    fn minimal_geometry() -> VerifierGeometry {
        VerifierGeometry {
            n_bits_ext: 4,
            arity: 4,
            n_queries: 1,
            stage_widths: vec![12],
            n_constants: 12,
            step_n_bits: vec![4],
            ..Default::default()
        }
    }

    /// One query over two width-12 trees of height 2^4: each costs one leaf hash plus a two-level
    /// path. Nothing else is configured, so the rest of the count is transcript only.
    #[test]
    fn the_query_phase_is_leaf_plus_path_per_tree() {
        let counts = verifier_hashes(&minimal_geometry(), "Poseidon1");
        assert_eq!(counts.leaf, 2, "one leaf hash per tree");
        assert_eq!(counts.merkle, 4, "two levels per tree");
        assert_eq!(counts.fri, 0, "a single FRI step has no folding tree");
    }

    /// Doubling the queries doubles the query phase and leaves the transcript alone.
    #[test]
    fn the_query_phase_scales_with_the_query_count() {
        let one = verifier_hashes(&minimal_geometry(), "Poseidon1");
        let mut geom = minimal_geometry();
        geom.n_queries = 8;
        let eight = verifier_hashes(&geom, "Poseidon1");

        assert_eq!(eight.leaf, one.leaf * 8);
        assert_eq!(eight.merkle, one.merkle * 8);
    }

    /// Each FRI step past the first commits a folded tree, which is then opened per query.
    #[test]
    fn each_fri_step_adds_a_folding_tree() {
        let mut geom = minimal_geometry();
        geom.step_n_bits = vec![4, 2];
        let counts = verifier_hashes(&geom, "Poseidon1");

        // The step-1 tree has 2^2 groups of 2^(4-2) = 4 extension elements: width 12, 1 level.
        assert_eq!(counts.fri, 2, "one leaf hash and one path level");
    }

    /// Grinding is one hash when it is configured and none when it is not.
    #[test]
    fn grinding_costs_one_hash() {
        let mut geom = minimal_geometry();
        assert_eq!(verifier_hashes(&geom, "Poseidon1").grinding, 0);
        geom.pow_bits = 20;
        assert_eq!(verifier_hashes(&geom, "Poseidon1").grinding, 1);
    }

    /// Checked against the instrumented native verifier, which counts its own hashes (see the report
    /// at the end of `starkVerify`). Three airs of the hashes example agreed to the unit:
    ///
    ///   Sha2    2^16 blowup 1  ->  27674   Blake2b 2^16 blowup 1  ->  29498
    ///   Blake3  2^20 blowup 2  ->  24183   (leaf 6042, merkle 9132, fri 8568, transcript 440)
    ///
    /// This pins the Blake3 one, whose geometry comes from its own `Blake3.starkinfo.json`: arity 2,
    /// 114 queries, lastLevelVerification 2, nBitsExt 22, steps [22,19,16,13,10,7,5], cm 214/174/6,
    /// 8 constants, hashCommits, no publics, challenges 2 and 1 in stages 2 and 3.
    #[test]
    fn the_measured_blake3_air_matches_the_native_verifier() {
        let geom = VerifierGeometry {
            n_bits_ext: 22,
            arity: 2,
            last_level_verification: 2,
            n_queries: 114,
            pow_bits: 16,
            hash_commits: true,
            stage_widths: vec![214, 174, 6],
            n_constants: 8,
            custom_commit_widths: vec![],
            step_n_bits: vec![22, 19, 16, 13, 10, 7, 5],
            n_publics: 0,
            n_evals: 531,
            stage_challenges: vec![2, 1],
            stage_air_values: vec![0, 0],
            final_pol_size: 1 << 5,
        };
        let counts = verifier_hashes(&geom, "blake3");

        assert_eq!(counts.leaf, 6042, "114 queries x 53 compressions");
        assert_eq!(counts.merkle, 9132, "4 trees x 114 x 20 levels + 4 root reductions");
        assert_eq!(counts.fri, 8568, "6 folding trees");
        assert_eq!(counts.transcript, 440, "replayed absorb/squeeze sequence");
        assert_eq!(counts.grinding, 1);
        assert_eq!(counts.total(), 24183, "what the native verifier reported");
    }

    /// The same air under Poseidon: identical widths and steps, arity 4 instead of 2. Also checked
    /// against the native verifier (12396 for Poseidon1 and Poseidon2 alike). Pinned separately
    /// because arity 4 is the case that exposes the narrower kept level -- see
    /// `an_odd_height_keeps_a_narrower_level`.
    #[test]
    fn the_measured_air_under_poseidon_matches_the_native_verifier() {
        let geom = VerifierGeometry {
            n_bits_ext: 22,
            arity: 4,
            last_level_verification: 2,
            n_queries: 114,
            pow_bits: 16,
            hash_commits: true,
            stage_widths: vec![214, 174, 6],
            n_constants: 8,
            custom_commit_widths: vec![],
            step_n_bits: vec![22, 19, 16, 13, 10, 7, 5],
            n_publics: 0,
            n_evals: 531,
            stage_challenges: vec![2, 1],
            stage_air_values: vec![0, 0],
            final_pol_size: 1 << 5,
        };

        for family in ["Poseidon1", "Poseidon2"] {
            let counts = verifier_hashes(&geom, family);
            assert_eq!(counts.leaf, 3990, "{family}");
            assert_eq!(counts.merkle, 4124, "{family}");
            assert_eq!(counts.fri, 4126, "{family}");
            assert_eq!(counts.transcript, 155, "{family}");
            assert_eq!(counts.total(), 12396, "{family}: what the native verifier reported");
        }
    }

    /// Blake3's binary trees make paths twice as long as arity-4 Poseidon's, which is the whole
    /// reason the family matters to this count. Each family carries its own arity, exactly as its
    /// own `stark_struct` does -- the caller must not pair a family with a foreign geometry.
    #[test]
    fn the_family_changes_the_count_not_just_the_price() {
        let poseidon = verifier_hashes(&minimal_geometry(), "Poseidon1");
        let blake3 =
            verifier_hashes(&VerifierGeometry { arity: merkle_tree_arity("blake3"), ..minimal_geometry() }, "blake3");

        assert_eq!(blake3.merkle, poseidon.merkle * 2, "binary paths are twice as long");
    }
}
