//! Hash invocations a verifier performs for one air. Geometry, not who checks it: native and
//! in-circuit run the same algorithm and differ only in the price of one hash.
//!
//! Each rule mirrors the verifier piece its test names. The transcript term is the standalone
//! (`challengesVadcop == false`) sequence, so an aggregated air's is ~1-3% lower.

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
    /// The transcript sponge's arity, which `StarkStruct` keeps independent of the tree's.
    pub transcript_arity: u64,
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
    /// Coefficients of the final polynomial FRI sends in clear, each a field-extension element.
    pub final_pol_size: u64,
}

/// Blake3 compressions over `bytes`: one per 64-byte block, plus one parent per extra 1024-byte
/// chunk. A 64-byte input — one Merkle node — is a single compression.
pub fn blake3_compressions(bytes: u64) -> u64 {
    // `hash_le64` forces nblocks = 1, so even an empty input costs one compression.
    let blocks = bytes.div_ceil(64).max(1);
    let chunks = bytes.div_ceil(1024).max(1);
    blocks + (chunks - 1)
}

/// Permutations to hash `width` field elements into one digest. Poseidon absorbs `sponge_rate`
/// per permutation (`linear_hash_seq`); Blake3 hashes the bytes in one call.
pub fn leaf_hashes(family: &str, arity: u64, width: u64) -> u64 {
    match family {
        "blake3" => blake3_compressions(width * 8),
        // sponge_rate is 0 at arity 1 and underflows at 0; the trees support 2..=4.
        _ if arity < 2 => 0,
        _ => width.div_ceil(sponge_rate(arity)),
    }
}

/// Node hashes on one Merkle path: `ceil(log_arity(height)) - last_level_verification`, matching
/// `MerkleTreeGL::getMerkleProofLength`. Each level is one permutation, since the sponge width is
/// exactly `arity * DIGEST_SIZE`.
pub fn merkle_path_permutations(n_bits_height: u64, arity: u64, last_level_verification: u64) -> u64 {
    if n_bits_height == 0 {
        return 0;
    }
    // The C++ divides the logs in floating point; mirror it so the two never disagree on an arity
    // that is not a power of two.
    let levels = (n_bits_height as f64 / (arity as f64).log2()).ceil() as u64;
    levels.saturating_sub(last_level_verification)
}

/// Hashes reducing the stored bottom level to the root, once per tree (`merkletreeReduce`). The kept
/// level is at *most* `arity^llv` wide, overshooting when the arity does not divide the height.
pub fn root_reduction_hashes(arity: u64, last_level_verification: u64, n_bits_height: u64) -> u64 {
    // The trees support arity 2..=4; below that the fold never converges. Both bounds are
    // user-configurable, so cap rather than overflow the pow or the shift.
    if arity < 2 || n_bits_height >= 64 {
        return 0;
    }
    let Some(stop) = arity.checked_pow(last_level_verification.min(u32::MAX as u64) as u32) else {
        return 0;
    };
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

/// Build the geometry from what the stats pipeline already has. Everything family-dependent
/// reaches it through `stark_struct` (the arity, and through it the FRI query count).
pub fn geometry_for_family(
    stark_struct: &crate::types::stark_struct::StarkStruct,
    setup: &crate::types::pilout_info::SetupResult,
    n_evals: usize,
) -> VerifierGeometry {
    use crate::types::security;
    use crate::types::security::pcs::{Batching, Fri, FriConfig};
    use crate::types::security::regimes::DecodingRegime;

    // The arity the setup actually builds: only blake3 has it forced, so any other family can be
    // configured away from the family default and the whole geometry moves with it.
    let arity = stark_struct.merkle_tree_arity as u64;

    // Same configuration `build_starkinfo_output` uses: the query count and grinding bits come
    // out of the security analysis, not out of the settings.
    let fri = Fri::new(FriConfig {
        field_size: security::goldilocks_safe_extension_field_size(),
        trace_length: 1u32 << stark_struct.n_bits,
        rate: 1.0 / (1u64 << (stark_struct.n_bits_ext - stark_struct.n_bits)) as f64,
        batch_size: n_evals.max(1) as u64,
        batching: Batching::Powers,
        log_folding_factors: crate::output::stark_info::compute_log_folding_factors(stark_struct),
        max_grinding_bits_query: stark_struct
            .low_degree_test
            .expect_fri("verifier hash accounting")
            .grinding_bits_queries as u64,
        use_max_grinding_bits_query: true,
        tree_arity: arity,
        hash_size_bits: 256,
        target_security_bits: 128,
        regime: DecodingRegime::Jbr,
    });
    let security = fri.security_params();
    let fri_struct = stark_struct.low_degree_test.expect_fri("verifier hash accounting");

    let width = |section: &str| setup.map_sections_n.get(section).copied().unwrap_or(0) as u64;

    VerifierGeometry {
        n_bits_ext: stark_struct.n_bits_ext as u64,
        arity,
        transcript_arity: stark_struct.transcript_arity as u64,
        last_level_verification: stark_struct.last_level_verification as u64,
        n_queries: security.n_queries,
        pow_bits: security.grinding_bits_query as u64,
        hash_commits: stark_struct.hash_commits,
        // One committed tree per stage, plus the quotient stage.
        stage_widths: (1..=setup.n_stages + 1).map(|s| width(&format!("cm{s}"))).collect(),
        n_constants: setup.n_constants as u64,
        custom_commit_widths: setup.custom_commits.iter().map(|c| width(&format!("{}0", c.name))).collect(),
        step_n_bits: fri_struct.log_domain_sizes.iter().map(|&b| b as u64).collect(),
        n_publics: setup.n_publics as u64,
        n_evals: n_evals as u64,
        stage_challenges: (2..=setup.n_stages + 1)
            .map(|s| setup.challenges_map.iter().filter(|c| c.stage == Some(s)).count() as u64)
            .collect(),
        stage_air_values: (2..=setup.n_stages + 1)
            .map(|s| setup.air_values_map.iter().filter(|v| v.stage == Some(s)).count() as u64)
            .collect(),
        final_pol_size: 1u64 << fri_struct.log_domain_sizes.last().copied().unwrap_or(0),
    }
}

/// Hashes the verifier performs for one air, under `family`.
pub fn verifier_hashes(geom: &VerifierGeometry, family: &str) -> HashCounts {
    let mut counts = HashCounts::default();

    // ── Query phase: every committed tree is opened at every query ──
    let open = |width: u64, n_bits_height: u64, leaf: &mut u64, merkle: &mut u64| {
        *leaf += geom.n_queries * leaf_hashes(family, geom.arity, width);
        *merkle += geom.n_queries * merkle_path_permutations(n_bits_height, geom.arity, geom.last_level_verification);
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
        // A deserialized starkinfo is not guaranteed to have decreasing steps.
        let group_size = 1u64 << geom.step_n_bits[step - 1].saturating_sub(n_bits).min(63);
        let (mut leaf, mut merkle) = (0, 0);
        open(group_size * FIELD_EXTENSION, n_bits, &mut leaf, &mut merkle);
        counts.fri += leaf + merkle;
    }

    counts.transcript = transcript_hashes(geom);
    // `starkVerify` runs the permutation unconditionally; powBits only picks the threshold.
    counts.grinding = 1;
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
    let mut t = TranscriptSim::new(geom.transcript_arity);
    // A hashed commit absorbs a digest of the values rather than the values, and computing that
    // digest costs a sponge of its own (`transcriptHash.getState`).
    let put_values = |t: &mut TranscriptSim, n: u64| {
        if geom.hash_commits {
            let mut inner = TranscriptSim::new(geom.transcript_arity);
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
    let mut queries = TranscriptSim::new(geom.transcript_arity);
    queries.put(FIELD_EXTENSION);
    queries.put(1);
    queries.get_permutations(geom.n_queries, geom.step_n_bits.first().copied().unwrap_or(0));

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
        // `hash_le64` compresses one padded block even for an empty input, so a zero-width tree
        // still costs a hash per query -- matching `blake3_core`'s `if (nblocks == 0) nblocks = 1`.
        assert_eq!(blake3_compressions(0), 1);
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
        assert_eq!(merkle_path_permutations(20, 4, 0), 10);
        assert_eq!(merkle_path_permutations(20, 2, 0), 20);
        assert_eq!(merkle_path_permutations(20, 4, 2), 8);
        assert_eq!(merkle_path_permutations(0, 4, 0), 0, "a height-1 tree has no path");
    }

    /// The kept bottom level is reduced to the root once, not once per query.
    #[test]
    fn the_root_reduction_is_paid_once_per_tree() {
        assert_eq!(root_reduction_hashes(4, 0, 20), 0);
        assert_eq!(root_reduction_hashes(4, 1, 20), 1, "4 nodes -> 1");
        assert_eq!(root_reduction_hashes(4, 2, 20), 5, "16 -> 4 -> 1");
        assert_eq!(root_reduction_hashes(2, 3, 20), 7, "8 -> 4 -> 2 -> 1");
    }

    /// An odd `n_bits` folded by 4 cannot land on 16, so the kept level is 8 wide and costs less.
    #[test]
    fn an_odd_height_keeps_a_narrower_level() {
        assert_eq!(root_reduction_hashes(4, 2, 22), 5, "2^22 folds to exactly 16");
        assert_eq!(root_reduction_hashes(4, 2, 19), 3, "2^19 folds to 8, not 16");
        assert_eq!(root_reduction_hashes(2, 2, 19), 3, "arity 2 always lands on 4");
    }

    /// The shipped blake3 default is arity 2 with llv 4, which neither measured case covers: the
    /// path drops 4 levels rather than 2 and the reduction folds a 16-node level instead of a 4-node
    /// one, so the two terms move in opposite directions.
    #[test]
    fn the_default_binary_geometry_trades_path_levels_for_root_reductions() {
        assert_eq!(merkle_path_permutations(22, 2, 4), 18, "22 levels less the 4 the kept level replaces");
        assert_eq!(merkle_path_permutations(22, 2, 2), 20, "the same tree at the old llv");
        // 16 -> 8 -> 4 -> 2 -> 1 = 8 + 4 + 2 + 1.
        assert_eq!(root_reduction_hashes(2, 4, 22), 15, "a 16-node kept level folds in 15");
        assert_eq!(root_reduction_hashes(2, 2, 22), 3, "a 4-node one in 3");
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
            transcript_arity: 4,
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

    /// One hash either way: `starkVerify` runs the permutation before looking at powBits, which
    /// only selects the threshold the result is compared against.
    #[test]
    fn grinding_costs_one_hash_whatever_the_threshold() {
        let mut geom = minimal_geometry();
        assert_eq!(verifier_hashes(&geom, "Poseidon1").grinding, 1);
        geom.pow_bits = 20;
        assert_eq!(verifier_hashes(&geom, "Poseidon1").grinding, 1);
    }

    /// Measured against an instrumented native verifier: 24183 for this Blake3 geometry.
    #[test]
    fn the_measured_blake3_air_matches_the_native_verifier() {
        let geom = VerifierGeometry {
            n_bits_ext: 22,
            arity: 2,
            transcript_arity: 2,
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
            transcript_arity: 4,
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
        let blake3_arity = merkle_tree_arity("blake3");
        let blake3 = verifier_hashes(
            &VerifierGeometry { arity: blake3_arity, transcript_arity: blake3_arity, ..minimal_geometry() },
            "blake3",
        );

        assert_eq!(blake3.merkle, poseidon.merkle * 2, "binary paths are twice as long");
    }
}
