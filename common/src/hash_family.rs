//! Hash-family registry.

use proofman_starks_lib_c::GOLDILOCKS_POSEIDON_MERKLE_TREE_ARITY;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateRole {
    PoseidonSponge,
    PoseidonCompression,
    TreeSelector,
    CMul,
    Fft4,
    EvPol4,
    SelectValArity4,
    SelectValArity2,
    Blake3Node,
    Blake3Compress,
}

pub const FAMILIES: &[&str] = &["Poseidon1", "Poseidon2", "blake3"];

/// The family a caller gets when it does not choose one: every `--hash` default, every `Default`
/// impl that carries a family, and the read-back value for a `globalInfo.json` with no `hash`.
pub const DEFAULT_HASH_ID: &str = "blake3";

/// Merkle-tree arity of `family`'s commitment trees over Goldilocks digests
pub fn merkle_tree_arity(family: &str) -> u64 {
    match family {
        "Poseidon1" | "Poseidon2" => GOLDILOCKS_POSEIDON_MERKLE_TREE_ARITY,
        "blake3" => 2,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Rows, as log2, that a recursive1 circuit may reach before the setup inserts a compressor
/// between it and the basic proof.
///
/// It is the family's natural recursion size, not a universal limit: a recursive1 that lands at or
/// under it needs no compressor, and everything in the airgroup is then padded to one shared size.
/// Poseidon recursion settles at 2^17. blake3's is 2^19, because its verification is dominated by
/// arity-2 Merkle path hashing at 56 AIR rows a compression -- reading 17 here would make every
/// blake3 air demand a compressor it does not need, and each compressor is a whole extra proof.
///
/// The A2 query bump fills a small circuit toward `2^(threshold - 1)`, so this also sets that
/// target.
pub fn recursive_bits_threshold(family: &str) -> usize {
    match family {
        "blake3" => 19,
        "Poseidon1" | "Poseidon2" => 17,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Whether the family's FRI schedule is solved for rather than folded at a fixed rate.
///
/// Worth it exactly where Merkle paths are expensive, which is where the tree is binary: a path
/// twice as long makes it cheaper to fold hard and keep fewer layers, and no uniform folding factor
/// expresses that. At arity 4 a fold of 3 and a fold of 4 tie, so poseidon gains nothing and keeps
/// the schedule its committed verifiers and circom fixtures encode. See
/// `stark_struct::optimal_fri_steps`.
pub fn uses_optimal_fri_schedule(family: &str) -> bool {
    merkle_tree_arity(family) == 2
}

/// Field elements one compression of `family` absorbs: a BLAKE3 block is 64 bytes, so 8 Goldilocks
/// elements. Not a rate -- these are compression functions that take a whole block per call and
/// carry the chaining value in the state, so a leaf of `w` elements costs `ceil(w / this)`.
pub fn compression_block_elements(family: &str) -> usize {
    match family {
        "blake3" => 8,
        "Poseidon1" | "Poseidon2" => 8,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// The FRI layer degree the recursion stops folding at, as log2.
///
/// Everything below the terminal is sent in the clear, and the verifier evaluates it at every query
/// -- which is `2^terminal` extension multiplications a query, arithmetic rather than hashing. Above
/// 7 that arithmetic outgrows the room the BLAKE3 block interiors leave for it.
pub fn fri_terminal_degree(family: &str) -> usize {
    match family {
        "blake3" => 7,
        "Poseidon1" | "Poseidon2" => 5,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Merkle levels the proof carries outright, for a recursion layer, or `None` to take the size-based
/// default.
///
/// Each level kept is one compression a query does not do, on every tree -- and it is bought with
/// `SelectValue` gates, `arity^llv - 1` of them per opening, which double for every level added. The
/// trade is worth taking exactly where paths are expensive and the gates are not: with a binary tree
/// the paths are twice as long, and since the BLAKE3 block interiors host the gates for free, a gate
/// row costs nothing the air was not already paying for.
///
/// blake3 takes one level above the size-based default (which is 4 at arity 2, from
/// `LAST_LEVEL_NODES = 16`); poseidon takes the default. The blake3 COMPRESSOR takes one more still,
/// and that is a per-template call rather than a per-family one -- see
/// `proving_key::recursive::recursive_last_level_verification`.
pub fn recursive_last_level_verification(family: &str) -> Option<usize> {
    match family {
        "blake3" => Some(5),
        "Poseidon1" | "Poseidon2" => None,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Proof-of-work bits the FINAL stage grinds for.
///
/// Its own function rather than `recursive_grinding_bits`, because the two differ for poseidon: the
/// recursion grinds 20 and the final stage 22, and both are pinned by what the committed verifiers
/// encode. blake3 grinds 24 throughout -- it hashes fast enough to afford it, and every bit comes
/// straight off the query count, which is what a verifier pays for per tree per query.
pub fn final_grinding_bits(family: &str) -> usize {
    match family {
        "blake3" => 24,
        "Poseidon1" | "Poseidon2" => 22,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Rows the `vadcop_final` air is pinned to, as log2, or `None` to let it size itself.
///
/// Pinned so one committed verifier per family serves every pilout: the verifier encodes nBits, so
/// airgroup count must not move it. Poseidon has no committed final verifier, so it sizes itself.
pub fn final_n_bits(family: &str) -> Option<usize> {
    match family {
        "blake3" => Some(19),
        "Poseidon1" | "Poseidon2" => None,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Blowup the final air is built at.
///
/// Blowup buys queries and costs memory, and the exchange rate is brutal: each step doubles the
/// extended domain -- every extended column and every Merkle node with it -- while the query count
/// only falls as `(128 - grinding) / log2(1/rate)`. Dropping one blowup therefore roughly halves the
/// prover's memory while adding a fraction to the query count and the proof with it. Halving memory
/// for a fraction more proof is the right side of that trade for a
/// family whose final air already sits at 99% of its BLAKE3 block capacity.
///
/// blake3 takes 2, but NOT on the trade the paragraph above describes -- measured, it does not halve
/// the memory. This air's constraints are naturally degree 8, so it needs no intermediate polynomials
/// at all; at 2 the cap falls to 5 and the packer adds 125 of them, 387 base-field columns against
/// 21. The extra columns cancel the halved domain almost exactly: 2^22 x 346 cells against
/// 2^21 x 712. What 2 does buy is a smaller extended domain to hold at once, and it costs queries --
/// 70 to 106 -- and the proof with them. Deliberate, not an oversight.
///
/// Poseidon keeps 4, which its committed verifiers and circom fixtures encode.
pub fn final_blowup_factor(family: &str) -> usize {
    match family {
        "blake3" => 2,
        "Poseidon1" | "Poseidon2" => 4,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// The highest constraint degree a blowup of `2^blowup` can carry.
///
/// A blowup of `2^b` lets the quotient span `2^b` chunks, so constraints may reach `2^b + 1`.
/// Compiling below that wastes the headroom -- and it is not free headroom: `std_sum` packs
/// `maxDeg - 1` bus terms behind each `im` pol, so a higher degree is directly fewer stage-2
/// columns -- on the final air the difference between the floor and the cap is close to half of them.
///
/// Capped at 8, which is the repo's own convention -- `Compressor.pil` pins
/// `set_max_constraint_degree(8)`. Blowup 3 and up could carry 9, worth about six more columns,
/// but nothing in the tree has been compiled there.
pub fn max_constraint_degree_for_blowup(blowup: usize) -> usize {
    const CAP: usize = 8;
    (2usize.pow(blowup as u32) + 1).min(CAP)
}

/// Whether the pipeline builds the `vadcop_final_compressed` stage by default.
///
/// The stage exists to shrink the final proof before whatever consumes it. Whether it does depends
/// on the tree: it trades Merkle path levels -- through a higher `lastLevelVerification` -- for data
/// sent in the clear, and with a binary tree that trade is close to even -- a couple of percent of
/// proof size for a whole extra recursion layer, and that layer's prover memory. Poseidon's arity-4
/// paths are half as long, so the same level costs it
/// less to keep and the compression is worth having.
///
/// Off is not the same as unavailable: `proofman-setup setup-compressed-final` adds the stage to an
/// existing proving key, so a key built without it can gain it later.
pub fn compressed_final_by_default(family: &str) -> bool {
    family != "blake3"
}

/// Proofs one recursive2 circuit aggregates, by family.
///
/// Every extra proof aggregated multiplies the verification work recursive2 does, and for blake3
/// that work is almost entirely hashing: on a fibonacci recursion, recursive2's BLAKE3 gate count is
/// exactly `arity` times recursive1's, and it alone decides the air's size. Poseidon absorbs a third
/// proof cheaply -- arity-4 Merkle paths, and a permutation that is not the bulk of the circuit --
/// so it keeps 3. Callers may override; both values stay valid for both families.
pub fn default_aggregation_arity(family: &str) -> usize {
    match family {
        "blake3" => 2,
        "Poseidon1" | "Poseidon2" => 3,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Proof-of-work bits the query phase grinds for. Per family because the prover searches `2^bits`,
/// so what one affords tracks how fast it hashes; the bits come off the query count.
pub fn default_grinding_bits(family: &str) -> usize {
    match family {
        "Poseidon1" | "Poseidon2" => 16,
        "blake3" => 24,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Proof-of-work bits every recursion layer grinds for.
///
/// Distinct from `default_grinding_bits`, which is the basic airs' figure: recursion grinds harder
/// because the bits come off the query count, and a query is far more expensive to verify inside a
/// circuit than outside one. In the fibonacci recursion one query costs ~179 BLAKE3 compressions
/// per proof verified, while the grinding itself costs the verifier a single hash -- it checks the
/// nonce, and the 2^bits is all prover work.
///
/// Poseidon's 20 is pinned by its committed native verifiers and circom fixtures, which encode the
/// query count it buys. blake3's fixtures are not committed to a figure, so it takes the 24 its own
/// family default asks for.
pub fn recursive_grinding_bits(family: &str) -> usize {
    match family {
        "Poseidon1" | "Poseidon2" => 20,
        "blake3" => 24,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Whether the family's `vadcop_final` proof can be wrapped in a BN128 SNARK.
///
/// The snark stage recurses the final Goldilocks proof into a circom circuit over BN128, and only
/// the poseidon families have that path built: the `recursivef` verifier and its circom templates
/// are poseidon-only. A blake3 proving key therefore has no snark stage, and `setup-snark` /
/// `prove-snark` refuse it up front rather than emit artifacts that cannot verify.
pub fn supports_snark(family: &str) -> bool {
    // Listed rather than `!= "blake3"`: this is a guard, so an unrecognised family has to fail
    // closed. The `!=` form its neighbours use is fine for a preference, not for a refusal.
    matches!(family, "Poseidon1" | "Poseidon2")
}

/// Fiat-Shamir transcript arity: the transcript squeezes through the same
/// sponge/compression as the trees, so it shares the Merkle arity.
pub fn transcript_arity(family: &str) -> u64 {
    merkle_tree_arity(family)
}

/// Elements in one digest, i.e. the sponge capacity (`nFieldElements` in the C++ trees).
pub const DIGEST_SIZE: u64 = 4;

/// Sponge width of `family`'s permutation at `arity`. The Merkle node hash absorbs
/// `arity * DIGEST_SIZE` children, and the trees pick the width that exactly fits them.
pub fn sponge_width(arity: u64) -> u64 {
    arity * DIGEST_SIZE
}

/// Elements absorbed per permutation by a linear (leaf) hash: the width less the capacity
/// carried between blocks. Mirrors `PoseidonGoldilocks::linear_hash_seq`.
pub fn sponge_rate(arity: u64) -> u64 {
    sponge_width(arity) - DIGEST_SIZE
}

/// Elements the transcript buffers before it permutes, and the size of the output FIFO it
/// then drains. Mirrors `TranscriptGL`'s `transcriptPendingSize` / `transcriptOutSize`.
pub fn transcript_pending_size(arity: u64) -> u64 {
    DIGEST_SIZE * (arity - 1)
}

pub fn transcript_out_size(arity: u64) -> u64 {
    DIGEST_SIZE * arity
}

/// Families with streaming-commit slot kernels (stream_commit.cu). The C side
/// re-checks via get_hash_family() and returns -15 on mismatch, so this list
/// must stay in sync with commit_witness_streaming_gpu's family gate.
pub fn supports_stream_commit(family: &str) -> bool {
    matches!(family, "Poseidon1" | "blake3")
}

/// True when the family's kernels support exactly one tree geometry
pub fn has_forced_tree_geometry(family: &str) -> bool {
    family == "blake3"
}

// (gate template name, role, owning family). `None` for family-agnostic gates.
const GATES: &[(&str, GateRole, Option<&str>)] = &[
    ("Poseidon1_16", GateRole::PoseidonSponge, Some("Poseidon1")),
    ("CustPoseidon1_16", GateRole::PoseidonCompression, Some("Poseidon1")),
    // TreeSelector's number is the gate's RADIX, not a hash arity: TreeSelector(nLevels, eSize)
    // selects one of 2^nLevels values by nLevels BINARY key bits, and TreeSelector4/8 just collapse
    // two or three levels of that binary tree into one gate. eSize is 3 -- extension elements, for
    // FRI -- so it has nothing to do with merkle_tree_arity. Which radix an air uses is a layout
    // choice (17 cells in one row vs 30 in two), so these are family-agnostic; the family is
    // identified by its hash gates, and an r1cs carrying both radices still trips the duplicate-role
    // check in to_plonk. Tagging them by family would make a blake3 r1cs that uses either radix look
    // like it mixed two families.
    ("TreeSelector8", GateRole::TreeSelector, None),
    ("Poseidon2_16", GateRole::PoseidonSponge, Some("Poseidon2")),
    ("CustPoseidon2_16", GateRole::PoseidonCompression, Some("Poseidon2")),
    ("TreeSelector4", GateRole::TreeSelector, None),
    ("CMul", GateRole::CMul, None),
    ("FFT4", GateRole::Fft4, None),
    ("EvPol4", GateRole::EvPol4, None),
    ("SelectValueArity4", GateRole::SelectValArity4, None),
    ("SelectValueArity2", GateRole::SelectValArity2, None),
    ("Blake3Node", GateRole::Blake3Node, Some("blake3")),
    ("Blake3Compress", GateRole::Blake3Compress, Some("blake3")),
];

pub fn lookup_gate(name: &str) -> Option<(GateRole, Option<&'static str>)> {
    GATES.iter().find(|(n, _, _)| *n == name).map(|(_, r, f)| (*r, *f))
}

pub fn is_known_family(id: &str) -> bool {
    FAMILIES.contains(&id)
}

/// Whether this family can be emitted as a native Rust verifier.
///
/// The generated verifier hashes through `proofman_fields::Hash`, which is a fixed-width sponge:
/// WIDTH / RATE / CAPACITY and an in-place permutation over a `[F; WIDTH]` state. Every Poseidon
/// variant is that shape. BLAKE3 is not -- it consumes a variable-length run of field elements and
/// produces four, which is why the C++ side reaches it through `Blake3Goldilocks::linearHash` and
/// `merkletree` rather than through the permutation interface. Supporting it here means
/// generalising that trait and the verifier template, not adding an impl, so until then a blake3
/// proving key is written without this artifact.
pub fn supports_native_rust_verifier(family: &str) -> bool {
    is_known_family(family)
}

pub fn rust_hash_type(family: &str, arity: u64) -> &'static str {
    match (family, arity * 4) {
        ("Poseidon1", 8) => "Poseidon1_8",
        ("Poseidon1", 12) => "Poseidon1_12",
        ("Poseidon1", 16) => "Poseidon1_16",
        ("Poseidon2", 4) => "Poseidon2_4",
        ("Poseidon2", 8) => "Poseidon2_8",
        ("Poseidon2", 12) => "Poseidon2_12",
        ("Poseidon2", 16) => "Poseidon2_16",
        // BLAKE3 has one width and it is the block, not a sponge width: eight field elements is a
        // 64-byte block, and at arity 2 a node is exactly two four-element digests. The arity is
        // still checked, because any other would mean a node the block cannot hold.
        ("blake3", 8) => "Blake3_8",
        (fam, width) => panic!("Unsupported hash type: {fam}_{width} (arity {arity})"),
    }
}

/// The Rust TRANSCRIPT type for a family, which is not always a sponge over a hash.
///
/// `rust_hash_type` answers "which hash", and for the Merkle and grinding slots that is the whole
/// answer. The transcript slot is different: Poseidon's is a sponge over its permutation, but
/// BLAKE3's absorbs eight words per compression with the state in the chaining value. Naming the
/// hash and wrapping it in a sponge produced different challenges from the prover's -- every
/// challenge, from the first squeeze -- which is why this returns the construction instead.
pub fn rust_transcript_type(family: &str, arity: u64) -> String {
    match family {
        "blake3" => "Blake3Transcript<Goldilocks>".to_string(),
        _ => format!("Transcript<Goldilocks, {}>", rust_hash_type(family, arity)),
    }
}

/// Type names `rust_transcript_type` needs imported, so the generator does not have to parse it.
pub fn rust_transcript_imports(family: &str, arity: u64) -> Vec<&'static str> {
    match family {
        "blake3" => vec!["Blake3Transcript"],
        _ => vec!["Transcript", rust_hash_type(family, arity)],
    }
}

pub fn rust_grinding_type(family: &str) -> &'static str {
    match family {
        "Poseidon1" => "Poseidon1_8",
        "Poseidon2" => "Poseidon2_8",
        "blake3" => "Blake3_8",
        fam => panic!("Unsupported grinding hash family: {fam}"),
    }
}

#[cfg(test)]
mod tests {
    /// Recursion grinds harder than the basic airs, and blake3 harder than poseidon: a query is
    /// expensive to verify in-circuit while the grinding costs the verifier one hash.
    #[test]
    fn recursive_grinding_bits_exceed_the_basic_defaults() {
        assert_eq!(super::recursive_grinding_bits("blake3"), 24);
        assert_eq!(super::recursive_grinding_bits("Poseidon1"), 20);
        assert_eq!(super::recursive_grinding_bits("Poseidon2"), 20);
        for f in super::FAMILIES {
            assert!(
                super::recursive_grinding_bits(f) >= super::default_grinding_bits(f),
                "{f} would grind less in recursion than in its basic airs"
            );
        }
    }

    /// Changing the pin means regenerating every verifier in `verifier/src/<family>/`.
    #[test]
    fn the_final_air_is_pinned_only_where_a_committed_verifier_encodes_it() {
        assert_eq!(super::final_n_bits("blake3"), Some(19));
        for f in ["Poseidon1", "Poseidon2"] {
            assert_eq!(super::final_n_bits(f), None, "{f}");
        }
    }

    /// The final stage grinds harder than the recursion for poseidon (22 vs 20) and the same for
    /// blake3 (24), and every family grinds at least as hard as its basic airs.
    #[test]
    fn final_grinding_bits_are_pinned_per_family() {
        assert_eq!(super::final_grinding_bits("blake3"), 24);
        assert_eq!(super::final_grinding_bits("Poseidon1"), 22);
        assert_eq!(super::final_grinding_bits("Poseidon2"), 22);
        for f in super::FAMILIES {
            assert!(super::final_grinding_bits(f) >= super::default_grinding_bits(f), "{f}");
        }
    }

    /// The blowup is per family, and whatever it is must still carry degree 5 -- the floor the final
    /// air's own constraints need -- so the degree cap never becomes the thing that forces it.
    #[test]
    fn final_blowup_is_per_family_and_carries_the_degree() {
        assert_eq!(super::final_blowup_factor("blake3"), 2);
        assert_eq!(super::final_blowup_factor("Poseidon2"), 4);
        for f in super::FAMILIES {
            let b = super::final_blowup_factor(f);
            assert!(super::max_constraint_degree_for_blowup(b) >= 5, "{f} at blowup {b} cannot carry degree 5");
        }
    }

    /// The degree follows the blowup, capped at the repo's 8. Blowup 2 lands on 5, which is what the
    /// recursion already compiles at -- so this changes nothing there and everything above it.
    #[test]
    fn max_constraint_degree_follows_the_blowup() {
        assert_eq!(super::max_constraint_degree_for_blowup(1), 3);
        assert_eq!(super::max_constraint_degree_for_blowup(2), 5);
        assert_eq!(super::max_constraint_degree_for_blowup(3), 8);
        assert_eq!(super::max_constraint_degree_for_blowup(4), 8);
        assert_eq!(super::max_constraint_degree_for_blowup(10), 8, "capped");
    }

    /// blake3 skips the compressed final because it measured a 2% saving for a whole extra layer;
    /// poseidon keeps it. Neither is unavailable -- the standalone subcommand adds it either way.
    #[test]
    fn compressed_final_is_off_only_for_blake3() {
        assert!(!super::compressed_final_by_default("blake3"));
        assert!(super::compressed_final_by_default("Poseidon1"));
        assert!(super::compressed_final_by_default("Poseidon2"));
    }

    /// blake3's recursion settles at 2^19 against poseidon's 2^17: reading poseidon's number for
    /// blake3 makes every air demand a compressor it does not need.
    #[test]
    fn recursive_bits_threshold_is_the_family_recursion_size() {
        assert_eq!(super::recursive_bits_threshold("blake3"), 19);
        assert_eq!(super::recursive_bits_threshold("Poseidon1"), 17);
        assert_eq!(super::recursive_bits_threshold("Poseidon2"), 17);
    }

    /// blake3 aggregates two proofs, poseidon three: recursive2's BLAKE3 gate count is `arity` times
    /// recursive1's, and for blake3 that count is what sizes the air.
    #[test]
    fn default_aggregation_arity_is_two_for_blake3_and_three_for_poseidon() {
        assert_eq!(super::default_aggregation_arity("blake3"), 2);
        assert_eq!(super::default_aggregation_arity("Poseidon1"), 3);
        assert_eq!(super::default_aggregation_arity("Poseidon2"), 3);
        for f in super::FAMILIES {
            let a = super::default_aggregation_arity(f);
            assert!(crate::global_info::is_valid_aggregation_arity(a), "{f} defaults to an invalid arity {a}");
        }
    }

    use super::*;

    #[test]
    fn blake3_gates_are_registered_to_the_blake3_family() {
        assert_eq!(lookup_gate("Blake3Node"), Some((GateRole::Blake3Node, Some("blake3"))));
        assert_eq!(lookup_gate("Blake3Compress"), Some((GateRole::Blake3Compress, Some("blake3"))));
    }

    /// `to_plonk.rs`'s `lookup_gate` panics on an unregistered name, so a family in `FAMILIES`
    /// with no gates cannot be packed at all -- which is what blocked blake3 recursion.
    /// A single air's gate set must resolve to exactly ONE family, because `to_plonk` panics with
    /// "r1cs mixes multiple hash families" otherwise.
    ///
    /// This is why TreeSelector4/8 are family-agnostic. Their number is the gate's radix over a
    /// binary selection tree, not a hash arity, so any family may use either; tagging TreeSelector4
    /// as Poseidon2 made a blake3 r1cs that used radix 4 look like it mixed blake3 with Poseidon2.
    /// The hash gates alone still identify the family, which is the property that has to hold.
    #[test]
    fn a_families_gate_set_resolves_to_exactly_one_family() {
        let plausible: &[(&str, &[&str])] = &[
            (
                "Poseidon1",
                &["Poseidon1_16", "CustPoseidon1_16", "TreeSelector8", "CMul", "FFT4", "EvPol4", "SelectValueArity4"],
            ),
            (
                "Poseidon2",
                &["Poseidon2_16", "CustPoseidon2_16", "TreeSelector4", "CMul", "FFT4", "EvPol4", "SelectValueArity4"],
            ),
            // blake3 is arity 2, and it uses radix 4 for the FRI selection tree
            (
                "blake3",
                &["Blake3Node", "Blake3Compress", "TreeSelector4", "CMul", "FFT4", "EvPol4", "SelectValueArity2"],
            ),
            // the other radix must be just as usable by any family
            ("blake3", &["Blake3Node", "TreeSelector8", "CMul"]),
            ("Poseidon2", &["Poseidon2_16", "TreeSelector8", "CMul"]),
        ];

        for (expected, gates) in plausible {
            let mut seen: Vec<&str> = Vec::new();
            for name in *gates {
                let (_, family) = lookup_gate(name).unwrap_or_else(|| panic!("unregistered gate {name}"));
                if let Some(f) = family {
                    if !seen.contains(&f) {
                        seen.push(f);
                    }
                }
            }
            assert_eq!(seen, vec![*expected], "gate set {gates:?} does not resolve to {expected} alone");
        }
    }

    #[test]
    fn every_family_has_at_least_one_gate() {
        for fam in FAMILIES {
            assert!(
                GATES.iter().any(|(_, _, f)| *f == Some(*fam)),
                "family {fam} is in FAMILIES but has no gate in GATES"
            );
        }
    }

    #[test]
    fn discriminating_gate_names_are_unique_per_family() {
        for (i, (name, role, fam)) in GATES.iter().enumerate() {
            if fam.is_none() {
                continue;
            }
            if !matches!(role, GateRole::PoseidonSponge | GateRole::PoseidonCompression | GateRole::TreeSelector) {
                continue;
            }
            for (other_name, _, other_fam) in &GATES[i + 1..] {
                if name == other_name {
                    panic!("gate {name:?} claimed by both {fam:?} and {other_fam:?}");
                }
            }
        }
    }

    #[test]
    fn rust_hash_type_tracks_arity_width() {
        // width = arity * 4; must match the committed generated verifier files.
        assert_eq!(rust_hash_type("Poseidon2", 4), "Poseidon2_16");
        assert_eq!(rust_hash_type("Poseidon2", 2), "Poseidon2_8");
        assert_eq!(rust_hash_type("Poseidon1", 4), "Poseidon1_16");
        assert_eq!(rust_hash_type("Poseidon1", 3), "Poseidon1_12");
        assert_eq!(rust_hash_type("Poseidon1", 2), "Poseidon1_8");
    }

    /// Every family needs a default, and the cheap-to-hash one grinds harder.
    #[test]
    fn every_family_has_grinding_bits_and_blake3_grinds_hardest() {
        for family in FAMILIES {
            assert!(default_grinding_bits(family) > 0, "{family} has no grinding default");
        }
        assert_eq!(default_grinding_bits("Poseidon1"), default_grinding_bits("Poseidon2"));
        assert!(default_grinding_bits("blake3") > default_grinding_bits("Poseidon1"));
    }

    /// The node hash absorbs every child in one permutation, so the width must be exactly
    /// `arity * DIGEST_SIZE` -- `merkletreeReduce` asserts the same thing in C++.
    #[test]
    fn the_sponge_width_fits_one_node_of_children() {
        for arity in [2, 3, 4] {
            assert_eq!(sponge_width(arity), arity * DIGEST_SIZE);
            assert_eq!(sponge_rate(arity), sponge_width(arity) - DIGEST_SIZE);
        }
        // The measured geometry: arity 4 is a width-16 sponge absorbing 12 per block.
        assert_eq!((sponge_width(4), sponge_rate(4)), (16, 12));
    }

    /// The transcript buffers one rate's worth before permuting and drains the full state.
    #[test]
    fn transcript_sizes_match_the_native_transcript() {
        assert_eq!((transcript_pending_size(4), transcript_out_size(4)), (12, 16));
        assert_eq!((transcript_pending_size(2), transcript_out_size(2)), (4, 8));
    }

    /// The BN128 wrap is poseidon-only; blake3 has no recursivef circom verifier.
    #[test]
    fn only_the_poseidon_families_can_be_wrapped_in_a_snark() {
        assert!(!super::supports_snark("blake3"));
        assert!(super::supports_snark("Poseidon1"));
        assert!(super::supports_snark("Poseidon2"));
        // A guard fails closed: a family this build does not know cannot be wrapped either.
        assert!(!super::supports_snark("Keccak"));
    }

    #[test]
    fn default_hash_id_is_registered() {
        assert!(is_known_family(DEFAULT_HASH_ID));
    }
}
