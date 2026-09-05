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
    SelectVal1,
}

pub const FAMILIES: &[&str] = &["Poseidon1", "Poseidon2", "Blake3"];
pub const DEFAULT_HASH_ID: &str = "Poseidon1";

/// Merkle-tree arity of `family`'s commitment trees over Goldilocks digests
pub fn merkle_tree_arity(family: &str) -> u64 {
    match family {
        "Poseidon1" | "Poseidon2" => GOLDILOCKS_POSEIDON_MERKLE_TREE_ARITY,
        "Blake3" => 2,
        fam => panic!("Unknown hash family: {fam}"),
    }
}

/// Proof-of-work bits the query phase grinds for. Per family because the prover searches `2^bits`,
/// so what one affords tracks how fast it hashes; the bits come off the query count.
pub fn default_grinding_bits(family: &str) -> usize {
    match family {
        "Poseidon1" | "Poseidon2" => 16,
        "Blake3" => 24,
        fam => panic!("Unknown hash family: {fam}"),
    }
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
    matches!(family, "Poseidon1" | "Blake3")
}

/// True when the family's kernels support exactly one tree geometry
pub fn has_forced_tree_geometry(family: &str) -> bool {
    family == "Blake3"
}

// (gate template name, role, owning family). `None` for family-agnostic gates.
const GATES: &[(&str, GateRole, Option<&str>)] = &[
    ("Poseidon1_16", GateRole::PoseidonSponge, Some("Poseidon1")),
    ("CustPoseidon1_16", GateRole::PoseidonCompression, Some("Poseidon1")),
    ("TreeSelector8", GateRole::TreeSelector, Some("Poseidon1")),
    ("Poseidon2_16", GateRole::PoseidonSponge, Some("Poseidon2")),
    ("CustPoseidon2_16", GateRole::PoseidonCompression, Some("Poseidon2")),
    ("TreeSelector4", GateRole::TreeSelector, Some("Poseidon2")),
    ("CMul", GateRole::CMul, None),
    ("FFT4", GateRole::Fft4, None),
    ("EvPol4", GateRole::EvPol4, None),
    ("SelectValue1", GateRole::SelectVal1, None),
];

pub fn lookup_gate(name: &str) -> Option<(GateRole, Option<&'static str>)> {
    GATES.iter().find(|(n, _, _)| *n == name).map(|(_, r, f)| (*r, *f))
}

pub fn is_known_family(id: &str) -> bool {
    FAMILIES.contains(&id)
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
        (fam, width) => panic!("Unsupported hash type: {fam}_{width} (arity {arity})"),
    }
}

pub fn rust_grinding_type(family: &str) -> &'static str {
    match family {
        "Poseidon1" => "Poseidon1_8",
        "Poseidon2" => "Poseidon2_8",
        fam => panic!("Unsupported grinding hash family: {fam}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(default_grinding_bits("Blake3") > default_grinding_bits("Poseidon1"));
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

    #[test]
    fn default_hash_id_is_registered() {
        assert!(is_known_family(DEFAULT_HASH_ID));
    }
}
