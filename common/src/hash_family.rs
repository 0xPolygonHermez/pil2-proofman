//! Hash-family registry.

use proofman_starks_lib_c::GOLDILOCKS_POSEIDON_MERKLE_TREE_ARITY;
use serde::{Deserialize, Serialize};
use HashFamily::{Blake3, Poseidon1, Poseidon2};

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
    /// circom's `PoseidonT(t)`: the original Poseidon over BN128, with every round's state as an
    /// output of the gate. The hash gate of the Poseidon1 BN128 wrap; of no STARK family.
    PoseidonT,
    /// circom's `Num2Bytes(nBits)`: its input `in` as `⌈nBits/16⌉` chunks of 16 bits, `out`, least
    /// significant first, so that `in < 2^(16·⌈nBits/16⌉)`. The range check of the final SNARK
    /// wraps, of any hash, which look every chunk up in a table of `[0, 2^16)`.
    RangeCheck,
}

/// Typed STARK hash family, all over Goldilocks. Parsing is case-sensitive (`Poseidon1`/`Poseidon2`/`blake3`);
/// serde reads and writes the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "&'static str")]
pub enum HashFamily {
    Poseidon1,
    Poseidon2,
    Blake3,
}

impl HashFamily {
    pub const ALL: [HashFamily; 3] = [Poseidon1, Poseidon2, Blake3];

    /// What every `--hash` default, every `Default` carrying a family and a `globalInfo.json` without `hash` take.
    pub const DEFAULT: HashFamily = Blake3;

    pub fn is_blake3(self) -> bool {
        self == Blake3
    }

    /// The id `set_hash_family_c` takes (C++ `HashFamily` in hash_family.hpp).
    pub const fn ffi_id(self) -> u8 {
        match self {
            Poseidon1 => 1,
            Poseidon2 => 2,
            Blake3 => 3,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Poseidon1 => "Poseidon1",
            Poseidon2 => "Poseidon2",
            Blake3 => "blake3",
        }
    }
}

impl std::fmt::Display for HashFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for HashFamily {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL.into_iter().find(|f| f.as_str() == s).ok_or_else(|| {
            let valid: Vec<_> = Self::ALL.iter().map(|f| f.as_str()).collect();
            format!("unknown hash family '{s}'; valid (case-sensitive): {}", valid.join(", "))
        })
    }
}

impl TryFrom<String> for HashFamily {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<HashFamily> for &'static str {
    fn from(f: HashFamily) -> Self {
        f.as_str()
    }
}

/// Merkle-tree arity of `family`'s commitment trees over Goldilocks digests
pub fn merkle_tree_arity(family: HashFamily) -> u64 {
    match family {
        Poseidon1 | Poseidon2 => GOLDILOCKS_POSEIDON_MERKLE_TREE_ARITY,
        Blake3 => 2,
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
pub fn recursive_bits_threshold(family: HashFamily) -> usize {
    match family {
        Blake3 => 19,
        Poseidon1 | Poseidon2 => 17,
    }
}

/// Whether the family's FRI schedule is solved for rather than folded at a fixed rate.
///
/// Worth it exactly where Merkle paths are expensive, which is where the tree is binary: a path
/// twice as long makes it cheaper to fold hard and keep fewer layers, and no uniform folding factor
/// expresses that. At arity 4 a fold of 3 and a fold of 4 tie, so poseidon gains nothing and keeps
/// its fixed-rate schedule. See `stark_struct::optimal_fri_steps`.
pub fn uses_optimal_fri_schedule(family: HashFamily) -> bool {
    merkle_tree_arity(family) == 2
}

/// Field elements one compression of `family` absorbs: a BLAKE3 block is 64 bytes, so 8 Goldilocks
/// elements. Not a rate -- these are compression functions that take a whole block per call and
/// carry the chaining value in the state, so a leaf of `w` elements costs `ceil(w / this)`.
pub fn compression_block_elements(family: HashFamily) -> usize {
    match family {
        Blake3 => 8,
        Poseidon1 | Poseidon2 => 8,
    }
}

/// The FRI layer the recursion stops folding at, as the log2 of its committed DOMAIN.
///
/// Everything below the terminal is sent in the clear, and the verifier evaluates it at every query
/// -- which is `2^terminal` extension multiplications a query, arithmetic rather than hashing. Above
/// 7 that arithmetic outgrows the room the BLAKE3 block interiors leave for it.
pub fn fri_terminal_degree(family: HashFamily) -> usize {
    match family {
        Blake3 => 7,
        Poseidon1 | Poseidon2 => 5,
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
pub fn recursive_last_level_verification(family: HashFamily) -> Option<usize> {
    match family {
        Blake3 => Some(5),
        Poseidon1 | Poseidon2 => None,
    }
}

/// Proof-of-work bits the FINAL stage grinds for.
///
/// Its own function rather than `recursive_grinding_bits`, because the two differ: poseidon's recursion
/// grinds 21 and its final stage 22, blake3's 24 and 27. blake3 grinds 27 here: at blowup 2^4 a query is worth ~2 bits, and the exact solver gives
/// 52 queries at 26, 51 at 27 and 28. The compressed final that verifies this proof sits at ~99% of
/// its hash blocks, so that one query is the difference between 2^18 and 2^19 rows there.
pub fn final_grinding_bits(family: HashFamily) -> usize {
    match family {
        Blake3 => 27,
        Poseidon1 | Poseidon2 => 22,
    }
}

/// Rows the `vadcop_final` air is pinned to, as log2, or `None` to let it size itself.
///
/// Pinned so the airgroup count does not move blake3's final air, which the compressed final verifies
/// at ~99% of its hash blocks. Poseidon sizes itself.
pub fn final_n_bits(family: HashFamily) -> Option<usize> {
    match family {
        Blake3 => Some(18),
        Poseidon1 | Poseidon2 => None,
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
/// blake3 takes 4 at 2^18 rows: the same extended domain as 2^19 at blowup 2, half the queries,
/// and degree-8 constraints with no intermediate polynomials. Its proof is what the compressed final
/// verifies, so fewer queries is what lets that layer fit 2^18 at two lanes.
///
/// Poseidon keeps 4.
pub fn final_blowup_factor(family: HashFamily) -> usize {
    match family {
        Blake3 => 4,
        Poseidon1 | Poseidon2 => 4,
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

/// Blowup the `vadcop_final_compressed` air is built at, as log2.
///
/// blake3's compressed proof is the one that ships, so it trades prover memory (~2x per step) for
/// proof size; it is proven once per block, so the memory is not held next to anything else.
/// Poseidon keeps the final air's blowup.
pub fn compressed_final_blowup_factor(family: HashFamily) -> usize {
    match family {
        Blake3 => 5,
        Poseidon1 | Poseidon2 => final_blowup_factor(family),
    }
}

/// Proof-of-work bits the `vadcop_final_compressed` stage grinds for.
///
/// blake3: at blowup 2^5 a query is worth ~2.5 bits, so the count only moves every few bits of
/// grinding: the exact solver gives 42 queries at 26, 41 at 27 and 28. 27 is the cheapest grind for 41.
pub fn compressed_final_grinding_bits(family: HashFamily) -> usize {
    match family {
        Blake3 => 27,
        Poseidon1 | Poseidon2 => final_grinding_bits(family),
    }
}

/// Whether proofman has a multiproof codec for `family` at this tree and transcript arity
/// (`proofman::compressed_proof`, which dispatches on exactly this).
pub fn has_multiproof_codec(family: HashFamily, merkle_arity: u64, transcript_arity: u64) -> bool {
    matches!((family, merkle_arity, transcript_arity), (Blake3, 2, _) | (Poseidon1 | Poseidon2, 2, 4))
}

/// The FRI terminal domain of the `vadcop_final_compressed` proof where no folds are pinned.
pub fn compressed_final_fri_terminal_degree(family: HashFamily) -> usize {
    match family {
        Blake3 => 8,
        Poseidon1 | Poseidon2 => 10,
    }
}

/// FRI folds of the `vadcop_final_compressed` proof, picked for multiproof size at the (air size,
/// blowup) given; under other parameters, or `None`, the solver's schedule. blake3 at 2^18: 272,927 B mean against
/// 277,812 for [4,4,4,3].
pub fn compressed_final_fri_folds(family: HashFamily) -> Option<((usize, usize), Vec<usize>)> {
    match family {
        Blake3 => Some(((18, 5), vec![4, 3, 3, 3, 3])),
        Poseidon1 | Poseidon2 => None,
    }
}

/// ((air size, blowup) measured at, folds, per-round grinding).
pub type StirSchedule = ((usize, usize), Vec<usize>, Vec<usize>);

/// Folds and per-round grinding of the `vadcop_final_compressed` proof when it is built with STIR
/// (`setup-compressed-final --low-degree-test STIR`), or `None` for the generic schedule.
///
/// blake3, measured on fibonacci over nine schedules: [4,3,3] (degree 2^18 to 2^8) is the smallest,
/// against 277,015 B for the generic [4,4,4,3] and 295,511 for FRI. Round 2 keeps 26 queries from 26
/// to 29 bits. Round 3 grinds 23: 22 queries, 264,935 B (27 bits would keep it at 21, 264,359 B).
/// The first entry is the (air size, blowup) it was measured at; under others the generic one is used.
pub fn compressed_final_stir_schedule(family: HashFamily) -> Option<StirSchedule> {
    match family {
        Blake3 => Some(((18, 5), vec![4, 3, 3], vec![27, 26, 23])),
        // At 2^15 with the family's 22 bits a round: the smallest of 1–4 rounds of 1–7 bits, flat or multiproof.
        Poseidon1 | Poseidon2 => Some(((15, 4), vec![3, 4], vec![22, 22])),
    }
}

/// blake3 lanes of the `vadcop_final_compressed` air, or `None` for the air's default.
///
/// It verifies one proof rather than the final air's two, so half the lanes hold it at 2^18.
pub fn compressed_final_blake3_lanes(family: HashFamily) -> Option<usize> {
    match family {
        Blake3 => Some(2),
        Poseidon1 | Poseidon2 => None,
    }
}

/// Proofs one recursive2 circuit aggregates, by family.
///
/// Every extra proof aggregated multiplies the verification work recursive2 does, and for blake3
/// that work is almost entirely hashing: on a fibonacci recursion, recursive2's BLAKE3 gate count is
/// exactly `arity` times recursive1's, and it alone decides the air's size. Poseidon absorbs a third
/// proof cheaply -- arity-4 Merkle paths, and a permutation that is not the bulk of the circuit --
/// so it keeps 3. Callers may override; both values stay valid for both families.
pub fn default_aggregation_arity(family: HashFamily) -> usize {
    match family {
        Blake3 => 2,
        Poseidon1 | Poseidon2 => 3,
    }
}

/// Proof-of-work bits the query phase grinds for. Per family because the prover searches `2^bits`,
/// so what one affords tracks how fast it hashes; the bits come off the query count.
pub fn default_grinding_bits(family: HashFamily) -> usize {
    match family {
        Poseidon1 | Poseidon2 => 16,
        Blake3 => 24,
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
/// Poseidon's 21 buys 72 queries: at the recursion's rate of 2^-3 a query is worth ~1.48 bits, so
/// the solver takes `ceil((128 - 21) / 1.48)`. It sits on a step -- 22 buys nothing further, the
/// next query comes off at 23. blake3 takes its own family default of 24.
///
/// Changing this changes the proof shape, and with it every proving key already built.
pub fn recursive_grinding_bits(family: HashFamily) -> usize {
    match family {
        Poseidon1 | Poseidon2 => 21,
        Blake3 => 24,
    }
}

/// Whether the family's final SNARK wraps a BN128 recursivef: the poseidon families', which any final
/// SNARK (PLONK, FFLONK, pilfflonk) takes. blake3's recursivef is a Goldilocks STARK of the key
/// (setup-recursivef), whose verifier only the pilfflonk wrap has.
pub fn has_bn128_recursivef(family: HashFamily) -> bool {
    match family {
        Poseidon1 | Poseidon2 => true,
        Blake3 => false,
    }
}

/// Fiat-Shamir transcript arity: the transcript squeezes through the same
/// sponge/compression as the trees, so it shares the Merkle arity.
pub fn transcript_arity(family: HashFamily) -> u64 {
    merkle_tree_arity(family)
}

/// Elements in one digest, i.e. the sponge capacity (`nFieldElements` in the C++ trees).
pub const DIGEST_SIZE: u64 = 4;

// The four helpers below describe a SPONGE and take an arity, not a family, so they answer for
// blake3 without being about it: at arity 2 they give a rate of 4 and a pending size of 4, while a
// BLAKE3 block absorbs 8. Anything on the blake3 path wants `compression_block_elements` instead --
// see `verifier_hashes.rs`, the one caller, which special-cases it.

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

/// True when the family's kernels support exactly one tree geometry
pub fn has_forced_tree_geometry(family: HashFamily) -> bool {
    family == Blake3
}

// (gate template name, role, owning family). `None` for family-agnostic gates.
const GATES: &[(&str, GateRole, Option<HashFamily>)] = &[
    ("Poseidon1_16", GateRole::PoseidonSponge, Some(Poseidon1)),
    ("CustPoseidon1_16", GateRole::PoseidonCompression, Some(Poseidon1)),
    // TreeSelector's number is the gate's RADIX, not a hash arity: TreeSelector(nLevels, eSize)
    // selects one of 2^nLevels values by nLevels BINARY key bits, and TreeSelector4/8 just collapse
    // two or three levels of that binary tree into one gate. eSize is 3 -- extension elements, for
    // FRI -- so it has nothing to do with merkle_tree_arity. Which radix an air uses is a layout
    // choice (17 cells in one row vs 30 in two), so these are family-agnostic; the family is
    // identified by its hash gates, and an r1cs carrying both radices still trips the duplicate-role
    // check in to_plonk. Tagging them by family would make a blake3 r1cs that uses either radix look
    // like it mixed two families.
    ("TreeSelector8", GateRole::TreeSelector, None),
    ("Poseidon2_16", GateRole::PoseidonSponge, Some(Poseidon2)),
    ("CustPoseidon2_16", GateRole::PoseidonCompression, Some(Poseidon2)),
    ("TreeSelector4", GateRole::TreeSelector, None),
    ("CMul", GateRole::CMul, None),
    ("FFT4", GateRole::Fft4, None),
    ("EvPol4", GateRole::EvPol4, None),
    ("SelectValueArity4", GateRole::SelectValArity4, None),
    ("SelectValueArity2", GateRole::SelectValArity2, None),
    ("Blake3Node", GateRole::Blake3Node, Some(Blake3)),
    ("Blake3Compress", GateRole::Blake3Compress, Some(Blake3)),
    // `PoseidonT(t)`: circom mints one gate id per width t, as for FFT4's parameters. The BN128
    // wraps' gates are of no STARK family.
    ("PoseidonT", GateRole::PoseidonT, None),
    // `Num2Bytes(nBits)`: one gate id per nBits, as for PoseidonT. Of no family: the wraps of every
    // hash range-check with it.
    ("Num2Bytes", GateRole::RangeCheck, None),
];

pub fn lookup_gate(name: &str) -> Option<(GateRole, Option<HashFamily>)> {
    GATES.iter().find(|(n, _, _)| *n == name).map(|(_, r, f)| (*r, *f))
}

pub fn rust_hash_type(family: HashFamily, arity: u64) -> &'static str {
    match (family, arity * 4) {
        (Poseidon1, 8) => "Poseidon1_8",
        (Poseidon1, 12) => "Poseidon1_12",
        (Poseidon1, 16) => "Poseidon1_16",
        (Poseidon2, 4) => "Poseidon2_4",
        (Poseidon2, 8) => "Poseidon2_8",
        (Poseidon2, 12) => "Poseidon2_12",
        (Poseidon2, 16) => "Poseidon2_16",
        // BLAKE3 has one width and it is the block, not a sponge width: eight field elements is a
        // 64-byte block, and at arity 2 a node is exactly two four-element digests. The arity is
        // still checked, because any other would mean a node the block cannot hold.
        (Blake3, 8) => "Blake3_8",
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
pub fn rust_transcript_type(family: HashFamily, arity: u64) -> String {
    match family {
        Blake3 => "Blake3Transcript<Goldilocks>".to_string(),
        Poseidon1 | Poseidon2 => format!("Transcript<Goldilocks, {}>", rust_hash_type(family, arity)),
    }
}

/// Type names `rust_transcript_type` needs imported, so the generator does not have to parse it.
pub fn rust_transcript_imports(family: HashFamily, arity: u64) -> Vec<&'static str> {
    match family {
        Blake3 => vec!["Blake3Transcript"],
        Poseidon1 | Poseidon2 => vec!["Transcript", rust_hash_type(family, arity)],
    }
}

pub fn rust_grinding_type(family: HashFamily) -> &'static str {
    match family {
        Poseidon1 => "Poseidon1_8",
        Poseidon2 => "Poseidon2_8",
        Blake3 => "Blake3_8",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-family knobs every key is built with, pinned: moving one invalidates every key of the family.
    #[test]
    fn per_family_knobs_are_pinned() {
        // (family, ffi id, grinding basic/recursive/final, compressed blowup/grinding, recursive bits, final bits, bn128)
        for (f, ffi, (basic, rec, fin), compressed, rec_bits, final_bits, bn128) in [
            (Poseidon1, 1, (16, 21, 22), (4, 22), 17, None, true),
            (Poseidon2, 2, (16, 21, 22), (4, 22), 17, None, true),
            (Blake3, 3, (24, 24, 27), (5, 27), 19, Some(18), false),
        ] {
            assert_eq!(f.ffi_id(), ffi, "{f}");
            assert_eq!(
                (default_grinding_bits(f), recursive_grinding_bits(f), final_grinding_bits(f)),
                (basic, rec, fin)
            );
            assert!(basic <= rec && rec <= fin, "{f}");
            assert_eq!((compressed_final_blowup_factor(f), compressed_final_grinding_bits(f)), compressed, "{f}");
            assert_eq!(
                (recursive_bits_threshold(f), final_n_bits(f), has_bn128_recursivef(f)),
                (rec_bits, final_bits, bn128)
            );
        }
    }

    /// The final blowup must carry degree 5, the floor of the final air's own constraints.
    #[test]
    fn final_blowup_carries_the_degree() {
        assert_eq!((final_blowup_factor(Blake3), final_blowup_factor(Poseidon2)), (4, 4));
        for f in HashFamily::ALL {
            let b = final_blowup_factor(f);
            assert!(max_constraint_degree_for_blowup(b) >= 5, "{f} at blowup {b} cannot carry degree 5");
        }
    }

    #[test]
    fn max_constraint_degree_follows_the_blowup() {
        let degrees = [1, 2, 3, 4, 10].map(max_constraint_degree_for_blowup);
        assert_eq!(degrees, [3, 5, 8, 8, 8], "capped at 8");
    }

    #[test]
    fn default_aggregation_arity_is_valid_and_pinned() {
        for (f, arity) in [(Blake3, 2), (Poseidon1, 3), (Poseidon2, 3)] {
            assert_eq!(default_aggregation_arity(f), arity, "{f}");
            assert!(crate::global_info::is_valid_aggregation_arity(arity), "{f}");
        }
    }

    /// Hash gates each own one family and every other gate is family-agnostic, so a gate set can only
    /// mix families by mixing hash gates (`to_plonk` panics on that).
    #[test]
    fn gates_are_owned_by_their_hash_family_or_by_none() {
        for (name, family) in [
            ("Poseidon1_16", Poseidon1),
            ("CustPoseidon1_16", Poseidon1),
            ("Poseidon2_16", Poseidon2),
            ("CustPoseidon2_16", Poseidon2),
            ("Blake3Node", Blake3),
            ("Blake3Compress", Blake3),
        ] {
            assert_eq!(lookup_gate(name).map(|(_, f)| f), Some(Some(family)), "{name}");
        }
        // TreeSelector4/8 are radices, not arities: tagging them made blake3 r1cs look mixed
        for name in [
            "TreeSelector4",
            "TreeSelector8",
            "CMul",
            "FFT4",
            "EvPol4",
            "SelectValueArity2",
            "SelectValueArity4",
            "PoseidonT",
            "Num2Bytes",
        ] {
            assert_eq!(lookup_gate(name).map(|(_, f)| f), Some(None), "{name}");
        }
        assert_eq!(lookup_gate("Nope"), None);
    }

    /// `lookup_gate` takes the first entry of a name, so a duplicate would hide the other.
    #[test]
    fn gate_names_are_unique() {
        let names: std::collections::HashSet<_> = GATES.iter().map(|(name, ..)| name).collect();
        assert_eq!(names.len(), GATES.len());
    }

    #[test]
    fn hash_geometry() {
        // width = arity * 4
        for (f, arity, name) in [
            (Poseidon2, 4, "Poseidon2_16"),
            (Poseidon2, 2, "Poseidon2_8"),
            (Poseidon1, 4, "Poseidon1_16"),
            (Poseidon1, 3, "Poseidon1_12"),
            (Poseidon1, 2, "Poseidon1_8"),
        ] {
            assert_eq!(rust_hash_type(f, arity), name);
        }
        assert_eq!((sponge_width(4), sponge_rate(4)), (16, 12));
        assert_eq!((transcript_pending_size(4), transcript_out_size(4)), (12, 16));
        assert_eq!((transcript_pending_size(2), transcript_out_size(2)), (4, 8));
    }

    /// Names parse exactly (unknown or differently-cased ones are refused) and serde uses the same names.
    #[test]
    fn parses_exact_names_only() {
        for f in HashFamily::ALL {
            let json = format!("\"{}\"", f.as_str());
            assert_eq!(f.as_str().parse::<HashFamily>().unwrap(), f);
            assert_eq!(f.is_blake3(), f == Blake3);
            assert_eq!(serde_json::to_string(&f).unwrap(), json);
            assert_eq!(serde_json::from_str::<HashFamily>(&json).unwrap(), f);
        }
        for bad in ["Blake3", "poseidon1", "POSEIDON2", "", "sha2", "Keccak"] {
            let e = bad.parse::<HashFamily>().unwrap_err();
            assert!(e.contains("Poseidon1") && e.contains("blake3"), "error must list valid names: {e}");
            assert!(serde_json::from_str::<HashFamily>(&format!("\"{bad}\"")).is_err(), "{bad}");
        }
    }
}
