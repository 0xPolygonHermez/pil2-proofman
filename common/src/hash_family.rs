//! Hash-family registry.

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

pub const FAMILIES: &[&str] = &["Poseidon1", "Poseidon2"];
pub const DEFAULT_HASH_ID: &str = "Poseidon1";

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
    //TODO: "blake3" is a GPU prover-only basic-proof family (no recursion/circom),
    // so it is accepted here but intentionally kept out of FAMILIES (which the
    // recursive-verifier / circom codegen iterates).
    FAMILIES.contains(&id) || id == "blake3"
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

    #[test]
    fn default_hash_id_is_registered() {
        assert!(is_known_family(DEFAULT_HASH_ID));
    }
}
