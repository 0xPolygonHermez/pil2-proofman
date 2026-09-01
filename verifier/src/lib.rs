#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[macro_use]
mod log;

mod proof;
mod verifier;

pub mod poseidon1 {
    pub mod recursive2_verifier;
    pub mod vadcop_final;
    pub mod vadcop_final_compressed;
}

pub mod poseidon2 {
    pub mod recursive2_verifier;
    pub mod vadcop_final;
    pub mod vadcop_final_compressed;
}

/// No `vadcop_final_compressed`: the stage is off for blake3, where it measured a 2% smaller proof
/// for a whole extra recursion layer (see `hash_family::compressed_final_by_default`). Turning it
/// back on means generating that verifier too and replacing the three stubs in `Blake3Verifier`.
///
/// `recursive2_verifier` and `vadcop_final` are byte-identical only because both airs are
/// `blake3/aggregator.pil` at the pinned N = 2^19 / LANES = 4. Do not collapse them into a
/// re-export: unpin the size or change the aggregation arity and they diverge.
pub mod blake3 {
    pub mod recursive2_verifier;
    pub mod vadcop_final;
}

pub use proof::*;
pub use verifier::*;

pub trait Verifier: Sync {
    fn verify_recursive2(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool;
    fn verify_recursive2_u64(&self, proof: &[u64], vk: &[u64]) -> bool;
    fn verify_vadcop_final(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool;
    fn verify_vadcop_final_u64(&self, proof: &[u64], vk: &[u64]) -> bool;
    fn verify_vadcop_final_compressed(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool;
    fn verify_vadcop_final_compressed_u64(&self, proof: &[u64], vk: &[u64]) -> bool;
    fn expected_vadcop_final_proof_bytes(&self) -> usize;
    fn expected_vadcop_final_compressed_proof_bytes(&self) -> usize;
}

pub struct Poseidon1Verifier;
pub struct Poseidon2Verifier;
pub struct Blake3Verifier;

impl Verifier for Poseidon1Verifier {
    fn verify_recursive2(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        poseidon1::recursive2_verifier::verify(proof, vk)
    }
    fn verify_recursive2_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        poseidon1::recursive2_verifier::verify_u64(proof, vk)
    }
    fn verify_vadcop_final(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        poseidon1::vadcop_final::verify(proof, vk)
    }
    fn verify_vadcop_final_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        poseidon1::vadcop_final::verify_u64(proof, vk)
    }
    fn verify_vadcop_final_compressed(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        poseidon1::vadcop_final_compressed::verify(proof, vk)
    }
    fn verify_vadcop_final_compressed_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        poseidon1::vadcop_final_compressed::verify_u64(proof, vk)
    }
    fn expected_vadcop_final_proof_bytes(&self) -> usize {
        poseidon1::vadcop_final::expected_proof_bytes()
    }
    fn expected_vadcop_final_compressed_proof_bytes(&self) -> usize {
        poseidon1::vadcop_final_compressed::expected_proof_bytes()
    }
}

impl Verifier for Poseidon2Verifier {
    fn verify_recursive2(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        poseidon2::recursive2_verifier::verify(proof, vk)
    }
    fn verify_recursive2_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        poseidon2::recursive2_verifier::verify_u64(proof, vk)
    }
    fn verify_vadcop_final(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        poseidon2::vadcop_final::verify(proof, vk)
    }
    fn verify_vadcop_final_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        poseidon2::vadcop_final::verify_u64(proof, vk)
    }
    fn verify_vadcop_final_compressed(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        poseidon2::vadcop_final_compressed::verify(proof, vk)
    }
    fn verify_vadcop_final_compressed_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        poseidon2::vadcop_final_compressed::verify_u64(proof, vk)
    }
    fn expected_vadcop_final_proof_bytes(&self) -> usize {
        poseidon2::vadcop_final::expected_proof_bytes()
    }
    fn expected_vadcop_final_compressed_proof_bytes(&self) -> usize {
        poseidon2::vadcop_final_compressed::expected_proof_bytes()
    }
}

impl Verifier for Blake3Verifier {
    fn verify_recursive2(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        blake3::recursive2_verifier::verify(proof, vk)
    }
    fn verify_recursive2_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        blake3::recursive2_verifier::verify_u64(proof, vk)
    }
    fn verify_vadcop_final(&self, proof: &VadcopFinalProof, vk: &[u64]) -> bool {
        blake3::vadcop_final::verify(proof, vk)
    }
    fn verify_vadcop_final_u64(&self, proof: &[u64], vk: &[u64]) -> bool {
        blake3::vadcop_final::verify_u64(proof, vk)
    }
    fn expected_vadcop_final_proof_bytes(&self) -> usize {
        blake3::vadcop_final::expected_proof_bytes()
    }

    // The three below have nothing to verify: blake3 does not build the compressed final stage.
    //
    // They panic rather than answer. `false` would read as "this proof is invalid" and send the
    // caller to debug a proof instead of a configuration; `true` would accept a proof without
    // verifying it; and returning a size of 0 is the worst of the three, because a caller would
    // size a buffer to nothing and read past it. Calling these is a programming error -- asking to
    // verify a stage that was never generated -- so it is reported as one.
    fn verify_vadcop_final_compressed(&self, _proof: &VadcopFinalProof, _vk: &[u64]) -> bool {
        unimplemented!("{}", NO_COMPRESSED_FINAL)
    }
    fn verify_vadcop_final_compressed_u64(&self, _proof: &[u64], _vk: &[u64]) -> bool {
        unimplemented!("{}", NO_COMPRESSED_FINAL)
    }
    fn expected_vadcop_final_compressed_proof_bytes(&self) -> usize {
        unimplemented!("{}", NO_COMPRESSED_FINAL)
    }
}

/// Says both halves of it: what is missing, and how to get it.
const NO_COMPRESSED_FINAL: &str = "blake3 proving keys are built without the vadcop_final_compressed \
     stage, so there is no verifier for it. Verify the vadcop_final proof instead. To build the \
     stage anyway, run `proofman-setup setup --compressed-final true` (or add it to an existing key \
     with `proofman-setup setup-compressed-final`), then generate verifier/src/blake3/\
     vadcop_final_compressed.rs and replace this stub.";

pub fn verifier(hash_id: &str) -> &'static dyn Verifier {
    match hash_id {
        "Poseidon1" => &Poseidon1Verifier,
        "Poseidon2" => &Poseidon2Verifier,
        "blake3" => &Blake3Verifier,
        // The hash impls exist; verifier/src/sha256/ does not, because only a recursive setup
        // emits it and sha256 has no PIL gates yet.
        "sha256" => panic!("sha256 has no generated native verifier yet"),
        other => panic!("Unknown hash family: {other:?}"),
    }
}

#[cfg(test)]
mod factory_tests {
    /// Every family the setup can produce a proving key for needs a verifier here, or proving
    /// succeeds and verification panics on an unknown family -- at the very end of a long run.
    #[test]
    fn the_factory_covers_every_known_family() {
        for family in ["Poseidon1", "Poseidon2", "blake3"] {
            let v = super::verifier(family);
            // Touching a method proves the vtable resolves, not just that the match arm exists.
            let _ = v.expected_vadcop_final_proof_bytes();
        }
    }

    /// blake3 has no compressed-final verifier, and asking for one must say so rather than answer.
    #[test]
    #[should_panic(expected = "built without the vadcop_final_compressed")]
    fn blake3_refuses_the_compressed_final_it_never_built() {
        super::verifier("blake3").expected_vadcop_final_compressed_proof_bytes();
    }
}
