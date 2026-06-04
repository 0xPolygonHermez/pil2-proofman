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

pub fn verifier(hash_id: &str) -> &'static dyn Verifier {
    match hash_id {
        "Poseidon1" => &Poseidon1Verifier,
        "Poseidon2" => &Poseidon2Verifier,
        other => panic!("Unknown hash family: {other:?}"),
    }
}
