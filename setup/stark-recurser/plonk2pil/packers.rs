//! Per-family setup dispatch. Every family is over Goldilocks, and so are its signatures.

use proofman_fields::Goldilocks;

use super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use super::setups;

pub fn pack_aggregation(r1cs: &R1csFile<Goldilocks>, opts: &PlonkOptions) -> SetupResult<Goldilocks> {
    match opts.hash_id.as_str() {
        "Poseidon1" => setups::poseidon1::aggregation::aggregation_compressor(r1cs, opts),
        "Poseidon2" => setups::poseidon2::aggregation::aggregation_compressor(r1cs, opts),
        "blake3" => setups::blake3::aggregation::aggregation_blake3(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}

pub fn pack_compressor(r1cs: &R1csFile<Goldilocks>, opts: &PlonkOptions) -> SetupResult<Goldilocks> {
    match opts.hash_id.as_str() {
        "Poseidon1" => setups::poseidon1::compressor::compressor(r1cs, opts),
        "Poseidon2" => setups::poseidon2::compressor::compressor(r1cs, opts),
        // blake3's compressor is the aggregator air at its own (N, LANES).
        "blake3" => setups::blake3::aggregation::compressor_blake3(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}
