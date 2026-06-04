//! Per-family setup dispatch.

use super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use super::setups;

pub fn pack_aggregation(r1cs: &R1csFile, opts: &PlonkOptions) -> SetupResult {
    match opts.hash_id.as_str() {
        "Poseidon1" => setups::poseidon1::aggregation::aggregation_compressor(r1cs, opts),
        "Poseidon2" => setups::poseidon2::aggregation::aggregation_compressor(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}

pub fn pack_compressor(r1cs: &R1csFile, opts: &PlonkOptions) -> SetupResult {
    match opts.hash_id.as_str() {
        "Poseidon1" => setups::poseidon1::compressor::compressor(r1cs, opts),
        "Poseidon2" => setups::poseidon2::compressor::compressor(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}
