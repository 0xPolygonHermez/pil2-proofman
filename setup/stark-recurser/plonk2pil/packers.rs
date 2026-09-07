//! Per-family setup dispatch.

use super::r1cs::types::{PlonkOptions, R1csFile, SetupResult};
use super::setups;

pub fn pack_aggregation(r1cs: &R1csFile, opts: &PlonkOptions) -> SetupResult {
    match opts.hash_id.as_str() {
        "Poseidon1" => setups::poseidon1::aggregation::aggregation_compressor(r1cs, opts),
        "Poseidon2" => setups::poseidon2::aggregation::aggregation_compressor(r1cs, opts),
        "blake3" => setups::blake3::aggregation::aggregation_blake3(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}

pub fn pack_compressor(r1cs: &R1csFile, opts: &PlonkOptions) -> SetupResult {
    match opts.hash_id.as_str() {
        "Poseidon1" => setups::poseidon1::compressor::compressor(r1cs, opts),
        "Poseidon2" => setups::poseidon2::compressor::compressor(r1cs, opts),
        // blake3 uses the aggregator AIR for both -- a recursion air is only a carrier for plonk
        // rows plus the custom gates, and both circuits draw on the same gate set. What differs is
        // the GEOMETRY: the aggregator is pinned because recursive1 and recursive2 must be
        // identical, a compressor picks its own (N, LANES). See setups::blake3::compressor.
        "blake3" => setups::blake3::compressor::compressor_blake3(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}
