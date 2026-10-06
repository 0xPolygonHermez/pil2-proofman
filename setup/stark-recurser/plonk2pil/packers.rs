//! Per-family setup dispatch. The STARK recursion's families are over Goldilocks, and so are their
//! signatures; the final SNARK wrap's, [`BN128_WRAP_FAMILY`], is over BN128.

use anyhow::{bail, Result};
use proofman_common::hash_family::{is_known_family, lookup_gate, GateRole, BLAKE3_BN128_WRAP_FAMILY, BN128_WRAP_FAMILY};
use proofman_fields::{Bn128, Goldilocks};

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
        // blake3 uses the aggregator AIR for both -- a recursion air is only a carrier for plonk
        // rows plus the custom gates, and both circuits draw on the same gate set. What differs is
        // the GEOMETRY: the aggregator is pinned because recursive1 and recursive2 must be
        // identical, a compressor picks its own (N, LANES). See setups::blake3::compressor.
        "blake3" => setups::blake3::compressor::compressor_blake3(r1cs, opts),
        other => panic!("Unknown hash family: {other}"),
    }
}

/// The final SNARK wrap's AIR, over BN128: the one family over that field.
pub fn pack_wrap(r1cs: &R1csFile<Bn128>, opts: &PlonkOptions) -> Result<SetupResult<Bn128>> {
    match opts.hash_id.as_str() {
        BN128_WRAP_FAMILY => setups::poseidon_bn128::wrap::wrap(r1cs, opts),
        BLAKE3_BN128_WRAP_FAMILY => setups::blake3_bn128::wrap::wrap(r1cs, opts),
        family if is_known_family(family) => bail!(
            "plonk2pil: the {family} family is over Goldilocks, and the r1cs is over BN128: the \
             {BN128_WRAP_FAMILY} family sets it up"
        ),
        family => {
            bail!("plonk2pil: unknown hash family {family:?}; an r1cs over BN128 is the {BN128_WRAP_FAMILY} family's")
        }
    }
}

/// Refuses a Goldilocks r1cs with a gate of the BN128 wraps, `PoseidonT` or `Num2Bytes`. The STARK
/// families place the gates they know and nothing else, so one would be left unconstrained; before
/// it had a role, it was an unknown gate, which they refuse.
pub fn refuse_bn128_gates(r1cs: &R1csFile<Goldilocks>) -> Result<()> {
    let of_the_wrap = |name: &str| {
        lookup_gate(name)
            .is_some_and(|(role, family)| family == Some(BN128_WRAP_FAMILY) || role == GateRole::RangeCheck)
    };
    if let Some(gate) = r1cs.custom_gates.iter().find(|g| of_the_wrap(&g.template_name)) {
        bail!(
            "plonk2pil: the r1cs is over Goldilocks and uses {}, a gate of the {BN128_WRAP_FAMILY} family, which is \
             over BN128",
            gate.template_name
        );
    }
    Ok(())
}
