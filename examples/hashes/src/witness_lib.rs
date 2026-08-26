use proofman_common::{PackedInfo, ProofmanResult};
use proofman::register_std;
use proofman_witness::{witness_library, witness_packed_info, WitnessLibrary, WitnessManager};
use pil2_std_lib::Std;
use proofman_fields::PrimeField64;
use proofman_fields::Goldilocks;

use crate::pil_helpers::PACKED_INFO;

use crate::blake2b::Blake2bAir;
use crate::blake3::Blake3Air;
use crate::sha2::Sha2Air;

witness_library!(WitnessLib, Goldilocks);

// All three components write through their `*TraceRowOps` trait, so the library can declare the
// packing pil-helpers generated for it, and `prove-air --packed` can size its buffers before this
// library is loaded as a WitnessLibrary.
witness_packed_info!(PACKED_INFO
    .iter()
    .map(|(airgroup_id, air_id, info)| (
        (*airgroup_id, *air_id),
        PackedInfo::new(info.is_packed, info.num_packed_words, info.unpack_info.to_vec())
    ))
    .collect());

impl<F: PrimeField64> WitnessLibrary<F> for WitnessLib {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()> {
        let std_lib = Std::new(wcm.get_pctx(), wcm.get_sctx(), true)?;
        register_std(wcm, &std_lib);

        let sha2_air = Sha2Air::new::<F>();
        wcm.register_component(sha2_air);

        let blake2b_air = Blake2bAir::new::<F>();
        wcm.register_component(blake2b_air);

        let blake3_air = Blake3Air::new::<F>();
        wcm.register_component(blake3_air);

        Ok(())
    }
}
