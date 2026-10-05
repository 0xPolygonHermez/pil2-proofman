//! The witness of the pilfflonk wrap: the stage-1 columns of the AIR plonk2pil makes of the final
//! SNARK circuit, from a zkin of the recursivef, as the original pil-fflonk computed them in its
//! `exec` mode (`pfProver true … .exec circuit.dat zkin`):
//! 1. the circuit's circom witness calculator, the `final.so` and `final.dat` setup-snark builds
//!    (`proofman_common::final_witness`), computes the circuit's witness over BN128's `Fr` from the
//!    zkin;
//! 2. the circuit's BN128 `.exec`, which plonk2pil writes, gathers the stage-1 columns and the
//!    publics out of it, with the STARK's `getCommitedPols` semantics
//!    ([`ExecFile::committed_pols`](proofman_common::exec_format::ExecFile::committed_pols));
//! 3. if the AIR has range checks (circom's `Num2Bytes`), the multiplicity of their table,
//!    `RANGE_MUL`, is counted from the chunk cells of the rows the exec's range-check bands name
//!    ([`RANGE_CHECK_BAND_KIND`](proofman_common::exec_format::RANGE_CHECK_BAND_KIND)), into the
//!    stage-1 column its band section's aux word names.
//!
//! pilfflonk proves the AIR as any other, and knows nothing of circom or of the exec: this crate is
//! the wrap's, between the three. It is used in two ways:
//! - **in the wrap's process**, through [`WrapWitness`]: [`WrapWitness::load`] with the
//!   [`WrapArtifacts`] of the key, then [`WrapWitness::witness`] for a zkin file, or
//!   [`WrapWitness::witness_from_json`] for the recursivef proof in memory, as the PLONK and FFLONK
//!   wraps hand it to their final circuit; [`witness_from_circom`] if the circuit's witness is at
//!   hand;
//! - **as a pilfflonk witness library** (pilfflonk/docs/README.md#witness), the cdylib this crate
//!   builds, `libpilfflonk_wrap_witness.so`: `pilfflonk prove -w libpilfflonk_wrap_witness.so -i
//!   inputs.json`, where `inputs.json` is a [`WrapInputs`] that names the zkin and the circuit's
//!   files. The library's Rust ABI ties it to the build of the `proofman-cli` that loads it, so it
//!   lives in that build, not beside the circuit's files in `provingKeySnark/`, and `-i` names them.
//!
//! What it cannot compute is an error, not a panic: a file missing, an exec that is not over BN128,
//! has gate bands that are not the wrap's range checks or does not fit the circuit's witness (one of
//! another compile of the circuit, whose wire count is not the one the exec records) or the AIR, a
//! chunk of a range check that is not below 2^16, a key whose shape is not one AIR with no
//! air values or proof values, a zkin that is not a JSON object, and a zkin the calculator fails
//! on. A key of the zkin that is not an input of the circuit is the exception: circom's
//! calculator stops the process on it, as it does in the PLONK and FFLONK wraps.

mod artifacts;
mod error;
mod witness;
mod zkin;

use std::path::Path;

use proofman_pilfflonk::{
    pilfflonk_witness_library, PilfflonkError, PilfflonkResult, PilfflonkWitnessLibrary, Witness, WitnessShape,
};

pub use artifacts::{WrapArtifacts, WrapInputs};
pub use error::{WrapWitnessError, WrapWitnessResult};
pub use witness::{witness_from_circom, WrapWitness};

pilfflonk_witness_library!(WrapWitnessLibrary);

/// The cdylib's witness: from the [`WrapInputs`] of `-i`, which it requires, the witness of the key
/// of `shape`.
impl PilfflonkWitnessLibrary for WrapWitnessLibrary {
    fn witness(&mut self, shape: &WitnessShape, public_inputs: Option<&Path>) -> PilfflonkResult<Witness> {
        let Some(path) = public_inputs else {
            return Err(PilfflonkError::InvalidFormat(
                "the wrap's witness library needs -i (--public-inputs): the JSON that names the zkin and the final \
                 circuit's witnessCalculator, dat and exec"
                    .to_string(),
            ));
        };
        let inputs = WrapInputs::read(path)?;
        Ok(WrapWitness::load(&inputs.artifacts)?.witness(shape, &inputs.zkin)?)
    }
}
