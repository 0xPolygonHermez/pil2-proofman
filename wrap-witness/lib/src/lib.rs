//! The wrap's witness as a pilfflonk witness library (pilfflonk/docs/README.md#witness),
//! `libpilfflonk_wrap_witness_lib.so`: `pilfflonk prove -w libpilfflonk_wrap_witness_lib.so -i
//! inputs.json`, where `inputs.json` is a [`WrapInputs`] that names the zkin and the circuit's files.
//! The library's Rust ABI ties it to the build of the `proofman-cli` that loads it, so it lives in
//! that build, not beside the circuit's files in `provingKeySnark/`, and `-i` names them.

use std::path::Path;

use pilfflonk_wrap_witness::{WrapInputs, WrapWitness};
use proofman_pilfflonk::{
    pilfflonk_witness_library, PilfflonkError, PilfflonkResult, PilfflonkWitnessLibrary, Witness, WitnessShape,
};

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
