//! The digest of the vkey (pilfflonk/docs/formats.md#digest):
//! `digest = keccak256("pilfflonk-v1" ‖ canonical(vkey without digest))`.
//!
//! The preimage is `Vkey::digest_preimage` (`proofman-pilfflonk`). The hash is Keccak-256, not
//! SHA3-256, through the C++ core: rapidsnark's `keccak_wrapper`, which the transcript hashes with
//! too (pilfflonk/docs/protocol.md#transcript). The workspace has no Keccak crate, and none is
//! added for this.

use proofman_pilfflonk::{Digest, Vkey};
use proofman_starks_lib_c::pilfflonk_keccak256_c;

use crate::error::SetupError;

/// The Keccak-256 hash of `data`.
pub fn keccak256(data: &[u8]) -> Result<[u8; 32], SetupError> {
    pilfflonk_keccak256_c(data).map_err(SetupError::native("cannot hash with keccak256"))
}

/// The digest of `vkey`: of every field but `digest`, which it does not depend on.
pub fn vkey_digest(vkey: &Vkey) -> Result<Digest, SetupError> {
    Ok(Digest(keccak256(&vkey.digest_preimage()?)?))
}

/// `vkey` with its digest set: what `Vkey::seal` does with this module's Keccak-256.
pub fn seal_vkey(vkey: Vkey) -> Result<Vkey, SetupError> {
    let digest = vkey_digest(&vkey)?;
    Ok(Vkey { digest, ..vkey })
}
