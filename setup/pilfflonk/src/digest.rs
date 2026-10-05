//! The digest of the vkey (pilfflonk/docs/formats.md#digest):
//! `digest = keccak256("pilfflonk-v1" ‖ canonical(vkey without digest))`.
//!
//! The preimage is `Vkey::digest_preimage` and the digest `Vkey::compute_digest`
//! (`proofman-pilfflonk`). The hash is Keccak-256, not SHA3-256, through the C++ core: rapidsnark's
//! `keccak_wrapper`, which the transcript hashes with too (pilfflonk/docs/protocol.md#transcript).
//! The workspace has no Keccak crate, and none is added for this.

use proofman_pilfflonk::Vkey;
use proofman_starks_lib_c::pilfflonk_keccak256_c;

use crate::error::SetupError;

/// The Keccak-256 hash of `data`.
pub fn keccak256(data: &[u8]) -> Result<[u8; 32], SetupError> {
    pilfflonk_keccak256_c(data).map_err(SetupError::native("cannot hash with keccak256"))
}

/// `vkey` with its digest set ([`Vkey::compute_digest`]), which does not depend on `digest`.
pub fn seal_vkey(vkey: Vkey) -> Result<Vkey, SetupError> {
    let digest = vkey.compute_digest()?;
    Ok(Vkey { digest, ..vkey })
}
