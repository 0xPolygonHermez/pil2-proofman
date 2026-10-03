//! The setup's steps (pilfflonk/docs/README.md#setup-pilfflonk), each in its module: what it
//! refuses in a pilout (pilfflonk/docs/README.md#what-the-setup-refuses), the fixed columns and
//! `<air>.const`, the SRS and the verkey, the vkey's digest (pilfflonk/docs/formats.md#digest), the
//! committed polynomials and the layout (pilfflonk/docs/protocol.md#degrees,
//! pilfflonk/docs/protocol.md#layout), the command that writes them in the `provingKey/`
//! (pilfflonk/docs/formats.md#provingkey), the Solidity verifier of `--solidity`
//! (pilfflonk/docs/verifier.md#solidity-verifier), and the fixed columns a pilout declares
//! `#pragma fixed_external` on compiled fixtures (pilfflonk/docs/formats.md#fixed-columns).
//!
//! One test crate, so that the pilouts built in code (`common`) are shared.

mod command;
mod common;
mod digest;
mod external_fixed;
mod fixed;
mod keys;
mod layout;
mod solidity;
mod validate;
