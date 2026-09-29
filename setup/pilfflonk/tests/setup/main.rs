//! The setup's steps (plan M15), each in its module: what it refuses in a pilout (spec §4.2.1),
//! the fixed columns and `<air>.const`, the SRS and the verkey, the vkey's digest (A.6), and the
//! command that writes them in the `provingKey/` (§4.2.6).
//!
//! One test crate, so that the pilouts built in code (`common`) are shared.

mod command;
mod common;
mod digest;
mod fixed;
mod keys;
mod validate;
