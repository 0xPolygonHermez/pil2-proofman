//! Symbolic analysis of a pilout, parameterised by the field: the symbolic passes of the STARK
//! setup, the `"chps"` bytecode container, temporary allocation and `globalConstraints.json`.

pub mod cfg;
pub mod error;
pub mod expr;
pub mod io;
pub mod output;
pub mod pil;
pub mod types;

pub use cfg::{FieldCfg, PilInfoCfg};
pub use error::{BinFileError, PilInfoError};
pub use pil::info::{run, PilInfoResult};
