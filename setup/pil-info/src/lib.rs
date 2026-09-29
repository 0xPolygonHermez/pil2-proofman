//! Symbolic analysis of a pilout, parameterised by the field, shared by the STARK and pilfflonk
//! setups: the symbolic passes, the `"chps"` bytecode container, temporary allocation and
//! `globalConstraints.json`.

pub mod expr;
pub mod io;
pub mod output;
pub mod pil;
pub mod types;
