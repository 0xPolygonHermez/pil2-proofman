//! Plonk-to-PIL setup generators, split by hash family.
//!
//! Each family lives in its own folder ([`poseidon1`], [`poseidon2`]) and
//! exposes the same surface (`aggregation`, `compressor`, a `PilTemplateParams`
//! struct and a `gen_pil_str` helper). Dispatch happens in [`super::packers`].

pub mod poseidon1;
pub mod poseidon2;
