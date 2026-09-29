//! The symbolic passes over BN254 (spec §4.2.2, §4.2.3): `pil_info::run` with
//! `PilInfoCfg::bn254()` and the degree search of D5, `Search { max: D }` for
//! `--max-constraint-degree D`.
//!
//! The passes recurse over the expression trees, which in large AIRs are thousands of levels
//! deep, so they run on a thread of their own with the stack the STARK setup gives them
//! ([`PASSES_STACK_SIZE`]). What they refuse comes back as [`SetupError::Passes`]. They still
//! index their expressions on invariants they keep themselves, which no pilout breaks: should a
//! bug of `pil-info` break one, the panic ends that thread only, and its message comes back as
//! [`SetupError::PassesPanicked`].

use std::any::Any;
use std::thread;

use pil2_pilout::pilout as pb;
use pil_info::pil::prepare::PrepareOptions;
use pil_info::{DegreePolicy, PilInfoCfg, PilInfoResult};

use crate::error::SetupError;
use crate::validate::ValidAir;

/// The stack of the thread the passes run on: the STARK setup's (`proofman-setup`'s rayon pool
/// and its `stats` thread).
pub const PASSES_STACK_SIZE: usize = 64 * 1024 * 1024;

/// The configuration of the passes for `--max-constraint-degree max_constraint_degree`.
pub fn cfg(max_constraint_degree: u64) -> Result<PilInfoCfg, SetupError> {
    let max =
        usize::try_from(max_constraint_degree).map_err(|_| SetupError::MaxConstraintDegree(max_constraint_degree))?;
    Ok(PilInfoCfg { degree_policy: DegreePolicy::Search { max }, ..PilInfoCfg::bn254() })
}

/// Runs the passes on the AIR of `pilout` that [`crate::validate::validate`] returned.
pub fn run_passes(pilout: &pb::PilOut, air: ValidAir, max_constraint_degree: u64) -> Result<PilInfoResult, SetupError> {
    let cfg = cfg(max_constraint_degree)?;
    let options = PrepareOptions::default();
    on_passes_thread(|| pil_info::run(pilout, air.airgroup_id, air.air_id, &cfg, &options))?.map_err(SetupError::Passes)
}

/// Runs `passes` on a thread with [`PASSES_STACK_SIZE`] of stack: what it returns, or its panic
/// as [`SetupError::PassesPanicked`].
fn on_passes_thread<T: Send>(passes: impl FnOnce() -> T + Send) -> Result<T, SetupError> {
    thread::scope(|scope| {
        let handle = thread::Builder::new()
            .name("pilfflonk-passes".into())
            .stack_size(PASSES_STACK_SIZE)
            .spawn_scoped(scope, passes)
            .map_err(SetupError::PassesThread)?;
        // A panic joined here does not reach the scope: it is this thread's result.
        handle.join().map_err(|payload| SetupError::PassesPanicked(panic_message(payload.as_ref())))
    })
}

/// The message of a panic, as the default hook prints it.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "a panic without a message".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_on_the_passes_thread_is_an_error() {
        let err = on_passes_thread(|| -> u32 { panic!("an invariant of the passes") }).unwrap_err();
        assert!(matches!(&err, SetupError::PassesPanicked(m) if m == "an invariant of the passes"), "{err}");
        assert_eq!(err.to_string(), "the symbolic passes (pil-info) panicked: an invariant of the passes");
    }

    #[test]
    fn the_passes_thread_returns_what_they_return() {
        assert_eq!(on_passes_thread(|| 7u32).unwrap(), 7);
    }
}
