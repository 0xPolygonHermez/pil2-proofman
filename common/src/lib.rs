mod air_instance;
mod air_instances_repository;
mod verbose_mode;
mod execution_ctx;
mod distribution_ctx;
mod proof_ctx;
mod prover;
mod extended_field;
mod setup;
mod setup_ctx;
mod std_mode;
mod custom_commits;
pub mod trace;
pub mod global_info;
pub mod stark_info;

pub use air_instance::*;
pub use air_instances_repository::*;
use proofman_starks_lib_c::set_log_level_c;
pub use verbose_mode::*;
pub use execution_ctx::*;
pub use distribution_ctx::*;
pub use proof_ctx::*;
pub use prover::*;
pub use extended_field::*;
pub use global_info::*;
pub use setup::*;
pub use setup_ctx::*;
pub use custom_commits::*;
pub use stark_info::*;
pub use std_mode::*;

pub fn initialize_logger(verbose_mode: VerboseMode) {
    env_logger::builder()
        .format_timestamp(None)
        .format_level(true)
        .format_target(false)
        .filter_level(verbose_mode.into())
        .init();
    set_log_level_c(verbose_mode.into());
}
