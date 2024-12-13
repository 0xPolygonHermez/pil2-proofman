use crate::{AirInstance, ProofCtx, VerboseMode};
use proofman_starks_lib_c::set_log_level_c;
use std::sync::Arc;
use p3_field::Field;

pub fn add_air_instance<F: Field>(air_instance: AirInstance<F>, pctx: Arc<ProofCtx<F>>) {
    let (is_mine, gid) = pctx.dctx.write().unwrap().add_instance(air_instance.airgroup_id, air_instance.air_id, 1);

    if is_mine {
        pctx.air_instance_repo.add_air_instance(air_instance, Some(gid));
    }
}

pub fn initialize_logger(verbose_mode: VerboseMode) {
    env_logger::builder()
        .format_timestamp(None)
        .format_level(true)
        .format_target(false)
        .filter_level(verbose_mode.into())
        .init();
    set_log_level_c(verbose_mode.into());
}
