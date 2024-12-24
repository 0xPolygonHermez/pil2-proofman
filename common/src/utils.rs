use crate::{AirInstance, ProofCtx, VerboseMode};
use proofman_starks_lib_c::set_log_level_c;
use std::sync::Arc;
use p3_field::Field;

pub fn add_air_instance<F: Field>(air_instance: AirInstance<F>, pctx: Arc<ProofCtx<F>>) -> Option<usize> {
    let (is_mine, gid) = pctx.dctx.write().unwrap().add_instance(air_instance.airgroup_id, air_instance.air_id, 1);

    if is_mine {
        return Some(pctx.air_instance_repo.add_air_instance(air_instance, Some(gid)));
    }

    None
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

pub fn format_bytes(mut num_bytes: f64) -> String {
    let units = ["Bytes", "KB", "MB", "GB"];
    let mut unit_index = 0;

    while num_bytes >= 0.01 && unit_index < units.len() - 1 {
        if num_bytes < 1024.0 {
            break;
        }
        num_bytes /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", num_bytes, units[unit_index])
}
