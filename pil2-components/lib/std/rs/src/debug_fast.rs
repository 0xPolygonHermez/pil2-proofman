use rustc_hash::{FxHashMap, FxHashSet};

use colored::Colorize;
use fields::PrimeField64;
use proofman_common::{ProofCtx, ProofmanError, ProofmanResult};

pub type DebugDataFast = FxHashMap<u64, SharedDataFast>; // opid -> sharedDataFast

#[derive(Clone, Debug, Default)]
pub struct SharedDataFast {
    pub global_values: FxHashSet<u64>, // store hashes for deduplication
    pub num_proves: u128,              // accumulation
    pub num_assumes: u128,             // accumulation
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data_fast(
    debug_data_fast: &mut DebugDataFast,
    opid: u64,
    hash: u64,
    is_proves: bool,
    times: u64,
    is_global: bool,
) -> ProofmanResult<()> {
    let bus = debug_data_fast.entry(opid).or_default();

    if is_global && !bus.global_values.insert(hash) {
        println!("Global value already exists for hash={}, skipping update.", hash);
        return Ok(());
    }

    // Compute contribution safely
    let contribution = (hash as u128)
        .checked_mul(times as u128)
        .ok_or_else(|| ProofmanError::ProofmanError("Overflow in update_debug_data_fast".to_string()))?;

    if is_proves {
        bus.num_proves = bus
            .num_proves
            .checked_add(contribution)
            .ok_or_else(|| ProofmanError::ProofmanError("Overflow in num_proves".to_string()))?;
    } else {
        bus.num_assumes = bus
            .num_assumes
            .checked_add(contribution)
            .ok_or_else(|| ProofmanError::ProofmanError("Overflow in num_assumes".to_string()))?;
    }

    Ok(())
}

pub fn check_invalid_opids<F: PrimeField64>(_pctx: &ProofCtx<F>, debug_data_fast: DebugDataFast) -> Vec<u64> {
    let mut invalid_opids = Vec::new();

    for (opid, bus) in debug_data_fast {
        if bus.num_proves != bus.num_assumes {
            invalid_opids.push(opid);
        }
    }

    if !invalid_opids.is_empty() {
        tracing::error!(
            "··· {}",
            format!("\u{2717} The following opids do not match {invalid_opids:?}").bright_red().bold()
        );
    } else {
        tracing::info!("··· {}", "\u{2713} All bus values match.".bright_green().bold());
    }

    invalid_opids
}
