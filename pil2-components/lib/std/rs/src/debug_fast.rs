use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use std::sync::RwLock;

use colored::Colorize;
use fields::PrimeField64;
use proofman_common::{ProofCtx, ProofmanError, ProofmanResult};
use proofman_hints::HintFieldOutput;

use crate::normalize_vals;

pub type DebugDataFast<F> = FxHashMap<F, SharedDataFast>; // opid -> sharedDataFast

#[derive(Clone, Debug, Default)]
pub struct SharedDataFast {
    pub global_values: FxHashSet<u64>, // store hashes for deduplication
    pub num_proves: u128,              // accumulation
    pub num_assumes: u128,             // accumulation
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data_fast<F: PrimeField64>(
    debug_data_fast: &mut DebugDataFast<F>,
    opid: F,
    vals: Vec<HintFieldOutput<F>>,
    is_proves: bool,
    times: F,
    is_global: bool,
) -> ProofmanResult<()> {
    let bus = debug_data_fast.entry(opid).or_default();

    let mut hasher = FxHasher::default();

    let norm_vals = normalize_vals(&vals);

    let mut values = Vec::new();
    for value in &norm_vals {
        match value {
            HintFieldOutput::Field(f) => values.push(*f),
            HintFieldOutput::FieldExtended(ef) => values.extend_from_slice(&ef.value),
        }
    }

    values.hash(&mut hasher);
    let hash_value = hasher.finish();

    if is_global && !bus.global_values.insert(hash_value) {
        return Ok(());
    }

    // Compute contribution safely
    let contribution = (hash_value as u128)
        .checked_mul(times.as_canonical_u64() as u128)
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

pub fn check_invalid_opids<F: PrimeField64>(
    _pctx: &ProofCtx<F>,
    debugs_data_fasts: &[RwLock<DebugDataFast<F>>],
) -> Vec<F> {
    let mut merged: FxHashMap<F, SharedDataFast> = FxHashMap::default();

    for map in debugs_data_fasts {
        let map = map.read().unwrap();
        for (opid, bus) in map.iter() {
            let entry = merged.entry(*opid).or_default();
            entry.num_proves = entry.num_proves.checked_add(bus.num_proves).expect("Overflow when merging num_proves");
            entry.num_assumes =
                entry.num_assumes.checked_add(bus.num_assumes).expect("Overflow when merging num_assumes");
            entry.global_values.extend(&bus.global_values);
        }
    }

    let mut invalid_opids = Vec::new();

    for (opid, bus) in &merged {
        if bus.num_proves != bus.num_assumes {
            invalid_opids.push(*opid);
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
