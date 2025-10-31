use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hasher, Hash},
};

use colored::Colorize;
use fields::PrimeField64;
use num_bigint::BigUint;
use num_traits::Zero;
use proofman_common::ProofCtx;
use proofman_hints::HintFieldOutput;

use crate::normalize_vals;

pub type DebugDataFast<F> = HashMap<F, SharedDataFast>; // opid -> sharedDataFast

#[derive(Clone, Debug)]
pub struct SharedDataFast {
    pub global_values: Vec<BigUint>,
    pub num_proves: BigUint,
    pub num_assumes: BigUint,
}

#[allow(clippy::too_many_arguments)]
pub fn update_debug_data_fast<F: PrimeField64>(
    debug_data_fast: &mut DebugDataFast<F>,
    opid: F,
    vals: Vec<HintFieldOutput<F>>,
    is_proves: bool,
    times: F,
    is_global: bool,
) {
    let bus_opid_times = debug_data_fast.entry(opid).or_insert_with(|| SharedDataFast {
        global_values: Vec::new(),
        num_proves: BigUint::zero(),
        num_assumes: BigUint::zero(),
    });

    // Normalize the vector of values
    let norm_vals = normalize_vals(&vals);

    let mut values = Vec::new();
    for value in norm_vals.iter() {
        match value {
            HintFieldOutput::Field(f) => values.push(*f),
            HintFieldOutput::FieldExtended(ef) => {
                values.push(ef.value[0]);
                values.push(ef.value[1]);
                values.push(ef.value[2]);
            }
        }
    }

    // Create hash of all values
    let mut hasher = DefaultHasher::new();
    values.hash(&mut hasher);
    let hash_value = BigUint::from(hasher.finish());

    // If the value is global but it was already processed, skip it
    if is_global {
        if bus_opid_times.global_values.contains(&hash_value) {
            return;
        }
        bus_opid_times.global_values.push(hash_value.clone());
    }

    // Update the number of proves or assumes
    if is_proves {
        bus_opid_times.num_proves += hash_value * times.as_canonical_biguint();
    } else {
        bus_opid_times.num_assumes += hash_value * times.as_canonical_biguint();
    }
}

pub fn check_invalid_opids<F: PrimeField64>(_pctx: &ProofCtx<F>, debugs_data_fasts: &mut [DebugDataFast<F>]) -> Vec<F> {
    let mut debug_data_fast = HashMap::new();

    for map in debugs_data_fasts {
        for (opid, bus) in map.iter() {
            if debug_data_fast.contains_key(opid) {
                let bus_fast: &mut SharedDataFast = debug_data_fast.get_mut(opid).unwrap();
                bus_fast.num_proves += bus.num_proves.clone();
                bus_fast.num_assumes += bus.num_assumes.clone();
            } else {
                debug_data_fast.insert(*opid, bus.clone());
            }
        }
    }

    // TODO: SINCRONIZATION IN DISTRIBUTED MODE

    let mut invalid_opids = Vec::new();

    // Check if there are any invalid opids
    for (opid, bus) in debug_data_fast.iter_mut() {
        if bus.num_proves != bus.num_assumes {
            invalid_opids.push(*opid);
        }
    }

    if !invalid_opids.is_empty() {
        tracing::error!(
            "··· {}",
            format!("\u{2717} The following opids does not match {invalid_opids:?}").bright_red().bold()
        );
    } else {
        tracing::info!("··· {}", "\u{2713} All bus values match.".bright_green().bold());
    }

    invalid_opids
}
