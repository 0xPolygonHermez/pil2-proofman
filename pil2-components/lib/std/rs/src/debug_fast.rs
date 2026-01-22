use std::collections::{HashSet, HashMap};

use colored::Colorize;
use fields::PrimeField64;
use proofman_common::{ProofCtx, ProofmanResult};

/// Small vec-based map for ~10 opids. Linear search is faster than hashing for small N.
#[derive(Clone, Debug, Default)]
pub struct DebugDataFast {
    entries: Vec<(u64, i128)>, // (opid, balance)
}

impl DebugDataFast {
    #[inline]
    pub fn new() -> Self {
        Self { entries: Vec::with_capacity(16) } // Pre-allocate for ~10 opids
    }

    #[inline(always)]
    pub fn get_or_insert(&mut self, opid: u64) -> &mut i128 {
        // Linear search - faster than HashMap for N < ~20
        let pos = self.entries.iter().position(|(id, _)| *id == opid);
        if let Some(idx) = pos {
            &mut self.entries[idx].1
        } else {
            self.entries.push((opid, 0));
            &mut self.entries.last_mut().unwrap().1
        }
    }

    #[inline(always)]
    pub fn merge_into(&self, other: &mut DebugDataFast) {
        for &(opid, balance) in &self.entries {
            if balance != 0 {
                let entry = other.get_or_insert(opid);
                *entry = entry.wrapping_add(balance);
            }
        }
    }
}

impl IntoIterator for DebugDataFast {
    type Item = (u64, i128);
    type IntoIter = std::vec::IntoIter<(u64, i128)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

pub type DebugDataFastGlobal = HashMap<u64, HashSet<u64>>;

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn update_debug_data_fast(
    debug_data_fast: &mut DebugDataFast,
    debug_data_fast_global: &mut DebugDataFastGlobal,
    opid: u64,
    hash: u64,
    is_proves: bool,
    times: u64,
    is_global: bool,
) -> ProofmanResult<()> {
    // Skip duplicate global values
    if is_global && !debug_data_fast_global.entry(opid).or_default().insert(hash) {
        return Ok(());
    }

    let contribution = (hash as u128).wrapping_mul(times as u128) as i128;
    let bus = debug_data_fast.get_or_insert(opid);

    if is_proves {
        *bus = bus.wrapping_add(contribution);
    } else {
        *bus = bus.wrapping_sub(contribution);
    }

    Ok(())
}

pub fn check_invalid_opids<F: PrimeField64>(_pctx: &ProofCtx<F>, debug_data_fast: DebugDataFast) -> Vec<u64> {
    let mut invalid_opids = Vec::new();

    for (opid, bus) in debug_data_fast.into_iter() {
        if bus != 0 {
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
