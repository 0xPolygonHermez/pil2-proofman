//! Splits the GPU budget left after the fixed regions between the witness prefetch zone, the
//! aggregation const cache and the proof streams (prefetch first, then cache, then streams).

use crate::{format_bytes, ProofmanError, ProofmanResult};

/// One prefetch unit is half the largest witness; a witness takes at most this many units.
pub const PREFETCH_UNIT_SPAN: u64 = 2;
pub const PREFETCH_UNITS_MIN: u32 = 2;
/// 6 units matched 8 on the RTX 5090 (2026-10-02 A/B); 4 lost 1.9% on the mid block.
pub const PREFETCH_UNITS_PLAN_MAX: u32 = 6;
/// LRU slots of the aggregation const cache, slot 0 included.
pub const CONST_CACHE_SLOTS_PLAN_MAX: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferPlanInput {
    /// Free GPU words after the headroom, and what the fixed regions take of them (for the error).
    pub free_words: u64,
    pub fixed_words: u64,
    /// Part of `fixed_words` that is the const cache's minimum (slot 0, the minimum normal slots, pins).
    pub cache_min_words: u64,
    /// Words left after the headroom, the fixed pols (minimum cache included) and a late region outside aux.
    pub budget_words: u64,
    /// The single stream that must exist: the largest proof of any kind.
    pub min_stream_words: u64,
    /// 0 disables the prefetch zone.
    pub prefetch_unit_words: u64,
    /// LRU slots, slot 0 included (min <= cap; both 0 = no cache). The minimum is in `cache_min_words`;
    /// every slot beyond it is a normal (recursive-sized) one of `cache_slot_words`.
    pub cache_slot_words: u64,
    pub cache_slots_min: usize,
    pub cache_slots_cap: usize,
    /// Final pins, taken only once the normal slots reach their cap (0 = none).
    pub optional_pin_words: u64,
    /// Basic packed fixed pols (+ custom commits): resident when they fit, else read from host.
    pub basic_fixed_words: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferPlan {
    pub prefetch_units: u32,
    /// LRU slots, slot 0 included.
    pub cache_slots: usize,
    pub pin_finals: bool,
    pub basic_resident: bool,
}

impl BufferPlan {
    /// Basic fixed words to carve on the device (0 in host mode).
    pub fn basic_words(&self, input: &BufferPlanInput) -> u64 {
        if self.basic_resident {
            input.basic_fixed_words
        } else {
            0
        }
    }

    pub fn prefetch_words(&self, unit_words: u64) -> u64 {
        self.prefetch_units as u64 * unit_words
    }

    /// Cache words beyond the minimum already counted in the budget.
    pub fn extra_cache_words(&self, input: &BufferPlanInput) -> u64 {
        (self.cache_slots - input.cache_slots_min) as u64 * input.cache_slot_words
            + if self.pin_finals { input.optional_pin_words } else { 0 }
    }
}

pub fn plan_buffer(input: &BufferPlanInput) -> ProofmanResult<BufferPlan> {
    let unit = input.prefetch_unit_words;
    let min_units = if unit == 0 { 0 } else { PREFETCH_UNITS_MIN };
    debug_assert!(input.cache_slots_min <= input.cache_slots_cap);
    let need = input.min_stream_words + min_units as u64 * unit;
    if input.budget_words < need {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "{} free on the GPU - {} fixed pols (incl. a const cache of {}) = {}, which cannot hold \
             the largest proof ({}) plus {} prefetch units ({}): short by {}",
            format_bytes(input.free_words as f64 * 8.0),
            format_bytes(input.fixed_words as f64 * 8.0),
            format_bytes(input.cache_min_words as f64 * 8.0),
            format_bytes(input.budget_words as f64 * 8.0),
            format_bytes(input.min_stream_words as f64 * 8.0),
            min_units,
            format_bytes((min_units as u64 * unit) as f64 * 8.0),
            format_bytes((need - input.budget_words) as f64 * 8.0),
        )));
    }
    let mut spare = input.budget_words - need;

    let basic_resident = spare >= input.basic_fixed_words;
    if basic_resident {
        spare -= input.basic_fixed_words;
    }

    let mut prefetch_units = min_units;
    if let Some(fit) = spare.checked_div(unit) {
        let extra = fit.min((PREFETCH_UNITS_PLAN_MAX - min_units) as u64);
        prefetch_units += extra as u32;
        spare -= extra * unit;
    }

    let mut cache_slots = input.cache_slots_min;
    if let Some(fit) = spare.checked_div(input.cache_slot_words) {
        let extra = fit.min((input.cache_slots_cap - input.cache_slots_min) as u64);
        cache_slots += extra as usize;
        spare -= extra * input.cache_slot_words;
    }
    let pin_finals =
        input.optional_pin_words > 0 && cache_slots == input.cache_slots_cap && spare >= input.optional_pin_words;

    Ok(BufferPlan { prefetch_units, cache_slots, pin_finals, basic_resident })
}

/// LRU slots a set of cache keys can keep busy: one per set that fits a normal slot, plus slot 0
/// for the ones that only fit there.
pub fn useful_lru_slots(set_words: &[u64], normal_words: u64) -> usize {
    let small = set_words.iter().filter(|&&w| w <= normal_words).count();
    small + usize::from(small < set_words.len())
}

pub fn prefetch_unit_words(witness_bytes: u64) -> u64 {
    if witness_bytes == 0 || !witness_bytes.is_multiple_of(PREFETCH_UNIT_SPAN * 8) {
        return 0;
    }
    witness_bytes / PREFETCH_UNIT_SPAN / 8
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB_WORDS: u64 = (1 << 30) / 8;
    const UNIT: u64 = 448 * (1 << 20) / 8; // ZisK: Main's packed trace / 2
    const SLOT: u64 = 100 * (1 << 20) / 8;

    fn input(budget_gib_x100: u64) -> BufferPlanInput {
        BufferPlanInput {
            free_words: budget_gib_x100 * GIB_WORDS / 100 + GIB_WORDS,
            fixed_words: GIB_WORDS,
            budget_words: budget_gib_x100 * GIB_WORDS / 100,
            min_stream_words: 2032 * GIB_WORDS / 100, // Keccakf, 20.32 GiB
            prefetch_unit_words: UNIT,
            cache_min_words: 2 * SLOT,
            cache_slot_words: SLOT,
            cache_slots_min: 2,
            cache_slots_cap: 10,
            optional_pin_words: 0,
            basic_fixed_words: 0,
        }
    }

    #[test]
    fn big_card_gets_the_maximums() {
        let p = plan_buffer(&input(2800)).unwrap();
        assert_eq!(p, BufferPlan { prefetch_units: 6, cache_slots: 10, pin_finals: false, basic_resident: true });
    }

    #[test]
    fn tight_card_keeps_the_minimums() {
        // 20.32 + 0.875 = 21.195: 0.1 GiB left, less than a unit, one slot.
        let p = plan_buffer(&input(2130)).unwrap();
        assert_eq!(p, BufferPlan { prefetch_units: 2, cache_slots: 3, pin_finals: false, basic_resident: true });
    }

    #[test]
    fn prefetch_is_filled_before_the_cache() {
        // Room for exactly one more unit and nothing else.
        let i = input(0);
        let i = BufferPlanInput { budget_words: i.min_stream_words + 3 * UNIT, ..i };
        assert_eq!(
            plan_buffer(&i).unwrap(),
            BufferPlan { prefetch_units: 3, cache_slots: 2, pin_finals: false, basic_resident: true }
        );
    }

    #[test]
    fn errors_when_minimums_do_not_fit() {
        let i = input(0);
        let i = BufferPlanInput { budget_words: i.min_stream_words + UNIT, ..i };
        let err = plan_buffer(&i).unwrap_err().to_string();
        assert!(err.contains("largest proof"), "{err}");
        assert!(err.contains("2 prefetch units"), "{err}");
        assert!(err.contains("short by"), "{err}");
        assert!(err.contains("free"), "{err}");
        assert!(err.contains("fixed pols"), "{err}");
        assert!(err.contains("const cache"), "{err}");
    }

    #[test]
    fn prefetch_disabled_has_no_minimum() {
        let i = BufferPlanInput { prefetch_unit_words: 0, ..input(0) };
        let i = BufferPlanInput { budget_words: i.min_stream_words, ..i };
        assert_eq!(
            plan_buffer(&i).unwrap(),
            BufferPlan { prefetch_units: 0, cache_slots: 2, pin_finals: false, basic_resident: true }
        );
    }

    #[test]
    fn cache_never_exceeds_its_cap() {
        // Few useful slots: the cap is the minimum, nothing beyond it.
        let i = BufferPlanInput { cache_slots_min: 2, cache_slots_cap: 2, ..input(2800) };
        let p = plan_buffer(&i).unwrap();
        assert_eq!(p.cache_slots, 2);
        assert_eq!(p.extra_cache_words(&i), 0);
    }

    #[test]
    fn no_cache_means_no_slots() {
        let i = BufferPlanInput { cache_slot_words: 0, cache_slots_min: 0, cache_slots_cap: 0, ..input(2800) };
        assert_eq!(plan_buffer(&i).unwrap().cache_slots, 0);
    }

    #[test]
    fn plan_words() {
        let i = input(2800);
        let p = BufferPlan { prefetch_units: 6, cache_slots: 10, pin_finals: false, basic_resident: true };
        assert_eq!(p.prefetch_words(UNIT), 6 * UNIT);
        assert_eq!(p.extra_cache_words(&i), 8 * SLOT);
    }

    #[test]
    fn finals_are_pinned_only_after_the_lru_is_full() {
        let pins = 100 * (1 << 20) / 8; // final + compressed, 50 MiB each
        let i = BufferPlanInput { optional_pin_words: pins, ..input(2800) };
        let p = plan_buffer(&i).unwrap();
        assert_eq!(p, BufferPlan { prefetch_units: 6, cache_slots: 10, pin_finals: true, basic_resident: true });
        assert_eq!(p.extra_cache_words(&i), 8 * SLOT + pins);
        // Room for the LRU but not for the pins.
        let i = input(0);
        let i = BufferPlanInput {
            budget_words: i.min_stream_words + 6 * UNIT + 8 * SLOT + pins - 1,
            optional_pin_words: pins,
            ..i
        };
        assert_eq!(
            plan_buffer(&i).unwrap(),
            BufferPlan { prefetch_units: 6, cache_slots: 10, pin_finals: false, basic_resident: true }
        );
    }

    #[test]
    fn finals_wait_for_the_lru_cap() {
        // Leftover below one normal slot but above the pins: the LRU is not full, so no pins.
        let pins = 20 * (1 << 20) / 8;
        let i = input(0);
        let i = BufferPlanInput {
            budget_words: i.min_stream_words + 6 * UNIT + pins + SLOT / 2,
            optional_pin_words: pins,
            ..i
        };
        assert_eq!(
            plan_buffer(&i).unwrap(),
            BufferPlan { prefetch_units: 6, cache_slots: 2, pin_finals: false, basic_resident: true }
        );
    }

    #[test]
    fn useful_lru_slots_counts_what_a_normal_slot_can_hold() {
        // 3 sets fit a normal slot, 2 only fit slot 0: 3 normal + slot 0.
        assert_eq!(useful_lru_slots(&[10, 10, 10, 30, 40], 10), 4);
        // Everything fits a normal slot: slot 0 is one of them.
        assert_eq!(useful_lru_slots(&[10, 5, 10], 10), 3);
        assert_eq!(useful_lru_slots(&[], 10), 0);
    }

    #[test]
    fn basic_fixed_is_resident_when_it_fits_before_prefetch_grows() {
        let basic = 1170 * (1 << 20) / 8;
        let i = BufferPlanInput { basic_fixed_words: basic, ..input(2800) };
        let p = plan_buffer(&i).unwrap();
        assert!(p.basic_resident);
        assert_eq!(p.basic_words(&i), basic);
        // Room for the minimums and less than the basic fixed: host mode, its words fund prefetch.
        let i = input(0);
        let i =
            BufferPlanInput { budget_words: i.min_stream_words + 2 * UNIT + basic - 1, basic_fixed_words: basic, ..i };
        let p = plan_buffer(&i).unwrap();
        assert!(!p.basic_resident);
        assert_eq!(p.basic_words(&i), 0);
        assert_eq!(p.prefetch_units, 4);
    }

    #[test]
    fn unit_words_requires_an_even_split() {
        assert_eq!(prefetch_unit_words(896 << 20), (448 << 20) / 8);
        assert_eq!(prefetch_unit_words(8), 0); // one word cannot be halved
        assert_eq!(prefetch_unit_words(0), 0);
    }
}
