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
/// `ConstSlotTable::MAX_SLOTS`: LRU slots and pins together.
pub const CONST_CACHE_MAX_SLOTS: usize = 64;

/// The aggregation const cache's pieces (words).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheInput {
    pub slot0_words: u64,
    /// Every LRU slot past slot 0.
    pub normal_words: u64,
    /// Pinned recursive2 slots, all airgroups.
    pub rec2_words: u64,
    /// Each recursive2 set fits slot 0, so it can go through the LRU when memory is short.
    pub rec2_fits_lru: bool,
    /// Final pins, taken only once the LRU reaches its cap.
    pub finals_words: u64,
    /// Slots each pin kind takes (one per set).
    pub rec2_sets: usize,
    pub finals_sets: usize,
    /// Useful LRU slots, slot 0 included (>= 2).
    pub slots_cap: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferPlanInput {
    /// Free GPU words after the headroom, and what the fixed regions take of them (for the error).
    pub free_words: u64,
    pub fixed_words: u64,
    /// Words left after the headroom, the fixed regions and the late region (the cache is
    /// planned here, so it is not taken out yet).
    pub budget_words: u64,
    /// The single stream that must exist: the largest proof of any kind.
    pub min_stream_words: u64,
    /// 0 disables the prefetch zone.
    pub prefetch_unit_words: u64,
    /// Basic packed fixed pols (+ custom commits): resident when they fit, else read from host.
    pub basic_fixed_words: u64,
    pub cache: Option<CacheInput>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferPlan {
    pub prefetch_units: u32,
    /// LRU slots, slot 0 included (0 = no cache).
    pub cache_slots: usize,
    pub pin_rec2: bool,
    pub pin_finals: bool,
    pub basic_resident: bool,
    /// The minimums did not fit and were relaxed (recursive2 unpinned, fewer slots or units).
    pub relaxed: bool,
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

    /// Words the const cache takes: its LRU slots and the pins.
    pub fn cache_words(&self, input: &BufferPlanInput) -> u64 {
        input.cache.as_ref().map_or(0, |c| {
            cache_min_words(c, self.cache_slots, self.pin_rec2) + if self.pin_finals { c.finals_words } else { 0 }
        })
    }
}

fn cache_min_words(c: &CacheInput, slots: usize, pin_rec2: bool) -> u64 {
    c.slot0_words + (slots as u64 - 1) * c.normal_words + if pin_rec2 { c.rec2_words } else { 0 }
}

/// A floor the plan must hold: the cache's slots and recursive2 pin, and prefetch units.
#[derive(Clone, Copy)]
struct Floor {
    slots: usize,
    pin_rec2: bool,
    units: u32,
}

/// Floors from the planned one down to the least that still proves, cheapest loss first: recursive2
/// through the LRU, a one-slot cache (1 rec1 slot measured within ~1% of 20), then prefetch units
/// (1 unit measured at +1.7-4.9%).
fn floors(input: &BufferPlanInput) -> Vec<Floor> {
    let units = if input.prefetch_unit_words == 0 { 0 } else { PREFETCH_UNITS_MIN };
    let Some(c) = &input.cache else {
        return (0..=units).rev().map(|units| Floor { slots: 0, pin_rec2: false, units }).collect();
    };
    // Too many recursive2 sets to pin go through the LRU (plan_buffer refuses them if they cannot).
    let pin_rec2 = c.rec2_words > 0 && 2 + c.rec2_sets <= CONST_CACHE_MAX_SLOTS;
    let pinned = Floor { slots: 2, pin_rec2, units };
    let mut floors = vec![pinned];
    let mut f = pinned;
    if f.pin_rec2 && c.rec2_fits_lru {
        f.pin_rec2 = false;
        floors.push(f);
    }
    f.slots = 1;
    floors.push(f);
    while f.units > 0 {
        f.units -= 1;
        floors.push(f);
    }
    floors
}

pub fn plan_buffer(input: &BufferPlanInput) -> ProofmanResult<BufferPlan> {
    let unit = input.prefetch_unit_words;
    let need = |f: &Floor| {
        input.min_stream_words
            + f.units as u64 * unit
            + input.cache.as_ref().map_or(0, |c| cache_min_words(c, f.slots, f.pin_rec2))
    };
    if let Some(c) = &input.cache {
        if c.rec2_words > 0 && !c.rec2_fits_lru && 2 + c.rec2_sets > CONST_CACHE_MAX_SLOTS {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "{} recursive2 const sets must be pinned but the const cache holds {} slots",
                c.rec2_sets, CONST_CACHE_MAX_SLOTS
            )));
        }
    }
    let floors = floors(input);
    let Some(i) = floors.iter().position(|f| need(f) <= input.budget_words) else {
        let f = floors.last().unwrap();
        return Err(ProofmanError::InvalidConfiguration(format!(
            "{} free on the GPU - {} fixed pols = {}, which cannot hold the largest proof ({}) plus the \
             smallest const cache ({}) and {} prefetch units ({}): short by {}",
            format_bytes(input.free_words as f64 * 8.0),
            format_bytes(input.fixed_words as f64 * 8.0),
            format_bytes(input.budget_words as f64 * 8.0),
            format_bytes(input.min_stream_words as f64 * 8.0),
            format_bytes(input.cache.as_ref().map_or(0, |c| cache_min_words(c, f.slots, f.pin_rec2)) as f64 * 8.0),
            f.units,
            format_bytes((f.units as u64 * unit) as f64 * 8.0),
            format_bytes((need(f) - input.budget_words) as f64 * 8.0),
        )));
    };
    let floor = floors[i];
    let mut spare = input.budget_words - need(&floor);

    let basic_resident = spare >= input.basic_fixed_words;
    if basic_resident {
        spare -= input.basic_fixed_words;
    }

    let mut prefetch_units = floor.units;
    if let Some(fit) = spare.checked_div(unit) {
        let extra = fit.min(PREFETCH_UNITS_PLAN_MAX.saturating_sub(floor.units) as u64);
        prefetch_units += extra as u32;
        spare -= extra * unit;
    }

    let mut cache_slots = floor.slots;
    let mut pin_finals = false;
    if let Some(c) = &input.cache {
        let pins = if floor.pin_rec2 { c.rec2_sets } else { 0 };
        let slots_cap = c.slots_cap.min(CONST_CACHE_MAX_SLOTS - pins);
        if let Some(fit) = spare.checked_div(c.normal_words) {
            let extra = fit.min(slots_cap.saturating_sub(floor.slots) as u64);
            cache_slots += extra as usize;
            spare -= extra * c.normal_words;
        }
        pin_finals = c.finals_words > 0
            && cache_slots >= slots_cap
            && spare >= c.finals_words
            && cache_slots + pins + c.finals_sets <= CONST_CACHE_MAX_SLOTS;
    }

    Ok(BufferPlan { prefetch_units, cache_slots, pin_rec2: floor.pin_rec2, pin_finals, basic_resident, relaxed: i > 0 })
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
    const KECCAKF: u64 = 2032 * GIB_WORDS / 100; // 20.32 GiB

    fn cache() -> CacheInput {
        CacheInput {
            slot0_words: SLOT,
            normal_words: SLOT,
            rec2_words: SLOT,
            rec2_fits_lru: true,
            finals_words: 0,
            rec2_sets: 1,
            finals_sets: 0,
            slots_cap: 10,
        }
    }

    fn input(budget_words: u64) -> BufferPlanInput {
        BufferPlanInput {
            free_words: budget_words + GIB_WORDS,
            fixed_words: GIB_WORDS,
            budget_words,
            min_stream_words: KECCAKF,
            prefetch_unit_words: UNIT,
            basic_fixed_words: 0,
            cache: Some(cache()),
        }
    }

    /// The minimums: Keccakf, 2 prefetch units, slot 0 + one normal slot + the recursive2 pin.
    const FLOOR: u64 = KECCAKF + 2 * UNIT + 3 * SLOT;

    fn plan(prefetch_units: u32, cache_slots: usize, pin_rec2: bool, pin_finals: bool, relaxed: bool) -> BufferPlan {
        BufferPlan { prefetch_units, cache_slots, pin_rec2, pin_finals, basic_resident: true, relaxed }
    }

    #[test]
    fn big_card_gets_the_maximums() {
        let p = plan_buffer(&input(28 * GIB_WORDS)).unwrap();
        assert_eq!(p, plan(6, 10, true, false, false));
    }

    #[test]
    fn tight_card_keeps_the_minimums() {
        // One more normal slot of spare, less than a unit.
        let p = plan_buffer(&input(FLOOR + SLOT)).unwrap();
        assert_eq!(p, plan(2, 3, true, false, false));
    }

    #[test]
    fn prefetch_is_filled_before_the_cache() {
        let p = plan_buffer(&input(FLOOR + UNIT)).unwrap();
        assert_eq!(p, plan(3, 2, true, false, false));
    }

    #[test]
    fn l4_relaxes_the_cache_before_prefetch() {
        // RTX L4, ZisK (2026-10-15): 21.32 GB left for the plan, 179 MB short of the full minimums.
        let p = plan_buffer(&input(2132 * GIB_WORDS / 100)).unwrap();
        assert_eq!(p, plan(2, 1, false, false, true));
        assert_eq!(p.cache_words(&input(0)), SLOT);
    }

    #[test]
    fn relaxation_order() {
        let p = plan_buffer(&input(FLOOR - 1)).unwrap();
        assert_eq!(p, plan(2, 2, false, false, true)); // recursive2 through the LRU first
        let p = plan_buffer(&input(FLOOR - SLOT - 1)).unwrap();
        assert_eq!(p, plan(2, 1, false, false, true)); // then a one-slot cache
        let p = plan_buffer(&input(KECCAKF + UNIT + SLOT)).unwrap();
        assert_eq!(p, plan(1, 1, false, false, true)); // then prefetch
        let p = plan_buffer(&input(KECCAKF + SLOT)).unwrap();
        assert_eq!(p, plan(0, 1, false, false, true));
    }

    #[test]
    fn recursive2_stays_pinned_when_it_cannot_use_the_lru() {
        let i = BufferPlanInput { cache: Some(CacheInput { rec2_fits_lru: false, ..cache() }), ..input(FLOOR - 1) };
        assert_eq!(plan_buffer(&i).unwrap(), plan(2, 1, true, false, true));
    }

    #[test]
    fn errors_when_even_the_least_does_not_fit() {
        let err = plan_buffer(&input(KECCAKF + SLOT - 1)).unwrap_err().to_string();
        for want in ["free", "fixed pols", "largest proof", "smallest const cache", "0 prefetch units", "short by"] {
            assert!(err.contains(want), "{want}: {err}");
        }
    }

    #[test]
    fn prefetch_disabled_has_no_minimum() {
        let i = BufferPlanInput { prefetch_unit_words: 0, ..input(KECCAKF + 3 * SLOT) };
        assert_eq!(plan_buffer(&i).unwrap(), plan(0, 2, true, false, false));
    }

    #[test]
    fn no_cache() {
        let i = BufferPlanInput { cache: None, ..input(KECCAKF + 2 * UNIT) };
        let p = plan_buffer(&i).unwrap();
        assert_eq!(p, plan(2, 0, false, false, false));
        assert_eq!(p.cache_words(&i), 0);
        let i = BufferPlanInput { cache: None, ..input(KECCAKF + UNIT) };
        assert_eq!(plan_buffer(&i).unwrap(), plan(1, 0, false, false, true));
    }

    #[test]
    fn cache_never_exceeds_its_cap() {
        let i = BufferPlanInput { cache: Some(CacheInput { slots_cap: 2, ..cache() }), ..input(28 * GIB_WORDS) };
        let p = plan_buffer(&i).unwrap();
        assert_eq!(p.cache_slots, 2);
        assert_eq!(p.cache_words(&i), 3 * SLOT);
    }

    #[test]
    fn finals_are_pinned_only_after_the_lru_is_full() {
        let pins = 100 * (1 << 20) / 8; // final + compressed, 50 MiB each
        let i = BufferPlanInput { cache: Some(CacheInput { finals_words: pins, ..cache() }), ..input(28 * GIB_WORDS) };
        let p = plan_buffer(&i).unwrap();
        assert_eq!(p, plan(6, 10, true, true, false));
        assert_eq!(p.cache_words(&i), 11 * SLOT + pins);
        // Room for the full LRU but not for the pins.
        let budget = FLOOR + 4 * UNIT + 8 * SLOT + pins - 1;
        let i = BufferPlanInput { cache: Some(CacheInput { finals_words: pins, ..cache() }), ..input(budget) };
        assert_eq!(plan_buffer(&i).unwrap(), plan(6, 10, true, false, false));
    }

    #[test]
    fn finals_wait_for_the_lru_cap() {
        // Spare below one normal slot but above the pins: the LRU is not full, so no pins.
        let pins = 20 * (1 << 20) / 8;
        let budget = FLOOR + 4 * UNIT + pins + SLOT / 2;
        let i = BufferPlanInput { cache: Some(CacheInput { finals_words: pins, ..cache() }), ..input(budget) };
        assert_eq!(plan_buffer(&i).unwrap(), plan(6, 2, true, false, false));
    }

    #[test]
    fn pins_respect_the_slot_table() {
        // 63 recursive2 sets cannot be pinned beside 2 LRU slots: they go through the LRU.
        let c = CacheInput { rec2_sets: 63, ..cache() };
        let i = BufferPlanInput { cache: Some(c.clone()), ..input(28 * GIB_WORDS) };
        assert!(!plan_buffer(&i).unwrap().pin_rec2);
        let i = BufferPlanInput { cache: Some(CacheInput { rec2_fits_lru: false, ..c }), ..input(28 * GIB_WORDS) };
        assert!(plan_buffer(&i).is_err());
        // 60 pinned: the LRU shrinks to 4, and the finals no longer fit.
        let c = CacheInput { rec2_sets: 60, finals_words: SLOT, finals_sets: 2, ..cache() };
        let p = plan_buffer(&BufferPlanInput { cache: Some(c), ..input(28 * GIB_WORDS) }).unwrap();
        assert_eq!((p.pin_rec2, p.cache_slots, p.pin_finals), (true, 4, false));
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
        let i = BufferPlanInput { basic_fixed_words: basic, ..input(28 * GIB_WORDS) };
        let p = plan_buffer(&i).unwrap();
        assert!(p.basic_resident);
        assert_eq!(p.basic_words(&i), basic);
        // Room for the minimums and less than the basic fixed: host mode, its words fund prefetch.
        let i = BufferPlanInput { basic_fixed_words: basic, ..input(FLOOR + basic - 1) };
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
