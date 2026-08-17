//! Cutting the per-GPU aux-trace budget into stream size classes.
//!
//! Per-air aux-trace requirements are far from uniform — one outlier air can need several times the
//! next largest — so sizing every stream to the maximum buys concurrency at the outlier's price. The
//! budget is instead cut into two classes: *large*, sized to the biggest air so nothing is homeless,
//! and *regular*, sized to whichever smaller air requirement carves best. A proof runs on any stream
//! at least as large as it needs, so an air between the two sizes has only the large streams.
//!
//! Streams are added keeping the classes in equilibrium — one large per [`REGULARS_PER_LARGE`]
//! regulars — so growth goes 1L, 1L+1R, 1L+2R, 2L+2R, … Whatever they leave becomes recursive streams.
//!
//! Every class is floored at the largest compressor / vadcop_final / recursive launch, since those
//! share the non-recursive streams with basics (see `RecursiveScheduler::next_nonrecursive`).

/// One aux-trace size class: `count` streams of `size` field elements each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamClass {
    pub size: usize,
    pub count: usize,
}

/// How one GPU's aux-trace budget is cut up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamLayout {
    /// The large and (when funded) regular class, in that order — at most two entries. Both host
    /// basic *and* compressor/vadcop_final launches. Never empty, and `basic[0].size` covers the
    /// largest air, so no proof is left without a home.
    pub basic: Vec<StreamClass>,
    /// Recursive1/recursive2-only streams.
    pub recursive: StreamClass,
    /// Budget left over: smaller than any class that could still have been cut.
    pub unused: usize,
}

/// Recursive streams kept funded before any basic class beyond the first is cut. Basic classes are
/// funded first, so without a floor they spend the last byte and aggregation's small proofs end up
/// running a handful at a time in whatever huge basic class is free. A floor, not a reservation:
/// the remainder still becomes recursive streams up to the caller's cap.
const RECURSIVE_STREAM_FLOOR: usize = 1;

/// How close the two class sizes must be before the split collapses to one uniform class. A split that
/// buys no extra stream is a pure loss — it only confines the airs above the smaller class — but
/// collapsing costs recursive streams (uniform classes are larger), so a genuine outlier keeps its split.
const UNIFORM_COLLAPSE_RATIO: f64 = 1.25;

/// Target regular streams per large stream. The carve adds whichever class is behind this ratio,
/// so neither the outlier nor the bulk airs end up with a single stream on a card that could
/// fund both.
const REGULARS_PER_LARGE: usize = 2;

/// Minimum useful class size: the largest air besides the outlier that defines the large
/// class. `class_floor` (compressor size) collapses to 0 without aggregation, letting the
/// count-maximizing carve buy classes too small to host most airs. Returns 0 if there is no second
/// distinct size.
fn non_outlier_floor(basic_sizes_desc: &[usize]) -> usize {
    let mut distinct = basic_sizes_desc.to_vec();
    distinct.sort_unstable_by(|a, b| b.cmp(a));
    distinct.dedup();
    distinct.get(1).copied().unwrap_or(0)
}

impl StreamLayout {
    pub fn n_basic_streams(&self) -> usize {
        self.basic.iter().map(|c| c.count).sum()
    }

    /// Buffer sizes for the basic streams, in stream order (largest class first).
    pub fn basic_stream_sizes(&self) -> Vec<usize> {
        self.basic.iter().flat_map(|c| std::iter::repeat_n(c.size, c.count)).collect()
    }
}

/// Streams bought for one (large, regular) pairing: counts plus the budget still unspent.
#[derive(Clone, Copy)]
struct Carve {
    n_large: usize,
    n_regular: usize,
    remaining: usize,
}

/// Grow both classes from the mandatory first large stream, always adding whichever is behind the
/// [`REGULARS_PER_LARGE`] equilibrium. A class that no longer fits yields to the other rather than
/// ending the carve — that turns 2L+2R into 1L+3R on a card that cannot afford a second large.
fn grow(budget: usize, large: usize, regular: usize, reserve: usize, max_basic_streams: usize) -> Carve {
    let mut carve = Carve { n_large: 1, n_regular: 0, remaining: budget - large };
    while carve.n_large + carve.n_regular < max_basic_streams {
        let regular_first = carve.n_regular < REGULARS_PER_LARGE * carve.n_large;
        // `regular == 0` means no candidate below the large size; those entries never fit.
        let wanted = if regular_first { [regular, large] } else { [large, regular] };
        let affordable = carve.remaining.saturating_sub(reserve);
        match wanted.iter().find(|&&s| s > 0 && s <= affordable) {
            // `regular` is strictly below `large` (distinct sizes), so the size identifies the class.
            Some(&size) => {
                carve.remaining -= size;
                if size == large {
                    carve.n_large += 1;
                } else {
                    carve.n_regular += 1;
                }
            }
            None => break,
        }
    }
    carve
}

/// Cut `budget` (in field elements, per GPU) into a large and a regular stream class.
///
/// `basic_sizes_desc` is the per-air basic requirement, largest first; duplicates are harmless.
/// `class_floor` is the smallest a basic class may be: it must hold anything besides a basic proof that
/// can land on one of these streams — today the compressor/vadcop_final launches that share them.
/// `recursive_size` sizes the recursive1/recursive2 class.
///
/// The large size is fixed by the biggest air. The regular size is chosen by carving once per
/// candidate air requirement and keeping the one that buys the most basic streams, ties going to the
/// larger size — a smaller regular exiles every air above it, so it must earn its place.
///
/// Basic streams are funded first — they are the only home for basics *and* compressors — and
/// aggregation takes the remainder, subject only to [`RECURSIVE_STREAM_FLOOR`].
///
/// Returns `None` when the budget cannot even hold one stream for the largest air.
pub fn plan_stream_layout(
    budget: usize,
    basic_sizes_desc: &[usize],
    class_floor: usize,
    recursive_size: usize,
    max_basic_streams: usize,
    max_recursive_streams: usize,
) -> Option<StreamLayout> {
    // Raise the floor so a class can host the bulk of the airs and not just a compressor (see
    // `non_outlier_floor`). Only ever a retry: a raised floor that costs the second basic stream
    // is worse than a small class, so that carve is discarded for the unraised one.
    let raised = class_floor.max(non_outlier_floor(basic_sizes_desc));
    if raised > class_floor {
        let covered =
            carve_classes(budget, basic_sizes_desc, raised, recursive_size, max_basic_streams, max_recursive_streams);
        if let Some(layout) = covered {
            if layout.n_basic_streams() >= 2 {
                return Some(layout);
            }
        }
    }
    carve_classes(budget, basic_sizes_desc, class_floor, recursive_size, max_basic_streams, max_recursive_streams)
}

/// The carve itself, at a fixed `class_floor`.
fn carve_classes(
    budget: usize,
    basic_sizes_desc: &[usize],
    class_floor: usize,
    recursive_size: usize,
    max_basic_streams: usize,
    max_recursive_streams: usize,
) -> Option<StreamLayout> {
    if max_basic_streams == 0 {
        return None;
    }

    // The candidate sizes are the distinct air requirements, each raised to the floor so either class
    // can also take a compressor and any contributions commit. Largest first.
    let mut sizes: Vec<usize> = basic_sizes_desc.iter().map(|&s| s.max(class_floor)).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.dedup();

    // The large class must hold anything that can land on a non-recursive stream.
    let &large = sizes.first()?;
    if large == 0 || budget < large {
        return None;
    }

    let reserve = recursive_size * RECURSIVE_STREAM_FLOOR.min(max_recursive_streams);
    // Most streams wins; the size in the key breaks ties toward the larger regular class.
    // Uniform: one class, sized to the largest air. Also the fallback when there is only one size.
    let uniform = grow(budget, large, 0, reserve, max_basic_streams);
    let (regular, carve) = sizes
        .iter()
        .skip(1)
        .map(|&regular| (regular, grow(budget, large, regular, reserve, max_basic_streams)))
        .chain(std::iter::once((0, uniform)))
        .max_by_key(|(size, c)| (c.n_large + c.n_regular, *size))
        .expect("the chained fallback is always a candidate");

    // No extra stream and sizes close: collapse, so nothing is confined to a subset of the streams.
    let split_streams = carve.n_large + carve.n_regular;
    let (regular, carve) =
        if regular > 0 && uniform.n_large >= split_streams && large as f64 <= regular as f64 * UNIFORM_COLLAPSE_RATIO {
            (0, uniform)
        } else {
            (regular, carve)
        };

    let mut basic = vec![StreamClass { size: large, count: carve.n_large }];
    if carve.n_regular > 0 {
        basic.push(StreamClass { size: regular, count: carve.n_regular });
    }

    let mut remaining = carve.remaining;
    let n_recursive = match recursive_size {
        0 => 0,
        size => max_recursive_streams.min(remaining / size),
    };
    remaining -= n_recursive * recursive_size;

    Some(StreamLayout { basic, recursive: StreamClass { size: recursive_size, count: n_recursive }, unused: remaining })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Measured from a zisk proving key, GB, largest first. Keccakf is the outlier: 14.57 vs 6.45.
    const ZISK_BASIC_GB: &[f64] = &[14.57, 6.45, 6.26, 5.82, 5.57, 4.74, 4.52, 4.43, 3.82, 2.63, 1.47, 0.75];
    const ZISK_COMPRESSOR_GB: f64 = 6.24;
    const ZISK_RECURSIVE_GB: f64 = 1.33;
    const ZISK_BUDGET_GB: f64 = 25.23;

    fn gb(v: f64) -> usize {
        (v * (1 << 30) as f64 / 8.0) as usize
    }

    fn zisk_layout(budget_gb: f64, max_basic: usize, max_recursive: usize) -> StreamLayout {
        let sizes: Vec<usize> = ZISK_BASIC_GB.iter().copied().map(gb).collect();
        plan_stream_layout(
            gb(budget_gb),
            &sizes,
            gb(ZISK_COMPRESSOR_GB),
            gb(ZISK_RECURSIVE_GB),
            max_basic,
            max_recursive,
        )
        .expect("zisk budget holds the largest air")
    }

    /// The carve is two sizes at most, whatever the budget or the air distribution.
    #[test]
    fn the_carve_never_has_more_than_two_sizes() {
        let uneven = [gb(14.5), gb(12.0), gb(9.1), gb(7.0), gb(3.2)];
        for budget in [20.0, 26.15, 33.0, 40.0, 60.0, 80.0, 200.0] {
            for sizes in [&ZISK_BASIC_GB.iter().copied().map(gb).collect::<Vec<_>>()[..], &uneven[..]] {
                let layout =
                    plan_stream_layout(gb(budget), sizes, gb(ZISK_COMPRESSOR_GB), gb(ZISK_RECURSIVE_GB), 16, 10);
                let Some(layout) = layout else { continue };
                assert!(layout.basic.len() <= 2, "budget {budget}: {:?}", layout.basic);
            }
        }
    }

    /// Neither class runs away with the card: streams are added 1L, 1L+1R, 1L+2R, 2L+2R, … Driven
    /// by the stream cap on a budget large enough that only the cap binds.
    #[test]
    fn classes_grow_in_equilibrium() {
        let expected = [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3), (2, 4), (3, 4)];
        for (max_basic, (n_large, n_regular)) in expected.into_iter().enumerate() {
            let layout = zisk_layout(120.0, max_basic + 1, 10);
            let got = (layout.basic[0].count, layout.basic.get(1).map_or(0, |c| c.count));
            assert_eq!(got, (n_large, n_regular), "max_basic {}: {:?}", max_basic + 1, layout.basic);
        }
    }

    /// Equilibrium wants a large stream next, but the remainder cannot fund one: the carve keeps
    /// buying regulars rather than stopping there.
    #[test]
    fn a_class_that_no_longer_fits_yields_to_the_other() {
        let layout = zisk_layout(70.0, 16, 10);
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 2 }, StreamClass { size: gb(6.45), count: 6 }]
        );
    }

    /// 6.26 GB regulars would fit just as many streams as 6.45 GB ones, and picking them would
    /// exile the 6.45 air to the large streams for nothing. Ties go to the larger size.
    #[test]
    fn a_smaller_regular_is_only_taken_when_it_buys_a_stream() {
        for budget in [ZISK_BUDGET_GB, 33.0, 40.0, 60.0, 80.0] {
            let layout = zisk_layout(budget, 16, 10);
            assert_eq!(layout.basic[1].size, gb(6.45), "budget {budget}: {:?}", layout.basic);
        }
    }

    /// The zisk budget buys a second class with no flag: the outlier gets its own, the rest share one.
    #[test]
    fn basic_classes_are_funded_before_recursive_streams() {
        let layout = zisk_layout(ZISK_BUDGET_GB, 16, 10);
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 1 }, StreamClass { size: gb(6.45), count: 1 }]
        );
        assert!(layout.basic[1].size >= gb(ZISK_COMPRESSOR_GB), "every basic class must hold a compressor");
        assert!(layout.recursive.count >= 3, "aggregation still gets the remainder: {:?}", layout.recursive);
    }

    /// Preloaded const trees push every non-outlier air below the compressor floor, so the floor
    /// sizes the second class. Matches zisk1: 14.57 + 6.24, four recursive.
    #[test]
    fn preloaded_sizes_put_the_second_class_at_the_compressor_floor() {
        let sizes = [gb(14.57), gb(5.9), gb(3.8), gb(2.6), gb(0.75)];
        let layout =
            plan_stream_layout(gb(26.15), &sizes, gb(ZISK_COMPRESSOR_GB), gb(ZISK_RECURSIVE_GB), 16, 10).unwrap();
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 1 }, StreamClass { size: gb(ZISK_COMPRESSOR_GB), count: 1 }]
        );
        assert_eq!(layout.recursive.count, 4);
    }

    /// The floor must never cost this card its second basic class — that undoes the whole measured
    /// gain (26.7s -> 21.5s) and the margin is thin, so pin it.
    #[test]
    fn the_recursive_floor_does_not_cost_the_second_class() {
        let sizes = [gb(14.57), gb(5.9), gb(3.8), gb(2.6), gb(0.75)];
        let layout =
            plan_stream_layout(gb(26.15), &sizes, gb(ZISK_COMPRESSOR_GB), gb(ZISK_RECURSIVE_GB), 16, 10).unwrap();
        assert_eq!(layout.n_basic_streams(), 2, "second class lost to the floor: {:?}", layout.basic);
        assert_eq!(layout.recursive.count, 4, "floor is a minimum, not a cap: {:?}", layout.recursive);
    }

    /// The floor is what stops a large budget from spending everything on classes.
    #[test]
    fn a_large_budget_still_gets_recursive_streams() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23), gb(3.8), gb(1.5)];
        for budget in [33.0, 40.0, 80.0] {
            let layout = plan_stream_layout(gb(budget), &sizes, gb(6.24), gb(ZISK_RECURSIVE_GB), 16, 10).unwrap();
            assert!(
                layout.recursive.count >= RECURSIVE_STREAM_FLOOR,
                "budget {budget}: {:?} / {:?}",
                layout.basic,
                layout.recursive
            );
        }
    }

    /// The recursive cap bounds the remainder: excess goes unused, not into extra streams.
    #[test]
    fn the_recursive_cap_still_bounds_the_remainder() {
        let layout = zisk_layout(ZISK_BUDGET_GB, 16, 1);
        assert_eq!(layout.recursive.count, 1);
        assert!(layout.unused > 0);
    }

    /// A card with room to spare keeps cutting classes. Funding basic first means a large card can
    /// leave aggregation nothing, so cap `max_basic_streams` to hold streams back.
    #[test]
    fn a_larger_budget_cuts_more_classes() {
        let layout = zisk_layout(80.0, 16, 8);
        assert!(layout.n_basic_streams() >= 4, "got {:?}", layout.basic);

        // Holding the basic classes down is what leaves room for aggregation streams.
        let capped = zisk_layout(80.0, 3, 8);
        assert_eq!(capped.n_basic_streams(), 3);
        assert_eq!(capped.recursive.count, 8);
    }

    /// Classes never drop below the compressor floor, even where a small air would fit.
    #[test]
    fn classes_never_fall_below_the_compressor_floor() {
        let layout = zisk_layout(ZISK_BUDGET_GB, 16, 0);
        for class in &layout.basic {
            assert!(class.size >= gb(ZISK_COMPRESSOR_GB), "class {class:?} cannot host a compressor");
        }
    }

    /// Aggregation off: `class_floor` is 0 (no compressor shares these streams), so only the
    /// non-outlier floor stops the carve from buying the smallest air on the list. Without it the
    /// card got 1 x 14.57 + 18 x 0.75 GB — a class hosting 6 of 42 airs — and every real proof
    /// serialized on the single large stream.
    #[test]
    fn a_no_aggregation_carve_still_hosts_the_airs() {
        let sizes: Vec<usize> = ZISK_BASIC_GB.iter().copied().map(gb).collect();
        let layout = plan_stream_layout(gb(29.05), &sizes, 0, 0, 20, 0).expect("budget holds the largest air");
        assert!(layout.n_basic_streams() >= 2, "no second stream to overlap with: {:?}", layout.basic);
        assert_eq!(layout.basic[1].size, gb(6.45), "second class cannot host the bulk airs: {:?}", layout.basic);
        assert!(
            layout.basic.iter().all(|c| c.size >= gb(6.45)),
            "a class below the non-outlier floor survived: {:?}",
            layout.basic
        );
    }

    /// The raised floor is a preference, not a mandate: where it would cost the second basic class
    /// (the one thing worth more than coverage) the carve falls back to the unraised floor.
    #[test]
    fn the_coverage_floor_never_costs_the_second_class() {
        // 12.0 is the non-outlier floor, and 14.5 + 12.0 does not fit — the fallback must engage.
        let sizes = [gb(14.5), gb(12.0), gb(3.2), gb(3.2), gb(3.2)];
        let layout = plan_stream_layout(gb(26.0), &sizes, 0, 0, 16, 0).expect("budget holds the largest air");
        assert!(layout.n_basic_streams() >= 2, "fallback did not engage: {:?}", layout.basic);
    }

    /// The carve is bounded by the stream cap, not only by memory.
    #[test]
    fn the_basic_stream_cap_is_respected() {
        let layout = zisk_layout(80.0, 2, 0);
        assert_eq!(layout.n_basic_streams(), 2);
    }

    /// Anything left over must be too small to have funded another class.
    #[test]
    fn leftover_is_smaller_than_any_further_class() {
        let layout = zisk_layout(ZISK_BUDGET_GB, 16, 3);
        assert!(layout.unused < gb(ZISK_COMPRESSOR_GB));
    }

    /// Prints the carve across budgets for a second near-outlier (14.5/12/6.23). Inspection only:
    /// `cargo test -p proofman-common carve_shape -- --nocapture`.
    #[test]
    fn carve_shape_with_a_second_near_outlier() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23), gb(3.8), gb(1.5)];
        for budget in [20.0, 26.15, 33.0, 40.0, 80.0] {
            let layout = plan_stream_layout(gb(budget), &sizes, gb(6.24), gb(ZISK_RECURSIVE_GB), 16, 10).unwrap();
            let classes: Vec<String> = layout
                .basic
                .iter()
                .map(|c| format!("{} x {:.2}", c.count, c.size as f64 * 8.0 / (1 << 30) as f64))
                .collect();
            println!(
                "budget {budget:>6.2} -> basic [{}]  recursive {}  unused {:.2}",
                classes.join(" + "),
                layout.recursive.count,
                layout.unused as f64 * 8.0 / (1 << 30) as f64,
            );
        }
    }

    /// A near-outlier the budget cannot give a regular class to falls back to the large streams.
    /// The candidate search must keep looking past it instead of settling for one class.
    #[test]
    fn an_air_too_big_for_the_regular_class_falls_back_to_the_large_one() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23)];
        let layout = plan_stream_layout(gb(26.15), &sizes, gb(6.24), gb(ZISK_RECURSIVE_GB), 16, 10).unwrap();

        // 14.5 + 12 overruns the budget, so the 12 GB air keeps the large class as its only home.
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(6.24), count: 1 }]
        );
        // The 6.23 GB air is below the floor, so its class is the floor -- never smaller.
        assert!(layout.basic[1].size >= gb(6.23));
        assert!(layout.recursive.count >= 3);
    }

    /// The measured ZisK key: 7.42 GB largest air against a 6.24 GB compressor floor (ratio 1.19). The
    /// split `1x7.42 + 2x6.24` and uniform `3x7.42` buy the same three streams, so the split bought
    /// nothing while confining the 7.42 air to one stream for the whole run. Collapse to uniform.
    #[test]
    fn a_split_that_buys_no_stream_collapses_to_uniform() {
        let sizes: Vec<usize> =
            [7.42, 5.99, 5.82, 5.57, 5.31, 4.52, 4.43, 3.82, 2.60, 1.47, 0.75].iter().map(|&g| gb(g)).collect();
        let layout = plan_stream_layout(gb(26.18), &sizes, gb(6.24), gb(1.33), 16, 10).unwrap();
        assert_eq!(layout.basic, vec![StreamClass { size: gb(7.42), count: 3 }], "should be uniform");
        // Every air can now use every stream -- that is the whole point.
        assert!(sizes.iter().all(|&s| s <= layout.basic[0].size));
        // Uniform costs recursive streams; the trade must stay visible, not silent.
        assert!(layout.recursive.count >= 1, "aggregation must keep at least the floor");
    }

    /// A genuine outlier keeps its own class: 14.57 against 6.45 is ratio 2.3, far outside the collapse
    /// window, and there the split really does buy streams.
    #[test]
    fn a_real_outlier_still_gets_its_own_class() {
        let layout = zisk_layout(70.0, 16, 10);
        assert_eq!(layout.basic.len(), 2, "outlier must not be collapsed: {:?}", layout.basic);
        assert_eq!(layout.basic[0].size, gb(14.57));
    }

    /// A budget under the largest air is a configuration error, not a silently truncated carve.
    #[test]
    fn too_small_a_budget_is_rejected() {
        let sizes = [gb(14.57)];
        assert!(plan_stream_layout(gb(10.0), &sizes, gb(6.24), gb(1.33), 16, 8).is_none());
    }

    /// Class 0 is sized by whichever of the largest air and the compressor floor is bigger.
    #[test]
    fn first_class_covers_both_the_largest_air_and_the_compressor() {
        let layout = plan_stream_layout(gb(40.0), &[gb(2.0)], gb(9.0), gb(1.0), 16, 0).unwrap();
        assert_eq!(layout.basic[0].size, gb(9.0));

        let layout = plan_stream_layout(gb(40.0), &[gb(12.0)], gb(9.0), gb(1.0), 16, 0).unwrap();
        assert_eq!(layout.basic[0].size, gb(12.0));
    }

    /// Stream order must match the class order the device carve walks.
    #[test]
    fn stream_sizes_expand_classes_largest_first() {
        let layout = StreamLayout {
            basic: vec![StreamClass { size: 9, count: 1 }, StreamClass { size: 4, count: 2 }],
            recursive: StreamClass { size: 1, count: 3 },
            unused: 0,
        };
        assert_eq!(layout.basic_stream_sizes(), vec![9, 4, 4]);
        assert_eq!(layout.n_basic_streams(), 3);
    }
}
