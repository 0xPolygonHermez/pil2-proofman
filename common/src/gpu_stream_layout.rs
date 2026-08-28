//! Cutting the per-GPU aux-trace budget into stream size classes.
//!
//! Sizing every stream to the largest air buys concurrency at that air's price, so the budget is cut
//! into two classes: *large*, sized to the biggest air so nothing is homeless, and *regular*, sized
//! to a smaller air. A proof runs on any stream at least as large as it needs.
//!
//! Which airs a proof instantiates, and how often, is not known this early — a proving key carries
//! airs a given run never proves — so nothing here weighs a carve by the work it would run.
//!
//! Every class is floored at the largest compressor / vadcop_final / recursive launch, since those
//! share these streams with basics (see `RecursiveScheduler::next_nonrecursive`).

/// One aux-trace size class: `count` streams of `size` field elements each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamClass {
    pub size: usize,
    pub count: usize,
}

/// How one GPU's aux-trace budget is cut up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamLayout {
    /// Large then regular, at most two entries. Never empty, and `basic[0].size` covers the largest
    /// air and the class floor, so no launch is left without a home.
    pub basic: Vec<StreamClass>,
    /// Aggregation-only streams; `count` is 0 when the pool does not pay for itself.
    pub recursive: StreamClass,
    /// Workers that take only aggregation work, or rec1/rec2 waits for the basic queue to drain.
    pub aggregation_workers: usize,
    /// Budget the class sizes and the stream caps leave unspendable.
    pub unused: usize,
}

/// Kept funded before any basic stream beyond the first, which would otherwise spend the last byte.
/// A floor, not a reservation: the remainder still becomes recursive streams.
const RECURSIVE_STREAM_FLOOR: usize = 1;

/// A class under `large / FLEET_FLOOR` is dust: it earns one spare stream, never a fleet. Fitted
/// between two measured shapes — 3.2 GB against 14.5 keeps its fleet, 1.0 GB against 14.5 does not.
const FLEET_FLOOR: usize = 8;

impl StreamLayout {
    pub fn n_basic_streams(&self) -> usize {
        self.basic.iter().map(|c| c.count).sum()
    }

    /// Buffer sizes for the basic streams, in stream order (largest class first).
    pub fn basic_stream_sizes(&self) -> Vec<usize> {
        self.basic.iter().flat_map(|c| std::iter::repeat_n(c.size, c.count)).collect()
    }
}

/// How many streams of `size` fit in `room`, at most `cap`. A size of 0 means no such class.
fn n_fit(room: usize, size: usize, cap: usize) -> usize {
    room.checked_div(size).map_or(0, |n| cap.min(n))
}

/// Buy streams of the two sizes out of `room`: at most `cap` in total, `cap_regular` regular ones.
///
/// One of each first so neither class starves, then the leftover buys the best thing it still
/// affords. The caller guarantees `large + regular` fits, so a large stream is always bought.
fn buy_streams(room: usize, large: usize, regular: usize, cap: usize, cap_regular: usize) -> (usize, usize) {
    debug_assert!(0 < regular && regular < large, "classes must be distinct sizes: {regular} vs {large}");
    let pairs = (room / (large + regular)).min(cap / 2).min(cap_regular);
    let (mut n_large, mut n_regular) = (pairs, pairs);
    let mut left = room - pairs * (large + regular);

    loop {
        let slot = n_large + n_regular < cap;
        // Best buy first (it hosts every air), then the cheaper half-measure — except that emptying
        // the class while a regular is affordable would strand its price.
        if slot && left >= large {
            left -= large;
            n_large += 1;
        } else if slot && n_regular < cap_regular && left >= regular && (2 * regular <= large || n_regular == 1) {
            left -= regular;
            n_regular += 1;
        } else if n_regular > 0 && left >= large - regular {
            left -= large - regular;
            n_regular -= 1;
            n_large += 1;
        } else {
            return (n_large, n_regular);
        }
    }
}

/// Cut `budget` (in field elements, per GPU) into a large and a regular stream class.
///
/// `basic_sizes` is the per-air basic requirement in any order; duplicates weigh twice.
/// `class_floor` is the smallest a basic class may be — it must hold anything besides a basic proof
/// that can land on one of these streams, today the compressor/vadcop_final launches that share them.
/// It is raised to `recursive_size` (which sizes the recursive1/recursive2 class), since those fall
/// back to these streams too.
///
/// The large class is sized by the biggest air. The regular class prefers the biggest air no larger
/// than half of it that earns its streams: shrinking from `large` to `s` frees `large - s`, which
/// only pays for itself if it buys another `s` stream. Failing that it takes the biggest air that
/// fits beside a large stream, which [`buy_streams`] trades back up when that is the better buy.
///
/// Aggregation gets streams of its own only while `recursive_size` is under half of `class_floor`,
/// where those bytes buy more than one restricted stream per basic stream's price; above it the
/// budget buys basic streams and `max_recursive_streams` bounds aggregation *workers* instead.
/// Basic streams are funded first either way, subject only to [`RECURSIVE_STREAM_FLOOR`].
///
/// Returns `None` when the budget cannot even hold one stream for the largest air.
pub fn plan_stream_layout(
    budget: usize,
    basic_sizes: &[usize],
    class_floor: usize,
    recursive_size: usize,
    max_basic_streams: usize,
    max_recursive_streams: usize,
) -> Option<StreamLayout> {
    if max_basic_streams == 0 {
        return None;
    }
    // A recursive launch falls back to a basic stream whenever the pool is busy or gone, so the floor
    // must cover it — enforced here rather than assumed of the caller.
    let class_floor = class_floor.max(recursive_size);
    let mut sizes: Vec<usize> = basic_sizes.iter().map(|&s| s.max(class_floor)).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.dedup();

    // The large class must hold anything that can land on a non-recursive stream.
    let &large = sizes.first()?;
    if large == 0 || budget < large {
        return None;
    }

    // Under 2x an aggregation-only stream costs about what a basic one does and runs far less; every
    // basic stream holds the launch anyway, so the bytes buy those and priority moves to the workers.
    let pool_is_redundant = class_floor < 2 * recursive_size;
    let pool_size = if pool_is_redundant { 0 } else { recursive_size };

    // The first large stream ignores the recursive floor — without it nothing can run at all.
    let room = budget.saturating_sub(pool_size * RECURSIVE_STREAM_FLOOR.min(max_recursive_streams));

    // A class must serve a strict majority of the airs, which is exactly "reaches the median air".
    // Strict, so a key growing precompiles to an exact split does not buy a fleet.
    let bulk_floor = {
        let mut asc = basic_sizes.to_vec();
        asc.sort_unstable();
        asc[asc.len() / 2]
    };
    // Sizes that still leave room for the large stream they run beside, largest first.
    let candidates = || sizes.iter().skip(1).copied().filter(|&s| s > 0 && large + s <= room);

    // Dust is never preferred: it falls through to the biggest class that fits instead.
    let earns_a_fleet = |s: usize| s >= bulk_floor && s * FLEET_FLOOR >= large;
    let candidate = candidates().find(|&s| 2 * s <= large && earns_a_fleet(s)).or_else(|| candidates().next());
    let (n_large, regular, n_regular) = match candidate {
        Some(regular) => {
            let cap_regular = if earns_a_fleet(regular) { max_basic_streams } else { 1 };
            let (n_large, n_regular) = buy_streams(room, large, regular, max_basic_streams, cap_regular);
            (n_large, regular, n_regular)
        }
        // Nothing fits beside a large stream, so the floor of one is that stream: the recursive floor
        // may have taken the room for it.
        None => (n_fit(room, large, max_basic_streams).max(1), 0, 0),
    };

    // The candidate is picked before the trade-ups run, so a carve left all large looks once more:
    // its change may still fund a smaller air.
    let (regular, n_regular) = if n_regular == 0 && n_large < max_basic_streams {
        let left = room.saturating_sub(n_large * large);
        candidates().find(|&s| s <= left).map_or((regular, 0), |spare| (spare, 1))
    } else {
        (regular, n_regular)
    };

    let mut basic = vec![StreamClass { size: large, count: n_large }];
    if n_regular > 0 {
        basic.push(StreamClass { size: regular, count: n_regular });
    }

    let spent = n_large * large + n_regular * regular;
    debug_assert!(spent <= budget, "carved {spent} out of a {budget} budget");
    let remaining = budget - spent;
    let n_pool = n_fit(remaining, pool_size, max_recursive_streams);
    // Dedicated streams get one worker each. Without any — the pool was dropped, or the budget could
    // not fund one — aggregation shares the basic streams, which costs a thread and not memory; but
    // never every stream at once, or the basic queue stalls behind them.
    let shared_workers = max_recursive_streams.min((n_large + n_regular).saturating_sub(1).max(1));
    Some(StreamLayout {
        basic,
        recursive: StreamClass { size: pool_size, count: n_pool },
        aggregation_workers: match n_pool {
            0 if recursive_size > 0 => shared_workers,
            n => n,
        },
        unused: remaining - n_pool * pool_size,
    })
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

    /// The air requirements arrive in whatever order the caller holds them; the carve must not
    /// depend on it, since the classes are picked from them largest first.
    #[test]
    fn the_input_order_of_the_airs_does_not_matter() {
        let sorted: Vec<usize> = ZISK_BASIC_GB.iter().copied().map(gb).collect();
        // Ascending, and an arbitrary rotation of it: the outlier neither first nor last.
        let ascending: Vec<usize> = sorted.iter().rev().copied().collect();
        let mut shuffled = sorted.clone();
        shuffled.rotate_left(5);

        for budget in [20.0, 26.15, 33.0, 40.0, 60.0, 80.0, 200.0] {
            let plan = |sizes: &[usize]| {
                plan_stream_layout(gb(budget), sizes, gb(ZISK_COMPRESSOR_GB), gb(ZISK_RECURSIVE_GB), 16, 10)
            };
            let expected = plan(&sorted);
            assert_eq!(plan(&ascending), expected, "budget {budget}: ascending input carved differently");
            assert_eq!(plan(&shuffled), expected, "budget {budget}: shuffled input carved differently");
        }
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

    /// Where the cap binds before memory every stream should be large: the pairs trade back up.
    #[test]
    fn a_roomy_budget_goes_uniform() {
        let layout = zisk_layout(120.0, 4, 10);
        assert_eq!(layout.basic, vec![StreamClass { size: gb(14.57), count: 4 }], "{:?}", layout.basic);
    }

    /// A budget that funds three of each gets three of each: the outlier is never flattened away.
    #[test]
    fn the_classes_grow_together() {
        let layout = zisk_layout(70.0, 16, 10);
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 3 }, StreamClass { size: gb(6.45), count: 3 }]
        );
    }

    /// What the pairs leave goes to whichever class still fits, largest first.
    #[test]
    fn the_remainder_buys_the_largest_stream_that_still_fits() {
        let layout = zisk_layout(40.0, 16, 10);
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 2 }, StreamClass { size: gb(6.45), count: 1 }],
            "a large stream fitted in the remainder: {:?}",
            layout.basic
        );

        let layout = zisk_layout(33.0, 16, 10);
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 1 }, StreamClass { size: gb(6.45), count: 2 }],
            "only a regular one fitted: {:?}",
            layout.basic
        );
    }

    /// The regular size is a property of the key, not the budget: 6.26 GB fits as many streams as
    /// 6.45 and exiles the 6.45 air for nothing. The old carve drifted with the budget; this must not.
    #[test]
    fn the_regular_size_does_not_drift_with_the_budget() {
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

    /// Preloaded const trees push every non-outlier air below the floor, so the floor sizes the second
    /// class. Matches zisk1, whose measured gain rests on that class surviving the recursive floor.
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

    /// Aggregation off, so only the bulk rule keeps the class off the smallest air: a carve free to
    /// take it bought 1 x 14.57 + 18 x 0.75 GB, hosting 6 of 42 airs.
    #[test]
    fn a_no_aggregation_carve_still_hosts_the_airs() {
        let sizes: Vec<usize> = ZISK_BASIC_GB.iter().copied().map(gb).collect();
        let layout = plan_stream_layout(gb(29.05), &sizes, 0, 0, 20, 0).expect("budget holds the largest air");
        assert!(layout.n_basic_streams() >= 2, "no second stream to overlap with: {:?}", layout.basic);
        assert_eq!(layout.basic[1].size, gb(6.45), "second class cannot host the bulk airs: {:?}", layout.basic);
    }

    /// Coverage is worth a lot, but never the second basic stream: where the budget cannot pair the
    /// largest air with the class just behind it, the small airs still define one the bulk can use.
    #[test]
    fn coverage_never_costs_the_second_class() {
        // 14.5 + 12.0 does not fit, so the only classes left are the ones the small airs define.
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

    /// 14.5 + 12.0 overruns the budget, so what is left funds a class hosting 1 of 3 airs: one spare.
    #[test]
    fn an_air_too_big_for_the_regular_class_falls_back_to_the_large_one() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23)];
        let layout = plan_stream_layout(gb(26.15), &sizes, gb(6.24), gb(ZISK_RECURSIVE_GB), 16, 10).unwrap();

        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(6.24), count: 1 }]
        );
        // The 6.23 GB air is below `class_floor`, so its class is the floor -- never smaller.
        assert!(layout.basic[1].size >= gb(6.23));
        assert!(layout.recursive.count >= 3);
    }

    /// 7.42 GB largest air against a 6.24 GB floor: the 6.24 class is bought and traded straight back
    /// up for 1.18 GB a stream, since the split buys no more streams than uniform.
    #[test]
    fn a_split_that_buys_no_stream_loses_to_uniform() {
        let sizes: Vec<usize> =
            [7.42, 5.99, 5.82, 5.57, 5.31, 4.52, 4.43, 3.82, 2.60, 1.47, 0.75].iter().map(|&g| gb(g)).collect();
        let layout = plan_stream_layout(gb(26.18), &sizes, gb(6.24), gb(1.33), 16, 10).unwrap();
        assert_eq!(layout.basic, vec![StreamClass { size: gb(7.42), count: 3 }], "should be uniform");
        // Every air can now use every stream -- that is the whole point.
        assert!(sizes.iter().all(|&s| s <= layout.basic[0].size));
        // Uniform costs recursive streams; the trade must stay visible, not silent.
        assert!(layout.recursive.count >= 1, "aggregation must keep at least the floor");
    }

    /// Below half of 14.57 there is only dust, so the class goes above half to the one that earns it.
    #[test]
    fn a_gap_below_half_puts_the_regular_class_above_it() {
        let sizes: Vec<usize> = [14.57f64, 8.0, 0.75].iter().copied().map(gb).collect();
        for (budget, want) in [(26.0, (1, 1)), (33.0, (1, 2)), (60.0, (3, 2)), (80.0, (4, 2))] {
            let layout = plan_stream_layout(gb(budget), &sizes, 0, 0, 20, 0).expect("budget holds the largest air");
            let (n_large, n_regular) = want;
            assert_eq!(
                layout.basic,
                vec![StreamClass { size: gb(14.57), count: n_large }, StreamClass { size: gb(8.0), count: n_regular }],
                "budget {budget}: {:?}",
                layout.basic
            );
        }
    }

    /// Three of these five airs outrun anything the leftover funds: one spare is all it is worth.
    #[test]
    fn a_class_the_bulk_cannot_use_is_worth_one_spare_stream() {
        let sizes: Vec<usize> = [14.5f64, 12.0, 12.0, 12.0, 1.0].iter().copied().map(gb).collect();
        let layout = plan_stream_layout(gb(26.0), &sizes, 0, 0, 20, 0).expect("budget holds the largest air");
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(1.0), count: 1 }],
            "{:?}",
            layout.basic
        );
    }

    /// Trading the last regular up ends at 2 x 14.5 with 11 GB dead; buying it is a 3rd stream.
    #[test]
    fn a_buyable_regular_is_bought_before_the_last_one_is_traded_up() {
        let sizes: Vec<usize> = [14.5f64, 12.0, 12.0, 12.0, 1.0].iter().copied().map(gb).collect();
        let layout = plan_stream_layout(gb(40.0), &sizes, 0, 0, 20, 0).expect("budget holds the largest air");
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(12.0), count: 2 }],
            "{:?}",
            layout.basic
        );
    }

    /// Tiny airs reaching exactly half the count must not flip the carve into a fleet of 1 GB streams.
    #[test]
    fn added_precompiles_do_not_flip_the_carve_to_a_tiny_fleet() {
        let sizes: Vec<usize> = [14.5f64, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0].iter().copied().map(gb).collect();
        let layout = plan_stream_layout(gb(40.0), &sizes, 0, 0, 20, 0).expect("budget holds the largest air");
        assert_eq!(layout.basic[1].size, gb(12.0), "a tiny fleet was bought: {:?}", layout.basic);
    }

    /// Cap-bound: 12.96 GB left over makes one regular large rather than sit unspent.
    #[test]
    fn a_cap_bound_carve_trades_its_regulars_up() {
        let layout = zisk_layout(55.0, 4, 0);
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.57), count: 3 }, StreamClass { size: gb(6.45), count: 1 }],
            "{:?}",
            layout.basic
        );
        assert!(layout.unused < gb(14.57 - 6.45), "another trade-up was affordable: {layout:?}");
    }

    /// A measured zisk proving key, largest first, GB. Two thirds are precompiles a given proof never
    /// instantiates — the reason no rule here may count a key's airs as work.
    const ZISK_KEY_GB: &[f64] = &[
        12.81, 5.99, 5.82, 5.69, 5.32, 4.52, 4.52, 4.43, 4.43, 4.32, 4.19, 4.07, 3.99, 3.89, 3.82, 3.65, 3.63, 3.63,
        3.24, 3.09, 2.94, 2.91, 2.88, 2.60, 2.60, 2.47, 2.36, 2.35, 2.28, 2.13, 2.08, 1.94, 1.92, 1.79, 1.79, 1.65,
        1.47, 1.43, 1.23, 1.17, 1.11, 0.74, 0.63,
    ];

    /// The regular class takes the top of the band it serves, whatever a smaller one would buy:
    /// work-scoring bought `1 x 12.81 + 3 x 4.52`, confining the 5.32-5.99 GB airs to one stream.
    #[test]
    fn the_regular_class_is_not_shaved_below_the_band_it_serves() {
        let sizes: Vec<usize> = ZISK_KEY_GB.iter().copied().map(gb).collect();
        for (budget, want) in [(28.21, (1, 2)), (40.0, (2, 2)), (60.0, (3, 3))] {
            let layout = plan_stream_layout(gb(budget), &sizes, 0, 0, 16, 0).expect("budget holds the largest air");
            let (n_large, n_regular) = want;
            assert_eq!(
                layout.basic,
                vec![StreamClass { size: gb(12.81), count: n_large }, StreamClass { size: gb(5.99), count: n_regular }],
                "budget {budget}: {:?}",
                layout.basic
            );
        }
    }

    /// Blake3: recursion at ~7.5 GB floors the basic classes too, so a dedicated stream costs what a
    /// basic one does. At 40 GB dropping the pool buys 3 universal streams, not 2 + 1.
    #[test]
    fn blake3_sized_recursion_shares_the_basic_streams() {
        let sizes: Vec<usize> = ZISK_BASIC_GB.iter().copied().map(gb).collect();
        for (budget, want_basic) in [(25.23, 2), (33.0, 3), (40.0, 3), (60.0, 5)] {
            let layout = plan_stream_layout(gb(budget), &sizes, gb(7.5), gb(7.5), 16, 10).unwrap();
            assert_eq!(layout.recursive.count, 0, "budget {budget}: pool not dropped: {:?}", layout.recursive);
            assert_eq!(layout.n_basic_streams(), want_basic, "budget {budget}: {:?}", layout.basic);
            // Aggregation keeps workers of its own, but never all the streams they share.
            assert_eq!(layout.aggregation_workers, want_basic - 1, "budget {budget}: {layout:?}");
            assert!(layout.aggregation_workers < layout.n_basic_streams(), "basics can stall: {layout:?}");
            // Every basic stream can host a recursive launch, so nothing is homeless.
            assert!(layout.basic.iter().all(|c| c.size >= gb(7.5)), "{:?}", layout.basic);
        }
    }

    /// Five 1 GB airs of nine make 1 GB the median, which on the count rule alone bought 16 streams
    /// only the dust could use. Dust never decides the carve, so one more precompile changes nothing.
    #[test]
    fn one_more_dust_air_does_not_change_the_carve() {
        let with_dust: Vec<usize> =
            [14.5f64, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0].iter().copied().map(gb).collect();
        let without: Vec<usize> = [14.5f64, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0].iter().copied().map(gb).collect();
        for budget in [40.0, 60.0] {
            let a = plan_stream_layout(gb(budget), &with_dust, 0, 0, 20, 0).unwrap();
            let b = plan_stream_layout(gb(budget), &without, 0, 0, 20, 0).unwrap();
            assert_eq!(a.basic, b.basic, "budget {budget}: one dust air moved the carve: {:?}", a.basic);
            // And the band that does earn streams gets them.
            assert!(a.basic.iter().all(|c| c.count == 1 || c.size >= gb(12.0)), "{:?}", a.basic);
        }

        // 40 GB: the 12 GB band takes two homes rather than the budget dying or buying dust.
        let layout = plan_stream_layout(gb(40.0), &with_dust, 0, 0, 20, 0).unwrap();
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(12.0), count: 2 }],
            "{:?}",
            layout.basic
        );
        assert!(layout.unused < gb(2.0), "budget left for dead: {layout:?}");
    }

    /// No dedicated stream must still mean workers, or rec1/rec2 waits behind every queued basic.
    #[test]
    fn a_pool_the_budget_cannot_fund_still_gets_workers() {
        let sizes = [gb(14.57), gb(6.45)];
        let layout = plan_stream_layout(gb(15.0), &sizes, gb(6.24), gb(1.33), 16, 10).unwrap();
        assert_eq!(layout.recursive.count, 0, "{layout:?}");
        assert!(layout.aggregation_workers > 0, "aggregation left with nothing: {layout:?}");
    }

    /// At 1.33 GB against a 6.24 GB floor the pool buys four streams for one basic stream's price.
    #[test]
    fn a_cheap_recursive_class_keeps_its_own_streams() {
        let layout = zisk_layout(ZISK_BUDGET_GB, 16, 10);
        assert!(layout.recursive.count >= 3, "cheap pool must survive: {:?}", layout.recursive);
        assert_eq!(layout.aggregation_workers, layout.recursive.count, "one worker per dedicated stream");
    }

    /// Raise the floor rather than rely on a pool: one the budget cannot fund leaves it homeless.
    #[test]
    fn an_oversized_recursive_launch_raises_the_floor() {
        let sizes: Vec<usize> = ZISK_BASIC_GB.iter().copied().map(gb).collect();
        let layout = plan_stream_layout(gb(60.0), &sizes, gb(6.24), gb(16.0), 16, 10).unwrap();
        assert!(layout.basic.iter().all(|c| c.size >= gb(16.0)), "no home for the launch: {:?}", layout.basic);
        assert!(layout.aggregation_workers > 0, "and it keeps its priority: {layout:?}");

        // Too small for even one such stream is a configuration error, not a carve that hangs later.
        assert!(plan_stream_layout(gb(15.0), &sizes, gb(6.24), gb(16.0), 16, 10).is_none());
    }

    /// At 36 the 12 GB class is bought and traded up; its 8 GB of change would otherwise die.
    #[test]
    fn an_all_large_carve_spends_its_change_on_a_spare() {
        let sizes: Vec<usize> = [14.0f64, 12.0, 8.0].iter().copied().map(gb).collect();
        for (budget, want) in [(36.0, Some(2)), (50.0, Some(3)), (30.0, None), (44.0, None)] {
            let layout = plan_stream_layout(gb(budget), &sizes, 0, 0, 16, 0).unwrap();
            match want {
                // The change covers an 8 GB stream: take it rather than let it die.
                Some(n_large) => assert_eq!(
                    layout.basic,
                    vec![StreamClass { size: gb(14.0), count: n_large }, StreamClass { size: gb(8.0), count: 1 }],
                    "budget {budget}: {:?}",
                    layout.basic
                ),
                // Under 8 GB of change: nothing to buy, and uniform hosts every air.
                None => assert_eq!(layout.basic.len(), 1, "budget {budget}: {:?}", layout.basic),
            }
        }
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
            aggregation_workers: 3,
            unused: 0,
        };
        assert_eq!(layout.basic_stream_sizes(), vec![9, 4, 4]);
        assert_eq!(layout.n_basic_streams(), 3);
    }

    /// Airs a class can host, of `sizes`.
    fn hosted(class: &StreamClass, sizes: &[usize]) -> usize {
        sizes.iter().filter(|&&s| s <= class.size).count()
    }

    /// A class at 12.0 costs nearly what a second large stream does and hosts one air fewer, so the
    /// regular class skips past it and the near-outlier rides the large streams.
    #[test]
    fn a_near_outlier_rides_the_large_class_rather_than_taking_one() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23), gb(5.9), gb(5.4), gb(3.8), gb(2.6), gb(1.5)];
        let layout = plan_stream_layout(gb(40.0), &sizes, 0, 0, 20, 0).unwrap();
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 2 }, StreamClass { size: gb(6.23), count: 1 }]
        );
    }

    /// Same with aggregation: the compressor floor only bounds a class from below, and the skip is
    /// the size rule's, not the budget's — 14.5 + 12.0 would have fitted here.
    #[test]
    fn aggregation_does_not_change_which_size_the_regular_class_takes() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23), gb(5.9), gb(5.4), gb(3.8), gb(2.6), gb(1.5)];
        let layout = plan_stream_layout(gb(29.05), &sizes, gb(6.24), gb(1.33), 20, 10).unwrap();
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(6.24), count: 2 }]
        );
    }

    /// No *fleet* may host less than half the airs — the old carve bought 7 x 1.50 GB ones hosting 1
    /// of 8. A single spare is the opposite shape, so only `count > 1` is held to it.
    #[test]
    fn no_carve_buys_a_fleet_below_half_the_airs() {
        let shapes: [&[f64]; 5] = [
            &[14.5, 12.0, 6.23, 5.9, 5.4, 3.8, 2.6, 1.5],
            ZISK_BASIC_GB,
            &[14.57, 5.9, 3.8, 2.6, 0.75],
            &[14.57, 8.0, 0.75],
            &[14.5, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for shape in shapes {
            let sizes: Vec<usize> = shape.iter().copied().map(gb).collect();
            for budget in [26.0, 26.15, 29.05, 33.0, 40.0, 60.0, 80.0] {
                for (floor, rec, max_rec) in [(0.0, 0.0, 0), (ZISK_COMPRESSOR_GB, ZISK_RECURSIVE_GB, 10)] {
                    let layout = plan_stream_layout(gb(budget), &sizes, gb(floor), gb(rec), 20, max_rec).unwrap();
                    for class in layout.basic.iter().filter(|c| c.count > 1) {
                        assert!(
                            hosted(class, &sizes) * 2 >= sizes.len(),
                            "budget {budget} floor {floor}: class {class:?} hosts {} of {}: {:?}",
                            hosted(class, &sizes),
                            sizes.len(),
                            layout.basic
                        );
                    }
                }
            }
        }
    }

    /// Overspending is a failed `cudaMalloc` at startup, so sweep every knob against every shape.
    #[test]
    fn no_carve_ever_overspends_its_budget() {
        let shapes: [&[f64]; 6] = [
            ZISK_KEY_GB,
            ZISK_BASIC_GB,
            &[14.5, 12.0, 6.23, 5.9, 5.4, 3.8, 2.6, 1.5],
            &[7.42, 5.99, 5.82, 5.57, 5.31, 4.52, 0.75],
            &[14.57, 8.0, 0.75],
            &[14.5, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for shape in shapes {
            let sizes: Vec<usize> = shape.iter().copied().map(gb).collect();
            for budget in [14.6, 20.0, 26.15, 28.21, 33.0, 40.0, 60.0, 80.0, 200.0] {
                for (floor, rec) in [(0.0, 0.0), (ZISK_COMPRESSOR_GB, ZISK_RECURSIVE_GB), (9.0, 0.5)] {
                    for (max_basic, max_rec) in [(1, 0), (2, 1), (4, 8), (16, 10), (20, 0)] {
                        let Some(layout) =
                            plan_stream_layout(gb(budget), &sizes, gb(floor), gb(rec), max_basic, max_rec)
                        else {
                            continue;
                        };
                        let where_ =
                            format!("shape {shape:?} budget {budget} floor {floor} caps {max_basic}/{max_rec}");

                        let spent: usize = layout.basic.iter().map(|c| c.size * c.count).sum::<usize>()
                            + layout.recursive.size * layout.recursive.count;
                        assert_eq!(spent + layout.unused, gb(budget), "{where_}: {layout:?}");

                        assert!(!layout.basic.is_empty(), "{where_}: no basic class");
                        assert_eq!(layout.basic[0].size, sizes[0].max(gb(floor)), "{where_}: {:?}", layout.basic);
                        assert!(layout.n_basic_streams() <= max_basic, "{where_}: {:?}", layout.basic);
                        assert!(layout.recursive.count <= max_rec, "{where_}: {:?}", layout.recursive);
                        assert!(layout.aggregation_workers <= max_rec, "{where_}: {layout:?}");
                        // Aggregation must always have somewhere to run first: dedicated streams, or
                        // workers on the shared ones. Neither is starvation behind the basic queue.
                        assert!(
                            rec == 0.0 || max_rec == 0 || layout.recursive.count > 0 || layout.aggregation_workers > 0,
                            "{where_}: aggregation has neither streams nor workers: {layout:?}"
                        );
                        assert!(
                            layout.basic.iter().all(|c| c.size >= gb(floor) && c.count > 0),
                            "{where_}: {:?}",
                            layout.basic
                        );
                    }
                }
            }
        }
    }
}
