//! Cutting the per-GPU aux-trace budget into stream size classes.
//!
//! Per-air aux-trace requirements are far from uniform — one outlier air can need several times the
//! next largest — so sizing every stream to the maximum buys concurrency at the outlier's price. The
//! budget is instead cut into two classes: *large*, sized to the biggest air so nothing is homeless,
//! and *regular*, sized to whichever smaller air requirement carves best. A proof runs on any stream
//! at least as large as it needs, so an air between the two sizes has only the large streams.
//!
//! The large size is fixed by the airs; the regular size and the split of streams between the two are
//! searched together by [`makespan_floor`]. Whatever they leave becomes recursive streams.
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

/// Recursive streams kept funded before any basic stream beyond the first is cut. Basic streams are
/// funded first, so without a floor they spend the last byte and aggregation's small proofs end up
/// running a handful at a time in whatever huge basic class is free. A floor, not a reservation:
/// the remainder still becomes recursive streams up to the caller's cap.
const RECURSIVE_STREAM_FLOOR: usize = 1;

/// How much worse than the best makespan floor a carve may be and still win by being more large-heavy
/// — up to the uniform carve, which drops the regular class altogether. A difference this small is
/// inside the noise of size-as-cost (a hot outlier air costs far more than one instance of it), and
/// the large streams are the only home the biggest airs have.
const LARGE_HEAVY_SLACK: f64 = 1.2;

impl StreamLayout {
    pub fn n_basic_streams(&self) -> usize {
        self.basic.iter().map(|c| c.count).sum()
    }

    /// Buffer sizes for the basic streams, in stream order (largest class first).
    pub fn basic_stream_sizes(&self) -> Vec<usize> {
        self.basic.iter().flat_map(|c| std::iter::repeat_n(c.size, c.count)).collect()
    }
}

/// The makespan floor: for each air, the work that *only* streams that big can take (every air at
/// least as large) over how many such streams there are. The largest of those ratios is what no
/// schedule can beat. Air size stands in for air cost — no instance is planned this early.
///
/// Counting hosted work instead rates `1 x 14.5 + 4 x 6.23` best on a 40 GB card: half the work — the
/// 14.5 and 12 GB airs — waits on the single large stream while the cheap airs get five.
/// Scaled, not divided, so the compare stays in exact integers. `basic_sizes_desc` must be sorted
/// largest first — the running `work` is the total of every air at least as large as this one;
/// [`plan_stream_layout`] sorts before calling.
fn makespan_floor(basic: &[StreamClass], basic_sizes_desc: &[usize]) -> u128 {
    debug_assert!(basic_sizes_desc.windows(2).all(|w| w[0] >= w[1]), "basic_sizes_desc must be descending");

    let (mut work, mut worst) = (0u128, 0u128);
    for &air in basic_sizes_desc {
        work += air as u128;
        // At least one: the large class covers every air.
        let streams: u128 = basic.iter().filter(|c| air <= c.size).map(|c| c.count as u128).sum();
        worst = worst.max((work << 20) / streams);
    }
    worst
}

/// How many streams of `size` fit in `room`, at most `cap`. A size of 0 means no such class.
fn n_fit(room: usize, size: usize, cap: usize) -> usize {
    room.checked_div(size).map_or(0, |n| cap.min(n))
}

/// Cut `budget` (in field elements, per GPU) into a large and a regular stream class.
///
/// `basic_sizes` is the per-air basic requirement in any order — sorted here, since the floor reads
/// it largest first; duplicates weigh twice. `class_floor` is the smallest a basic class may be: it
/// must hold anything besides a basic proof that can land on one of these streams — today the
/// compressor/vadcop_final launches that share them. `recursive_size` sizes the recursive1/recursive2
/// class.
///
/// The large size is fixed by the biggest air; the regular size and the split of streams between the
/// two are searched by [`makespan_floor`], the most large-heavy carve winning ties (see
/// [`LARGE_HEAVY_SLACK`]).
///
/// Basic streams are funded first — they are the only home for basics *and* compressors — and
/// aggregation takes the remainder, subject only to [`RECURSIVE_STREAM_FLOOR`].
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
    // [`makespan_floor`] reads the airs largest first; normalize once rather than trust the caller.
    let mut basic_sizes_desc: Vec<usize> = basic_sizes.to_vec();
    basic_sizes_desc.sort_unstable_by(|a, b| b.cmp(a));

    // The candidate sizes are the distinct air requirements, each raised to the floor so either class
    // can also take a compressor and any contributions commit. Largest first.
    let mut sizes: Vec<usize> = basic_sizes_desc.iter().map(|&s| s.max(class_floor)).collect();
    sizes.dedup();

    // The large class must hold anything that can land on a non-recursive stream.
    let &large = sizes.first()?;
    if large == 0 || budget < large {
        return None;
    }

    // The first large stream ignores the recursive floor — without it nothing can run at all.
    let for_extra_streams = budget.saturating_sub(recursive_size * RECURSIVE_STREAM_FLOOR.min(max_recursive_streams));

    // Every air requirement a second class could be funded at, plus 0 for the uniform carve. Searched,
    // not picked up front: streams afforded and airs stranded are exactly what the floor measures.
    let regulars = sizes.iter().skip(1).copied().filter(|&s| large + s <= for_extra_streams).chain([0]);

    // The slack compares splits of the same two classes, so it stays inside one candidate size.
    let best_for = |regular: usize| -> Option<(u128, StreamLayout)> {
        let carves: Vec<(u128, StreamLayout)> = (1..=max_basic_streams)
            .filter_map(|n_large| {
                let spent = n_large * large;
                if spent > budget || (n_large > 1 && spent > for_extra_streams) {
                    return None;
                }
                let n_regular = n_fit(for_extra_streams.saturating_sub(spent), regular, max_basic_streams - n_large);

                let mut basic = vec![StreamClass { size: large, count: n_large }];
                if n_regular > 0 {
                    basic.push(StreamClass { size: regular, count: n_regular });
                }
                let remaining = budget - spent - n_regular * regular;
                let n_recursive = n_fit(remaining, recursive_size, max_recursive_streams);
                Some((
                    makespan_floor(&basic, &basic_sizes_desc),
                    StreamLayout {
                        basic,
                        recursive: StreamClass { size: recursive_size, count: n_recursive },
                        unused: remaining - n_recursive * recursive_size,
                    },
                ))
            })
            .collect();
        let best = carves.iter().map(|(floor, _)| *floor).min()?;
        carves
            .into_iter()
            .filter(|(floor, _)| *floor as f64 <= best as f64 * LARGE_HEAVY_SLACK)
            .max_by_key(|(_, layout)| (layout.basic[0].count, layout.recursive.count))
    };

    // Ties go large-heavy, then to more streams, then (iteration order) to the larger size —
    // shrinking a class the floor is indifferent to only exiles an air.
    regulars
        .filter_map(best_for)
        .min_by_key(|(floor, layout)| {
            (*floor, std::cmp::Reverse(layout.basic[0].count), std::cmp::Reverse(layout.n_basic_streams()))
        })
        .map(|(_, layout)| layout)
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
    /// depend on it, since the floor reads them largest first.
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

    /// The split only beats scarcity: where the cap binds before memory, every stream should be large
    /// so every air can use all of them.
    #[test]
    fn a_roomy_budget_goes_uniform() {
        let layout = zisk_layout(120.0, 4, 10);
        assert_eq!(layout.basic, vec![StreamClass { size: gb(14.57), count: 4 }], "{:?}", layout.basic);
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

    /// Coverage is worth a lot, but never the second basic stream: where the budget cannot pair the
    /// largest air with a class that hosts most airs, a smaller class still beats having only one.
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

    /// The measured ZisK key: 7.42 GB largest air against a 6.24 GB compressor floor. The split
    /// `1x7.42 + 2x6.24` and uniform `3x7.42` buy the same three streams, so the split bought nothing
    /// while confining the 7.42 air to one stream for the whole run — it must lose on parallelism.
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

    /// A genuine outlier keeps its own class: on a budget this tight the split really does buy
    /// streams (2L+6R against 4 uniform), so it must not be flattened into one size.
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

    /// Airs a class can host, of `sizes`.
    fn hosted(class: &StreamClass, sizes: &[usize]) -> usize {
        sizes.iter().filter(|&&s| s <= class.size).count()
    }

    /// One big air, one close behind, the rest the bulk. The regular class is sized for the bulk
    /// (6.23), but the near-outlier rides the large class — and 14.5 + 12.0 is half the work, so a
    /// second large stream is worth more than a fourth regular one.
    #[test]
    fn the_classes_are_balanced_by_the_work_each_must_run() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23), gb(5.9), gb(5.4), gb(3.8), gb(2.6), gb(1.5)];
        let layout = plan_stream_layout(gb(40.0), &sizes, 0, 0, 20, 0).unwrap();
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 2 }, StreamClass { size: gb(6.23), count: 1 }]
        );
    }

    /// Same with aggregation: the compressor floor only bounds a class from below.
    #[test]
    fn aggregation_sizes_the_regular_class_so_no_air_is_stranded() {
        let sizes = [gb(14.5), gb(12.0), gb(6.23), gb(5.9), gb(5.4), gb(3.8), gb(2.6), gb(1.5)];
        let layout = plan_stream_layout(gb(29.05), &sizes, gb(6.24), gb(1.33), 20, 10).unwrap();
        // A 6.24 regular affords one more stream but strands the 12 GB air: 2.1% worse floor.
        assert_eq!(
            layout.basic,
            vec![StreamClass { size: gb(14.5), count: 1 }, StreamClass { size: gb(12.0), count: 1 }]
        );
    }

    /// No *fleet* of streams may host less than half the airs — the old carve bought 7 x 1.50 GB ones
    /// hosting 1 of 8. A single top-up stream is the opposite shape and can be optimal.
    #[test]
    fn no_carve_buys_a_fleet_below_half_the_airs() {
        let shapes: [&[f64]; 3] =
            [&[14.5, 12.0, 6.23, 5.9, 5.4, 3.8, 2.6, 1.5], ZISK_BASIC_GB, &[14.57, 5.9, 3.8, 2.6, 0.75]];
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
}
