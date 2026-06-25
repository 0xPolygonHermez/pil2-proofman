use fields::PrimeField64;
use proofman_common::{BucketRule, Classifier};
use proofman_hints::HintFieldOutput;

/// Evaluate the bucket rule for a given opid against a bus value tuple.
///
/// Returns `Some(bucket_key)` for rows that should be tracked, `None` for rows that the
/// rule filters out (currently only possible with `Classifier::Value` when an explicit
/// `values` filter list is set and the column value isn't in it).
///
/// Bucket key meanings:
/// - `Classifier::Value` — the raw column value canonicalized to u64.
/// - `Classifier::Range` — the index of the matching range bucket.
/// - `Classifier::Prefix` — the index of the matching prefix, or `prefixes.len()` for
///   the implicit "no match" bucket.
/// - `Classifier::Step` — the index of the matching step bucket, or `step_oor_index(...)`
///   for the implicit "out of range" bucket.
///
/// Panics if `rule.column` is out of bounds for `bus_value` — this would indicate a
/// configuration mismatch and should be caught during proving-key validation, not silently
/// suppressed at runtime.
pub fn evaluate_bucket<F: PrimeField64>(rule: &BucketRule, bus_value: &[HintFieldOutput<F>]) -> Option<u64> {
    assert!(
        rule.column < bus_value.len(),
        "Bucket rule for opid {} references column {} but bus value only has {} components",
        rule.opid,
        rule.column,
        bus_value.len()
    );

    let col = column_as_u64(&bus_value[rule.column]);

    match &rule.classifier {
        Classifier::Value { values: None } => Some(col),
        Classifier::Value { values: Some(allowed) } => {
            // Filter mode: drop rows whose column value isn't in the allowed list.
            if allowed.contains(&col) {
                Some(col)
            } else {
                None
            }
        }
        Classifier::Range { ranges, filter } => {
            for (i, r) in ranges.iter().enumerate() {
                let above_min = r.min.is_none_or(|lo| col >= lo);
                let below_max = r.max.is_none_or(|hi| col < hi);
                if above_min && below_max {
                    return Some(i as u64);
                }
            }
            // No range matched. In default mode parse-time validation guarantees this is
            // unreachable; in filter mode the row is dropped.
            if *filter {
                None
            } else {
                unreachable!("range ranges should cover all u64 values when filter=false (validated at parse time)");
            }
        }
        Classifier::Prefix { prefixes, filter } => {
            for (i, p) in prefixes.iter().enumerate() {
                let shift = 64 - p.bits;
                let top = if shift == 64 { 0 } else { col >> shift };
                if top == p.value {
                    return Some(i as u64);
                }
            }
            // No prefix matched.
            if *filter {
                None
            } else {
                Some(prefixes.len() as u64)
            }
        }
        Classifier::Step { start, stop, step, filter } => {
            if col < *start || col >= *stop {
                if *filter {
                    None
                } else {
                    Some(step_oor_index(*start, *stop, *step))
                }
            } else {
                Some((col - *start) / *step)
            }
        }
    }
}

/// Index of the "out of range" bucket for a `Step` classifier.
/// Equals `ceil((stop - start) / step)`, computed without overflow.
#[inline]
pub fn step_oor_index(start: u64, stop: u64, step: u64) -> u64 {
    let span = stop - start;
    span / step + if span.is_multiple_of(step) { 0 } else { 1 }
}

/// Canonicalize a field output to u64. For extended-field components, uses the base
/// component. The redesign explicitly states bucketing operates on non-extended columns;
/// callers configuring bucketing on extended columns get the base only.
fn column_as_u64<F: PrimeField64>(out: &HintFieldOutput<F>) -> u64 {
    match out {
        HintFieldOutput::Field(f) => f.as_canonical_u64(),
        HintFieldOutput::FieldExtended(ef) => ef.value[0].as_canonical_u64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fields::Goldilocks;
    use proofman_common::{BucketPrefix, BucketRange};

    fn rule(column: usize, classifier: Classifier) -> BucketRule {
        BucketRule { opid: 0, column, classifier }
    }

    fn val(n: u64) -> HintFieldOutput<Goldilocks> {
        HintFieldOutput::Field(Goldilocks::from_u64(n))
    }

    #[test]
    fn value_classifier_returns_raw_column() {
        let r = rule(0, Classifier::Value { values: None });
        assert_eq!(evaluate_bucket(&r, &[val(42)]), Some(42));
        assert_eq!(evaluate_bucket(&r, &[val(0xABCD)]), Some(0xABCD));
    }

    #[test]
    fn value_classifier_uses_correct_column() {
        let r = rule(2, Classifier::Value { values: None });
        assert_eq!(evaluate_bucket(&r, &[val(1), val(2), val(99), val(4)]), Some(99));
    }

    /// Regression: callers must pass the un-normalized bus value to `evaluate_bucket`.
    /// `normalize_vals` trims trailing zeros, which would shrink the slice and make
    /// `column: 2` out of bounds. The bucket evaluator itself doesn't normalize — it
    /// trusts the caller — so this test simply documents the invariant that the column
    /// index references the full tuple.
    #[test]
    fn evaluate_bucket_handles_full_tuple_including_trailing_zeros() {
        let r = rule(2, Classifier::Value { values: None });
        // Bus value with a trailing zero in column 2 — column 2 must still be accessible.
        assert_eq!(evaluate_bucket(&r, &[val(1), val(2), val(0)]), Some(0));
    }

    #[test]
    fn value_classifier_with_filter_list_keeps_listed_values() {
        // Only 0x42 and 0xff are tracked; everything else is dropped.
        let r = rule(0, Classifier::Value { values: Some(vec![0x42, 0xff]) });

        assert_eq!(evaluate_bucket(&r, &[val(0x42)]), Some(0x42));
        assert_eq!(evaluate_bucket(&r, &[val(0xff)]), Some(0xff));
        // Bucket key for a matching value is the raw value, not the index.
    }

    #[test]
    fn value_classifier_with_filter_list_drops_unlisted_values() {
        let r = rule(0, Classifier::Value { values: Some(vec![0x42, 0xff]) });

        assert_eq!(evaluate_bucket(&r, &[val(0)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0x41)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0x100)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(u64::MAX)]), None);
    }

    #[test]
    fn range_classifier_buckets_by_index() {
        let r = rule(
            0,
            Classifier::Range {
                ranges: vec![
                    BucketRange { min: None, max: Some(0x10000) },
                    BucketRange { min: Some(0x10000), max: Some(0x100000000) },
                    BucketRange { min: Some(0x100000000), max: None },
                ],
                filter: false,
            },
        );

        assert_eq!(evaluate_bucket(&r, &[val(0)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0xFFFF)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0x10000)]), Some(1));
        assert_eq!(evaluate_bucket(&r, &[val(0xFFFFFFFF)]), Some(1));
        assert_eq!(evaluate_bucket(&r, &[val(0x100000000)]), Some(2));
    }

    #[test]
    fn range_classifier_filter_mode_drops_outside_values() {
        // Single explicit range, filter mode → drops everything outside [0x100, 0x200).
        let r = rule(
            0,
            Classifier::Range { ranges: vec![BucketRange { min: Some(0x100), max: Some(0x200) }], filter: true },
        );

        assert_eq!(evaluate_bucket(&r, &[val(0x100)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0x1FF)]), Some(0));
        // Outside the range → dropped.
        assert_eq!(evaluate_bucket(&r, &[val(0)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0xFF)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0x200)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0xFFFF_FFFF)]), None);
    }

    #[test]
    fn prefix_classifier_matches_top_bits() {
        let r = rule(
            0,
            Classifier::Prefix {
                prefixes: vec![BucketPrefix { value: 0xFF, bits: 8 }, BucketPrefix { value: 0xAB, bits: 8 }],
                filter: false,
            },
        );

        assert_eq!(evaluate_bucket(&r, &[val(0xFF00_0000_0000_0000)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0xAB12_3456_789A_BCDE)]), Some(1));
        // 0x12... matches neither — implicit catch-all bucket (index = prefixes.len()).
        assert_eq!(evaluate_bucket(&r, &[val(0x1234_5678_9ABC_DEF0)]), Some(2));
    }

    #[test]
    fn prefix_classifier_filter_mode_drops_no_match() {
        let r = rule(0, Classifier::Prefix { prefixes: vec![BucketPrefix { value: 0xFF, bits: 8 }], filter: true });

        assert_eq!(evaluate_bucket(&r, &[val(0xFF00_0000_0000_0000)]), Some(0));
        // No prefix matches → dropped instead of going into catch-all.
        assert_eq!(evaluate_bucket(&r, &[val(0x1234_5678_9ABC_DEF0)]), None);
    }

    #[test]
    fn prefix_classifier_handles_non_byte_aligned_bits() {
        let r = rule(0, Classifier::Prefix { prefixes: vec![BucketPrefix { value: 0b1010, bits: 4 }], filter: false });

        assert_eq!(evaluate_bucket(&r, &[val(0xA000_0000_0000_0000)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0xB000_0000_0000_0000)]), Some(1));
    }

    #[test]
    fn step_classifier_buckets_uniformly() {
        let r = rule(0, Classifier::Step { start: 0, stop: 0x1000, step: 0x100, filter: false });

        assert_eq!(evaluate_bucket(&r, &[val(0)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0xFF)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0x100)]), Some(1));
        assert_eq!(evaluate_bucket(&r, &[val(0x1FF)]), Some(1));
        assert_eq!(evaluate_bucket(&r, &[val(0xF00)]), Some(15));
        assert_eq!(evaluate_bucket(&r, &[val(0xFFF)]), Some(15));
        assert_eq!(evaluate_bucket(&r, &[val(0x1000)]), Some(16));
        assert_eq!(evaluate_bucket(&r, &[val(0xFFFF_FFFF)]), Some(16));
    }

    #[test]
    fn step_classifier_filter_mode_drops_oor() {
        let r = rule(0, Classifier::Step { start: 0x100, stop: 0x500, step: 0x100, filter: true });

        // In-range values still get bucketed.
        assert_eq!(evaluate_bucket(&r, &[val(0x100)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0x4FF)]), Some(3));
        // Out-of-range values dropped instead of OOR bucket.
        assert_eq!(evaluate_bucket(&r, &[val(0)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0xFF)]), None);
        assert_eq!(evaluate_bucket(&r, &[val(0x500)]), None);
    }

    #[test]
    fn step_classifier_handles_offset_start() {
        let r = rule(0, Classifier::Step { start: 0x100, stop: 0x500, step: 0x100, filter: false });

        assert_eq!(evaluate_bucket(&r, &[val(0)]), Some(4));
        assert_eq!(evaluate_bucket(&r, &[val(0xFF)]), Some(4));
        assert_eq!(evaluate_bucket(&r, &[val(0x100)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0x4FF)]), Some(3));
        assert_eq!(evaluate_bucket(&r, &[val(0x500)]), Some(4));
    }

    #[test]
    fn step_classifier_handles_non_divisible_span() {
        let r = rule(0, Classifier::Step { start: 0, stop: 10, step: 3, filter: false });

        assert_eq!(evaluate_bucket(&r, &[val(0)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(2)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(3)]), Some(1));
        assert_eq!(evaluate_bucket(&r, &[val(8)]), Some(2));
        assert_eq!(evaluate_bucket(&r, &[val(9)]), Some(3));
        assert_eq!(evaluate_bucket(&r, &[val(10)]), Some(4));
    }

    #[test]
    fn prefix_classifier_handles_full_64_bits() {
        let r = rule(
            0,
            Classifier::Prefix {
                prefixes: vec![BucketPrefix { value: 0x1234_5678_9ABC_DEF0, bits: 64 }],
                filter: false,
            },
        );

        assert_eq!(evaluate_bucket(&r, &[val(0x1234_5678_9ABC_DEF0)]), Some(0));
        assert_eq!(evaluate_bucket(&r, &[val(0x1234_5678_9ABC_DEF1)]), Some(1));
    }
}
