//! Small data-parallel helpers that switch between rayon and sequential
//! iteration on the `parallel` feature, so callers stay free of `#[cfg]` noise.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Below this many items, run sequentially even under the `parallel` feature —
/// rayon's fork/join overhead dominates for the small tail (late sumcheck /
/// fold rounds), while the large early rounds still parallelize.
#[cfg(feature = "parallel")]
const SEQ_THRESHOLD: usize = 1 << 12;

/// Collect `f` over `0..n`, in parallel when `parallel` is enabled and `n` is large.
#[cfg(feature = "parallel")]
pub(crate) fn map_range<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    if n < SEQ_THRESHOLD {
        (0..n).map(f).collect()
    } else {
        (0..n).into_par_iter().map(f).collect()
    }
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn map_range<R>(n: usize, f: impl Fn(usize) -> R) -> Vec<R> {
    (0..n).map(f).collect()
}

/// Chunked parallel map-reduce over `0..n` with per-worker reusable state.
///
/// `init` builds a fresh accumulator (and scratch) per chunk, `fold` folds one
/// index into it, `combine` merges two accumulators. Runs sequentially when
/// `parallel` is off or `n` is small. Used for the zerocheck round polynomial,
/// where each row does real compute (constraint-cone evaluation) and the
/// per-worker scratch avoids per-row allocation.
#[cfg(feature = "parallel")]
pub(crate) fn fold_reduce<A: Send>(
    n: usize,
    init: impl Fn() -> A + Sync + Send,
    fold: impl Fn(&mut A, usize) + Sync + Send,
    combine: impl Fn(A, A) -> A + Sync + Send,
) -> A {
    if n < SEQ_THRESHOLD {
        let mut acc = init();
        for i in 0..n {
            fold(&mut acc, i);
        }
        return acc;
    }
    (0..n)
        .into_par_iter()
        .fold(&init, |mut acc, i| {
            fold(&mut acc, i);
            acc
        })
        .reduce(&init, &combine)
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn fold_reduce<A>(
    n: usize,
    init: impl Fn() -> A,
    fold: impl Fn(&mut A, usize),
    combine: impl Fn(A, A) -> A,
) -> A {
    let _ = combine;
    let mut acc = init();
    for i in 0..n {
        fold(&mut acc, i);
    }
    acc
}

/// Collect `f` over a slice, in parallel when `parallel` is enabled.
#[cfg(feature = "parallel")]
pub(crate) fn map_slice<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    items.par_iter().map(f).collect()
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn map_slice<T, R>(items: &[T], f: impl Fn(&T) -> R) -> Vec<R> {
    items.iter().map(f).collect()
}
