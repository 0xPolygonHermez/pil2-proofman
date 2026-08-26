//! Small data-parallel helpers that switch between rayon and sequential
//! iteration on the `parallel` feature, so callers stay free of `#[cfg]` noise.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Below this many items, run sequentially even under the `parallel` feature.
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

/// Collect `f` over a slice, in parallel when `parallel` is enabled.
/// Unlike [`map_range`] there is no sequential threshold: use it when each
/// item is a big unit of work (e.g. one whole column).
#[cfg(feature = "parallel")]
pub(crate) fn map_slice<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    items.par_iter().map(f).collect()
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn map_slice<T, R>(items: &[T], f: impl Fn(&T) -> R) -> Vec<R> {
    items.iter().map(f).collect()
}

/// Partition `0..n` into chunks and collect `f(start, end)` per chunk — the
/// building block for parallel reductions: each chunk returns a partial
/// accumulator the caller combines. Sequentially this is one `f(0, n)` call.
#[cfg(feature = "parallel")]
pub(crate) fn map_chunks<R: Send>(n: usize, f: impl Fn(usize, usize) -> R + Sync + Send) -> Vec<R> {
    if n < SEQ_THRESHOLD {
        return vec![f(0, n)];
    }
    let n_chunks = (rayon::current_num_threads() * 4).min(n.div_ceil(1024)).max(1);
    let chunk = n.div_ceil(n_chunks);
    (0..n_chunks).into_par_iter().map(|c| f(c * chunk, ((c + 1) * chunk).min(n))).collect()
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn map_chunks<R>(n: usize, f: impl Fn(usize, usize) -> R) -> Vec<R> {
    vec![f(0, n)]
}

/// Run two closures, in parallel when `parallel` is enabled.
#[cfg(feature = "parallel")]
pub(crate) fn join<RA: Send, RB: Send>(a: impl FnOnce() -> RA + Send, b: impl FnOnce() -> RB + Send) -> (RA, RB) {
    rayon::join(a, b)
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn join<RA, RB>(a: impl FnOnce() -> RA, b: impl FnOnce() -> RB) -> (RA, RB) {
    (a(), b())
}
/// Element-wise `f(&mut a[i], &mut b[i])` over two equal-length slices,
/// chunked in parallel when large.
#[cfg(feature = "parallel")]
pub(crate) fn zip2_for_each_mut<A: Send, B: Send>(a: &mut [A], b: &mut [B], f: impl Fn(&mut A, &mut B) + Sync + Send) {
    debug_assert_eq!(a.len(), b.len());
    if a.len() < SEQ_THRESHOLD {
        a.iter_mut().zip(b.iter_mut()).for_each(|(x, y)| f(x, y));
    } else {
        let chunk = a.len().div_ceil(rayon::current_num_threads() * 4).max(1024);
        a.par_chunks_mut(chunk).zip(b.par_chunks_mut(chunk)).for_each(|(ca, cb)| {
            ca.iter_mut().zip(cb.iter_mut()).for_each(|(x, y)| f(x, y));
        });
    }
}
#[cfg(not(feature = "parallel"))]
pub(crate) fn zip2_for_each_mut<A, B>(a: &mut [A], b: &mut [B], f: impl Fn(&mut A, &mut B)) {
    a.iter_mut().zip(b.iter_mut()).for_each(|(x, y)| f(x, y));
}
