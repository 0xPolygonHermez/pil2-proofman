//! Airs whose stage-1 witness is produced by a caller-supplied GPU kernel.
//!
//! Declared before `ProofMan::new`, like `packed_info`, because the host trace
//! pool and the witness prefetch zone are sized from it before any witness exists.
//! A declared air's pinned pool buffer holds only its staged kernel inputs; the
//! kernel writes the commit slot directly and no trace is uploaded.
//!
//! The declaration is a guarantee: nothing fills that air's trace on the host,
//! so a declared air that cannot reach its kernel is a setup error.

/// A kernel's entry point, re-exported from the FFI crate.
pub use proofman_starks_lib_c::GpuWitnessFillFn;

use proofman_fields::PrimeField64;

use crate::{AirInstance, ProofmanError, ProofmanResult, TraceInfo};

/// Write a kernel's inputs into `buffer` and wrap it as `decl`'s air's `AirInstance`.
///
/// `decl` is the prover's declaration (`ProofCtx::gpu_witness_air`): the commit uploads
/// `ops * decl.bytes_per_op` bytes and stages at most `decl.input_bytes_per_instance`, so `Op`
/// must be exactly `bytes_per_op` wide and the ops must fit that bound. The buffer is used as raw bytes (`Goldilocks` is
/// not `repr(transparent)`), so ops are written through a raw pointer. `Op` must be `Copy` (no drop
/// glue: the buffer is later dropped as `F`) and should be a `#[repr(C)]` mirror of the kernel's op struct.
///
/// `n_cols` is the air's real column count and does not match `trace.len()`;
/// the commit path takes its geometry from the setup.
pub fn stage_gpu_witness<F: PrimeField64, Op, I>(
    decl: &GpuWitnessAir,
    num_rows: usize,
    n_cols: usize,
    mut buffer: Vec<F>,
    inputs: &[Vec<I>],
) -> ProofmanResult<(AirInstance<F>, u64)>
where
    Op: Copy,
    for<'a> Op: From<&'a I>,
{
    let (airgroup_id, air_id) = (decl.airgroup_id, decl.air_id);
    if std::mem::size_of::<Op>() as u64 != decl.bytes_per_op {
        return Err(ProofmanError::InvalidParameters(format!(
            "air {airgroup_id}:{air_id}: staged op is {} bytes but the kernel declares {}",
            std::mem::size_of::<Op>(),
            decl.bytes_per_op
        )));
    }
    let capacity = std::mem::size_of_val(buffer.as_slice()).min(decl.input_bytes_per_instance as usize);
    // Checked: zero-sized inputs can count past usize without allocating, and the writes trust `needed`.
    let num_ops = inputs.iter().try_fold(0usize, |n, chunk| n.checked_add(chunk.len()));
    let Some((num_ops, needed)) = num_ops.and_then(|n| Some((n, n.checked_mul(std::mem::size_of::<Op>())?))) else {
        return Err(ProofmanError::InvalidParameters(format!(
            "air {airgroup_id}:{air_id}: the staged operation count overflows"
        )));
    };
    // Both commit paths treat a zero-op instance as fatal, so refuse it while it is still recoverable.
    if num_ops == 0 {
        return Err(ProofmanError::InvalidParameters(format!(
            "air {airgroup_id}:{air_id}: no operations to stage; the kernel needs at least one"
        )));
    }
    if needed > capacity {
        return Err(ProofmanError::InvalidParameters(format!(
            "air {airgroup_id}:{air_id}: {num_ops} staged operations need {needed} bytes but the \
             trace buffer and the declaration allow {capacity}"
        )));
    }

    // SAFETY: `needed <= capacity` was checked, the buffer is uniquely owned, and
    // offsets never overlap. Only the H2D reads the bytes back.
    let base = buffer.as_mut_ptr() as *mut u8;
    let mut written = 0usize;
    for chunk in inputs {
        for input in chunk {
            unsafe {
                std::ptr::write_unaligned(base.add(written * std::mem::size_of::<Op>()) as *mut Op, Op::from(input));
            }
            written += 1;
        }
    }
    debug_assert_eq!(written, num_ops);

    let mut air_instance = AirInstance::new(TraceInfo::new(airgroup_id, air_id, n_cols, num_rows, buffer, true, false));
    air_instance.gpu_witness_ops = num_ops as u64;
    Ok((air_instance, num_ops as u64))
}

/// Where a kernel writes, and therefore what the prover runs afterwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TraceLayout {
    /// Packed cm1 rows, as the host upload would deliver them; unpack runs after.
    PackedCm1 = 0,
    /// Row-major plain cm1; the row-to-column-major transform runs after.
    PlainCm1 = 1,
}

/// One air's GPU witness declaration.
#[derive(Clone, Copy, Debug)]
pub struct GpuWitnessAir {
    pub airgroup_id: usize,
    pub air_id: usize,
    /// Bytes of staged inputs per instance. Sizes the host pool buffer and the
    /// prefetch slot; the kernel cannot allocate its own.
    pub input_bytes_per_instance: u64,
    /// Size of one staged op. Must match the kernel's op struct exactly.
    pub bytes_per_op: u64,
    /// What the kernel writes into the commit slot. Must agree with the air's packing.
    pub emits: TraceLayout,
    /// The kernel itself, carried with the declaration so it cannot be missing.
    pub kernel: GpuWitnessFillFn,
}

impl GpuWitnessAir {
    pub fn new(
        airgroup_id: usize,
        air_id: usize,
        input_bytes_per_instance: u64,
        bytes_per_op: u64,
        emits: TraceLayout,
        kernel: GpuWitnessFillFn,
    ) -> Self {
        assert!(bytes_per_op > 0, "a staged operation cannot be zero bytes wide");
        Self { airgroup_id, air_id, input_bytes_per_instance, bytes_per_op, emits, kernel }
    }
}

/// The declared set. Empty (the normal case) means every witness comes from the host.
#[derive(Clone, Debug, Default)]
pub struct GpuWitnessAirs {
    airs: Vec<GpuWitnessAir>,
}

impl GpuWitnessAirs {
    pub fn new(airs: Vec<GpuWitnessAir>) -> Self {
        Self { airs }
    }

    pub fn is_empty(&self) -> bool {
        self.airs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.airs.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &GpuWitnessAir> {
        self.airs.iter()
    }

    /// Whether this air's witness is produced on the device.
    pub fn contains(&self, airgroup_id: usize, air_id: usize) -> bool {
        self.get(airgroup_id, air_id).is_some()
    }

    pub fn get(&self, airgroup_id: usize, air_id: usize) -> Option<&GpuWitnessAir> {
        self.airs.iter().find(|a| a.airgroup_id == airgroup_id && a.air_id == air_id)
    }

    /// Replace this prover's C++ registry (keyed by its `d_buffers`) with these declarations.
    pub fn register(&self, d_buffers: *mut std::ffi::c_void) {
        proofman_starks_lib_c::gpu_witness_clear_c(d_buffers);
        for a in &self.airs {
            proofman_starks_lib_c::gpu_witness_register_c(
                d_buffers,
                a.airgroup_id as u64,
                a.air_id as u64,
                a.bytes_per_op,
                a.emits as i32,
                a.kernel,
            );
        }
    }

    /// Device bytes for one instance's staged inputs: a `max` over airs, since the
    /// reservation is per concurrent commit. Zero when nothing is declared.
    pub fn max_input_bytes(&self) -> u64 {
        self.airs.iter().map(|a| a.input_bytes_per_instance).max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_fields::Goldilocks;

    unsafe extern "C" fn never_called(
        _: *const std::ffi::c_void,
        _: u64,
        _: *mut u64,
        _: i32,
        _: *mut std::ffi::c_void,
    ) -> i32 {
        unreachable!("declaration test: the kernel is never invoked")
    }

    fn airs() -> GpuWitnessAirs {
        GpuWitnessAirs::new(vec![
            GpuWitnessAir::new(0, 36, 3_123_792, 216, TraceLayout::PackedCm1, never_called),
            GpuWitnessAir::new(0, 2, 512, 16, TraceLayout::PlainCm1, never_called),
        ])
    }

    #[test]
    fn an_empty_declaration_claims_nothing() {
        let none = GpuWitnessAirs::default();
        assert!(none.is_empty());
        assert!(!none.contains(0, 36));
        assert_eq!(none.max_input_bytes(), 0);
    }

    #[test]
    fn lookup_is_by_the_full_air_key() {
        let airs = airs();
        assert!(airs.contains(0, 36));
        assert!(!airs.contains(1, 36), "air id alone must not match across airgroups");
        assert!(!airs.contains(0, 37));
    }

    fn decl(input_bytes: u64, bytes_per_op: u64) -> GpuWitnessAir {
        GpuWitnessAir::new(0, 36, input_bytes, bytes_per_op, TraceLayout::PackedCm1, never_called)
    }

    #[test]
    fn input_staging_is_the_max_not_the_sum() {
        assert_eq!(airs().max_input_bytes(), 3_123_792);
    }

    /// One op per input, contiguous across chunk boundaries.
    #[test]
    fn staged_ops_are_contiguous_across_chunks() {
        #[repr(C)]
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct Op {
            a: u64,
            b: u64,
        }
        impl From<&u32> for Op {
            fn from(v: &u32) -> Self {
                Op { a: *v as u64, b: (*v as u64) << 32 }
            }
        }

        let inputs = vec![vec![1u32, 2], vec![3], vec![], vec![4, 5]];
        let buffer = vec![Goldilocks::default(); 64];
        let (air_instance, ops) =
            stage_gpu_witness::<Goldilocks, Op, u32>(&decl(512, 16), 8, 4, buffer, &inputs).expect("stages");

        assert_eq!(ops, 5, "every input becomes one operation");
        assert_eq!(air_instance.gpu_witness_ops, 5, "the instance carries the count for the prover");

        let base = air_instance.trace.as_ptr() as *const Op;
        for (i, expected) in [1u32, 2, 3, 4, 5].iter().enumerate() {
            assert_eq!(unsafe { *base.add(i) }, Op::from(expected), "op {i}");
        }
    }

    /// Overrunning the pool buffer would corrupt the next one handed out.

    #[test]
    fn staging_more_than_the_buffer_holds_is_refused() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Wide([u64; 8]);
        impl From<&u32> for Wide {
            fn from(v: &u32) -> Self {
                Wide([*v as u64; 8])
            }
        }

        // Room for two ops, asked for three.
        let buffer = vec![Goldilocks::default(); 16];
        let err = stage_gpu_witness::<Goldilocks, Wide, u32>(&decl(4096, 64), 8, 4, buffer, &[vec![1, 2, 3]])
            .expect_err("must refuse rather than overrun");
        assert!(err.to_string().contains("192 bytes"), "reports what was needed: {err}");
    }

    #[test]
    fn an_op_count_that_overflows_is_refused() {
        #[allow(dead_code)] // only its size matters
        #[derive(Clone, Copy)]
        struct Op(u64);
        impl From<&()> for Op {
            fn from(_: &()) -> Self {
                Op(0)
            }
        }
        // Zero-sized inputs reach usize::MAX ops without allocating.
        let buffer = vec![Goldilocks::default(); 16];
        let inputs = vec![vec![(); usize::MAX], vec![(); 1]];
        let err = stage_gpu_witness::<Goldilocks, Op, ()>(&decl(128, 8), 8, 4, buffer, &inputs)
            .expect_err("must refuse rather than wrap");
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    #[test]
    fn an_empty_operation_set_is_refused() {
        #[allow(dead_code)] // only its size matters
        #[derive(Clone, Copy)]
        struct Op(u64);
        impl From<&u32> for Op {
            fn from(v: &u32) -> Self {
                Op(*v as u64)
            }
        }
        let buffer = vec![Goldilocks::default(); 16];
        let err = stage_gpu_witness::<Goldilocks, Op, u32>(&decl(128, 8), 8, 4, buffer, &[vec![], vec![]])
            .expect_err("an empty batch must not reach the commit");
        assert!(err.to_string().contains("no operations"), "{err}");
    }

    #[test]
    fn an_op_wider_or_narrower_than_declared_is_refused() {
        #[allow(dead_code)] // only its size matters
        #[derive(Clone, Copy)]
        struct Op(u64);
        impl From<&u32> for Op {
            fn from(v: &u32) -> Self {
                Op(*v as u64)
            }
        }
        // The commit copies ops * bytes_per_op: an 8-byte op under a 16-byte declaration would ship garbage.
        let buffer = vec![Goldilocks::default(); 16];
        let err = stage_gpu_witness::<Goldilocks, Op, u32>(&decl(128, 16), 8, 4, buffer, &[vec![1]])
            .expect_err("must refuse a layout mismatch");
        assert!(err.to_string().contains("declares 16"), "{err}");
    }

    #[test]
    fn staging_past_the_declared_bound_is_refused_even_in_a_bigger_buffer() {
        #[allow(dead_code)] // only its size matters
        #[derive(Clone, Copy)]
        struct Op(u64);
        impl From<&u32> for Op {
            fn from(v: &u32) -> Self {
                Op(*v as u64)
            }
        }
        // The pool buffer is sized for the largest air; this air declared room for two ops.
        let buffer = vec![Goldilocks::default(); 64];
        let err = stage_gpu_witness::<Goldilocks, Op, u32>(&decl(16, 8), 8, 4, buffer, &[vec![1, 2, 3]])
            .expect_err("must refuse past the declaration");
        assert!(err.to_string().contains("allow 16"), "{err}");
    }

    #[test]
    fn the_emitted_layout_is_carried_per_air() {
        let airs = airs();
        assert_eq!(airs.get(0, 36).unwrap().emits, TraceLayout::PackedCm1);
        assert_eq!(airs.get(0, 2).unwrap().emits, TraceLayout::PlainCm1);
    }
}
