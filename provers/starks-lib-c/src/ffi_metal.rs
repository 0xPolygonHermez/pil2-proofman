// Apple-Silicon-only FFI to the Metal-backed unified-memory allocator
// exposed by pil2-stark/libstarksmetal.a. Linked only when the crate is
// compiled with the `metal` feature.

#![cfg(feature = "metal")]

use std::os::raw::c_void;

extern "C" {
    /// Allocate a Metal-backed shared-storage buffer of `bytes` bytes and
    /// return its CPU-accessible `.contents` pointer. Metal kernels can
    /// bind the same buffer zero-copy via the internal registry the
    /// pil2-stark Metal bridge consults. Caller must eventually release
    /// the returned pointer with [`pil2_metal_free_shared`]. Returns null
    /// on `bytes == 0`; aborts on allocation failure.
    pub fn pil2_metal_alloc_shared(bytes: u64) -> *mut c_void;

    /// Release a buffer returned by [`pil2_metal_alloc_shared`]. Must be
    /// called with the exact pointer handed out by alloc (not a derived
    /// pointer inside the allocation). Silent on unknown pointers
    /// (logs to stderr).
    pub fn pil2_metal_free_shared(ptr: *mut c_void);

    /// Diagnostic: non-zero when `ptr` is exactly the base of a
    /// registered allocation.
    pub fn pil2_metal_is_shared_base(ptr: *const c_void) -> i32;
}
