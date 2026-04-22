pub mod cli;
pub mod timer_macro;
use std::ffi::c_void;

use std::mem::MaybeUninit;

mod proof;
pub use proof::*;

pub fn create_buffer_fast<F>(buffer_size: usize) -> Vec<F> {
    let mut buffer: Vec<MaybeUninit<F>> = Vec::with_capacity(buffer_size);
    unsafe {
        buffer.set_len(buffer_size);
    }
    let buffer: Vec<F> = unsafe { std::mem::transmute(buffer) };
    buffer
}

/// Apple-Silicon Metal unified-memory allocator. Returns a `Vec<F>`
/// backed by shared-storage memory allocated through
/// `pil2_metal_alloc_shared` so Metal kernels can bind the buffer
/// zero-copy. Only available under the `metal` feature.
///
/// Lifetime note: the returned `Vec` MUST NOT be dropped through
/// Rust's `Global` allocator — it is owned by the Metal allocator.
/// Callers must `std::mem::forget` a clone of the owning `Arc` so
/// the Rust allocator never runs on these pages.
#[cfg(feature = "metal")]
pub fn metal_alloc_unified<F>(buffer_size: usize) -> Vec<F> {
    if buffer_size == 0 {
        return Vec::new();
    }
    extern "C" {
        fn pil2_metal_alloc_shared(bytes: u64) -> *mut core::ffi::c_void;
    }
    let bytes = (buffer_size * std::mem::size_of::<F>()) as u64;
    let ptr = unsafe { pil2_metal_alloc_shared(bytes) };
    assert!(!ptr.is_null(), "pil2_metal_alloc_shared returned null");
    unsafe { Vec::from_raw_parts(ptr as *mut F, buffer_size, buffer_size) }
}

/// Set the current thread's Metal stream id. Must be called from every
/// proof-worker thread before it invokes any Metal code; without this,
/// all threads land on scratch row 0 and concurrent Metal calls race.
/// `id` must be in [0, 8). No-op on non-Metal builds.
#[cfg(feature = "metal")]
pub fn metal_set_stream_id(id: i32) {
    extern "C" {
        fn pil2_metal_set_stream_id(id: i32);
    }
    unsafe { pil2_metal_set_stream_id(id) };
}

#[cfg(not(feature = "metal"))]
pub fn metal_set_stream_id(_id: i32) {}

pub fn create_buffer_fast_u8(buffer_size: usize) -> Vec<u8> {
    let mut buffer: Vec<MaybeUninit<u8>> = Vec::with_capacity(buffer_size);
    unsafe {
        buffer.set_len(buffer_size);
    }
    let buffer: Vec<u8> = unsafe { std::mem::transmute(buffer) };
    buffer
}

#[derive(Default)]
pub struct DeviceBuffer(pub *mut c_void);
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn get_ptr(&self) -> *mut c_void {
        self.0
    }
}
