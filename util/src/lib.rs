pub mod cli;

/// The starkinfo key format: the setup writes it, the provers refuse any other (2: canonical expression and im order).
pub const KEY_VERSION: u32 = 2;
pub mod timer_macro;
use std::ffi::c_void;

use std::mem::MaybeUninit;

pub fn create_buffer_fast<F>(buffer_size: usize) -> Vec<F> {
    let mut buffer: Vec<MaybeUninit<F>> = Vec::with_capacity(buffer_size);
    unsafe {
        buffer.set_len(buffer_size);
    }
    let buffer: Vec<F> = unsafe { std::mem::transmute(buffer) };
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
