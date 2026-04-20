mod ffi_goldilocks;
mod ffi_starks;
pub use ffi_goldilocks::*;
pub use ffi_starks::*;

#[cfg(feature = "metal")]
mod ffi_metal;
#[cfg(feature = "metal")]
pub use ffi_metal::*;
