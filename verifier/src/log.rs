#[cfg(feature = "std")]
macro_rules! v_debug {
    ($($arg:tt)*) => { tracing::debug!($($arg)*) };
}

#[cfg(not(feature = "std"))]
macro_rules! v_debug {
    ($($arg:tt)*) => {{}};
}

#[cfg(feature = "std")]
macro_rules! v_error {
    ($($arg:tt)*) => { tracing::error!($($arg)*) };
}

#[cfg(not(feature = "std"))]
macro_rules! v_error {
    ($($arg:tt)*) => {{}};
}
