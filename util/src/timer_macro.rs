#[macro_export]
macro_rules! timer_start_info {
    ($name:ident) => {
        timer_start_info!($name, "");
    };
    ($name:ident, $arg:expr) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::info!("{}>>> {}{}{}", "\x1b[2m", stringify!($name), $arg, "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_start_debug {
    ($name:ident) => {
        timer_start_debug!($name, "");
    };
    ($name:ident, $arg:expr) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::debug!("{}>>> {}{}{}", "\x1b[2m", stringify!($name), $arg, "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_start_trace {
    ($name:ident) => {
        timer_start_trace!($name, "");
    };
    ($name:ident, $arg:expr) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::trace!("{}>>> {}{}{}", "\x1b[2m", stringify!($name), $arg, "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_info {
    ($name:ident) => {
        timer_stop_and_log_info!($name, "");
    };
    ($name:ident, $arg:expr) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::info!("{}<<< {}{} {}ms{}", "\x1b[2m", stringify!($name), $arg, $name.as_millis(), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_debug {
    ($name:ident) => {
        timer_stop_and_log_debug!($name, "");
    };
    ($name:ident, $arg:expr) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::debug!("{}<<< {}{} {}ms{}", "\x1b[2m", stringify!($name), $arg, $name.as_millis(), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_trace {
    ($name:ident) => {
        timer_stop_and_log_trace!($name, "");
    };
    ($name:ident, $arg:expr) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::trace!("{}<<< {}{} {}ms{}", "\x1b[2m", stringify!($name), $arg, $name.as_millis(), "\x1b[37;0m");
    };
}
