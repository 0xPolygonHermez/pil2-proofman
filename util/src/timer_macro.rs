#[macro_export]
macro_rules! timer_start_info {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::info!("{}>>> {}{}", "\x1b[2m", stringify!($name), "\x1b[37;0m");
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::info!("{}>>> {}{}", "\x1b[2m", format!($($arg)+), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_info {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::info!("{}<<< {} ({}ms){}", "\x1b[2m", stringify!($name), $name.as_millis(), "\x1b[37;0m");
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::info!("{}<<< {} ({}ms){}", "\x1b[2m", format!($($arg)+), $name.as_millis(), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_start_debug {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::debug!("{}>>> {}{}", "\x1b[2m", stringify!($name), "\x1b[37;0m");
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::debug!("{}>>> {}{}", "\x1b[2m", format!($($arg)+), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_debug {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::debug!("{}<<< {} ({}ms){}", "\x1b[2m", stringify!($name), $name.as_millis(), "\x1b[37;0m");
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::debug!("{}<<< {} ({}ms){}", "\x1b[2m", format!($($arg)+), $name.as_millis(), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_start_trace {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::trace!("{}>>> {}{}", "\x1b[2m", stringify!($name), "\x1b[37;0m");
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::trace!("{}>>> {}{}", "\x1b[2m", format!($($arg)+), "\x1b[37;0m");
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_trace {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::trace!("{}<<< {} ({}ms){}", "\x1b[2m", stringify!($name), $name.as_millis(), "\x1b[37;0m");
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::trace!("{}<<< {} ({}ms){}", "\x1b[2m", format!($($arg)+), $name.as_millis(), "\x1b[37;0m");
    };
}
