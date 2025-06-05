#[macro_export]
macro_rules! timer_start_info {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::info!(">>> {}", stringify!($name));
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::info!(">>> {}", format!($($arg)+));
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_info {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::info!("<<< {} ({}ms)", stringify!($name), $name.as_millis());
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::info!("<<< {} ({}ms)", format!($($arg)+), $name.as_millis());
    };
}

#[macro_export]
macro_rules! timer_start_debug {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::debug!(">>> {}", stringify!($name));
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::debug!(">>> {}", format!($($arg)+));
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_debug {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::debug!("<<< {} ({}ms)", stringify!($name), $name.as_millis());
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::debug!("<<< {} ({}ms)", format!($($arg)+), $name.as_millis());
    };
}

#[macro_export]
macro_rules! timer_start_trace {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::trace!(">>> {}", stringify!($name));
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        tracing::trace!(">>> {}", format!($($arg)+));
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_trace {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::trace!("<<< {} ({}ms)", stringify!($name), $name.as_millis());
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        tracing::trace!("<<< {} ({}ms)", format!($($arg)+), $name.as_millis());
    };
}
