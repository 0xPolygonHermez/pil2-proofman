#[macro_export]
macro_rules! timer_start_info {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::info!("{}>>> {}{}", escape_in, stringify!($name), escape_out);
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::info!("{}>>> {}{}", escape_in, format!($($arg)+), escape_out);
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_info {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::info!("{}<<< {} ({}ms){}", escape_in, stringify!($name), $name.as_millis(), escape_out);
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::info!("{}<<< {} ({}ms){}", escape_in, format!($($arg)+), $name.as_millis(), escape_out);
    };
}

#[macro_export]
macro_rules! timer_start_debug {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::debug!("{}>>> {}{}", escape_in, stringify!($name), escape_out);
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::debug!("{}>>> {}{}", escape_in, format!($($arg)+), escape_out);
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_debug {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::debug!("{}<<< {} ({}ms){}", escape_in, stringify!($name), $name.as_millis(), escape_out);
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::debug!("{}<<< {} ({}ms){}", escape_in, format!($($arg)+), $name.as_millis(), escape_out);
    };
}

#[macro_export]
macro_rules! timer_start_trace {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::trace!("{}>>> {}{}", escape_in, stringify!($name), escape_out);
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now();
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::trace!("{}>>> {}{}", escape_in, format!($($arg)+), escape_out);
    };
}

#[macro_export]
macro_rules! timer_stop_and_log_trace {
    ($name:ident) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::trace!("{}<<< {} ({}ms){}", escape_in, stringify!($name), $name.as_millis(), escape_out);
    };
    ($name:ident, $($arg:tt)+) => {
        #[allow(non_snake_case)]
        let $name = std::time::Instant::now() - $name;
        let escape_in = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[2m",
            false => "",
        };
        let escape_out = match std::io::IsTerminal::is_terminal(&std::io::stdout()) {
            true => "\x1b[37;0m",
            false => "",
        };
        tracing::trace!("{}<<< {} ({}ms){}", escape_in, format!($($arg)+), $name.as_millis(), escape_out);
    };
}
