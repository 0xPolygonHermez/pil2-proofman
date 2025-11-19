use tracing_subscriber::filter::LevelFilter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerboseMode {
    Info,
    Debug,
    Trace,
}

impl From<u8> for VerboseMode {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Info,
            1 => Self::Debug,
            _ => Self::Trace,
        }
    }
}

impl From<VerboseMode> for u64 {
    fn from(val: VerboseMode) -> Self {
        match val {
            VerboseMode::Info => 3,
            VerboseMode::Debug => 4,
            VerboseMode::Trace => 5,
        }
    }
}

impl From<VerboseMode> for LevelFilter {
    fn from(val: VerboseMode) -> Self {
        match val {
            VerboseMode::Info => LevelFilter::INFO,
            VerboseMode::Debug => LevelFilter::DEBUG,
            VerboseMode::Trace => LevelFilter::TRACE,
        }
    }
}
