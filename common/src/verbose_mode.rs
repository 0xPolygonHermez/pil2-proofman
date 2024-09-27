use log::LevelFilter;

#[derive(Debug, Clone)]
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

impl Into<u64> for VerboseMode {
    fn into(self) -> u64 {
        match self {
            VerboseMode::Info => 3,
            VerboseMode::Debug => 4,
            VerboseMode::Trace => 5,
        }
    }
}

impl From<VerboseMode> for LevelFilter {
    fn from(val: VerboseMode) -> Self {
        match val {
            VerboseMode::Info => LevelFilter::Info,
            VerboseMode::Debug => LevelFilter::Debug,
            VerboseMode::Trace => LevelFilter::Trace,
        }
    }
}
