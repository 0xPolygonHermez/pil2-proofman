use log::LevelFilter;

pub enum VerboseMode {
    Info,
    Debug,
    Trace,
}

impl VerboseMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Info,
            1 => Self::Debug,
            _ => Self::Trace,
        }
    }
}

impl Into<LevelFilter> for VerboseMode {
    fn into(self) -> LevelFilter {
        match self {
            Self::Info => LevelFilter::Info,
            Self::Debug => LevelFilter::Debug,
            Self::Trace => LevelFilter::Trace,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    Disabled,
    Error,
    XXX,
    Trace,
}

impl DebugMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Disabled,
            1 => Self::Error,
            2 => Self::XXX,
            3 => Self::Trace,
            _ => Self::Disabled,
        }
    }
}
