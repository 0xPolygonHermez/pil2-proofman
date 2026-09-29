//! The errors of this crate (spec §5.4: `thiserror`, following `common/src/error_manager.rs`).

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum PilfflonkError {
    /// A file cannot be read or written.
    #[error("IO error on {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A file, or a string, is not the JSON its type expects.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The JSON is well formed but a value breaks the format: a constant that must be fixed, an
    /// integer out of range, an index to nothing, two structures that disagree.
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// An error from the content of a file, with the file it comes from.
    #[error("{}: {source}", path.display())]
    InFile {
        path: PathBuf,
        #[source]
        source: Box<PilfflonkError>,
    },
}

pub type PilfflonkResult<T> = Result<T, PilfflonkError>;

/// `Err(InvalidFormat)` built with `format!`.
macro_rules! invalid {
    ($($arg:tt)*) => {
        Err($crate::error::PilfflonkError::InvalidFormat(format!($($arg)*)))
    };
}
pub(crate) use invalid;
