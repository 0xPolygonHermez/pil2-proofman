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

    /// The JS verifier (spec §4.5) cannot give a verdict: no Node.js, its dependencies missing and
    /// `npm install` unable to install them, inputs it cannot read, or a failure of its own.
    #[error("JS verifier: {0}")]
    JsVerifier(String),

    /// A witness library (spec §4.3, D4, `crate::witness_library`) that cannot be loaded, or that is
    /// not a pilfflonk one.
    #[error("witness library {}: {reason}", path.display())]
    WitnessLibrary { path: PathBuf, reason: String },

    /// The witness does not satisfy the AIR's constraints: the prover's constraint polynomial `Q` is
    /// not a polynomial of its degree (A.1). The C++ core's message says where.
    #[error("{0}")]
    Unsatisfied(String),

    /// A call to the C++ core failed; `context` says which.
    #[error("{context}: {source}")]
    Native {
        context: String,
        #[source]
        source: proofman_starks_lib_c::PilFflonkError,
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
