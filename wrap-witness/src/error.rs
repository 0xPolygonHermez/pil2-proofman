//! The errors of this crate: `thiserror`, as `common/src/error_manager.rs` and pilfflonk's.

use std::path::PathBuf;

use proofman_common::ProofmanError;
use proofman_pilfflonk::PilfflonkError;

#[derive(Debug, thiserror::Error)]
pub enum WrapWitnessError {
    /// One of the files the witness is computed from is not there.
    #[error("the {what} {} is not a file", path.display())]
    Missing { what: &'static str, path: PathBuf },

    /// A file cannot be read.
    #[error("IO error on {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The JSON the library reads, or a zkin, is not JSON of its shape.
    #[error("{}: {source}", path.display())]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// A zkin that its witness calculator's nlohmann/json cannot hold.
    #[error("the zkin {}: {reason}", path.display())]
    Zkin { path: PathBuf, reason: String },

    /// The exec cannot be read, is not over BN128, or does not fit the circuit or the AIR.
    #[error("{0}")]
    Exec(#[source] ProofmanError),

    /// The final circuit's witness calculator cannot be loaded, or cannot compute a witness from a
    /// zkin, which is then the file `zkin`, if it was read from one. Its `getWitness` writes why to
    /// stderr.
    #[error(
        "witness calculator {}{}: {source}",
        path.display(),
        zkin.as_ref().map(|zkin| format!(", on the zkin {} (the calculator's reason is on stderr)", zkin.display())).unwrap_or_default()
    )]
    Calculator {
        path: PathBuf,
        zkin: Option<PathBuf>,
        #[source]
        source: ProofmanError,
    },

    /// The witness calculator, the exec and the AIR do not fit together.
    #[error("{0}")]
    Mismatch(String),

    /// pilfflonk refuses the witness.
    #[error(transparent)]
    Pilfflonk(#[from] PilfflonkError),
}

pub type WrapWitnessResult<T> = Result<T, WrapWitnessError>;

/// What a pilfflonk witness library returns, for the cdylib: pilfflonk's errors as they are, a
/// file's with the file, and the rest as an invalid format, the variant of an input that breaks
/// what it is read as. [`PilfflonkError`] is part of the witness library ABI, so it gains no
/// variant of its own for this crate.
impl From<WrapWitnessError> for PilfflonkError {
    fn from(error: WrapWitnessError) -> Self {
        match error {
            WrapWitnessError::Io { path, source } => PilfflonkError::Io { path, source },
            WrapWitnessError::Json { path, source } => {
                PilfflonkError::InFile { path, source: Box::new(PilfflonkError::Json(source)) }
            }
            WrapWitnessError::Pilfflonk(error) => error,
            error => PilfflonkError::InvalidFormat(error.to_string()),
        }
    }
}
