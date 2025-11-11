/// Coordinator-specific error types with proper security boundaries
#[derive(Debug, thiserror::Error)]
pub enum ProofmanError {
    #[error("Proof error: {0}")]
    InvalidProof(String),

    #[error("MPI cancellation detected: {0}")]
    MpiCancellation(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    #[error("Invalid instances assignation: {0}")]
    InvalidAssignation(String),

    #[error("Dynamic library error: {0}")]
    LibraryError(#[from] libloading::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Setup error: {0}")]
    InvalidSetup(String),

    #[error("Hints error: {0}")]
    InvalidHints(String),

    #[error("STD error: {0}")]
    StdError(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Slice conversion error: {0}")]
    SliceConversion(#[from] std::array::TryFromSliceError),

    #[error("Proofman error: {0}")]
    ProofmanError(String),

    #[error("Cancelled")]
    Cancelled,
}

pub type ProofmanResult<T> = Result<T, ProofmanError>;

unsafe impl Send for ProofmanError {}
unsafe impl Sync for ProofmanError {}
