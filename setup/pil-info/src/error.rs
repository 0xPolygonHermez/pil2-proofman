//! The errors of `pil-info`, with `thiserror` following `common/src/error_manager.rs`
//! (pilfflonk/docs/README.md#conventions): [`PilInfoError`] for the passes and the code types,
//! [`BinFileError`] for the `"chps"` container's writer. The setups report them with their own
//! errors.

/// What the passes refuse in a pilout, or find broken in what they build themselves.
#[derive(Debug, thiserror::Error)]
pub enum PilInfoError {
    // --- A pilout the passes cannot process --------------------------------------------------
    #[error("the pilout has no air {air_id} in airgroup {airgroup_id}")]
    NoSuchAir { airgroup_id: usize, air_id: usize },

    /// A pilout that breaks its own format: an index to nothing.
    #[error("invalid pilout: {0}")]
    InvalidPilout(String),

    // --- What the constraints need that the passes cannot give --------------------------------
    #[error("no choice of intermediate polynomials brings the constraints down to a degree in 2..={max}")]
    NoFeasibleDegree { max: usize },

    /// The passes are pilfflonk's: the STARK's DEEP setup is `stark_setup` over the core.
    #[error("the passes open with SHPLONK only; the STARK's DEEP setup runs through the core")]
    DeepPasses,

    // --- An invariant of the passes broken: a bug of pil-info, not of the pilout -------------
    #[error("{pass}: unknown expression op `{op}`")]
    UnknownOp { pass: &'static str, op: String },

    #[error("a hint field of op `{0}`, which no hint holds")]
    UnknownHintOp(String),

    #[error("op {op} uses temporary {id}, and the code has {max_id}")]
    TmpOutOfRange { op: usize, id: u64, max_id: usize },

    #[error("op {op} uses temporary {id} of dimension {dim}, neither 1 nor the extension's {ext_dim}")]
    TmpDimension { op: usize, id: usize, dim: u64, ext_dim: u64 },

    /// Building or lowering pilfflonk's constraint DAG (the qVerifier's source) failed.
    #[error("the constraint DAG: {0}")]
    ConstraintDag(String),

    // --- The code types ------------------------------------------------------------------------
    #[error("unknown opType `{0}`")]
    UnknownOpType(String),
}

/// A failure of [`crate::io::bin_file_writer::BinFileWriter`].
#[derive(Debug, thiserror::Error)]
pub enum BinFileError {
    #[error("the type of a binary file is 4 bytes, and `{0}` is not")]
    FileType(String),

    #[error("a section is being written already")]
    SectionOpen,

    #[error("no section is being written")]
    NoSectionOpen,

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// The `Result` of `pil-info`: a [`PilInfoError`] unless it says otherwise.
pub type Result<T, E = PilInfoError> = std::result::Result<T, E>;
