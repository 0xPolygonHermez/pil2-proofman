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

    /// A hint field with neither an operand, a string nor an array of fields.
    #[error("invalid pilout: a field of hint `{hint}` has no value")]
    HintFieldWithoutValue { hint: String },

    /// The columns of a custom commit are of stage 0.
    #[error(
        "invalid pilout: symbol `{name}` is a custom column of stage {stage}, and the columns of a custom commit \
         are of stage 0"
    )]
    CustomColumnStage { name: String, stage: usize },

    /// An evaluation of a column that has no symbol: the FRI polynomial needs its stage and
    /// dimension.
    #[error("invalid pilout: the constraints evaluate {entry_type} {id}, which has no symbol")]
    NoSymbolForEvaluation { entry_type: String, id: usize },

    // --- What the constraints need that the passes cannot give --------------------------------
    #[error("no choice of intermediate polynomials brings the constraints down to a degree in 2..={max}")]
    NoFeasibleDegree { max: usize },

    /// The FRI polynomial combines the evaluations at each opening point, and there are none.
    #[error("the FRI polynomial has no evaluation to combine: the constraints evaluate no column")]
    FriWithoutEvaluations,

    // --- An invariant of the passes broken: a bug of pil-info, not of the pilout -------------
    #[error("{pass}: unknown expression op `{op}`")]
    UnknownOp { pass: &'static str, op: String },

    #[error("constraint boundary `{0}` is not supported")]
    UnsupportedBoundary(String),

    #[error("an evaluation of a `{0}`, which is neither a cm, a const nor a custom column")]
    UnknownEvaluation(String),

    #[error("a hint field of op `{0}`, which no hint holds")]
    UnknownHintOp(String),

    #[error("no code was generated for the FRI polynomial, expression {0}")]
    FriCodeMissing(usize),

    /// Verifier code whose ops all write temporaries, as `fix_dimensions_verifier` requires.
    #[error(
        "verifier code: op {index} is a `{op}` into a `{dest}`, and verifier code only adds, subtracts, multiplies \
         or copies into temporaries"
    )]
    VerifierCode { index: usize, op: String, dest: String },

    #[error("op {op} uses temporary {id}, and the code has {max_id}")]
    TmpOutOfRange { op: usize, id: u64, max_id: usize },

    #[error("op {op} uses temporary {id} of dimension {dim}, neither 1 nor the extension's {ext_dim}")]
    TmpDimension { op: usize, id: usize, dim: u64, ext_dim: u64 },

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
