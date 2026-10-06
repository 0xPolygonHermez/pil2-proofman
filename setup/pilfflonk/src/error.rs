//! The errors of the library part of this crate: `thiserror`, following
//! `common/src/error_manager.rs` (pilfflonk/docs/README.md#conventions). The command,
//! [`crate::command`], reports them with `anyhow`.

use std::path::PathBuf;

use pil_info::PilInfoError;
use proofman_pilfflonk::{PilfflonkError, BN128_R};
use proofman_starks_lib_c::PilFflonkError;

use crate::grouping::GroupingError;

#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    // --- What the setup refuses in a pilout (pilfflonk/docs/README.md#what-the-setup-refuses) ---
    /// The pilout is over Goldilocks: what pil2com writes when it ignores `--field`, as the pinned
    /// compiler does (pilfflonk/docs/README.md#compile-pil).
    #[error(
        "the pilout is over Goldilocks, not BN128: compile it with `--field bn128`, and with a pil2com that has \
         `--field` (PIL2C_EXEC); the pinned one ignores it and compiles over Goldilocks"
    )]
    GoldilocksPilout,

    #[error("the pilout's base field is {base_field}, not BN128's r = {BN128_R}")]
    NotBn128 { base_field: String },

    /// A pilfflonk proof has one instance of one AIR (pilfflonk/docs/README.md#scope).
    #[error("the pilout has {n_airs} AIRs, and a pilfflonk proof holds exactly one instance of one AIR")]
    AirCount { n_airs: usize },

    /// Air values: pil-fflonk has none, and neither has v1.
    #[error("air {air} has {n} air values, which pilfflonk does not support (pilfflonk/docs/README.md#scope)")]
    AirValues { air: String, n: usize },

    /// Airgroup values. The std's buses declare one each unless the std is in
    /// `STD_MODE_ONE_INSTANCE`.
    #[error(
        "the pilout has {n} airgroup values, which pilfflonk does not support (pilfflonk/docs/README.md#scope); the \
         std's buses declare them unless it is compiled with set_std_mode(STD_MODE_ONE_INSTANCE)"
    )]
    AirgroupValues { n: usize },

    /// Proof values.
    #[error("the pilout has {n} proof values, which pilfflonk does not support (pilfflonk/docs/README.md#scope)")]
    ProofValues { n: usize },

    /// Global constraints: a pilfflonk proof holds one AIR, and its `pilout.globalConstraints.json`
    /// none.
    #[error(
        "the pilout has {n} global constraints, which pilfflonk does not support (pilfflonk/docs/README.md#scope)"
    )]
    GlobalConstraints { n: usize },

    #[error("air {air} has {num_rows} rows, not a power of two of at most 2^28 (pilfflonk/docs/protocol.md#notation)")]
    NumRows { air: String, num_rows: u32 },

    #[error(
        "an AIR of 2^{n_bits} rows: BN128's roots of unity allow at most 2^28 (pilfflonk/docs/protocol.md#notation)"
    )]
    NBits { n_bits: u64 },

    #[error("air {air} has {n} custom commits, which pilfflonk does not support (pilfflonk/docs/README.md#scope)")]
    CustomCommits { air: String, n: usize },

    #[error("air {air} has {n} periodic columns, which pilfflonk does not support")]
    PeriodicColumns { air: String, n: usize },

    #[error("the pilout has {n} public tables, which pilfflonk does not support")]
    PublicTables { n: usize },

    /// The prover hint `im_airval` (pilfflonk/docs/README.md#what-the-setup-refuses), which
    /// computes an air value: v1 has none.
    #[error(
        "{location} has the prover hint `im_airval`, which computes an air value, and pilfflonk has none \
         (pilfflonk/docs/README.md#scope): the std adds one for a term of a bus that is a constant"
    )]
    ImAirvalHint { location: String },

    /// An `im_col`, `gsum_col` or `gprod_col` that the prover cannot compute as the STARK's
    /// `calculateImHints` and `calculateWitnessSTD` do (`crate::validate::check_prover_hints`).
    #[error("air {air}: its hint `{hint}` cannot be computed: {reason}")]
    ProverHint { hint: String, air: String, reason: String },

    /// A hint that is neither a prover hint nor one of the witness and debug hints the setup
    /// ignores (pilfflonk/docs/README.md#what-the-setup-refuses).
    #[error("{location} has the hint `{name}`, which is neither a prover hint nor a witness or debug hint")]
    UnknownHint { name: String, location: String },

    /// Columns of stage 2 or above that no `gsum_col` or `gprod_col` produces: the prover computes
    /// those stages from the hints alone (pilfflonk/docs/README.md#what-the-setup-refuses).
    #[error(
        "air {air} has {n_columns} columns of stage {stage} that no hint pilfflonk supports (gsum_col, gprod_col) \
         produces"
    )]
    StageWithoutHint { air: String, stage: usize, n_columns: u32 },

    #[error("{location} is {value}, which is not below r")]
    ConstantNotBelowR { location: String, value: String },

    /// Two columns that the proof and the layout would name alike, with the im pols and the columns
    /// that share a name and have no indices already indexed (`crate::air_info`): two arrays of the
    /// same name, or a column named as the setup names another.
    #[error(
        "air {air}: {first} and {second} are both named {name}, and the proof names each evaluation by its column \
         (pilfflonk/docs/formats.md#proof-names); the setup only indexes the im pols and the columns that share a \
         name and have no indices"
    )]
    ColumnName { air: String, name: String, first: String, second: String },

    /// The extended domain does not fit in the 2-adicity of BN128
    /// (pilfflonk/docs/protocol.md#degrees).
    #[error("the extended domain has 2^{n_bits_ext} points, and BN128's roots of unity allow at most 2^28")]
    ExtendedDomain { n_bits_ext: u64 },

    /// A fixed column without a value per row, which `<air>.const` needs.
    #[error(
        "fixed column {column} has {n_values} values, not one per row ({n_rows}); a pilout compiled with its fixed \
         columns to a file, or without their values, has none"
    )]
    FixedValues { column: usize, n_values: usize, n_rows: usize },

    /// A pilout that breaks its own format: an index to nothing, an enum value out of range, a
    /// column without a symbol.
    #[error("invalid pilout: {0}")]
    InvalidPilout(String),

    // --- The external fixed columns (pilfflonk/docs/formats.md#fixed-columns) ------------------
    /// A fixed column that the pilout has no values of, as one it declares `#pragma fixed_external`,
    /// and that no external column fills (`crate::fixed::FixedColumns::from_air_with_external`).
    #[error(
        "fixed column {column} of the AIR, {name} (index {index}), has no values in the pilout and no external \
         column gives them: a column declared `#pragma fixed_external` takes its values from the caller of the \
         setup (pilfflonk/docs/formats.md#fixed-columns)"
    )]
    ExternalFixedMissing { column: usize, name: String, index: usize },

    /// An external fixed column whose name and index are those of no fixed column of the AIR.
    #[error("the external fixed column {name} (index {index}) is not a fixed column of air {air}")]
    ExternalFixedUnknown { air: String, name: String, index: usize },

    /// An external fixed column for a column that has its values in the pilout.
    #[error(
        "the external fixed column {name} (index {index}) is fixed column {column} of the AIR, which has its values \
         in the pilout: only a column declared `#pragma fixed_external` takes external values"
    )]
    ExternalFixedNotEmpty { name: String, index: usize, column: usize },

    #[error("the external fixed column {name} (index {index}) is given twice")]
    ExternalFixedTwice { name: String, index: usize },

    #[error("the external fixed column {name} (index {index}) has {n_values} values, not one per row ({n_rows})")]
    ExternalFixedValues { name: String, index: usize, n_values: usize, n_rows: usize },

    // --- What the passes decide (pilfflonk/docs/protocol.md#degree-search) ----------------------
    /// The symbolic passes refused the AIR: a pilout that refers to nothing, or constraints they
    /// cannot process.
    #[error("the symbolic passes (pil-info) failed: {0}")]
    Passes(#[source] PilInfoError),

    /// The symbolic passes panicked: an invariant of their own broken, a bug of `pil-info` and not
    /// of the pilout ([`crate::passes`]). The setup reports the panic's message.
    #[error("the symbolic passes (pil-info) panicked: {0}")]
    PassesPanicked(String),

    /// The thread the passes run on, with their stack, could not be started.
    #[error("cannot start the thread the symbolic passes (pil-info) run on: {0}")]
    PassesThread(#[source] std::io::Error),

    /// What the passes returned does not fit pilfflonk: a value of dimension other than 1, an
    /// evaluation of an unknown kind, the quotient's pieces out of place. A bug of the passes or
    /// of this crate, not of the pilout.
    #[error("the symbolic passes (pil-info) returned what pilfflonk cannot use: {0}")]
    PassesOutput(String),

    /// The constraints' degree in the columns is 0: `qDeg` would be negative.
    #[error(
        "the constraints give qDeg = {0}: some constraint must depend on a column \
         (pilfflonk/docs/protocol.md#degree-search)"
    )]
    QDegree(i64),

    // --- What the setup refuses in its arguments (pilfflonk/docs/README.md#setup-pilfflonk) -----
    #[error("--max-constraint-degree {0}: the degree search starts at 2 (pilfflonk/docs/protocol.md#degree-search)")]
    MaxConstraintDegree(u64),

    /// The grouping of the committed polynomials in `f_i`
    /// (pilfflonk/docs/protocol.md#grouping-rules) refused them with this `--extra-muls`: too many,
    /// a search too large, or no valid partition (pilfflonk/docs/protocol.md#grouping-errors). Its
    /// messages say which, and what `--extra-muls` would do.
    #[error(transparent)]
    Grouping(#[from] GroupingError),

    // --- Files and the C++ core --------------------------------------------------------------
    #[error("IO error on {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{} is not a pilout: {source}", path.display())]
    Pilout {
        path: PathBuf,
        #[source]
        source: prost::DecodeError,
    },

    /// An `<air>.const` that is not the one its shape expects.
    #[error("{}: {reason}", path.display())]
    ConstFile { path: PathBuf, reason: String },

    /// A layout that does not fit the fixed columns it is applied to.
    #[error("invalid layout: {0}")]
    Layout(String),

    /// From the pilfflonk file types (`proofman-pilfflonk`).
    #[error(transparent)]
    Pilfflonk(#[from] PilfflonkError),

    /// The Solidity verifier (pilfflonk/docs/verifier.md#solidity-verifier) cannot be generated
    /// from the vkey: one the JS verifier accepts no proof of, or a template that does not render.
    #[error("cannot generate the Solidity verifier: {0}")]
    Solidity(String),

    /// A call to the C++ core failed; `context` says which and on what.
    #[error("{context}: {source}")]
    Native {
        context: String,
        #[source]
        source: PilFflonkError,
    },
}

impl SetupError {
    pub(crate) fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Self {
        let path = path.into();
        move |source| SetupError::Io { path, source }
    }

    pub(crate) fn native(context: impl Into<String>) -> impl FnOnce(PilFflonkError) -> Self {
        let context = context.into();
        move |source| SetupError::Native { context, source }
    }
}
