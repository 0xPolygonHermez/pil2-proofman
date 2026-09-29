//! The errors of the library part of this crate (spec §5.4: `thiserror`, following
//! `common/src/error_manager.rs`). The command, [`crate::command`], reports them with `anyhow`.

use std::path::PathBuf;

use proofman_pilfflonk::{PilfflonkError, BN254_R};
use proofman_starks_lib_c::PilFflonkError;

#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    // --- What the setup refuses in a pilout (spec §4.2.1) -------------------------------------
    /// The pilout is over Goldilocks: what pil2com writes when it ignores `prime`, as the pinned
    /// compiler does (plan M13).
    #[error(
        "the pilout is over Goldilocks, not BN254: compile it with `-P <config>` whose `prime` is r, and with a \
         pil2com that honours `prime` (PIL2C_EXEC); the pinned one ignores it and compiles over Goldilocks"
    )]
    GoldilocksPilout,

    #[error("the pilout's base field is {base_field}, not BN254's r = {BN254_R}")]
    NotBn254 { base_field: String },

    /// A pilfflonk proof has one instance of one AIR (spec §7.1, D2).
    #[error("the pilout has {n_airs} AIRs, and a pilfflonk proof holds exactly one instance of one AIR")]
    AirCount { n_airs: usize },

    /// Air values: pil-fflonk has none, and neither has v1 (spec §7.1, D2).
    #[error("air {air} has {n} air values, which pilfflonk does not support (spec D2)")]
    AirValues { air: String, n: usize },

    /// Airgroup values (spec §7.1, D2).
    #[error("the pilout has {n} airgroup values, which pilfflonk does not support (spec D2)")]
    AirgroupValues { n: usize },

    /// Proof values (spec §7.1, D2).
    #[error("the pilout has {n} proof values, which pilfflonk does not support (spec D2)")]
    ProofValues { n: usize },

    /// Global constraints (spec §7.1, D2): a pilfflonk proof holds one AIR, and its
    /// `pilout.globalConstraints.json` none.
    #[error("the pilout has {n} global constraints, which pilfflonk does not support (spec D2)")]
    GlobalConstraints { n: usize },

    #[error("air {air} has {num_rows} rows, not a power of two of at most 2^28 (spec P2)")]
    NumRows { air: String, num_rows: u32 },

    #[error("an AIR of 2^{n_bits} rows: BN254's roots of unity allow at most 2^28 (spec P2)")]
    NBits { n_bits: u64 },

    #[error("air {air} has {n} custom commits, which pilfflonk does not support (spec P5)")]
    CustomCommits { air: String, n: usize },

    #[error("air {air} has {n} periodic columns, which pilfflonk does not support")]
    PeriodicColumns { air: String, n: usize },

    #[error("the pilout has {n} public tables, which pilfflonk does not support")]
    PublicTables { n: usize },

    /// A prover hint (spec §3.4). Fase 1 supports none of them (plan R2).
    #[error(
        "{location} has the prover hint `{name}`, which pilfflonk does not support yet: it proves stage 1 only, \
         with no prover hint (the std's buses come in Fase 2)"
    )]
    UnsupportedProverHint { name: String, location: String },

    /// A hint that is neither a prover hint nor one of the witness and debug hints the setup
    /// ignores (spec §3.4).
    #[error("{location} has the hint `{name}`, which is neither a prover hint nor a witness or debug hint")]
    UnknownHint { name: String, location: String },

    #[error(
        "air {air} has {n_columns} columns of stage {stage}, and no hint pilfflonk supports produces them: it \
         proves stage 1 only"
    )]
    StageWithoutHint { air: String, stage: usize, n_columns: u32 },

    #[error("{location} is {value}, which is not below r")]
    ConstantNotBelowR { location: String, value: String },

    /// The extended domain does not fit in the 2-adicity of BN254 (spec A.1).
    #[error("the extended domain has 2^{n_bits_ext} points, and BN254's roots of unity allow at most 2^28")]
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

    // --- What the passes decide (spec §4.2.2, §4.2.3) ----------------------------------------
    /// The symbolic passes stopped: `pil-info` panics on what it cannot process (plan R6, until
    /// M27), and the setup reports the panic's message.
    #[error("the symbolic passes (pil-info) failed: {0}")]
    Passes(String),

    /// What the passes returned does not fit pilfflonk: a value of dimension other than 1, an
    /// evaluation of an unknown kind, the quotient's pieces out of place. A bug of the passes or
    /// of this crate, not of the pilout.
    #[error("the symbolic passes (pil-info) returned what pilfflonk cannot use: {0}")]
    PassesOutput(String),

    /// The constraints' degree in the columns is 0: `qDeg` (A.1) would be negative.
    #[error("the constraints give qDeg = {0}: some constraint must depend on a column (spec A.1)")]
    QDegree(i64),

    // --- What the setup refuses in its arguments (spec §4.2) ----------------------------------
    #[error("--max-constraint-degree {0}: the degree search starts at 2 (spec A.1, D5)")]
    MaxConstraintDegree(u64),

    /// Splitting `Q` is not implemented yet (plan R3, until M33).
    #[error("--max-q-degree {0}: splitting Q is not implemented yet, leave it at 0")]
    QSplitting(u64),

    /// The grouping packs nothing yet (plan R1, until M22).
    #[error("packing the f_i is not implemented yet: pass --no-packing, which forces k = 1")]
    Packing,

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
