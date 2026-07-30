use thiserror::Error;

#[derive(Error, Debug)]
pub enum MlError {
    #[error("sumcheck round {round} inconsistent: g(0) + g(1) != previous claim")]
    SumcheckRound { round: usize },
    #[error("final sumcheck check failed: {0}")]
    FinalCheck(String),
    #[error("merkle path verification failed ({0})")]
    MerklePath(String),
    #[error("FRI fold consistency check failed at query {query}, level {level}")]
    FoldConsistency { query: usize, level: usize },
    #[error("constraint {constraint} does not vanish: {0}", constraint = .1)]
    Constraint(String, usize),
    #[error("malformed proof: {0}")]
    Malformed(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("io error: {0}")]
    Io(String),
}
