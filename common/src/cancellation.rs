use std::sync::RwLock;

use tokio_util::sync::CancellationToken;

use crate::ProofmanError;

/// The cancellation of the current proof. One `RwLock<CancellationInfo>` is shared by `Proofman`,
/// which cancels and resets it, and by `ProofCtx`, through which the witness components read it.
#[derive(Debug, Default)]
pub struct CancellationInfo {
    /// Replaced on every [`Self::reset`], so a clone stays with the proof it was taken in: whoever
    /// follows the current proof reads the token through the shared lock.
    pub token: CancellationToken,
    pub error: Option<ProofmanError>,
}

impl CancellationInfo {
    pub fn cancel(&mut self, error: Option<ProofmanError>) {
        self.token.cancel();
        if self.error.is_some() {
            return;
        }
        if let Some(err) = error {
            self.error = Some(err);
        }
    }

    pub fn reset(&mut self) {
        self.token = CancellationToken::new();
        self.error = None;
    }
}

/// Poison-tolerant access to the shared cancellation hub: recovering the guard avoids a poison
/// storm aborting teardown on every thread. Safe because `token` is atomic and `error` a plain
/// field, so the worst a reader sees after a mid-write panic is a stale error, never UB.
pub trait CancellationInfoExt {
    fn read_recover(&self) -> std::sync::RwLockReadGuard<'_, CancellationInfo>;
    fn write_recover(&self) -> std::sync::RwLockWriteGuard<'_, CancellationInfo>;
}

impl CancellationInfoExt for RwLock<CancellationInfo> {
    fn read_recover(&self) -> std::sync::RwLockReadGuard<'_, CancellationInfo> {
        self.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write_recover(&self) -> std::sync::RwLockWriteGuard<'_, CancellationInfo> {
        self.write().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// `ProofCtx` holds the shared lock and not a token, because a reset replaces the token: a reader
    /// of the lock sees the proof cancelled and the next one not, a token clone stays cancelled.
    #[test]
    fn a_reader_of_the_shared_lock_follows_the_proof_across_a_reset() {
        let info = Arc::new(RwLock::new(CancellationInfo::default()));
        let reader = info.clone();
        let token_of_first_proof = info.read_recover().token.clone();
        assert!(!reader.read_recover().token.is_cancelled(), "a new proof starts uncancelled");

        info.write_recover().cancel(None);
        assert!(reader.read_recover().token.is_cancelled());

        info.write_recover().reset();
        assert!(!reader.read_recover().token.is_cancelled());
        assert!(token_of_first_proof.is_cancelled());
    }
}
