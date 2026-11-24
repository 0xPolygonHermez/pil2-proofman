//! Shared helpers for the proving-key generators.

use anyhow::{bail, Result};
use std::path::Path;

/// Exact log2 of a row count: `Err` when `n` is 0 or not a power of two.
pub fn exact_log2(n: u64) -> Result<usize> {
    if !n.is_power_of_two() {
        bail!("{n} is not a power of two");
    }
    Ok(n.trailing_zeros() as usize)
}

/// A circuit's `.const` file must exist to compute its verkey: `Err` naming `name` and the path.
pub fn require_const_file(name: &str, path: &Path) -> Result<()> {
    if !path.exists() {
        bail!("{name}: const file {} missing; cannot compute the verkey", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_log2_rejects_non_powers() {
        assert!(exact_log2(3).is_err());
        assert!(exact_log2(0).is_err());
        assert!(exact_log2((1 << 17) + 1).is_err());
        assert_eq!(exact_log2(1 << 17).unwrap(), 17);
        assert_eq!(exact_log2(1).unwrap(), 0);
    }

    #[test]
    fn missing_const_file_is_an_error() {
        let e = require_const_file("Circ", Path::new("/nonexistent/dir/Circ.const")).unwrap_err().to_string();
        assert!(e.contains("Circ") && e.contains("Circ.const") && e.contains("missing"), "{e}");
        let f = std::env::temp_dir().join("pk_util_const_present.const");
        std::fs::write(&f, b"x").unwrap();
        assert!(require_const_file("Circ", &f).is_ok());
    }
}
