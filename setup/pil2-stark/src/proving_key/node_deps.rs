//! Self-bootstrapping resolution of the Node.js tooling dependencies
//! (pil2-compiler, snarkjs, circomlib) declared in this crate's `package.json`
//! (`setup/pil2-stark/package.json`).
//!
//! Search/bootstrap order:
//!   1. An already-populated `node_modules` at the compile-time-baked crate dir,
//!      the cwd, or any ancestor of the running executable — no npm needed.
//!   2. The baked crate dir's own `package.json` → `npm install` there.
//!   3. A per-user cache dir seeded with the manifest embedded in this binary —
//!      makes a prebuilt binary self-sufficient on machines without the source tree.
//!
//! Air-gapped machines can't bootstrap (npm needs the network); the per-tool env
//! overrides (`PIL2C_EXEC`, `SNARKJS_PATH`, `CIRCOMLIB_PATH`) remain the escape hatch.

use std::path::{Path, PathBuf};

/// This crate's Node manifest, embedded at compile time so it travels with the
/// binary and the crate is self-contained for publishing (used to seed the
/// per-user cache when running off-tree).
const EMBEDDED_PACKAGE_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/package.json"));

/// Return the first root whose `node_modules/<probe>` exists, canonicalized.
fn find_in_roots(roots: &[PathBuf], probe: &str) -> Option<PathBuf> {
    roots.iter().find(|r| r.join("node_modules").join(probe).exists()).and_then(|r| r.canonicalize().ok())
}

/// Run `npm install` (program `npm`) in `root`, logging failures. True on success.
fn npm_install(root: &Path, npm: &str) -> bool {
    match std::process::Command::new(npm).arg("install").current_dir(root).status() {
        Ok(status) if status.success() => true,
        Ok(status) => {
            tracing::warn!("`npm install` in {} exited with {status}", root.display());
            false
        }
        Err(e) => {
            tracing::warn!("failed to run `npm install` in {}: {e}", root.display());
            false
        }
    }
}

/// Ensure `root` (a dir owned by us, created on demand) holds `manifest` as its
/// `package.json` and a `node_modules/<probe>`, running npm when missing or when
/// the manifest on disk differs from `manifest` (binary upgraded). Concurrent
/// callers serialize on an advisory file lock inside `root`.
fn ensure_cache_deps(root: &Path, manifest: &str, npm: &str, probe: &str) -> Option<PathBuf> {
    std::fs::create_dir_all(root).ok()?;
    let lock = std::fs::File::create(root.join(".install.lock")).ok()?;
    lock.lock().ok()?;

    let manifest_path = root.join("package.json");
    let fresh = std::fs::read_to_string(&manifest_path).is_ok_and(|on_disk| on_disk == manifest);
    if fresh && root.join("node_modules").join(probe).exists() {
        return Some(root.to_path_buf());
    }
    if !fresh {
        std::fs::write(&manifest_path, manifest).ok()?;
        // A lockfile resolved from the old manifest would pin old versions.
        let _ = std::fs::remove_file(root.join("package-lock.json"));
    }
    if !npm_install(root, npm) {
        return None;
    }
    root.join("node_modules").join(probe).exists().then(|| root.to_path_buf())
}

/// Per-user cache location: `$XDG_CACHE_HOME` or `~/.cache`, under `pil2-proofman/node-deps`.
fn user_cache_root() -> Option<PathBuf> {
    if let Ok(x) = std::env::var("XDG_CACHE_HOME") {
        if !x.is_empty() {
            return Some(PathBuf::from(x).join("pil2-proofman/node-deps"));
        }
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/pil2-proofman/node-deps"))
}

/// Return a directory whose `node_modules/<probe>` exists, bootstrapping with
/// `npm install` if necessary (see module docs for the order). `probe` is a
/// path relative to `node_modules`, e.g. `.bin/pil2com` or `circomlib/circuits`.
pub(crate) fn ensure_node_deps(probe: &str) -> Option<PathBuf> {
    // node_modules / package.json live in this crate's dir (setup/pil2-stark).
    const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

    let mut roots = vec![PathBuf::from(CRATE_ROOT)];
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent();
        while let Some(d) = dir {
            roots.push(d.to_path_buf());
            dir = d.parent();
        }
    }
    if let Some(root) = find_in_roots(&roots, probe) {
        return Some(root);
    }

    // The baked crate dir's package.json is this crate's own manifest (source
    // checkout or cargo git checkout), so installing there is always correct;
    // a package.json at the cwd could belong to the consumer's unrelated
    // project, so we never npm-install anywhere else.
    let baked = Path::new(CRATE_ROOT);
    let baked = baked.canonicalize().unwrap_or_else(|_| baked.to_path_buf());
    if baked.join("package.json").is_file() {
        tracing::info!("Node deps not found; running `npm install` in {}", baked.display());
        if npm_install(&baked, "npm") && baked.join("node_modules").join(probe).exists() {
            return Some(baked);
        }
    }

    let cache = user_cache_root()?;
    tracing::info!("Bootstrapping Node deps into {}", cache.display());
    ensure_cache_deps(&cache, EMBEDDED_PACKAGE_JSON, "npm", probe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Write an executable fake-npm shell script and return its path.
    /// The script runs with cwd set to the install root and appends a line to
    /// `runs.log` there, so tests can assert how many times npm was invoked.
    fn fake_npm(dir: &Path, extra: &str) -> String {
        let p = dir.join("fake-npm.sh");
        std::fs::write(&p, format!("#!/bin/sh\necho run >> runs.log\n{extra}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn npm_runs(root: &Path) -> usize {
        std::fs::read_to_string(root.join("runs.log")).map(|s| s.lines().count()).unwrap_or(0)
    }

    #[test]
    fn find_in_roots_picks_first_populated_root() {
        let tmp = tempfile::tempdir().unwrap();
        let empty = tmp.path().join("empty");
        let populated = tmp.path().join("populated");
        std::fs::create_dir_all(populated.join("node_modules/snarkjs")).unwrap();
        std::fs::create_dir_all(&empty).unwrap();

        let roots = vec![empty, populated.clone()];
        let found = find_in_roots(&roots, "snarkjs").unwrap();
        assert_eq!(found, populated.canonicalize().unwrap());
    }

    #[test]
    fn find_in_roots_returns_none_when_probe_missing() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("a/node_modules/other")).unwrap();
        let roots = vec![tmp.path().join("a"), tmp.path().join("does-not-exist")];
        assert!(find_in_roots(&roots, "snarkjs").is_none());
    }

    #[test]
    fn cache_bootstrap_writes_manifest_and_installs() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let npm = fake_npm(tmp.path(), "mkdir -p node_modules/.bin && touch node_modules/.bin/pil2com");

        let got = ensure_cache_deps(&cache, "{\"v\":1}", &npm, ".bin/pil2com").unwrap();
        assert_eq!(got, cache);
        assert_eq!(std::fs::read_to_string(cache.join("package.json")).unwrap(), "{\"v\":1}");
        assert_eq!(npm_runs(&cache), 1);
        assert!(cache.join("node_modules/.bin/pil2com").is_file());
    }

    #[test]
    fn cache_skips_npm_when_fresh() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        std::fs::create_dir_all(cache.join("node_modules/.bin")).unwrap();
        std::fs::write(cache.join("node_modules/.bin/pil2com"), "").unwrap();
        std::fs::write(cache.join("package.json"), "{\"v\":1}").unwrap();
        let npm = fake_npm(tmp.path(), "");

        let got = ensure_cache_deps(&cache, "{\"v\":1}", &npm, ".bin/pil2com").unwrap();
        assert_eq!(got, cache);
        assert_eq!(npm_runs(&cache), 0);
    }

    #[test]
    fn cache_reinstalls_when_manifest_stale() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        std::fs::create_dir_all(cache.join("node_modules/.bin")).unwrap();
        std::fs::write(cache.join("node_modules/.bin/pil2com"), "").unwrap();
        std::fs::write(cache.join("package.json"), "{\"v\":1}").unwrap();
        std::fs::write(cache.join("package-lock.json"), "{}").unwrap();
        let npm = fake_npm(tmp.path(), "mkdir -p node_modules/.bin && touch node_modules/.bin/pil2com");

        let got = ensure_cache_deps(&cache, "{\"v\":2}", &npm, ".bin/pil2com").unwrap();
        assert_eq!(got, cache);
        assert_eq!(npm_runs(&cache), 1);
        assert_eq!(std::fs::read_to_string(cache.join("package.json")).unwrap(), "{\"v\":2}");
        // stale lockfile from the old manifest must not pin old versions
        assert!(!cache.join("package-lock.json").exists());
    }

    #[test]
    fn cache_fails_when_npm_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let npm = fake_npm(tmp.path(), "exit 1");
        assert!(ensure_cache_deps(&cache, "{}", &npm, ".bin/pil2com").is_none());
    }

    /// Real npm + network + git: full bootstrap of the embedded manifest into a
    /// scratch dir, as a prebuilt binary on a clean machine would do.
    /// Run explicitly: cargo test real_npm_bootstrap -- --ignored
    #[test]
    #[ignore]
    fn real_npm_bootstrap_of_embedded_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("node-deps");
        let got = ensure_cache_deps(&cache, EMBEDDED_PACKAGE_JSON, "npm", ".bin/pil2com").unwrap();
        assert!(got.join("node_modules/.bin/pil2com").exists());
        assert!(got.join("node_modules/snarkjs").is_dir());
        assert!(got.join("node_modules/circomlib/circuits").is_dir());
    }

    #[test]
    fn cache_fails_when_npm_does_not_produce_probe() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");
        let npm = fake_npm(tmp.path(), "");
        assert!(ensure_cache_deps(&cache, "{}", &npm, ".bin/pil2com").is_none());
    }
}
