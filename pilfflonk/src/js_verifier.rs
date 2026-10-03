//! The JS verifier (pilfflonk/docs/verifier.md#js-verifier), run with Node as `verify_snark_proof`
//! (`proofman/src/snark_wrapper.rs`) runs snarkjs: `node <js>/bin/verify.js <vkey> <publics>
//! <proof>`, the arguments of `snarkjs fflonk verify`, whose exit status is the verdict.
//! `proofman-cli pilfflonk verify` calls [`verify`].
//!
//! The verifier is this crate's `js/` directory, whose path is fixed at compile time, or the
//! directory [`JS_DIR_VAR`] names: a copy of it, for a binary run away from the source tree. Its
//! dependencies, the `dependencies` of its `package.json` (`ffjavascript` and `@noble/hashes`),
//! are looked for as Node looks for the packages the verifier imports: in a `node_modules/` of
//! that directory or of one of its ancestors. If one is missing, `npm install` runs there, on its
//! own `package.json`, as the STARK setup's `node_deps::ensure_node_deps` does in its crate's
//! directory (`setup/pil2-stark/src/proving_key/node_deps.rs`). The per-user cache that
//! `ensure_node_deps` falls back to does not apply: Node resolves the verifier's imports from its
//! own directory, and never from another one's `node_modules/`.

use std::ffi::OsString;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

use crate::error::{PilfflonkError, PilfflonkResult};

/// The environment variable that names the verifier's directory instead of this crate's `js/`.
pub const JS_DIR_VAR: &str = "PILFFLONK_JS";

/// The verifier's command, relative to its directory: exit status 0 if the proof verifies, 1 if
/// it does not (a malformed input included), 2 if a file cannot be read, or not as JSON.
const VERIFY_SCRIPT: &str = "bin/verify.js";

/// The programs the verifier runs with.
#[derive(Clone, Copy, Debug)]
struct Tools<'a> {
    node: &'a str,
    npm: &'a str,
}

fn js_error(message: String) -> PilfflonkError {
    PilfflonkError::JsVerifier(message)
}

/// Whether the proof at `proof` verifies against the vkey at `vkey` (`pilfflonk.vkey.json`) and
/// the publics at `publics`, by the JS verifier: `Ok(false)` for a proof it rejects, malformed
/// inputs included, with its reasons on stderr. An error if it cannot give a verdict (see
/// [`PilfflonkError::JsVerifier`]).
pub fn verify(vkey: &Path, publics: &Path, proof: &Path) -> PilfflonkResult<bool> {
    let dir = verifier_dir(std::env::var_os(JS_DIR_VAR))?;
    run_verifier(Tools { node: "node", npm: "npm" }, &dir, [vkey, publics, proof])
}

/// The verifier's directory, canonical: `from_env` if set and not empty, and this crate's `js/`
/// otherwise.
fn verifier_dir(from_env: Option<OsString>) -> PilfflonkResult<PathBuf> {
    let dir = match from_env {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("js"),
    };
    let dir = dir.canonicalize().unwrap_or(dir);
    if !dir.join(VERIFY_SCRIPT).is_file() || !dir.join("package.json").is_file() {
        return Err(js_error(format!(
            "{} is not the pilfflonk verifier, which has {VERIFY_SCRIPT} and package.json: set {JS_DIR_VAR} to a \
             copy of pilfflonk/js",
            dir.display()
        )));
    }
    Ok(dir)
}

/// Runs the verifier of `dir` on `[vkey, publics, proof]`, after checking that Node runs and that
/// the verifier's dependencies are installed.
fn run_verifier(tools: Tools, dir: &Path, inputs: [&Path; 3]) -> PilfflonkResult<bool> {
    check_node(tools.node)?;
    ensure_dependencies(dir, tools.npm)?;
    let script = dir.join(VERIFY_SCRIPT);
    let status = Command::new(tools.node)
        .arg(&script)
        .args(inputs)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| js_error(format!("cannot run `{} {}`: {e}", tools.node, script.display())))?;
    match status.code() {
        Some(0) => Ok(true),
        Some(1) => Ok(false),
        Some(2) => Err(js_error(format!(
            "{VERIFY_SCRIPT} cannot read the vkey, the publics or the proof (see its message above)"
        ))),
        _ => Err(js_error(format!("{} failed: {status}", script.display()))),
    }
}

/// Checks that `node` runs: the verifier is JS.
fn check_node(node: &str) -> PilfflonkResult<()> {
    let status = Command::new(node).arg("--version").stdin(Stdio::null()).stdout(Stdio::null()).status();
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(js_error(format!("`{node} --version` exited with {status}"))),
        Err(e) if e.kind() == ErrorKind::NotFound => Err(js_error(format!(
            "Node.js is not installed (no `{node}` on the PATH), and the pilfflonk verifier is JS: install Node.js \
             18 or later, with npm"
        ))),
        Err(e) => Err(js_error(format!("cannot run `{node}`: {e}"))),
    }
}

/// The packages of the `dependencies` of the verifier's `package.json`.
fn dependencies(dir: &Path) -> PilfflonkResult<Vec<String>> {
    let path = dir.join("package.json");
    let text = fs::read_to_string(&path).map_err(|source| PilfflonkError::Io { path: path.clone(), source })?;
    let manifest: Value = serde_json::from_str(&text).map_err(|e| PilfflonkError::from(e).in_file(&path))?;
    match manifest.get("dependencies").and_then(Value::as_object) {
        Some(dependencies) => Ok(dependencies.keys().cloned().collect()),
        None => Err(js_error(format!("{} has no dependencies", path.display()))),
    }
}

/// The packages of `dependencies` that Node would not find from `dir`: those in no
/// `node_modules/` of `dir` or of its ancestors.
fn missing_dependencies(dir: &Path, dependencies: &[String]) -> Vec<String> {
    let installed = |package: &String| {
        dir.ancestors().any(|root| root.join("node_modules").join(package).join("package.json").is_file())
    };
    dependencies.iter().filter(|package| !installed(package)).cloned().collect()
}

/// Installs the verifier's dependencies with `npm install` in `dir` if one is missing.
fn ensure_dependencies(dir: &Path, npm: &str) -> PilfflonkResult<()> {
    let dependencies = dependencies(dir)?;
    let missing = missing_dependencies(dir, &dependencies);
    if missing.is_empty() {
        return Ok(());
    }
    tracing::info!("The JS verifier needs {}: running `{npm} install` in {}", missing.join(", "), dir.display());
    match Command::new(npm).arg("install").current_dir(dir).stdin(Stdio::null()).status() {
        Ok(status) if status.success() => {}
        Ok(status) => return Err(js_error(format!("`{npm} install` in {} exited with {status}", dir.display()))),
        Err(e) => {
            return Err(js_error(format!(
                "cannot run `{npm} install` in {} to install {} ({e}): install npm, or those packages there",
                dir.display(),
                missing.join(", ")
            )))
        }
    }
    let missing = missing_dependencies(dir, &dependencies);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(js_error(format!("`{npm} install` in {} did not install {}", dir.display(), missing.join(", "))))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use super::*;

    /// Held by every test that runs a program or writes one, so that none writes a program while
    /// another forks: the child of the fork would hold the file open for writing until it execs,
    /// and running the program then fails with ETXTBSY ("Text file busy").
    fn serial() -> MutexGuard<'static, ()> {
        static SERIAL: Mutex<()> = Mutex::new(());
        SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A fresh directory of its own for a test, removed when it is dropped.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("proofman_pilfflonk_js_{test}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir.canonicalize().unwrap())
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// `root/node_modules/<package>` as npm leaves it.
    fn install(root: &Path, package: &str) {
        let dir = root.join("node_modules").join(package);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("package.json"), "{}").unwrap();
    }

    /// A verifier directory with this crate's package.json and a script.
    fn verifier(root: &Path) -> PathBuf {
        let dir = root.join("js");
        fs::create_dir_all(dir.join("bin")).unwrap();
        fs::write(dir.join(VERIFY_SCRIPT), "").unwrap();
        fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("js/package.json"), dir.join("package.json")).unwrap();
        dir
    }

    const DEPENDENCIES: [&str; 2] = ["@noble/hashes", "ffjavascript"];

    #[test]
    fn the_dependencies_are_those_of_the_package_json() {
        let tmp = TempDir::new("dependencies");
        let mut dependencies = dependencies(&verifier(&tmp.0)).unwrap();
        dependencies.sort();
        assert_eq!(dependencies, DEPENDENCIES);
        fs::write(tmp.0.join("js/package.json"), "{\"name\": \"x\"}").unwrap();
        assert!(super::dependencies(&tmp.0.join("js")).unwrap_err().to_string().contains("has no dependencies"));
    }

    /// As Node finds them: in the verifier's node_modules or an ancestor's, not a sibling's.
    #[test]
    fn the_dependencies_are_found_where_node_finds_them() {
        let tmp = TempDir::new("found");
        let dir = verifier(&tmp.0);
        let all: Vec<String> = DEPENDENCIES.map(String::from).to_vec();
        assert_eq!(missing_dependencies(&dir, &all), all);
        install(&dir, "ffjavascript");
        install(&tmp.0.join("sibling"), "@noble/hashes");
        assert_eq!(missing_dependencies(&dir, &all), ["@noble/hashes"]);
        install(&tmp.0, "@noble/hashes");
        assert!(missing_dependencies(&dir, &all).is_empty());
        // A directory without its package.json is not an installed package.
        fs::remove_file(tmp.0.join("node_modules/@noble/hashes/package.json")).unwrap();
        assert_eq!(missing_dependencies(&dir, &all), ["@noble/hashes"]);
    }

    #[test]
    fn the_verifier_is_this_crates_js_or_the_directory_named() {
        let js = Path::new(env!("CARGO_MANIFEST_DIR")).join("js").canonicalize().unwrap();
        assert_eq!(verifier_dir(None).unwrap(), js);
        assert_eq!(verifier_dir(Some(OsString::new())).unwrap(), js);
        let tmp = TempDir::new("dir");
        let dir = verifier(&tmp.0);
        assert_eq!(verifier_dir(Some(dir.clone().into_os_string())).unwrap(), dir);
        let err = verifier_dir(Some(tmp.0.clone().into_os_string())).unwrap_err().to_string();
        assert!(err.contains("is not the pilfflonk verifier") && err.contains(JS_DIR_VAR), "{err}");
    }

    #[test]
    fn no_node_is_a_clear_error() {
        let _serial = serial();
        let err = check_node("/nonexistent/node").unwrap_err().to_string();
        assert!(err.contains("Node.js is not installed"), "{err}");
    }

    #[cfg(unix)]
    mod with_fake_programs {
        use std::os::unix::fs::PermissionsExt;

        use super::*;

        /// An executable shell script at `dir/name` that appends a line to `dir/runs.log` and then
        /// runs `body` in the directory it is run in.
        fn program(dir: &Path, name: &str, body: &str) -> String {
            let path = dir.join(name);
            let log = dir.join("runs.log");
            fs::write(&path, format!("#!/bin/sh\necho \"$@\" >> '{}'\n{body}\n", log.display())).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
            path.to_string_lossy().into_owned()
        }

        fn runs(dir: &Path) -> Vec<String> {
            fs::read_to_string(dir.join("runs.log")).map(|s| s.lines().map(String::from).collect()).unwrap_or_default()
        }

        const INSTALLS: &str = "mkdir -p node_modules/ffjavascript node_modules/@noble/hashes && \
            touch node_modules/ffjavascript/package.json node_modules/@noble/hashes/package.json";

        #[test]
        fn npm_does_not_run_when_the_dependencies_are_installed() {
            let _serial = serial();
            let tmp = TempDir::new("installed");
            let dir = verifier(&tmp.0);
            DEPENDENCIES.iter().for_each(|p| install(&dir, p));
            let npm = program(&tmp.0, "npm", "exit 1");
            ensure_dependencies(&dir, &npm).unwrap();
            assert!(runs(&tmp.0).is_empty());
        }

        #[test]
        fn npm_installs_the_missing_dependencies_in_the_verifiers_directory() {
            let _serial = serial();
            let tmp = TempDir::new("install");
            let dir = verifier(&tmp.0);
            let npm = program(&tmp.0, "npm", INSTALLS);
            ensure_dependencies(&dir, &npm).unwrap();
            assert_eq!(runs(&tmp.0), ["install"]);
            assert!(missing_dependencies(&dir, &super::dependencies(&dir).unwrap()).is_empty());
            // Once installed, npm does not run again.
            ensure_dependencies(&dir, &npm).unwrap();
            assert_eq!(runs(&tmp.0), ["install"]);
        }

        #[test]
        fn an_npm_that_fails_or_installs_nothing_is_an_error() {
            let _serial = serial();
            let tmp = TempDir::new("npm_fails");
            let dir = verifier(&tmp.0);
            let err = ensure_dependencies(&dir, &program(&tmp.0, "failing-npm", "exit 3")).unwrap_err().to_string();
            assert!(err.contains("install` in") && err.contains("exited with"), "{err}");
            let err = ensure_dependencies(&dir, &program(&tmp.0, "idle-npm", "")).unwrap_err().to_string();
            assert!(err.contains("did not install @noble/hashes, ffjavascript"), "{err}");
            let err = ensure_dependencies(&dir, "/nonexistent/npm").unwrap_err().to_string();
            assert!(err.contains("cannot run"), "{err}");
        }

        /// The verdict is the exit status: 0 verifies, 1 does not, anything else is no verdict.
        #[test]
        fn the_exit_status_is_the_verdict() {
            let _serial = serial();
            let tmp = TempDir::new("verdict");
            let dir = verifier(&tmp.0);
            DEPENDENCIES.iter().for_each(|p| install(&dir, p));
            let inputs = [Path::new("vkey.json"), Path::new("publics.json"), Path::new("proof.json")];
            let node = |code: u8| {
                let body = format!("[ \"$1\" = --version ] && exit 0\nexit {code}");
                program(&tmp.0, &format!("node{code}"), &body)
            };
            let verdict = |code| run_verifier(Tools { node: &node(code), npm: "/nonexistent/npm" }, &dir, inputs);
            assert!(verdict(0).unwrap());
            assert!(!verdict(1).unwrap());
            assert!(verdict(2).unwrap_err().to_string().contains("cannot read the vkey, the publics or the proof"));
            assert!(verdict(3).unwrap_err().to_string().contains("failed"));
            // The verifier gets its script and the three paths, in the order of snarkjs.
            let script = dir.join(VERIFY_SCRIPT);
            assert!(runs(&tmp.0).contains(&format!("{} vkey.json publics.json proof.json", script.display())));
        }
    }
}
