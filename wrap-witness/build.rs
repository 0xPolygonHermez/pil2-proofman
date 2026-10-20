//! Compiles `src/zkin.cpp` against the system's nlohmann/json, the one `setup/final_snark_circom/`
//! compiles final.so against: the zkin it makes crosses into final.so's `getWitness` as a C++ object.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/zkin.cpp");
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").warnings(true).extra_warnings(true).file("src/zkin.cpp");
    // Apple's clang does not search Homebrew's prefix, where `brew install nlohmann-json` puts it.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let prefix = Command::new("brew")
            .arg("--prefix")
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "/opt/homebrew".to_string());
        build.include(format!("{prefix}/include")).include(format!("{prefix}/opt/nlohmann-json/include"));
    }
    build.compile("zkin");
}
