use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_MPI").is_err() {
        return;
    }

    // Probe for MPI via pkg-config first, then fall back to mpicc
    let found = probe_pkg_config() || probe_mpicc();

    if !found {
        panic!(
            "MPI not found. Install an MPI toolchain (e.g. `apt install libopenmpi-dev` or \
             `brew install open-mpi`) or build without MPI: `cargo build --no-default-features`"
        );
    }
}

fn probe_pkg_config() -> bool {
    Command::new("pkg-config")
        .args(["--exists", "ompi", "ompi-c", "mpich"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
        || Command::new("pkg-config").args(["--exists", "ompi"]).status().map(|s| s.success()).unwrap_or(false)
        || Command::new("pkg-config").args(["--exists", "mpich"]).status().map(|s| s.success()).unwrap_or(false)
}

fn probe_mpicc() -> bool {
    Command::new("mpicc").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}
