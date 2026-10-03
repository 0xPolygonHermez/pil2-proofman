//! The inputs of the pilfflonk benchmark (`pilfflonk/bench/bench.sh`,
//! pilfflonk/docs/performance.md#reproducing): a test ptau, and the witness directory of a benchmark
//! program at the size of its key. It is the example `pilfflonk_bench_inputs` of `proofman-cli`,
//! whose dev-dependencies it uses: the test ptau of `pilfflonk-setup` (feature `test-ptau`,
//! pilfflonk/docs/README.md#tests) and the fixtures' witness generators of `pilfflonk/tests/data/`,
//! as the CLI tests include them.
//!
//! ```text
//! cargo run --release --features proofman-starks-lib-c/cpu-only -p proofman-cli \
//!     --example pilfflonk_bench_inputs -- ptau <powers> <out.ptau>
//! cargo run --release --features proofman-starks-lib-c/cpu-only -p proofman-cli \
//!     --example pilfflonk_bench_inputs -- witness <fibonacci|all> <provingKey> <out dir>
//! ```
//!
//! - `ptau`: `fixed_tau_ptau` of that many powers `[τ^i]₁`, with the tests' `τ` (`TEST_TAU`): a
//!   full-width `τ`, so the MSMs see the points of a real SRS. Never a ptau to prove anything with:
//!   its `τ` is public.
//! - `witness`: the witness of `pilfflonk/bench/<program>.pil` for the `2^nBits` rows of the key,
//!   written as a witness directory (pilfflonk/docs/formats.md#witness-directory) for
//!   `pilfflonk prove --witness`: the Fibonacci's of `tests/data/fibonacci.rs` and `all`'s of
//!   `tests/data/all.rs` (`witness_of_size`), both with the inputs `[1, 2]`. It reads the key's JSON
//!   files only, not its SRS or `.const`.

#[allow(dead_code)]
#[path = "../tests/data/all.rs"]
mod all;
#[allow(dead_code)]
#[path = "../tests/data/connection.rs"]
mod connection;
#[path = "../tests/data/fibonacci.rs"]
mod fibonacci;
#[allow(dead_code)]
#[path = "../tests/data/permutation.rs"]
mod permutation;
#[allow(dead_code)]
#[path = "../tests/data/plookup.rs"]
mod plookup;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pilfflonk_setup::test_ptau::{test_tau, write_fixed_tau_ptau};
use proofman_pilfflonk::{AirFile, JsonFile, PilfflonkGlobalInfo, PilfflonkInfo, WitnessShape};

const USAGE: &str = "usage: pilfflonk_bench_inputs ptau <powers> <out.ptau>\n       \
                     pilfflonk_bench_inputs witness <fibonacci|all> <provingKey> <out dir>";

/// The Fibonacci's inputs `[in1, in2]`, those of the fixtures.
const INPUTS: [u64; 2] = [1, 2];

fn ptau(powers: &str, path: &Path) -> Result<(), String> {
    let n_g1: usize = powers.parse().map_err(|_| format!("{powers} is not a number of powers"))?;
    if n_g1 == 0 {
        return Err("a ptau has at least one power".into());
    }
    write_fixed_tau_ptau(path, n_g1, &test_tau()).map_err(|e| format!("{}: {e}", path.display()))
}

fn witness(program: &str, proving_key: &Path, dir: &Path) -> Result<(), String> {
    let global_info = PilfflonkGlobalInfo::from_proving_key(proving_key).map_err(|e| e.to_string())?;
    let info_path = global_info.air_file(proving_key, 0, 0, AirFile::PilfflonkInfo).map_err(|e| e.to_string())?;
    let info = PilfflonkInfo::read(&info_path).map_err(|e| e.to_string())?;
    let shape = WitnessShape::from_proving_key(&global_info, &[&info]).map_err(|e| e.to_string())?;
    let n_bits = u32::try_from(info.n_bits).map_err(|_| format!("nBits = {} is not a size", info.n_bits))?;
    let generated = match program {
        "fibonacci" => fibonacci::witness(n_bits, INPUTS),
        "all" => all::witness_of_size(n_bits, INPUTS),
        _ => return Err(format!("no benchmark program {program}: fibonacci or all")),
    };
    generated.write(dir, &shape).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["ptau", powers, path] => ptau(powers, &PathBuf::from(path)),
        ["witness", program, proving_key, dir] => witness(program, &PathBuf::from(proving_key), &PathBuf::from(dir)),
        _ => Err(USAGE.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
