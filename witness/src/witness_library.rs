use crate::WitnessManager;
use libloading::Library;
use proofman_fields::PrimeField64;
use proofman_common::{PackedInfo, ProofCtx, ProofmanError, ProofmanResult, VerboseMode, RankInfo};
use std::collections::HashMap;
use std::path::Path;

/// This is the type of the function that is used to load a witness library.
pub type WitnessLibInitFn<F> = fn(VerboseMode, Option<RankInfo>) -> ProofmanResult<Box<dyn WitnessLibrary<F>>>;

pub trait WitnessLibrary<F: PrimeField64> {
    fn register_witness(&mut self, wcm: &WitnessManager<F>) -> ProofmanResult<()>;

    /// Returns the weight indicating the complexity of the witness computation.
    ///
    /// Used as a heuristic for estimating computational cost.
    fn get_witness_weight(&self, _pctx: &ProofCtx<F>, _global_id: usize) -> ProofmanResult<usize> {
        Ok(1)
    }
}

#[macro_export]
macro_rules! witness_library {
    ($lib_name:ident, $field_type:ty) => {
        // Define the struct
        pub struct $lib_name;

        // Define the init_library function
        #[no_mangle]
        pub extern "Rust" fn init_library(
            verbose_mode: proofman_common::VerboseMode,
            rank: Option<proofman_common::RankInfo>,
        ) -> proofman_common::ProofmanResult<Box<dyn $crate::WitnessLibrary<$field_type>>> {
            proofman_common::initialize_logger(verbose_mode, rank.as_ref());

            Ok(Box::new($lib_name))
        }
    };
}

/// Type of the optional `packed_info` entry point a witness library may export. A standalone symbol
/// because packing sizes the buffer pool before the library is loaded as a `WitnessLibrary`; Rust
/// ABI, like the `init_library` vtable beside it, so both sides must share a toolchain.
pub type WitnessLibPackedInfoFn = fn() -> HashMap<(usize, usize), PackedInfo>;

/// Read a witness library's per-air trace packing, or an empty map if it exports none.
///
/// The handle is deliberately leaked: the returned map is plain owned data, but `ProofMan`
/// re-opens the same path later and dropping ours first would unload the library.
pub fn load_packed_info(witness_lib_path: &Path) -> ProofmanResult<HashMap<(usize, usize), PackedInfo>> {
    if !witness_lib_path.exists() {
        return Err(ProofmanError::InvalidParameters(format!(
            "Witness computation dynamic library not found at path: {witness_lib_path:?}"
        )));
    }

    let library = unsafe { Library::new(witness_lib_path) }
        .map_err(|e| ProofmanError::InvalidParameters(format!("Failed to load {witness_lib_path:?}: {e}")))?;

    let packed_info = match unsafe { library.get::<WitnessLibPackedInfoFn>(b"packed_info") } {
        Ok(f) => f(),
        Err(_) => HashMap::new(),
    };

    std::mem::forget(library);
    Ok(packed_info)
}

/// Export a witness library's per-air trace packing as the `packed_info` symbol that
/// [`load_packed_info`] looks up. Optional: omit it and the prover runs unpacked.
#[macro_export]
macro_rules! witness_packed_info {
    ($body:expr) => {
        #[no_mangle]
        pub extern "Rust" fn packed_info() -> std::collections::HashMap<(usize, usize), proofman_common::PackedInfo> {
            $body
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `packed_info` export/import contract, end to end against a real witness library. Skips
    /// when the example has not been built, since it needs the `.so`.
    #[test]
    fn a_library_exporting_packed_info_round_trips_through_load() {
        let lib = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/release/libhashes.so");
        if !lib.exists() {
            eprintln!("skipping: build it with `cargo build -p hashes --release`");
            return;
        }

        let packed = load_packed_info(&lib).expect("loads");
        if packed.is_empty() {
            return;
        }
        for ((airgroup_id, air_id), info) in &packed {
            assert!(info.is_packed, "air ({airgroup_id}, {air_id}) is declared but not packed");
            assert!(info.num_packed_words > 0, "air ({airgroup_id}, {air_id}) packs into no words");
            assert!(
                info.num_packed_words < info.unpack_info.len() as u64,
                "air ({airgroup_id}, {air_id}) packs {} words for {} columns, which saves nothing",
                info.num_packed_words,
                info.unpack_info.len()
            );
        }
    }

    /// A library with no such symbol is not an error: packing is opt-in.
    #[test]
    fn a_library_without_the_symbol_reports_no_packing() {
        let lib = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/release/libfibonacci_square.so");
        if !lib.exists() {
            eprintln!("skipping: build it with `cargo build -p fibonacci-square --release`");
            return;
        }
        assert!(load_packed_info(&lib).expect("loads").is_empty());
    }
}
