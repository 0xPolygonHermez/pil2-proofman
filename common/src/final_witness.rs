//! The witness library of the final SNARK circuit: the `final.so` (`final.dylib` on macOS) that
//! setup-snark builds from the circuit's circom C++ and `setup/final_snark_circom/`, with its
//! `final.dat`. Its `getWitness` computes the circuit's witness over BN254's `Fr` from a zkin, a
//! `nlohmann::json`. The PLONK and FFLONK wraps hand it the recursivef proof
//! (`proofman::generate_witness_final_snark`); the pilfflonk wrap's witness (`pilfflonk-wrap-witness`)
//! hands it that proof too, or a zkin read from a file.

use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::path::Path;

use libloading::{Library, Symbol};

use crate::{GetSizeWitnessFunc, ProofmanError, ProofmanResult};

/// `getWitness(zkin, datFile, pWitness, nMutexes)` (setup/final_snark_circom/main.cpp): 0 on
/// success, and then `pWitness` holds the witness.
pub type GetWitnessFinalFunc =
    unsafe extern "C" fn(zkin: *mut c_void, dat_file: *const c_char, witness: *mut c_void, n_mutexes: u64) -> i64;

/// Bytes of a value of the witness: an element of `Fr`, canonical, little-endian.
pub const FINAL_WITNESS_VALUE_BYTES: usize = 32;

/// A loaded witness library of the final circuit, and the `.dat` it computes with. The library is
/// unloaded when this is dropped.
#[derive(Debug)]
pub struct FinalWitnessLibrary {
    library: Library,
    dat: CString,
}

impl FinalWitnessLibrary {
    /// Loads the library at `library`, which will compute with the `.dat` at `dat`. Refuses a
    /// library that does not exist, and one that cannot be loaded.
    pub fn load(library: &Path, dat: &Path) -> ProofmanResult<Self> {
        if !library.exists() {
            return Err(ProofmanError::InvalidSetup(format!(
                "Rust lib dynamic library not found at path: {library:?}"
            )));
        }
        // SAFETY: loading the library runs its initializers: it is the circuit's witness code,
        // which setup-snark built.
        let library = unsafe { Library::new(library)? };
        let dat = CString::new(dat.as_os_str().as_encoded_bytes()).map_err(|_| {
            ProofmanError::InvalidParameters(format!("the dat path {} holds a NUL byte", dat.display()))
        })?;
        Ok(Self { library, dat })
    }

    /// The witness of the circuit for `zkin`: `getSizeWitness()` values,
    /// [`FINAL_WITNESS_VALUE_BYTES`] each, indexed by witness index (wire 0 the constant one).
    /// `getWitness` runs with up to 8 mutexes, and fails on a zkin that is not one of the circuit's
    /// inputs: its reason is on stderr.
    ///
    /// # Safety
    ///
    /// `zkin` must point to a live `nlohmann::json` object of the nlohmann/json the library was
    /// compiled against, which nothing else uses during the call.
    pub unsafe fn witness(&self, zkin: *mut c_void) -> ProofmanResult<Vec<u8>> {
        let get_size_witness: Symbol<GetSizeWitnessFunc> = self.library.get(b"getSizeWitness\0")?;
        let size_witness = get_size_witness();

        let mut witness: Vec<u8> = vec![0; (size_witness * FINAL_WITNESS_VALUE_BYTES as u64) as usize];
        let witness_ptr = witness.as_mut_ptr();

        let get_witness_final: Symbol<GetWitnessFinalFunc> = self.library.get(b"getWitness\0")?;
        let nmutex = std::cmp::min(8, rayon::current_num_threads());
        let res = get_witness_final(zkin, self.dat.as_ptr(), witness_ptr as *mut c_void, nmutex as u64);
        if res != 0 {
            return Err(ProofmanError::InvalidProof("Error generating final witness from rust".into()));
        }
        Ok(witness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The errors `proofman::generate_witness_final_snark` has always given: a library that is not
    /// there is an invalid setup, with this message, and a file that is not a library a library
    /// error.
    #[test]
    fn a_library_that_is_not_there_or_not_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("proofman_common_final_witness_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (missing, not_a_library) = (dir.join("final.so"), dir.join("not_a_library.so"));
        std::fs::write(&not_a_library, b"not an ELF file").unwrap();

        match FinalWitnessLibrary::load(&missing, &dir.join("final.dat")) {
            Err(ProofmanError::InvalidSetup(message)) => {
                assert_eq!(message, format!("Rust lib dynamic library not found at path: {missing:?}"))
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            FinalWitnessLibrary::load(&not_a_library, &dir.join("final.dat")),
            Err(ProofmanError::LibraryError(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
