//! A zkin, the input of the final circuit, as the `nlohmann::json` its witness calculator takes
//! (`FinalWitnessLibrary::witness`): made by `src/zkin.cpp`, compiled against the nlohmann/json
//! final.so is compiled against, as libstarks's recursivef proof is in the PLONK and FFLONK wraps.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fs;
use std::os::raw::c_char;
use std::path::Path;
use std::ptr::NonNull;

use serde::de::IgnoredAny;

use crate::error::{WrapWitnessError, WrapWitnessResult};

extern "C" {
    fn pilfflonk_wrap_zkin_parse(text: *const c_char, len: usize) -> *mut c_void;
    fn pilfflonk_wrap_zkin_free(zkin: *mut c_void);
}

/// A zkin on the C++ heap, freed when dropped.
pub(crate) struct Zkin(NonNull<c_void>);

impl Zkin {
    /// The zkin in the file `path`: a JSON object whose keys are input signals of the circuit and
    /// whose values are their values, as circom's calculator reads them (decimal strings, or
    /// arrays of them).
    ///
    /// Anything but a JSON object is refused here, where the error can say where: the calculator
    /// takes the object's keys, and would throw on another value. A key that is not an input of the
    /// circuit is not refused: the calculator stops the process on it (an `assert` of circom's
    /// `getInputSignalHashPosition`), as it does in the PLONK and FFLONK wraps.
    pub(crate) fn read(path: &Path) -> WrapWitnessResult<Self> {
        let text = fs::read(path).map_err(|source| WrapWitnessError::Io { path: path.to_path_buf(), source })?;
        serde_json::from_slice::<BTreeMap<String, IgnoredAny>>(&text)
            .map_err(|source| WrapWitnessError::Json { path: path.to_path_buf(), source })?;
        // SAFETY: `text` is `text.len()` bytes, which the function only reads, during the call.
        let zkin = unsafe { pilfflonk_wrap_zkin_parse(text.as_ptr().cast(), text.len()) };
        NonNull::new(zkin).map(Self).ok_or_else(|| WrapWitnessError::Zkin {
            path: path.to_path_buf(),
            reason: "nlohmann/json cannot parse it, or there is not memory enough to hold it".to_string(),
        })
    }

    /// The `nlohmann::json`, for `getWitness`.
    pub(crate) fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for Zkin {
    fn drop(&mut self) {
        // SAFETY: the pointer is the one `pilfflonk_wrap_zkin_parse` returned, freed once, here.
        unsafe { pilfflonk_wrap_zkin_free(self.0.as_ptr()) }
    }
}
