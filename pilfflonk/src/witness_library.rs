//! Witness libraries (pilfflonk/docs/README.md#witness): a dynamic library that computes the
//! witness of a proof over `Fr`, as a STARK witness library computes one over Goldilocks
//! (`proofman_witness`'s `WitnessLibrary`), but on a path of its own, with nothing of the STARK's
//! runtime (pilfflonk/docs/README.md#design, principle 1): it takes the shape of the witness and the
//! public inputs, and returns a [`Witness`], which the prover reads as any other
//! [`WitnessSource`](crate::WitnessSource).
//!
//! - **The library** implements [`PilfflonkWitnessLibrary`] and exports it with
//!   [`pilfflonk_witness_library!`](crate::pilfflonk_witness_library), which defines its entry point,
//!   the symbol `pilfflonk_init_library` ([`INIT_SYMBOL`]), of type [`PilfflonkWitnessLibInitFn`]. It
//!   computes its traces in the rows over [`Bn128`](proofman_fields::Bn128) that `pil-helpers`
//!   generates for a BN128 pilout, which has no packed rows (pilfflonk does not accept packed traces
//!   yet), and makes the stage-1 witness of each instance with
//!   [`Stage1Witness::from_rows`](crate::Stage1Witness::from_rows).
//! - **The host** loads it with [`load_witness_library`], and runs it with [`compute_witness`], which
//!   checks what it returns against the shape.
//!
//! A STARK witness library exports `init_library`, not `pilfflonk_init_library`: neither loader takes
//! the other's libraries. As the STARK's, the entry point has the Rust ABI (`extern "Rust"`, a
//! `Box<dyn …>` and this crate's types), so a library and its host must be built with the same
//! toolchain and the same version of this crate.
//!
//! **Public inputs.** As a STARK library does, a pilfflonk library reads its public inputs itself,
//! from the JSON file the host names, if any (`--public-inputs`), with [`read_public_inputs`]: for
//! a program's publics, into the `<Program>Publics` of its `pil-helpers`, whose publics are `Bn128`
//! values, written as decimal strings (pilfflonk/docs/formats.md#json-encoding).

use std::fs;
use std::path::Path;

use libloading::Library;
use serde::de::DeserializeOwned;

use crate::error::{PilfflonkError, PilfflonkResult};
use crate::witness::{Witness, WitnessShape};

/// The symbol of a pilfflonk witness library's entry point, a [`PilfflonkWitnessLibInitFn`].
pub const INIT_SYMBOL: &str = "pilfflonk_init_library";

/// The symbol of a STARK witness library's entry point (`proofman_witness::witness_library!`), to
/// tell one apart.
const STARK_INIT_SYMBOL: &str = "init_library";

/// The type of a pilfflonk witness library's entry point, `pilfflonk_init_library`: it takes the
/// host's verbosity, the number of `-v` its command was given (`proofman_common::VerboseMode::from`),
/// sets up the library's logger with it, and returns the library.
pub type PilfflonkWitnessLibInitFn = fn(u8) -> PilfflonkResult<Box<dyn PilfflonkWitnessLibrary>>;

/// A library that computes the witness of a proof over `Fr` (see the module).
pub trait PilfflonkWitnessLibrary {
    /// The witness of a proof of the AIRs of `shape`, the shape of the witness of the proving key
    /// ([`ProvingKey::witness_shape`](crate::ProvingKey::witness_shape)), from the public inputs in
    /// the JSON file `public_inputs`, if the host was given one. The host checks the witness against
    /// `shape` ([`compute_witness`]).
    fn witness(&mut self, shape: &WitnessShape, public_inputs: Option<&Path>) -> PilfflonkResult<Witness>;
}

/// Exports the pilfflonk witness library `$lib_name`, a unit struct it defines, which must
/// implement [`PilfflonkWitnessLibrary`](crate::witness_library::PilfflonkWitnessLibrary): it
/// defines the entry point `pilfflonk_init_library`, which sets up the library's logger, as the
/// STARK's `witness_library!` does, and returns the library. For the logger, the library depends on
/// `proofman-common`, as every library whose traces `pil-helpers` generates does.
///
/// Unlike the STARK's, the entry point sets up the logger once, with the verbosity of its first
/// call: a library may be loaded more than once, from several threads, and `initialize_logger`
/// panics if two threads set it up at once. A panic does not unwind out of a dynamic library, which
/// has a Rust runtime of its own: it aborts the process.
#[macro_export]
macro_rules! pilfflonk_witness_library {
    ($lib_name:ident) => {
        pub struct $lib_name;

        #[no_mangle]
        pub extern "Rust" fn pilfflonk_init_library(
            verbose: u8,
        ) -> $crate::PilfflonkResult<Box<dyn $crate::witness_library::PilfflonkWitnessLibrary>> {
            static LOGGER: std::sync::Once = std::sync::Once::new();
            LOGGER.call_once(|| proofman_common::initialize_logger(verbose.into(), None));

            Ok(Box::new($lib_name))
        }
    };
}

/// Loads the witness library at `path` and calls its entry point with `verbose` (see
/// [`PilfflonkWitnessLibInitFn`]).
///
/// Refuses a path that is not a file, a file that is not a library, and a library with no
/// `pilfflonk_init_library`, and says so of a STARK witness library.
///
/// The library stays loaded until the process ends, as `proofman_witness::load_packed_info` leaves
/// its own: what it returns (the witness, and errors that may hold its vtables) and its thread-local
/// destructors may outlive any handle to it.
pub fn load_witness_library(path: &Path, verbose: u8) -> PilfflonkResult<Box<dyn PilfflonkWitnessLibrary>> {
    let error = |reason: String| PilfflonkError::WitnessLibrary { path: path.to_path_buf(), reason };
    if !path.is_file() {
        return Err(error(if path.exists() { "it is not a file" } else { "there is no such file" }.to_string()));
    }
    // SAFETY: loading the library runs its initializers: it is code the user asked to run, as a
    // STARK witness library is.
    let library = unsafe { Library::new(path) }.map_err(|e| error(format!("it cannot be loaded: {e}")))?;
    // SAFETY: the symbol is the entry point `pilfflonk_witness_library!` defines, of this type; the
    // pointer stays valid because the library is never unloaded (below).
    let init = unsafe { library.get::<PilfflonkWitnessLibInitFn>(INIT_SYMBOL.as_bytes()) }.map(|init| *init);
    // SAFETY: only whether the symbol exists is looked at.
    let stark = init.is_err() && unsafe { library.get::<*const ()>(STARK_INIT_SYMBOL.as_bytes()) }.is_ok();
    std::mem::forget(library);
    match init {
        Ok(init) => init(verbose),
        Err(_) if stark => Err(error(format!(
            "it is a STARK witness library ({STARK_INIT_SYMBOL}), not a pilfflonk one ({INIT_SYMBOL})"
        ))),
        Err(_) => Err(error(format!("it exports no {INIT_SYMBOL}: it is not a pilfflonk witness library"))),
    }
}

/// The public inputs in the JSON file `path`, for a library to read (see the module), or their
/// default if the host was given no file: as `proofman_common::load_from_json` reads a STARK
/// library's, with errors in place of its panics.
pub fn read_public_inputs<T: Default + DeserializeOwned>(path: Option<&Path>) -> PilfflonkResult<T> {
    let Some(path) = path else {
        return Ok(T::default());
    };
    let text = fs::read_to_string(path).map_err(|source| PilfflonkError::Io { path: path.to_path_buf(), source })?;
    serde_json::from_str(&text).map_err(|e| PilfflonkError::from(e).in_file(path))
}

/// Runs `library` for the witness of `shape` from the public inputs in `public_inputs`, and checks
/// the witness against `shape`, as [`FileWitnessSource::open`](crate::FileWitnessSource::open) checks
/// a directory: the instances, their order and their traces' sizes, and the numbers of air values,
/// publics and proof values. Its values are below `r` by construction.
pub fn compute_witness(
    library: &mut dyn PilfflonkWitnessLibrary,
    shape: &WitnessShape,
    public_inputs: Option<&Path>,
) -> PilfflonkResult<Witness> {
    let witness = library.witness(shape, public_inputs)?;
    shape.check(&witness)?;
    Ok(witness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::FrBytes;
    use crate::witness::{AirInstanceRef, AirShape, InstanceWitness, Stage1Witness};

    /// A library of an AIR of 2 rows and 1 column, whose one public is the number of instances it
    /// is told to make.
    struct Instances(usize);

    impl PilfflonkWitnessLibrary for Instances {
        fn witness(&mut self, _shape: &WitnessShape, _public_inputs: Option<&Path>) -> PilfflonkResult<Witness> {
            let stage1 = Stage1Witness::from_columns(2, &[vec![FrBytes::ZERO; 2]], vec![])?;
            let instance = InstanceWitness { air: AirInstanceRef { airgroup_id: 0, air_id: 0 }, stage1 };
            Ok(Witness {
                instances: vec![instance; self.0],
                publics: vec![FrBytes::from_u64(self.0 as u64)],
                proof_values: vec![],
            })
        }
    }

    fn shape() -> WitnessShape {
        WitnessShape::new(vec![AirShape { airgroup_id: 0, air_id: 0, n_bits: 1, n_cols: 1, n_air_values: 0 }], 1, 0)
            .unwrap()
    }

    #[test]
    fn the_witness_of_a_library_is_checked_against_the_shape() {
        let witness = compute_witness(&mut Instances(1), &shape(), None).unwrap();
        assert_eq!(witness.publics, [FrBytes::from_u64(1)]);
        // An instance too few, and one of an AIR the shape has not.
        assert!(compute_witness(&mut Instances(0), &shape(), None).is_err());
        let other = WitnessShape::new(
            vec![AirShape { airgroup_id: 0, air_id: 1, n_bits: 1, n_cols: 1, n_air_values: 0 }],
            1,
            0,
        )
        .unwrap();
        let err = compute_witness(&mut Instances(1), &other, None).unwrap_err();
        assert!(err.to_string().contains("air 0/0"), "{err}");
    }

    #[derive(Debug, Default, PartialEq, serde::Deserialize)]
    struct Publics {
        #[serde(default)]
        a: proofman_fields::Bn128,
        #[serde(default)]
        b: proofman_fields::Bn128,
    }

    /// As `proofman_common::load_from_json`: no file, the default; a public missing, its default.
    #[test]
    fn public_inputs_are_read_from_their_json_file() {
        use proofman_fields::QuotientMap;

        let dir = std::env::temp_dir().join(format!("proofman_pilfflonk_public_inputs_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("inputs.json");
        assert_eq!(read_public_inputs::<Publics>(None).unwrap(), Publics::default());
        std::fs::write(
            &path,
            r#"{"a": "21888242871839275222246405745257275088548364400416034343698204186575808495616"}"#,
        )
        .unwrap();
        let read: Publics = read_public_inputs(Some(&path)).unwrap();
        assert_eq!(read, Publics { a: proofman_fields::Bn128::from_int(-1i64), b: Default::default() });
        // A JSON number is not a Bn128, and a file that is not there is an error.
        std::fs::write(&path, r#"{"a": 1}"#).unwrap();
        assert!(matches!(read_public_inputs::<Publics>(Some(&path)), Err(PilfflonkError::InFile { .. })));
        assert!(matches!(read_public_inputs::<Publics>(Some(&dir.join("none.json"))), Err(PilfflonkError::Io { .. })));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_path_that_is_not_a_library_is_refused() {
        let dir = std::env::temp_dir().join(format!("proofman_pilfflonk_witness_library_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("missing.so");
        let not_a_library = dir.join("not_a_library.so");
        std::fs::write(&not_a_library, b"not an ELF file").unwrap();

        let reason = |path: &Path| match load_witness_library(path, 0) {
            Err(PilfflonkError::WitnessLibrary { path: at, reason }) if at == path => reason,
            Err(e) => panic!("{e}"),
            Ok(_) => panic!("{} loaded", path.display()),
        };
        assert_eq!(reason(&missing), "there is no such file");
        assert_eq!(reason(&dir), "it is not a file");
        assert!(reason(&not_a_library).starts_with("it cannot be loaded"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
