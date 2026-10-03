//! `proofman-cli pilfflonk`: the pilfflonk backend's commands (pilfflonk/docs/README.md#commands),
//! nested as `pilout`'s are.

pub mod pilfflonk_calldata;
pub mod pilfflonk_check;
pub mod pilfflonk_prove;
pub mod pilfflonk_verify;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use clap::{ArgGroup, Args, Parser, Subcommand};
use proofman_pilfflonk::{
    compute_witness, gpu_available, load_witness_library, AirInstanceRef, Device, FileWitnessSource, FrBytes,
    PilfflonkResult, ProvingKey, ProvingKeyFiles, Stage1Witness, Witness, WitnessShape, WitnessSource,
};

use self::pilfflonk_calldata::PilfflonkCalldataCmd;
use self::pilfflonk_check::PilfflonkCheckCmd;
use self::pilfflonk_prove::PilfflonkProveCmd;
use self::pilfflonk_verify::PilfflonkVerifyCmd;

#[derive(Parser)]
pub struct PilfflonkCmd {
    #[command(subcommand)]
    pub pilfflonk_commands: PilfflonkSubcommands,
}

#[derive(Subcommand)]
pub enum PilfflonkSubcommands {
    Prove(PilfflonkProveCmd),
    Verify(PilfflonkVerifyCmd),
    Check(PilfflonkCheckCmd),
    Calldata(PilfflonkCalldataCmd),
}

// Where the witness of `prove` and `check` comes from (pilfflonk/docs/README.md#witness): a witness
// directory (pilfflonk/docs/formats.md#witness-directory), or a witness library that computes it
// over Fr, exactly one, with the flags of `prove` (`-w`, `-i`). The public inputs are the
// library's.
#[derive(Args)]
#[command(group(ArgGroup::new("witness_source").required(true).args(["witness", "witness_lib"])))]
pub struct PilfflonkWitnessArgs {
    /// The witness directory: instances.json, instance_<ag>_<a>_<t>.bin, publics.json and proof_values.json
    #[clap(long)]
    pub witness: Option<PathBuf>,

    /// Witness computation dynamic library path: a pilfflonk witness library, which computes the
    /// witness over Fr
    #[clap(short = 'w', long)]
    pub witness_lib: Option<PathBuf>,

    /// Public inputs path: the JSON file the witness library reads. Its values are decimal strings
    /// ("5", not 5), below r: a JSON number is refused
    #[clap(short = 'i', long, requires = "witness_lib", conflicts_with = "witness")]
    pub public_inputs: Option<PathBuf>,
}

impl PilfflonkWitnessArgs {
    /// The witness of a proof of `pk`: the witness directory, opened and checked against the shape
    /// of the key, or the witness the library computes from the public inputs, checked the same way
    /// (`compute_witness`). The library is loaded with `verbose`, the number of `-v`.
    pub fn open(
        &self,
        pk: &ProvingKey,
        verbose: u8,
    ) -> Result<PilfflonkWitness, Box<dyn std::error::Error + Send + Sync>> {
        let shape = pk.witness_shape()?;
        // clap's group `witness_source` makes exactly one of these Some.
        match (&self.witness, &self.witness_lib) {
            (Some(dir), None) => Ok(PilfflonkWitness::Dir(FileWitnessSource::open(dir, &shape)?, Mutex::new(None))),
            (None, Some(path)) => {
                let mut library = load_witness_library(path, verbose)?;
                let witness = compute_witness(&mut *library, &shape, self.public_inputs.as_deref())?;
                Ok(PilfflonkWitness::Library(witness))
            }
            _ => Err("exactly one of --witness and --witness-lib is required".into()),
        }
    }

    /// The provingKey/ at `proving_key`, loaded on `device` ([`ProvingKey::load_on`]), and the
    /// witness of a proof of it, as [`open`](Self::open) gives it, sooner
    /// (pilfflonk/docs/performance.md#the-start-of-a-proof): a witness directory is opened, and the
    /// trace of its first instance read, while the C++ core loads the key, and with
    /// [`Device::Gpu`] CUDA's initialisation starts before any file is read. The errors are those
    /// of the key first and then those of the witness, as with `load_on` and then `open`. A witness
    /// library is loaded and run after the key, as `open` does.
    pub fn load_with_key(
        &self,
        proving_key: &Path,
        device: Device,
        verbose: u8,
    ) -> Result<(ProvingKey, PilfflonkWitness), Box<dyn std::error::Error + Send + Sync>> {
        std::thread::scope(|scope| {
            if device == Device::Gpu {
                // CUDA's first call creates its context; the C++ core's first call waits for this one.
                scope.spawn(gpu_available);
            }
            let files = ProvingKeyFiles::read(proving_key)?;
            // A shape the key cannot give is for `open` to refuse, after the C++ core's errors.
            let ahead = match (&self.witness, files.witness_shape()) {
                (Some(dir), Ok(shape)) => Some(scope.spawn(move || PilfflonkWitness::read_ahead(dir, &shape))),
                _ => None,
            };
            let pk = files.load_on(device);
            let witness = ahead.map(|thread| thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)));
            let pk = pk?;
            let witness = match witness {
                Some(witness) => witness?,
                None => self.open(&pk, verbose)?,
            };
            Ok((pk, witness))
        })
    }
}

/// The witness of `prove` and `check`, from where [`PilfflonkWitnessArgs`] says: read from its
/// directory as the prover asks for it, or in memory, as the library computed it.
pub enum PilfflonkWitness {
    /// A witness directory, and the trace of its first instance (or the error reading it) if
    /// [`PilfflonkWitnessArgs::load_with_key`] read it ahead: the first `stage1(0)` takes it, and
    /// later ones read the directory again.
    Dir(FileWitnessSource, Mutex<Option<PilfflonkResult<Stage1Witness>>>),
    Library(Witness),
}

impl PilfflonkWitness {
    /// The witness directory `dir`, opened against `shape`, with the trace of its first instance
    /// read.
    fn read_ahead(dir: &Path, shape: &WitnessShape) -> PilfflonkResult<Self> {
        let source = FileWitnessSource::open(dir, shape)?;
        let first = source.stage1(0);
        Ok(Self::Dir(source, Mutex::new(Some(first))))
    }
}

impl WitnessSource for PilfflonkWitness {
    fn instances(&self) -> Vec<AirInstanceRef> {
        match self {
            Self::Dir(source, _) => source.instances(),
            Self::Library(witness) => witness.instances(),
        }
    }

    fn stage1(&self, instance: usize) -> PilfflonkResult<Stage1Witness> {
        match self {
            Self::Dir(source, first) => {
                let ahead =
                    if instance == 0 { first.lock().unwrap_or_else(PoisonError::into_inner).take() } else { None };
                ahead.unwrap_or_else(|| source.stage1(instance))
            }
            Self::Library(witness) => witness.stage1(instance),
        }
    }

    fn publics(&self) -> PilfflonkResult<Vec<FrBytes>> {
        match self {
            Self::Dir(source, _) => source.publics(),
            Self::Library(witness) => witness.publics(),
        }
    }

    fn proof_values(&self) -> PilfflonkResult<Vec<FrBytes>> {
        match self {
            Self::Dir(source, _) => source.proof_values(),
            Self::Library(witness) => witness.proof_values(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;

    fn parse(args: &str) -> Result<PilfflonkCmd, clap::Error> {
        PilfflonkCmd::try_parse_from(std::iter::once("pilfflonk").chain(args.split_whitespace()))
    }

    fn witness_args(args: &str) -> PilfflonkWitnessArgs {
        match parse(args).unwrap().pilfflonk_commands {
            PilfflonkSubcommands::Prove(cmd) => cmd.witness,
            PilfflonkSubcommands::Check(cmd) => cmd.witness,
            PilfflonkSubcommands::Verify(_) | PilfflonkSubcommands::Calldata(_) => unreachable!("{args}"),
        }
    }

    /// Exactly one of `--witness` and `--witness-lib`, and `--public-inputs` only with the library,
    /// in both commands, with `prove`'s short flags.
    #[test]
    fn the_witness_comes_from_a_directory_or_a_library() {
        for command in ["prove -k key -o out", "check -k key"] {
            let dir = witness_args(&format!("{command} --witness dir"));
            assert_eq!((dir.witness, dir.witness_lib, dir.public_inputs), (Some("dir".into()), None, None));
            let lib = witness_args(&format!("{command} -w lib.so -i inputs.json"));
            assert_eq!(
                (lib.witness, lib.witness_lib, lib.public_inputs),
                (None, Some("lib.so".into()), Some("inputs.json".into()))
            );
            let lib = witness_args(&format!("{command} --witness-lib lib.so --public-inputs inputs.json"));
            assert_eq!((lib.witness_lib, lib.public_inputs), (Some("lib.so".into()), Some("inputs.json".into())));
            assert_eq!(witness_args(&format!("{command} -w lib.so")).public_inputs, None);

            for (args, kind) in [
                ("--witness dir --witness-lib lib.so", ErrorKind::ArgumentConflict),
                ("", ErrorKind::MissingRequiredArgument),
                ("--witness dir --public-inputs inputs.json", ErrorKind::ArgumentConflict),
                ("-i inputs.json", ErrorKind::MissingRequiredArgument),
            ] {
                let err = parse(&format!("{command} {args}")).err().unwrap_or_else(|| panic!("{command} {args}"));
                assert_eq!(err.kind(), kind, "{command} {args}: {err}");
            }
        }
    }
}
