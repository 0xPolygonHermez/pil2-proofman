//! `proofman-cli pilfflonk`: the pilfflonk backend's commands (spec §4.4, §4.5), nested as
//! `pilout`'s are.

pub mod pilfflonk_calldata;
pub mod pilfflonk_check;
pub mod pilfflonk_prove;
pub mod pilfflonk_verify;

use std::path::PathBuf;

use clap::{ArgGroup, Args, Parser, Subcommand};
use proofman_pilfflonk::{
    compute_witness, load_witness_library, AirInstanceRef, FileWitnessSource, FrBytes, PilfflonkResult, ProvingKey,
    Stage1Witness, Witness, WitnessSource,
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

// Where the witness of `prove` and `check` comes from (spec §4.3): a witness directory (spec A.6),
// or a witness library that computes it over Fr (D4, plan M38c), exactly one, with the flags of
// `prove` (`-w`, `-i`). The public inputs are the library's.
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
            (Some(dir), None) => Ok(PilfflonkWitness::Dir(FileWitnessSource::open(dir, &shape)?)),
            (None, Some(path)) => {
                let mut library = load_witness_library(path, verbose)?;
                let witness = compute_witness(&mut *library, &shape, self.public_inputs.as_deref())?;
                Ok(PilfflonkWitness::Library(witness))
            }
            _ => Err("exactly one of --witness and --witness-lib is required".into()),
        }
    }
}

/// The witness of `prove` and `check`, from where [`PilfflonkWitnessArgs`] says: read from its
/// directory as the prover asks for it, or in memory, as the library computed it.
pub enum PilfflonkWitness {
    Dir(FileWitnessSource),
    Library(Witness),
}

impl WitnessSource for PilfflonkWitness {
    fn instances(&self) -> Vec<AirInstanceRef> {
        match self {
            Self::Dir(source) => source.instances(),
            Self::Library(witness) => witness.instances(),
        }
    }

    fn stage1(&self, instance: usize) -> PilfflonkResult<Stage1Witness> {
        match self {
            Self::Dir(source) => source.stage1(instance),
            Self::Library(witness) => witness.stage1(instance),
        }
    }

    fn publics(&self) -> PilfflonkResult<Vec<FrBytes>> {
        match self {
            Self::Dir(source) => source.publics(),
            Self::Library(witness) => witness.publics(),
        }
    }

    fn proof_values(&self) -> PilfflonkResult<Vec<FrBytes>> {
        match self {
            Self::Dir(source) => source.proof_values(),
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
