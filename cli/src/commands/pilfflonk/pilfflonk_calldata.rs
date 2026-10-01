use clap::{Args, ValueEnum};
use colored::Colorize;
use proofman_common::initialize_logger;
use proofman_pilfflonk::Calldata;
use std::fs;
use std::path::PathBuf;

// The calldata encoder of spec §4.5 ("Calldata", plan M41): the arguments of the Solidity verifier's
// verifyProof for a proof, as snarkjs's `zkey export soliditycalldata` prints those of its FFLONK
// verifier. It takes the files of `pilfflonk verify`, and reads them as the JS verifier does.
/// Print the calldata of a pilfflonk proof for its Solidity verifier's verifyProof, as snarkjs's `zkey export soliditycalldata`
#[derive(Args)]
pub struct PilfflonkCalldataCmd {
    /// pilfflonk.vkey.json, the vkey the Solidity verifier was generated from
    #[clap(short = 'k', long)]
    pub vkey: PathBuf,

    /// The proof: proof.json, or its bytes (spec A.6) in a file whose name ends in .bin
    #[clap(short = 'p', long)]
    pub proof: PathBuf,

    /// publics.json
    #[clap(long)]
    pub publics: PathBuf,

    /// How the calldata is written
    #[clap(long, value_enum, default_value_t = CalldataFormat::Solidity)]
    pub format: CalldataFormat,

    /// Write the calldata to this file, and nothing else, instead of printing it
    #[clap(short = 'o', long)]
    pub output: Option<PathBuf>,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

/// How `pilfflonk calldata` writes the calldata (`proofman_pilfflonk::calldata`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CalldataFormat {
    /// The arguments of verifyProof as snarkjs prints them: [0x…,…],[0x…,…], the words of the proof
    /// and then the publics (none if there are none), each 0x and 64 hexadecimal digits
    Solidity,
    /// The ABI-encoded calldata of the call, selector first, as 0x and hexadecimal digits: the data
    /// of an eth_call, or of `cast call <address> --data <hex>`
    Hex,
}

impl PilfflonkCalldataCmd {
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("{} Pilfflonk calldata subcommand", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let calldata = Calldata::read(&self.vkey, &self.proof, &self.publics)?;
        let text = match self.format {
            CalldataFormat::Solidity => calldata.to_solidity(),
            CalldataFormat::Hex => calldata.to_hex()?,
        };
        match &self.output {
            Some(path) => {
                fs::write(path, format!("{text}\n")).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
                tracing::info!(
                    "    {} {} ({})",
                    "\u{2713} pilfflonk calldata written to".bright_green().bold(),
                    path.display(),
                    calldata.signature()
                );
            }
            None => println!("{text}"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{PilfflonkCmd, PilfflonkSubcommands};
    use super::*;
    use clap::error::ErrorKind;
    use clap::Parser;

    fn parse(args: &str) -> Result<PilfflonkCalldataCmd, clap::Error> {
        let cmd = PilfflonkCmd::try_parse_from(["pilfflonk", "calldata"].into_iter().chain(args.split_whitespace()))?;
        match cmd.pilfflonk_commands {
            PilfflonkSubcommands::Calldata(cmd) => Ok(cmd),
            _ => unreachable!("{args}"),
        }
    }

    /// The files of `pilfflonk verify`, with the flags of `pilfflonk-solidity` (`-k`, `-o`) and of
    /// `verify-snark` (`-p`); the Solidity form by default, and the calldata printed.
    #[test]
    fn the_calldata_takes_the_files_of_verify() {
        let cmd = parse("-k vkey.json -p proof.json --publics publics.json").unwrap();
        assert_eq!(
            (cmd.vkey, cmd.proof, cmd.publics, cmd.format, cmd.output),
            ("vkey.json".into(), "proof.json".into(), "publics.json".into(), CalldataFormat::Solidity, None)
        );
        let cmd = parse("--vkey v --proof p.bin --publics u --format hex --output out.txt").unwrap();
        assert_eq!((cmd.proof, cmd.format, cmd.output), ("p.bin".into(), CalldataFormat::Hex, Some("out.txt".into())));
        assert_eq!(parse("-k v -p p --publics u --format solidity -o o").unwrap().format, CalldataFormat::Solidity);

        for (args, kind) in [
            ("-k v -p p", ErrorKind::MissingRequiredArgument),
            ("-p p --publics u", ErrorKind::MissingRequiredArgument),
            ("-k v --publics u", ErrorKind::MissingRequiredArgument),
            ("-k v -p p --publics u --format abi", ErrorKind::InvalidValue),
        ] {
            let err = parse(args).err().unwrap_or_else(|| panic!("{args}"));
            assert_eq!(err.kind(), kind, "{args}: {err}");
        }
    }
}
