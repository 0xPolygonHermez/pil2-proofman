//! `proofman-cli pilfflonk`: the pilfflonk backend's commands (spec §4.4, §4.5), nested as
//! `pilout`'s are. `prove` is to come (plan M18).

pub mod pilfflonk_verify;

use clap::{Parser, Subcommand};

use self::pilfflonk_verify::PilfflonkVerifyCmd;

#[derive(Parser)]
pub struct PilfflonkCmd {
    #[command(subcommand)]
    pub pilfflonk_commands: PilfflonkSubcommands,
}

#[derive(Subcommand)]
pub enum PilfflonkSubcommands {
    Verify(PilfflonkVerifyCmd),
}
