//! `proofman-cli pilfflonk`: the pilfflonk backend's commands (spec §4.4, §4.5), nested as
//! `pilout`'s are.

pub mod pilfflonk_check;
pub mod pilfflonk_prove;
pub mod pilfflonk_verify;

use clap::{Parser, Subcommand};

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
}
