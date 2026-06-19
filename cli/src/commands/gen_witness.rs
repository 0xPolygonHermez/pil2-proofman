// extern crate env_logger;
use clap::Parser;
use regex::Regex;
use proofman_common::{initialize_logger, SetupsVadcop, MpiCtx, ProofCtx, VerboseMode, ProofmanError, ProofType};
use std::fs::File;
use std::io::Read;
use colored::Colorize;
use proofman_fields::{Field, Goldilocks};
use std::os::raw::c_void;
use std::path::PathBuf;
use std::sync::Arc;
use std::error::Error;
use std::str::FromStr;
use proofman_util::{timer_start_info, timer_stop_and_log_info};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct GenWitnessCmd {
    #[clap(short = 'p', long)]
    pub proof: PathBuf,

    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl GenWitnessCmd {
    pub fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("{} GenWitness", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(VerboseMode::Info, None);

        let pctx: ProofCtx<Goldilocks> =
            ProofCtx::create_ctx(self.proving_key.clone(), true, self.verbose.into(), Arc::new(MpiCtx::new()), false)?;

        let setups_vadcop: Arc<SetupsVadcop<Goldilocks>> =
            Arc::new(SetupsVadcop::new(&pctx.global_info, false, true, &[], false)?);

        let mut zkin_file = File::open(&self.proof)?;
        let mut zkin_u8 = Vec::new();
        zkin_file.read_to_end(&mut zkin_u8)?;
        if !zkin_u8.len().is_multiple_of(8) {
            return Err(Box::new(ProofmanError::InvalidProof(format!(
                "Proof file size ({} bytes) is not a multiple of 8",
                zkin_u8.len()
            ))));
        }
        let mut zkin: Vec<u64> = zkin_u8.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();

        let re = Regex::new(r"ag(\d+)_air(\d+)_t([A-Za-z0-9]+)").unwrap();

        let info = re.captures(self.proof.to_str().unwrap()).unwrap();
        let airgroup_id = info[1].parse::<usize>().unwrap();
        let air_id = info[2].parse::<usize>().unwrap();
        let proof_type = &ProofType::from_str(&info[3]).unwrap();

        let setup = setups_vadcop.get_setup(airgroup_id, air_id, proof_type)?;

        let witness_size = setup.get_circom_witness_size();

        let mut witness: Vec<Goldilocks> = vec![Goldilocks::ZERO; witness_size];

        let state = setup.circom_state.read().unwrap();
        let circom_circuit_ptr = match state.circuit {
            Some(ptr) => ptr,
            None => return Err(Box::new(ProofmanError::InvalidSetup("circom_circuit is not initialized".into()))),
        };

        // let publics_circom_size: usize =
        //     pctx.global_info.n_publics + pctx.global_info.n_proof_values.iter().sum::<usize>() * 3 + 3 + 4;

        // let publics_aggregation = n_publics_aggregation(&pctx, 0);
        // let null_proof_size = setup.proof_size as usize + publics_aggregation;

        // zkin[publics_circom_size..(publics_circom_size + null_proof_size)].fill(0);
        // zkin[publics_circom_size + null_proof_size..publics_circom_size + 2*null_proof_size].fill(0);
        // zkin[publics_circom_size + 2*null_proof_size..].fill(0);

        let get_witness_fn =
            state.get_witness_fn.ok_or(ProofmanError::InvalidSetup("GetWitness function not loaded".to_string()))?;

        timer_start_info!(WITNESS_GENERATION);
        let res =
            unsafe { get_witness_fn(zkin.as_mut_ptr(), circom_circuit_ptr, witness.as_mut_ptr() as *mut c_void, 1) };
        drop(state);
        timer_stop_and_log_info!(WITNESS_GENERATION);

        if res != 0 {
            Err(Box::new(ProofmanError::InvalidProof("Error generating witness".into())))
        } else {
            tracing::info!("    {}", "\u{2713} Witness generated successfully".bright_green().bold());
            Ok(())
        }
    }
}
