// extern crate env_logger;
use clap::Parser;
use regex::Regex;
use proofman_common::{initialize_logger, ParamsGPU, SetupsVadcop, MpiCtx, ProofCtx, VerboseMode, ProofmanError, ProofType};
use proofman::{GetWitnessFunc, initialize_witness_circom};
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;
use colored::Colorize;
use fields::{Field, Goldilocks};
use libloading::Symbol;
use std::os::raw::c_void;
use std::path::PathBuf;
use bytemuck::cast_slice_mut;
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

        let pctx: ProofCtx<Goldilocks> = ProofCtx::create_ctx(
            self.proving_key.clone(),
            HashMap::new(),
            true,
            false,
            self.verbose.into(),
            Arc::new(MpiCtx::new()),
        )?;

        let gpu_params = ParamsGPU::new(false);

        let setups_vadcop: Arc<SetupsVadcop<Goldilocks>> =
            Arc::new(SetupsVadcop::new(&pctx.global_info, false, true, false, &gpu_params, &[]));
        initialize_witness_circom(&pctx, &setups_vadcop)?;

        let mut zkin_file = File::open(&self.proof)?;
        let mut zkin_u8 = Vec::new();
        zkin_file.read_to_end(&mut zkin_u8)?;
        let zkin: &mut [u64] = cast_slice_mut::<u8, u64>(&mut zkin_u8);

        let re = Regex::new(r"ag(\d+)_air(\d+)_t([A-Za-z0-9]+)").unwrap();

        let info = re.captures(self.proof.to_str().unwrap()).unwrap();
        let airgroup_id = info[1].parse::<usize>().unwrap();
        let air_id = info[2].parse::<usize>().unwrap();
        let proof_type = &ProofType::from_str(&info[3]).unwrap();

        let setup = setups_vadcop.get_setup(airgroup_id, air_id, proof_type)?;

        let mut witness_size = setup.size_witness.read().unwrap().unwrap();
        witness_size += *setup.exec_data.read().unwrap().as_ref().unwrap().first().unwrap();

        let witness: Vec<Goldilocks> = vec![Goldilocks::ZERO; witness_size as usize];

        let circom_circuit_guard = setup.circom_circuit.read().unwrap();
        let circom_circuit_ptr = match *circom_circuit_guard {
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

        timer_start_info!(WITNESS_GENERATION);
        let res = unsafe {
            let library_guard = setup.circom_library.read().unwrap();
            let library =
                library_guard.as_ref().ok_or(ProofmanError::InvalidSetup("Circom library not loaded".to_string()))?;
            let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0")?;
            get_witness(zkin.as_ptr() as *mut u64, circom_circuit_ptr, witness.as_ptr() as *mut c_void, 1)
        };
        timer_stop_and_log_info!(WITNESS_GENERATION);

        if res != 0 {
            Err(Box::new(ProofmanError::InvalidProof("Error generating witness".into())))
        } else {
            tracing::info!("    {}", "\u{2713} Witness generated successfully".bright_green().bold());
            Ok(())
        }
    }
}
