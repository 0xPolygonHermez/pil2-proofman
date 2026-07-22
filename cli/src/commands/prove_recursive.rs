// extern crate env_logger;
use clap::Parser;
use regex::Regex;
use proofman_common::{
    calculate_fixed_tree, init_gpu_setup, initialize_logger, SetupCtx, SetupsVadcop, MpiCtx, ProofCtx, ProofmanError,
    ProofType,
};
use proofman::{n_publics_aggregation, verify_proof};
use proofman_starks_lib_c::{
    add_publics_aggregation_c, gen_recursive_proof_c, get_committed_pols_c, get_stream_id_proof_c,
    load_device_const_pols_c, load_device_setup_c, read_exec_file_c,
};
use libloading::{Library, Symbol};
use std::fs::File;
use std::io::Read;
use colored::Colorize;
use fields::{Field, Goldilocks};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::error::Error;
use std::str::FromStr;
use proofman_util::{timer_start_info, timer_stop_and_log_info};

// Circom witness-library entry points (mirror examples/test-recursive/src/recursive.rs).
type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, circom_circuit: *mut c_void, witness: *mut c_void, n_mutexes: u64) -> i64;
type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;
type GetCircomCircuitFunc = unsafe extern "C" fn(dat_file: *const c_char) -> *mut c_void;

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveRecursiveCmd {
    #[clap(short = 'p', long)]
    pub proof: PathBuf,

    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Stop after generating the recursion witness (legacy gen-witness behavior).
    #[clap(long)]
    pub emit_witness_only: bool,

    /// Run the recursive prover on the GPU.
    #[clap(long)]
    pub gpu: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl ProveRecursiveCmd {
    pub fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("{} ProveRecursive", format!("{: >12}", "Command").bright_green().bold());
        println!();

        initialize_logger(self.verbose.into(), None);

        let mut pctx: ProofCtx<Goldilocks> = ProofCtx::create_ctx(
            self.proving_key.clone(),
            true,
            self.verbose.into(),
            Arc::new(MpiCtx::new()),
            self.gpu,
        )?;

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

        // prove-recursive only ever proves the single recursive AIR named by the proof
        // file (airgroup, air, proof_type). Build exactly that one setup -- do NOT build
        // the full SetupsVadcop aggregation stack (compressor + recursive1 + recursive2 +
        // vadcop_final), which would fail on proving keys that only contain this AIR.
        //
        // Load the setup for the ACTUAL parsed proof_type so global_info.get_air_setup_path
        // resolves the correct on-disk layout: Basic -> airs/<Air>/air/<Air>, Recursive1 ->
        // airs/<Air>/recursive1/recursive1, etc. (global_info.rs:151-181). The prior code
        // hardcoded Basic, which only works for test proving keys that place recursive
        // artifacts under air/; on a full zisk key the recursive1 .so/.dat/.exec/.const live
        // under recursive1/, so loading as Basic looked for a nonexistent air/<Air>.so.
        // (Compressor with has_compressor unset still yields an empty setup, but a recursive
        // proof file never names that case here.)
        let sctx: SetupCtx<Goldilocks> = SetupCtx::new(&pctx.global_info, proof_type, false, &[], self.gpu)?;

        // Initialize the GPU (set_gpu_mode_c + init_gpu_setup_c). Without this the CUDA
        // context is not selected and check_device_memory_c (used by set_device_buffers)
        // returns 0. Mirrors proofman.rs:670-682 / common::init_gpu_setup.
        init_gpu_setup(sctx.max_n_bits_ext as u64, self.gpu)?;

        let setup = sctx.get_setup(airgroup_id, air_id)?;

        // Regenerate the const tree if missing/stale. calculate_fixed_tree validates the
        // existing .consttree(_gpu) (size + verkey root) and rebuilds + rewrites it when
        // invalid, so a stale tree (e.g. from a prior setup at a different layout) doesn't
        // crash the prover's const loader. Host-only; mirrors proofman.rs:684-686.
        calculate_fixed_tree(setup);

        // Load the circom witness library directly from the setup path, rather than
        // relying on Setup's circom_state (which is only populated for proof types whose
        // has_compressor flag is set -- a standalone recursive AIR like this leaves it
        // empty). Mirrors examples/test-recursive/src/recursive.rs.
        let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
        let rust_lib_filename = setup.setup_path.display().to_string() + lib_extension;
        let rust_lib_path = Path::new(&rust_lib_filename);
        if !rust_lib_path.exists() {
            return Err(Box::new(ProofmanError::InvalidSetup(format!(
                "Circom witness library not found at {rust_lib_path:?}"
            ))));
        }

        let dat_filename = setup.setup_path.display().to_string() + ".dat";
        let dat_filename_str = std::ffi::CString::new(dat_filename)?;
        let dat_filename_ptr = dat_filename_str.as_ptr() as *mut c_char;

        // Pre-load the .exec file (header = n_adds, n_smap; body follows) for the
        // committed-pols extraction below.
        let exec_filename = setup.setup_path.display().to_string() + ".exec";
        let mut exec_header_file = File::open(&exec_filename)?;
        let mut bytes = [0u8; 8];
        exec_header_file.read_exact(&mut bytes)?;
        let n_adds = u64::from_le_bytes(bytes);
        exec_header_file.read_exact(&mut bytes)?;
        let n_smap = u64::from_le_bytes(bytes);
        drop(exec_header_file);

        let n_cols = setup.stark_info.map_sections_n["cm1"];
        let exec_data_size = 2 + n_adds * 4 + n_smap * n_cols;
        let mut exec_file_data: Vec<u64> = vec![0; exec_data_size as usize];
        read_exec_file_c(exec_file_data.as_mut_ptr(), exec_filename.as_str(), n_cols);

        let library: Library = unsafe { Library::new(rust_lib_path)? };

        let circom_circuit_ptr = unsafe {
            let init_circom_circuit: Symbol<GetCircomCircuitFunc> = library.get(b"initCircuit\0")?;
            init_circom_circuit(dat_filename_ptr)
        };

        let size_witness = unsafe {
            let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0")?;
            get_size_witness()
        };

        // Total circom witness size = circuit witness + the n_adds from the exec header.
        let witness_size = (size_witness + exec_file_data[0]) as usize;
        let mut witness: Vec<Goldilocks> = vec![Goldilocks::ZERO; witness_size];

        timer_start_info!(WITNESS_GENERATION);
        let res = unsafe {
            let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0")?;
            get_witness(zkin.as_mut_ptr(), circom_circuit_ptr, witness.as_mut_ptr() as *mut c_void, 1)
        };
        timer_stop_and_log_info!(WITNESS_GENERATION);

        if res != 0 {
            return Err(Box::new(ProofmanError::InvalidProof("Error generating witness".into())));
        }

        if self.emit_witness_only {
            tracing::info!("    {}", "\u{2713} Witness generated successfully".bright_green().bold());
            return Ok(());
        }

        // --- Recursive prove continuation -------------------------------------------------
        // Drive gen_recursive_proof_c directly with the witness we just generated,
        // bypassing contributions / basic proofs. Mirrors
        // proofman::recursion::generate_recursive_proof (recursion.rs:273).

        // Device buffers. gen_recursive_proof_gpu reads the AIR's const pols from the
        // *aggregation* const buffer (d_constPolsAggregation, starks_api.cu:976), which
        // set_device_buffers only allocates when aggregation=true (sizing it from the
        // SetupsVadcop const totals). We don't have a full vadcop stack, so build an empty
        // (aggregation=false) SetupsVadcop and patch in just this AIR's const sizes, then
        // call set_device_buffers with aggregation=true so the aggregation const area is
        // allocated. The recursive-stream loop still allocates 0 recursive streams because
        // a compressor proof runs on a regular stream (recursive buffer sizes left 0).
        let load_tree = setup.preallocate;
        let mut setups_vadcop: SetupsVadcop<Goldilocks> =
            SetupsVadcop::new(&pctx.global_info, false, false, &[], self.gpu)?;
        setups_vadcop.total_const_pols_size = setup.const_pols_size_packed;
        if load_tree {
            setups_vadcop.total_const_tree_size = setup.const_tree_size;
        }
        pctx.set_device_buffers(&sctx, &setups_vadcop, true, self.gpu, 1, 1)?;

        // Register this AIR's setup + upload its const pols into the aggregation const
        // buffer under the same proofType gen_recursive_proof_c uses. Mirrors
        // proofman::utils::load_device_setups / load_device_const_pols (the aggregation
        // branch), but for the single AIR we are proving.
        let proof_type_str: &str = (*proof_type).clone().into();
        let d_buffers = pctx.get_device_buffers_ptr();
        load_device_setup_c(
            airgroup_id as u64,
            air_id as u64,
            proof_type_str,
            (&setup.p_setup).into(),
            d_buffers,
            setup.verkey.as_ptr() as *mut u8,
            std::ptr::null_mut(),
        );
        let tree_path = if load_tree { setup.const_pols_tree_path.as_str() } else { "" };
        load_device_const_pols_c(
            airgroup_id as u64,
            air_id as u64,
            0,
            d_buffers,
            &setup.const_pols_path,
            setup.const_pols_size_packed as u64,
            tree_path,
            setup.const_tree_size as u64,
            proof_type_str,
            false,
        );

        // vadcop/instance/proof_type follow recursion.rs:293-298 for non-final proofs.
        // (prove-recursive targets compressor/recursive1/recursive2 — the vadcop tail
        // goes through a different entry point and is out of scope here.)
        let n = 1u64 << setup.stark_info.stark_struct.n_bits;

        let mut trace: Vec<Goldilocks> = vec![Goldilocks::ZERO; (n_cols * n) as usize];
        let mut publics: Vec<Goldilocks> = vec![Goldilocks::ZERO; setup.stark_info.n_publics as usize];

        get_committed_pols_c(
            witness.as_ptr() as *mut u8,
            exec_file_data.as_mut_ptr(),
            trace.as_mut_ptr() as *mut u8,
            publics.as_mut_ptr() as *mut u8,
            size_witness,
            n,
            setup.stark_info.n_publics,
            n_cols,
        );

        // Output proof buffer: proof_size + publics_aggregation (recursion.rs:258-263).
        // The aggregation publics live in [0..publics_aggregation); the recursive proof
        // is written starting at initial_idx = publics_aggregation.
        let publics_aggregation = n_publics_aggregation(&pctx, airgroup_id);
        let proof_buffer_size = setup.proof_size as usize + publics_aggregation;
        let mut proof_buffer: Vec<u64> = vec![0u64; proof_buffer_size];

        add_publics_aggregation_c(
            proof_buffer.as_ptr() as *mut u8,
            0,
            publics.as_ptr() as *mut u8,
            publics_aggregation as u64,
        );

        let aux_trace: Vec<Goldilocks> = vec![Goldilocks::ZERO; setup.prover_buffer_size as usize];

        // Const pols/tree pointers. On GPU they are loaded device-side from the paths, so
        // pass NULL host ptrs (recursion.rs:347-351). On CPU, gen_recursive_proof_cpu
        // loadFileParallel()-fills the host buffers from constPolsPath/constTreePath
        // (starks_api.cpp:848-849), so we must hand it real allocations of the right size
        // (NULL would segfault in genProof). These buffers must outlive the FFI call.
        // (empty on GPU; sized on CPU). Kept in scope so they outlive the FFI call.
        let mut const_pols_cpu: Vec<Goldilocks> =
            if self.gpu { Vec::new() } else { vec![Goldilocks::ZERO; (setup.stark_info.n_constants * n) as usize] };
        let mut const_tree_cpu: Vec<Goldilocks> =
            if self.gpu { Vec::new() } else { vec![Goldilocks::ZERO; setup.const_tree_size] };
        let (const_pols_ptr, const_tree_ptr) = if self.gpu {
            (std::ptr::null_mut(), std::ptr::null_mut())
        } else {
            (const_pols_cpu.as_mut_ptr() as *mut u8, const_tree_cpu.as_mut_ptr() as *mut u8)
        };

        let p_setup: *mut c_void = (&setup.p_setup).into();

        timer_start_info!(GEN_RECURSIVE_PROOF);
        let stream_id = gen_recursive_proof_c(
            p_setup,
            trace.as_ptr() as *mut u8,
            aux_trace.as_ptr() as *mut u8,
            const_pols_ptr,
            const_tree_ptr,
            publics.as_ptr() as *mut u8,
            proof_buffer[publics_aggregation..].as_mut_ptr(),
            "",
            airgroup_id as u64,
            air_id as u64,
            0,
            true,
            pctx.get_device_buffers_ptr(),
            &setup.const_pols_path,
            &setup.const_pols_tree_path,
            proof_type_str,
            false,
            "",
        );

        // The recursive prover writes its output asynchronously; the result is only in
        // proof_buffer after the stream drains (recursion.rs:516/648/717 pattern).
        get_stream_id_proof_c(pctx.get_device_buffers_ptr(), stream_id);
        timer_stop_and_log_info!(GEN_RECURSIVE_PROOF);

        // Verify the generated STARK proof against this AIR's own verkey. The proof buffer
        // is [agg_publics (publics_aggregation)] ++ [stark_proof]; verify the stark_proof
        // part (proofman.rs:2990-2991 split). The transcript challenge is recomputed via
        // Fiat-Shamir (challenges=None).
        timer_start_info!(VERIFY_RECURSIVE_PROOF);
        let stark_info_path = setup.setup_path.display().to_string() + ".starkinfo.json";
        let expressions_bin_path = setup.setup_path.display().to_string() + ".verifier.bin";
        let verkey_path = setup.setup_path.display().to_string() + ".verkey.json";
        let valid = verify_proof::<Goldilocks>(
            proof_buffer[publics_aggregation..].as_mut_ptr(),
            stark_info_path,
            expressions_bin_path,
            verkey_path,
            Some(publics.clone()),
            None,
            None,
        );
        timer_stop_and_log_info!(VERIFY_RECURSIVE_PROOF);

        if !valid {
            tracing::info!("··· {}", "\u{2717} Recursive proof was NOT verified".bright_red().bold());
            return Err(Box::new(ProofmanError::InvalidProof("Recursive proof verification failed".into())));
        }
        tracing::info!("    {}", "\u{2713} Recursive proof verified".bright_green().bold());

        Ok(())
    }
}
