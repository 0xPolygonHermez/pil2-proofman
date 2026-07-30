// extern crate env_logger;
use clap::Parser;
use regex::Regex;
use proofman_common::{
    calculate_fixed_tree, init_gpu_setup, initialize_logger, GetWitnessTraceFunc, ProofmanOptions, SetupCtx,
    SetupsVadcop, MpiCtx, ProofCtx, ProofmanError, ProofType,
};
use proofman::{n_publics_aggregation, verify_proof, ProofMan};
use proofman_witness::load_packed_info;
use proofman_starks_lib_c::{
    add_publics_aggregation_c, gen_recursive_proof_c, get_stream_id_proof_c, load_device_const_pols_c,
    load_device_setup_c, read_exec_file_c,
};
use libloading::{Library, Symbol};
use std::fs::File;
use std::io::Read;
use colored::Colorize;
use proofman_fields::{Field, Goldilocks};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::error::Error;
use std::str::FromStr;
use proofman_util::{timer_start_info, timer_stop_and_log_info};

// Circom witness-library entry points (mirror examples/test-recursive/src/recursive.rs).
// `GetWitnessTraceFunc` is shared with proofman_common so the FFI signature has one owner.
type GetCircomCircuitFunc = unsafe extern "C" fn(dat_file: *const c_char) -> *mut c_void;

/// Proves ONE air on its own: the transcript is seeded from its verkey + publics, so the proof
/// verifies standalone. The trace comes either from a recursion air's circom library (`--proof`)
/// or from the project's Rust witness library (`--witness-lib --air`).
#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct ProveAirCmd {
    /// Recursion input: zkin file whose name encodes ag<N>_air<M>_t<ProofType>.
    #[clap(short = 'p', long, conflicts_with = "witness_lib", required_unless_present = "witness_lib")]
    pub proof: Option<PathBuf>,

    /// Witness computation dynamic library path.
    #[clap(short = 'w', long, requires = "air")]
    pub witness_lib: Option<PathBuf>,

    /// Name of the air to prove, as it appears in the pilout (e.g. Blake3).
    #[clap(short = 'a', long, requires = "witness_lib")]
    pub air: Option<String>,

    /// Public inputs path (witness-lib mode).
    #[clap(short = 'i', long, requires = "witness_lib")]
    pub public_inputs: Option<PathBuf>,

    #[clap(short = 'k', long)]
    pub proving_key: PathBuf,

    /// Stop after generating the recursion witness (legacy gen-witness behavior).
    #[clap(long, conflicts_with = "witness_lib")]
    pub emit_witness_only: bool,

    /// Run the prover on the GPU.
    #[clap(long)]
    pub gpu: bool,

    /// Force a packed trace (witness-lib mode). On `--gpu` this is already the default whenever
    /// the witness library exports `packed_info`; pass it to pack on the CPU too.
    #[clap(long, requires = "witness_lib")]
    pub packed: bool,

    /// Never pack the trace, even on `--gpu`.
    #[clap(long, conflicts_with = "packed", requires = "witness_lib")]
    pub no_packed: bool,

    /// Skip verifying the generated proof (witness-lib mode; useful for timing runs).
    #[clap(long, requires = "witness_lib")]
    pub no_verify: bool,

    /// Verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, help = "Increase verbosity level")]
    pub verbose: u8, // Using u8 to hold the number of `-v`
}

impl ProveAirCmd {
    pub fn run(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("{} ProveAir", format!("{: >12}", "Command").bright_green().bold());
        println!();

        // clap's `requires` / `required_unless_present` make exactly one of these Some.
        match (&self.witness_lib, &self.air, &self.proof) {
            (Some(witness_lib), Some(air), _) => self.run_witness_lib(witness_lib, air),
            (_, _, Some(proof)) => self.run_recursion(proof),
            _ => Err(Box::new(ProofmanError::InvalidParameters(
                "either --proof or --witness-lib together with --air is required".into(),
            ))),
        }
    }

    /// Regular air: ProofMan plans every instance but only the requested air gets a witness.
    fn run_witness_lib(&self, witness_lib: &Path, air: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut options = ProofmanOptions::new();
        options.no_aggregation();
        if self.gpu {
            options.gpu();
        }
        // Exporting `packed_info` is the library's declaration that its components honour packing,
        // so the GPU takes the ~5x smaller H2D by default.
        let packed_info = load_packed_info(witness_lib)?;
        if self.packed && packed_info.is_empty() {
            return Err(Box::new(ProofmanError::InvalidParameters(format!(
                "--packed requires the witness library to export `packed_info`, but {witness_lib:?} does not"
            ))));
        }
        if self.packed || (!self.no_packed && self.gpu && !packed_info.is_empty()) {
            options.packed();
            options.packed_info(packed_info);
        } else {
            // gpu() implies packing; the user asked for none.
            options.no_packed();
        }
        options.verbose_mode(self.verbose.into());

        let proofman = ProofMan::<Goldilocks>::new(self.proving_key.clone(), options)?;
        proofman.generate_air_proof(
            witness_lib.to_path_buf(),
            self.public_inputs.clone(),
            air,
            self.verbose.into(),
            !self.no_verify,
        )?;

        Ok(())
    }

    fn run_recursion(&self, proof_path: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        initialize_logger(self.verbose.into(), None);

        let mut pctx: ProofCtx<Goldilocks> = ProofCtx::create_ctx(
            self.proving_key.clone(),
            true,
            self.verbose.into(),
            Arc::new(MpiCtx::new()),
            self.gpu,
        )?;

        let mut zkin_file = File::open(proof_path)?;
        let mut zkin_u8 = Vec::new();
        zkin_file.read_to_end(&mut zkin_u8)?;
        if !zkin_u8.len().is_multiple_of(8) {
            return Err(Box::new(ProofmanError::InvalidProof(format!(
                "Proof file size ({} bytes) is not a multiple of 8",
                zkin_u8.len()
            ))));
        }
        let mut zkin: Vec<u64> = zkin_u8.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();

        // Match the file name, not the whole path: a directory that also encodes a proof type would
        // otherwise win. The type is snake_case, so `_` has to be in the class.
        let name = proof_path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
            ProofmanError::InvalidParameters(format!("Proof file name is not valid UTF-8: {proof_path:?}"))
        })?;
        let stem = name.strip_suffix(".bin").unwrap_or(name);
        let re = Regex::new(r"ag(\d+)_air(\d+)_t([A-Za-z0-9_]+)$").unwrap();
        let info = re.captures(stem).ok_or_else(|| {
            ProofmanError::InvalidParameters(format!(
                "Proof file name {name:?} does not match [zkin_]ag<N>_air<M>_t<proof_type>.bin"
            ))
        })?;
        let parse_id = |raw: &str, what: &str| -> Result<usize, ProofmanError> {
            raw.parse::<usize>()
                .map_err(|e| ProofmanError::InvalidParameters(format!("Invalid {what} {raw:?} in {name:?}: {e}")))
        };
        let airgroup_id = parse_id(&info[1], "airgroup id")?;
        let air_id = parse_id(&info[2], "air id")?;
        let proof_type = &ProofType::from_str(&info[3])
            .map_err(|_| ProofmanError::InvalidParameters(format!("Unknown proof type {:?} in {name:?}", &info[3])))?;

        // A recursive-test key holds its single AIR in the Basic slot, so fall back to that -- but
        // only if it exists, else a missing setup silently proves a different circuit.
        let has_setup = |t: &ProofType| {
            Path::new(&format!(
                "{}.starkinfo.json",
                pctx.global_info.get_air_setup_path(airgroup_id, air_id, t).display()
            ))
            .exists()
        };
        let setup_proof_type = if has_setup(proof_type) {
            *proof_type
        } else if has_setup(&ProofType::Basic) {
            tracing::debug!("no {proof_type:?} setup for air {air_id}; loading the Basic layout");
            ProofType::Basic
        } else {
            return Err(Box::new(ProofmanError::InvalidSetup(format!(
                "Proving key has no {proof_type:?} (nor Basic) setup for airgroup {airgroup_id} air {air_id}"
            ))));
        };

        let sctx: SetupCtx<Goldilocks> =
            SetupCtx::new(&pctx.global_info, &setup_proof_type, false, &[], &[], self.gpu)?;

        // Without this the CUDA context is unselected and check_device_memory_c returns 0.
        init_gpu_setup(&pctx.global_info.hash, self.gpu)?;

        let setup = sctx.get_setup(airgroup_id, air_id)?;

        // A tree left over from a setup at a different layout would crash the const loader.
        calculate_fixed_tree(setup);

        // From the setup path, not Setup's circom_state: that is only populated when
        // has_compressor is set, which a standalone recursive AIR leaves unset.
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

        // Header is n_adds then n_smap, body follows.
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

        // getWitnessTrace scatters straight into the trace + publics; no witness buffer.
        let n = 1u64 << setup.stark_info.stark_struct.n_bits;
        let n_publics = setup.stark_info.n_publics;
        let mut trace: Vec<Goldilocks> = vec![Goldilocks::ZERO; (n_cols * n) as usize];
        let mut publics: Vec<Goldilocks> = vec![Goldilocks::ZERO; n_publics as usize];

        timer_start_info!(WITNESS_GENERATION);
        let res = unsafe {
            let get_witness_trace: Symbol<GetWitnessTraceFunc> = library.get(b"getWitnessTrace\0")?;
            get_witness_trace(
                zkin.as_mut_ptr(),
                circom_circuit_ptr,
                exec_file_data.as_mut_ptr(),
                trace.as_mut_ptr() as *mut c_void,
                publics.as_mut_ptr() as *mut c_void,
                n,
                n_publics,
                n_cols,
                1,
                std::ptr::null_mut(), // no signalValues pool for a one-shot CLI run
            )
        };
        timer_stop_and_log_info!(WITNESS_GENERATION);

        if res != 0 {
            return Err(Box::new(ProofmanError::InvalidProof("Error generating witness".into())));
        }

        if self.emit_witness_only {
            tracing::info!("    {}", "\u{2713} Witness generated successfully".bright_green().bold());
            return Ok(());
        }

        // gen_recursive_proof_gpu reads const pols from the *aggregation* buffer, which
        // set_device_buffers only allocates under aggregation=true -- hence an empty SetupsVadcop
        // patched with this AIR's const sizes, then set_device_buffers(aggregation: true).
        let load_tree = setup.preallocate;
        let mut setups_vadcop: SetupsVadcop<Goldilocks> =
            SetupsVadcop::new(&pctx.global_info, false, false, &[], self.gpu)?;
        setups_vadcop.total_const_pols_size = setup.const_pols_size_packed;
        if load_tree {
            setups_vadcop.total_const_tree_size = setup.const_tree_size;
        }
        pctx.set_device_buffers(&sctx, &setups_vadcop, true, self.gpu, 1, 1)?;

        // The proofType must match the one gen_recursive_proof_c reads the const pols under.
        let proof_type_str: &str = (*proof_type).into();
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
            // Single AIR, single slot: nothing to share with.
            false,
        );

        // Non-final proofs only: the vadcop tail goes through a different entry point.

        // Layout: aggregation publics in [0..publics_aggregation), then the proof itself.
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

        // NULL on GPU (loaded device-side from the paths); on CPU genProof fills these itself and
        // would segfault on NULL. Kept in scope so they outlive the FFI call.
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
            u64::MAX, // one-off launch: reserve stream internally
        );

        // Async: proof_buffer is only filled once the stream drains.
        get_stream_id_proof_c(pctx.get_device_buffers_ptr(), stream_id);
        timer_stop_and_log_info!(GEN_RECURSIVE_PROOF);

        // challenges=None: the verifier reseeds from verkey + publics, as the prover did.
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
