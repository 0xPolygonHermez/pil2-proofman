use std::sync::{Arc, RwLock};
use std::env;

use std::ffi::{c_void, c_char};
use proofman_common::{
    load_exec_file, AirInstance, BufferPool, GetWitnessTraceFunc, ProofCtx, ProofmanResult, SetupCtx, TraceInfo,
};
use proofman_witness::WitnessComponent;
use proofman_fields::PrimeField64;
use proofman_starks_lib_c::expand_gate_bands_c;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::ffi::CString;
use libloading::{Library, Symbol};

pub struct Compressor {}

impl Compressor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

type GetCircomCircuitFunc = unsafe extern "C" fn(dat_file: *const c_char) -> *mut c_void;

impl<F: PrimeField64> WitnessComponent<F> for Compressor {
    fn execute(
        &self,
        pctx: Arc<ProofCtx<F>>,
        _sctx: Arc<SetupCtx<F>>,
        global_ids: &RwLock<Vec<usize>>,
    ) -> ProofmanResult<()> {
        pctx.add_instance(0, 0)?;
        global_ids.write().unwrap().push(0);
        Ok(())
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) -> ProofmanResult<()> {
        if stage == 1 {
            let setup = sctx.get_setup(0, 0)?;
            let hash_family = pctx.global_info.hash.to_lowercase();
            let current_dir = env::current_dir()
                .expect("Failed to get current directory")
                .join("examples/test-recursive")
                .join(&hash_family);
            // The inner proof this fixture recurses over. Which recursion level it is depends on the
            // family's fixture -- poseidon2 recurses over a recursive2 zkin, blake3 over a
            // recursive1 one -- and either builds the same production aggregator AIR. What matters
            // is that the .bin and the test.circom come from the SAME setup: the zkin is just the
            // circom's input vector, so a mismatched pair generates a witness that fails an assert
            // and every constraint that reads it then reports as broken. The name keeps
            // `tCompressor` because prove-air parses the proof type out of it to resolve the setup
            // path, and the harness names every recursive-test AIR "Compressor".
            let proof_path = current_dir.join("ag0_air0_tCompressor.bin");

            let mut file = File::open(proof_path).unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();

            assert!(buffer.len().is_multiple_of(8), "proof file length is not a multiple of 8");
            let proof: Vec<u64> = buffer.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();

            let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
            let rust_lib_filename = setup.setup_path.display().to_string() + lib_extension;
            let rust_lib_path = Path::new(&rust_lib_filename);

            let dat_filename = setup.setup_path.display().to_string() + ".dat";
            let dat_filename_str = CString::new(dat_filename).unwrap();
            let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

            let exec_filename = setup.setup_path.display().to_string() + ".exec";
            let n_cols = setup.stark_info.map_sections_n["cm1"];
            // Whole file, gate-band tail included.
            let mut exec_file_data = load_exec_file(&exec_filename, n_cols)?;
            let exec_words = exec_file_data.len() as u64;

            let library: Library = unsafe { Library::new(rust_lib_path).unwrap() };

            let circom_circuit = unsafe {
                let init_circom_circuit: Symbol<GetCircomCircuitFunc> = library.get(b"initCircuit\0").unwrap();
                init_circom_circuit(dat_filename_ptr)
            };

            let publics = vec![F::ZERO; setup.stark_info.n_publics as usize];
            let trace = vec![F::ZERO; n_cols as usize * (1 << setup.stark_info.stark_struct.n_bits) as usize];
            let n_rows: u64 = 1 << setup.stark_info.stark_struct.n_bits;

            let res = unsafe {
                let get_witness_trace: Symbol<GetWitnessTraceFunc> = library.get(b"getWitnessTrace\0").unwrap();
                let nmutex = std::cmp::min(8, rayon::current_num_threads()) as u64;
                get_witness_trace(
                    proof.as_ptr() as *mut u64,
                    circom_circuit,
                    exec_file_data.as_mut_ptr(),
                    trace.as_ptr() as *mut c_void,
                    publics.as_ptr() as *mut c_void,
                    n_rows,
                    setup.stark_info.n_publics,
                    // Full width, NOT recursion_trace_stride: expand_gate_bands_c below runs on the
                    // host unconditionally and rebuilds the interiors in place, so it needs the real
                    // layout. This fixture hands the finished trace to the ordinary prove path, which
                    // never reaches gen_recursive_proof_gpu's compact reader.
                    n_cols,
                    nmutex,
                    std::ptr::null_mut(), // no signalValues pool here: self-allocate
                )
            };
            // Otherwise a failed solve yields an all-zero trace and an opaque error later.
            assert_eq!(res, 0, "getWitnessTrace failed for the compressor witness");
            // The hash gates map only their boundary; fill the rest from it. No-op on an exec
            // file without a band section.
            expand_gate_bands_c(
                trace.as_ptr() as *mut u8,
                exec_file_data.as_mut_ptr(),
                n_cols,
                exec_words,
                1 << setup.stark_info.stark_struct.n_bits,
            );

            for (index, public) in publics.iter().enumerate() {
                pctx.set_public_value(F::as_canonical_u64(public), index);
            }

            let air_instance = AirInstance::new(TraceInfo::new(
                0,
                0,
                n_cols as usize,
                1 << (setup.stark_info.stark_struct.n_bits),
                trace,
                false,
                false,
            ));
            pctx.add_air_instance(air_instance, 0);
        }
        Ok(())
    }
}
