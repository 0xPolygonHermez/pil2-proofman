use std::sync::{Arc, RwLock};
use std::env;

use std::ffi::{c_void, c_char};
use proofman_common::{AirInstance, BufferPool, ProofCtx, SetupCtx, TraceInfo};
use witness::WitnessComponent;
use std::path::PathBuf;
use fields::PrimeField64;
use proofman_starks_lib_c::{read_exec_file_c, get_committed_pols_c};

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::ffi::CString;
use bytemuck::cast_slice;
use libloading::{Library, Symbol};

pub struct RecursiveC12 {}

impl RecursiveC12 {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

type GetWitnessFunc =
    unsafe extern "C" fn(zkin: *mut u64, circom_circuit: *mut c_void, witness: *mut c_void, n_mutexes: u64);

type GetSizeWitnessFunc = unsafe extern "C" fn() -> u64;

type GetCircomCircuitFunc = unsafe extern "C" fn(dat_file: *const c_char) -> *mut c_void;

impl<F: PrimeField64> WitnessComponent<F> for RecursiveC12 {
    fn execute(&self, pctx: Arc<ProofCtx<F>>, global_ids: &RwLock<Vec<usize>>, _input_data_path: Option<PathBuf>) {
        pctx.add_instance(0, 0, 1);
        global_ids.write().unwrap().push(0);
    }

    fn calculate_witness(
        &self,
        stage: u32,
        pctx: Arc<ProofCtx<F>>,
        sctx: Arc<SetupCtx<F>>,
        _instance_ids: &[usize],
        _n_cores: usize,
        _buffer_pool: &dyn BufferPool<F>,
    ) {
        if stage == 1 {
            let setup = sctx.get_setup(0, 0);
            let current_dir =
                env::current_dir().expect("Failed to get current directory").join("examples/test-recursive-c12");
            let proof_path = current_dir.join("proof.bin");

            let mut file = File::open(proof_path).unwrap();
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();

            let proof_slice: &[u64] = cast_slice(&buffer);
            let proof: Vec<u64> = proof_slice.to_vec();

            let lib_extension = if cfg!(target_os = "macos") { ".dylib" } else { ".so" };
            let rust_lib_filename = setup.setup_path.display().to_string() + lib_extension;
            let rust_lib_path = Path::new(&rust_lib_filename);

            let dat_filename = setup.setup_path.display().to_string() + ".dat";
            let dat_filename_str = CString::new(dat_filename).unwrap();
            let dat_filename_ptr = dat_filename_str.as_ptr() as *mut std::os::raw::c_char;

            let exec_filename = setup.setup_path.display().to_string() + ".exec";
            let exec_filename_str = CString::new(exec_filename.clone()).unwrap();
            let exec_filename_ptr = exec_filename_str.as_ptr() as *mut std::os::raw::c_char;

            let mut file = File::open(exec_filename.clone()).unwrap();

            let mut bytes = [0u8; 8];

            file.read_exact(&mut bytes).unwrap();
            let n_adds = u64::from_le_bytes(bytes);

            file.read_exact(&mut bytes).unwrap();
            let n_smap = u64::from_le_bytes(bytes);

            let n_cols = 12;

            let exec_data_size = 2 + n_adds * 4 + n_smap * n_cols;
            let mut exec_file_data: Vec<u64> = vec![0; exec_data_size as usize];
            read_exec_file_c(exec_file_data.as_mut_ptr(), exec_filename_ptr, n_cols);

            let library: Library = unsafe { Library::new(rust_lib_path).unwrap() };

            let circom_circuit = unsafe {
                let init_circom_circuit: Symbol<GetCircomCircuitFunc> = library.get(b"initCircuit\0").unwrap();
                init_circom_circuit(dat_filename_ptr)
            };

            let size_witness = unsafe {
                let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0").unwrap();
                get_size_witness()
            };

            let witness_size = size_witness + exec_file_data.first().unwrap();

            let witness: Vec<F> = vec![F::ZERO; witness_size as usize];

            unsafe {
                let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0").unwrap();
                get_witness(proof.as_ptr() as *mut u64, circom_circuit, witness.as_ptr() as *mut c_void, 1);
            }

            let publics = vec![F::ZERO; setup.stark_info.n_publics as usize];
            let trace = vec![F::ZERO; n_cols as usize * (1 << setup.stark_info.stark_struct.n_bits) as usize];

            get_committed_pols_c(
                witness.as_ptr() as *mut u8,
                exec_file_data.as_mut_ptr(),
                trace.as_ptr() as *mut u8,
                publics.as_ptr() as *mut u8,
                size_witness,
                1 << (setup.stark_info.stark_struct.n_bits),
                setup.stark_info.n_publics,
                n_cols,
            );

            for (index, public) in publics.iter().enumerate() {
                pctx.set_public_value(F::as_canonical_u64(public), index);
            }

            let air_instance =
                AirInstance::new(TraceInfo::new(0, 0, 1 << (setup.stark_info.stark_struct.n_bits), trace, false));
            pctx.add_air_instance(air_instance, 0);
        }
    }
}
