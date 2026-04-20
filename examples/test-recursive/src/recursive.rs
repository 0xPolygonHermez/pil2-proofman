use std::sync::{Arc, RwLock};
use std::env;

use proofman_common::{
    AirInstance, BufferPool, ProofCtx, ProofmanResult, SetupCtx, TraceInfo, GetCircomCircuitFunc,
    GetCircomCircuitCalcWitFunc, GetSizeWitnessFunc, GetWitnessFunc, GetSignalValuesFunc, GetWitness2SignalListFunc,
};
use witness::WitnessComponent;
use fields::PrimeField64;
use proofman_starks_lib_c::{read_exec_file_c, get_committed_pols_c};
use proofman_util::{timer_start_info, timer_stop_and_log_info};

use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::ffi::CString;
use bytemuck::cast_slice;
use libloading::{Library, Symbol};

pub struct Compressor {}

impl Compressor {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

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
            let current_dir =
                env::current_dir().expect("Failed to get current directory").join("examples/test-recursive");
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

            let mut file = File::open(exec_filename.clone()).unwrap();

            let mut bytes = [0u8; 8];

            file.read_exact(&mut bytes).unwrap();
            let n_adds = u64::from_le_bytes(bytes);

            file.read_exact(&mut bytes).unwrap();
            let n_smap = u64::from_le_bytes(bytes);

            let n_cols = setup.stark_info.map_sections_n["cm1"];

            let exec_data_size = 2 + n_adds * 4 + n_smap * n_cols;
            let mut exec_file_data: Vec<u64> = vec![0; exec_data_size as usize];
            read_exec_file_c(exec_file_data.as_mut_ptr(), exec_filename.as_str(), n_cols);

            let library: Library = unsafe { Library::new(rust_lib_path).unwrap() };

            let circom_circuit = unsafe {
                let init_circom_circuit: Symbol<GetCircomCircuitFunc> = library.get(b"initCircuit\0").unwrap();
                init_circom_circuit(dat_filename_ptr)
            };

            let circom_calc_wit = unsafe {
                let init_circom_calc_wit: Symbol<GetCircomCircuitCalcWitFunc> = library.get(b"initCalcWit\0").unwrap();
                init_circom_calc_wit(circom_circuit, rayon::current_num_threads() as u64)
            };

            let size_witness = unsafe {
                let get_size_witness: Symbol<GetSizeWitnessFunc> = library.get(b"getSizeWitness\0").unwrap();
                get_size_witness()
            };

            timer_start_info!(WITNESS_GENERATION);
            unsafe {
                let get_witness: Symbol<GetWitnessFunc> = library.get(b"getWitness\0").unwrap();
                get_witness(proof.as_ptr() as *mut u64, circom_calc_wit);
            }
            timer_stop_and_log_info!(WITNESS_GENERATION);

            let signal_values_ptr = unsafe {
                let get_signal_values: Symbol<GetSignalValuesFunc> = library.get(b"getSignalValues\0").unwrap();
                get_signal_values(circom_calc_wit)
            };
            let w2s_ptr = unsafe {
                let get_w2s: Symbol<GetWitness2SignalListFunc> = library.get(b"getWitness2SignalList\0").unwrap();
                get_w2s(circom_calc_wit)
            };

            let mut publics = vec![F::ZERO; setup.stark_info.n_publics as usize];
            let mut trace = vec![F::ZERO; n_cols as usize * (1 << setup.stark_info.stark_struct.n_bits) as usize];

            timer_start_info!(COMMITTED_POLS);
            get_committed_pols_c(
                signal_values_ptr,
                w2s_ptr,
                exec_file_data.as_mut_ptr(),
                trace.as_mut_ptr() as *mut u8,
                publics.as_mut_ptr() as *mut u8,
                size_witness,
                1 << (setup.stark_info.stark_struct.n_bits),
                setup.stark_info.n_publics,
                n_cols,
            );
            timer_stop_and_log_info!(COMMITTED_POLS);

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
