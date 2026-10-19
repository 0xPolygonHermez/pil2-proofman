use proofman_common::{
    format_bytes, GlobalInfoAir, ProofmanError, ProofmanResult, ProofType, PublicsInfo, Setup,
    calculate_fixed_tree_snark, load_const_pols_recursivef, load_const_pols_tree, MemoryHandlerRecursive, VerboseMode,
    initialize_logger,
};
use proofman_util::{timer_start_info, timer_stop_and_log_info, timer_start_debug, timer_stop_and_log_debug};
use proofman_verifier::VadcopFinalProof;
use proofman_fields::PrimeField64;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::process::Command;
use colored::Colorize;
use std::io::Read;
use std::ffi::c_void;
use crate::check_const_tree;
use proofman_starks_lib_c::{
    init_final_snark_prover_c, free_final_snark_prover_c, get_snark_protocol_id_c, snark_proof_bytes_to_json_c,
    get_unified_buffer_gpu_for_recursivef_c, pre_allocate_final_snark_prover_c, free_device_buffers_recursivef_c,
    gen_device_buffers_recursivef_c, set_gpu_mode_c, get_num_gpus_c, init_gpu_setup_c, get_unified_buffer_gpu_size_c,
};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::{
    verify_proof_bn128, generate_witness_final_snark, generate_recursivef_proof, generate_snark_proof, RecursivefProof,
    PROVING,
};
use crate::pilfflonk_wrap::{self, PilfflonkWrapProver};
use proofman_pilfflonk::global_info::PROVING_KEY_DIR;
use proofman_pilfflonk::PilfflonkGlobalInfo;
use serde::{Deserialize, Serialize};

/// Sets GPU mode and verifies that a usable GPU is available when `gpu` is requested.
/// Returns an error if the library was built without CUDA support, or if GPU mode was
/// requested but no GPUs were found.
pub fn ensure_gpu_available(gpu: bool) -> ProofmanResult<()> {
    if !set_gpu_mode_c(gpu) {
        return Err(ProofmanError::InvalidConfiguration(
            "GPU mode requested but library was built without CUDA support".into(),
        ));
    }
    if gpu && get_num_gpus_c() == 0 {
        return Err(ProofmanError::InvalidConfiguration("No GPUs found".into()));
    }
    Ok(())
}

/// The protocol of the final SNARK: rapidsnark's PLONK or FFLONK, whose zkey names it, or pilfflonk
/// (pilfflonk/docs/README.md), whose key is a `provingKey/` ([`FinalSnarkKey`]).
pub enum SnarkProtocol {
    Fflonk,
    Plonk,
    Pilfflonk,
}

/// The protocol id of a pilfflonk proof in a [`SnarkProof`]. A zkey names the ids of PLONK (2) and
/// FFLONK (10), snarkjs's, and none for pilfflonk, so this one is proofman's own, far from those:
/// `0x7066`, the "pf" of pilfflonk's tags (the high half of its bytecode's version,
/// pilfflonk/docs/formats.md#bytecode).
pub const PILFFLONK_PROTOCOL_ID: u64 = 0x7066;

impl SnarkProtocol {
    pub fn protocol_id(&self) -> u64 {
        match self {
            SnarkProtocol::Fflonk => 10,
            SnarkProtocol::Plonk => 2,
            SnarkProtocol::Pilfflonk => PILFFLONK_PROTOCOL_ID,
        }
    }

    /// Serialized proof length: 24 big-endian 32-byte words for both protocols (PLONK: 9 G1
    /// points and 6 evaluations; FFLONK: 4 points and 16 evaluations). None for pilfflonk, whose
    /// length is its key's: only its vkey says where the parts of its bytes end.
    pub fn proof_bytes(&self) -> Option<usize> {
        match self {
            SnarkProtocol::Plonk | SnarkProtocol::Fflonk => Some(24 * 32),
            SnarkProtocol::Pilfflonk => None,
        }
    }

    pub fn protocol_name(&self) -> &'static str {
        match self {
            SnarkProtocol::Plonk => "plonk",
            SnarkProtocol::Fflonk => "fflonk",
            SnarkProtocol::Pilfflonk => "pilfflonk",
        }
    }

    pub fn from_protocol_id(protocol_id: u64) -> ProofmanResult<Self> {
        match protocol_id {
            2 => Ok(SnarkProtocol::Plonk),
            10 => Ok(SnarkProtocol::Fflonk),
            PILFFLONK_PROTOCOL_ID => Ok(SnarkProtocol::Pilfflonk),
            _ => Err(ProofmanError::InvalidConfiguration(format!("Unsupported snark protocol id: {}", protocol_id))),
        }
    }

    /// The protocol of a loaded final SNARK prover: the one its zkey names, PLONK or FFLONK.
    fn from_snark_prover(snark_prover: *mut c_void) -> ProofmanResult<Self> {
        match Self::from_protocol_id(get_snark_protocol_id_c(snark_prover))? {
            SnarkProtocol::Pilfflonk => Err(ProofmanError::InvalidConfiguration(
                "A zkey names PLONK or FFLONK, and this one the id of pilfflonk, whose key is a provingKey/".into(),
            )),
            protocol => Ok(protocol),
        }
    }

    /// What `snarkjs <protocol> verify` prints when the proof verifies. None for pilfflonk, which
    /// snarkjs does not verify: its JS verifier's verdict is its exit status.
    fn verified_message(&self) -> Option<&'static str> {
        match self {
            SnarkProtocol::Plonk => Some("OK!"),
            SnarkProtocol::Fflonk => Some("PROOF VERIFIED SUCCESSFULLY"),
            SnarkProtocol::Pilfflonk => None,
        }
    }
}

/// The key of the final SNARK in a setup's `provingKeySnark/final/`, which setup-snark writes for
/// one protocol (`--final-snark`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalSnarkKey {
    /// `final.zkey`: rapidsnark's PLONK or FFLONK key, which names its protocol.
    Zkey(PathBuf),
    /// `provingKey/`: a pilfflonk key, whose globalInfo says `"backend": "pilfflonk"`.
    Pilfflonk(PathBuf),
}

impl FinalSnarkKey {
    /// The key of the setup whose final SNARK files have the stem `setup_snark_path`
    /// (`provingKeySnark/final/final`): `final.zkey` or `provingKey/` beside them, exactly one. The
    /// zkey is read when its prover loads; the `provingKey/` must be a pilfflonk one now, by its
    /// globalInfo.
    pub fn find(setup_snark_path: &Path) -> ProofmanResult<Self> {
        let mut zkey = setup_snark_path.as_os_str().to_os_string();
        zkey.push(".zkey");
        let zkey = PathBuf::from(zkey);
        // setup-snark sets the final circuit up with `setup-pilfflonk -b provingKeySnark/final`.
        let proving_key = setup_snark_path.with_file_name(PROVING_KEY_DIR);
        match (zkey.exists(), proving_key.exists()) {
            (true, false) => Ok(Self::Zkey(zkey)),
            (false, true) => match PilfflonkGlobalInfo::from_proving_key(&proving_key) {
                Ok(_) => Ok(Self::Pilfflonk(proving_key)),
                Err(e) => Err(ProofmanError::InvalidSetup(format!(
                    "{} is not a pilfflonk key, the only final SNARK key that is a provingKey/: {e}",
                    proving_key.display()
                ))),
            },
            (true, true) => Err(ProofmanError::InvalidSetup(format!(
                "There are two final SNARK keys, {} (PLONK or FFLONK) and {} (pilfflonk): setup-snark writes the key \
                 of one protocol and removes the other's, so running it again with the --final-snark to prove with \
                 leaves only that one",
                zkey.display(),
                proving_key.display()
            ))),
            (false, false) => Err(ProofmanError::InvalidSetup(format!(
                "There is no final SNARK key: neither {} (PLONK or FFLONK) nor {} (pilfflonk)",
                zkey.display(),
                proving_key.display()
            ))),
        }
    }
}

/// [`SnarkWrapper`] of a poseidon key: the recursivef proves the vadcop_final proof, and then the final
/// SNARK of `provingKeySnark/final/` ([`FinalSnarkKey`]), rapidsnark's PLONK or FFLONK or pilfflonk,
/// proves the recursivef's verifier circuit.
///
/// **One proof at a time.** The wrapper is `Send` and `Sync`, and proofs of it from several threads
/// run one after another: each proof uses its buffers and provers alone.
///
/// **GPU memory with pilfflonk.** As for PLONK, the recursivef's device buffers are the wrapper's
/// (in `d_buffers`, proofman's unified buffer, if there is one). With the unified buffer, all of
/// pilfflonk's key (the SRS's powers, the fixed columns' coefficients, its proofs' arena) is in its
/// aux area (`pilfflonk_wrap::load_poseidon_wrap`), which the recursivef writes over: preloaded, the
/// key keeps a pinned host copy of it and writes it back before each proof. Without it, pilfflonk's
/// proofs keep their data in the recursivef's prover buffer (`pilfflonk_wrap::wrap_arena`), where
/// `pre_allocate_final_snark_prover_c` carves the PLONK prover's, and the key's own memory is beside it.
pub(crate) struct PoseidonWrap<F: PrimeField64> {
    pub setup_snark_path: PathBuf,
    pub setup_recursivef: Setup<F>,
    pub vadcop_final_verkey: Vec<u64>,
    pub aux_trace: Arc<Vec<F>>,
    pub recursivef_const_pols: Arc<Vec<F>>,
    pub recursivef_const_tree: Arc<Vec<F>>,
    pub d_buffers: Option<*mut c_void>,
    pub reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
    /// rapidsnark's prover of `final.zkey`, with `preload`; never with pilfflonk.
    pub snark_prover: Option<*mut c_void>,
    /// The recursivef's device buffers (null on the CPU), out of which PLONK's GPU prover carves its
    /// own, and pilfflonk's its arena without `d_buffers`.
    pub d_buffers_recursivef: *mut c_void,
    pub proving_key_path: PathBuf,
    pub memory_handler_recursive_witness: Arc<MemoryHandlerRecursive<F>>,
    pub gpu: bool,
    pub final_snark_key: FinalSnarkKey,
    /// Held by each proof throughout ([`generate_final_snark_proof`](Self::generate_final_snark_proof)),
    /// so that the wrapper proves one at a time: two proofs at once would both write the
    /// recursivef's prover buffer `aux_trace` and its device buffers, and rapidsnark's prover.
    /// With `preload`, it holds pilfflonk's prover, a C++ handle that is neither `Send` nor `Sync`:
    /// it is only used under this lock, which the wrapper's `Send` and `Sync` rely on.
    proving: Mutex<Option<PilfflonkWrapProver>>,
}

/// The BN254 scalar order `r`, big-endian.
const BN254_R_BE: [u8; 32] = [
    0x30, 0x64, 0x4e, 0x72, 0xe1, 0x31, 0xa0, 0x29, 0xb8, 0x50, 0x45, 0xb6, 0x81, 0x81, 0x58, 0x5d, 0x28, 0x33, 0xe8,
    0x48, 0x79, 0xb9, 0x70, 0x91, 0x43, 0xe1, 0xf5, 0x93, 0xf0, 0x00, 0x00, 0x01,
];

/// The first 32-byte big-endian word at or above `r`, if any. Words of equal length compare as
/// numbers.
fn first_word_not_below_r(bytes: &[u8]) -> Option<usize> {
    bytes.as_chunks::<32>().0.iter().position(|word| word >= &BN254_R_BE)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnarkProof {
    pub proof_bytes: Vec<u8>,
    pub public_bytes: Vec<u8>,
    pub public_snark_bytes: Vec<u8>,
    pub protocol_id: u64,
}

impl SnarkProof {
    pub fn new(proof_bytes: Vec<u8>, public_bytes: Vec<u8>, public_snark_bytes: Vec<u8>, protocol_id: u64) -> Self {
        Self { proof_bytes, public_bytes, public_snark_bytes, protocol_id }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = File::create(path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to create file for saving SNARK proof: {}: {}", path.display(), e),
            )
        })?;

        bincode::serde::encode_into_std_write(self, &mut file, bincode::config::standard())?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut file = File::open(path.as_ref()).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to open file for loading SNARK proof: {}: {}", path.as_ref().display(), e),
            )
        })?;
        let proof: SnarkProof = bincode::serde::decode_from_std_read(&mut file, bincode::config::standard())?;
        Ok(proof)
    }

    /// The JSON views of a PLONK or FFLONK proof and of its publics, as snarkjs reads them. A
    /// pilfflonk proof's bytes do not say where their parts end, its vkey does, and it is refused:
    /// [`convert_to_json_with_vkey`](Self::convert_to_json_with_vkey).
    pub fn convert_to_json(
        &self,
    ) -> Result<(serde_json::Value, serde_json::Value), Box<dyn std::error::Error + Send + Sync>> {
        // The C parser throws on an unknown protocol or a short proof, which aborts the process
        // across the FFI, and reads the digest in 32-byte words past a ragged end. Refuse here.
        let protocol = SnarkProtocol::from_protocol_id(self.protocol_id)?;
        let Some(proof_bytes) = protocol.proof_bytes() else {
            return Err("A pilfflonk proof's bytes do not say where their parts end, its vkey does: \
                 convert it with convert_to_json_with_vkey"
                .into());
        };
        if self.proof_bytes.len() != proof_bytes {
            return Err(format!(
                "{} proof is {} bytes, expected {}",
                protocol.protocol_name(),
                self.proof_bytes.len(),
                proof_bytes
            )
            .into());
        }
        if self.public_snark_bytes.is_empty() || !self.public_snark_bytes.len().is_multiple_of(32) {
            return Err(format!(
                "SNARK public input is {} bytes, not whole 32-byte field elements",
                self.public_snark_bytes.len()
            )
            .into());
        }
        // The Solidity verifiers snarkjs generates for both protocols require every proof word
        // below `r` (their `checkField`), while the parser reads coordinates mod q and
        // evaluations mod r: a word past either bound is another encoding of the same proof,
        // which would verify here and be rejected on-chain. A digest is reduced mod `r` too.
        if let Some(i) = first_word_not_below_r(&self.proof_bytes) {
            return Err(
                format!("{} proof word {i} is not below the BN254 scalar order", protocol.protocol_name()).into()
            );
        }
        if let Some(i) = first_word_not_below_r(&self.public_snark_bytes) {
            return Err(format!("SNARK public input {i} is not below the BN254 scalar order").into());
        }
        let (proof_json, publics_json) =
            snark_proof_bytes_to_json_c(&self.proof_bytes, &self.public_snark_bytes, self.protocol_id as i32);

        let proof_json_value: serde_json::Value = serde_json::from_str(&proof_json)?;
        let publics_json_value: serde_json::Value = serde_json::from_str(&publics_json)?;

        Ok((proof_json_value, publics_json_value))
    }

    /// The JSON views of the proof and of its publics, as its verifier reads them, for every
    /// protocol: for PLONK and FFLONK, [`convert_to_json`](Self::convert_to_json)'s, which do not
    /// read `vkey`; for pilfflonk, `proof.json` and `publics.json` (pilfflonk/docs/formats.md#proof),
    /// the proof's bytes named by the vkey at `vkey`, `pilfflonk.vkey.json`.
    pub fn convert_to_json_with_vkey(
        &self,
        vkey: &Path,
    ) -> Result<(serde_json::Value, serde_json::Value), Box<dyn std::error::Error + Send + Sync>> {
        match SnarkProtocol::from_protocol_id(self.protocol_id)? {
            SnarkProtocol::Pilfflonk => {
                let vkey = pilfflonk_wrap::read_vkey(vkey)?;
                let (proof, publics) = pilfflonk_wrap::json_views(&self.proof_bytes, &self.public_snark_bytes, &vkey)?;
                Ok((serde_json::to_value(proof)?, serde_json::to_value(publics)?))
            }
            SnarkProtocol::Plonk | SnarkProtocol::Fflonk => self.convert_to_json(),
        }
    }

    pub fn get_public_bytes(&self) -> &[u8] {
        &self.public_bytes
    }
}

impl<F: PrimeField64> Drop for PoseidonWrap<F> {
    fn drop(&mut self) {
        if let Some(snark_prover) = self.snark_prover {
            free_final_snark_prover_c(snark_prover);
        }
        // pilfflonk's key first, whose arena may be the recursivef's prover buffer.
        drop(self.proving.get_mut().unwrap_or_else(|poisoned| poisoned.into_inner()).take());
        free_device_buffers_recursivef_c(self.d_buffers_recursivef);
    }
}

impl<F: PrimeField64> PoseidonWrap<F> {
    pub fn new_with_preallocated_buffers(
        proving_key_path: &Path,
        verbose_mode: VerboseMode,
        _aux_trace: Option<Arc<Vec<F>>>,
        d_buffers: Option<*mut c_void>,
        reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
        preload: bool,
        gpu: bool,
    ) -> ProofmanResult<Self> {
        initialize_logger(verbose_mode, None);

        ensure_gpu_available(gpu)?;

        if d_buffers.is_some() && reload_fixed_pols_gpu.is_none() {
            return Err(ProofmanError::InvalidConfiguration(
                "A shared device buffer needs its owner's reload flag: the wrap overwrites it".to_string(),
            ));
        }

        let setup_recursivef_path =
            PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));
        let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
        let final_snark_key = FinalSnarkKey::find(&setup_snark_path)?;

        let vadcop_final_verkey_path =
            PathBuf::from(format!("{}/vadcop_final.verkey.json", proving_key_path.display()));

        let mut file = File::open(&vadcop_final_verkey_path).expect("Unable to open file");
        let mut json_str = String::new();
        file.read_to_string(&mut json_str).expect("Unable to read file");
        let vadcop_final_verkey: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");

        timer_start_info!(LOADING_RECURSIVE_F_SETUP);

        let setup_recursivef = Setup::new(
            &setup_recursivef_path,
            0,
            0,
            &GlobalInfoAir::new("RecursiveF".to_string()),
            &ProofType::RecursiveF,
            false,
            gpu,
            None,
            &std::collections::HashMap::new(),
            false,
        )?;

        {
            // On a shared buffer the tree is built over its aux area: not while a ProofMan job runs.
            let _proving = d_buffers.map(|_| PROVING.lock().unwrap_or_else(|e| e.into_inner()));
            check_const_tree(&setup_recursivef, &d_buffers)?;
        }

        let mut recursivef_const_pols_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_pols_size];
        load_const_pols_recursivef(&setup_recursivef, &mut recursivef_const_pols_buf);
        let recursivef_const_pols: Arc<Vec<F>> = Arc::new(recursivef_const_pols_buf);
        let mut recursivef_const_tree_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_tree_size];
        load_const_pols_tree(&setup_recursivef, &mut recursivef_const_tree_buf);
        let recursivef_const_tree: Arc<Vec<F>> = Arc::new(recursivef_const_tree_buf);

        timer_stop_and_log_info!(LOADING_RECURSIVE_F_SETUP);

        let aux_trace = recursivef_prover_buffer(_aux_trace, gpu, setup_recursivef.prover_buffer_size as usize);

        let d_buffers_vadcop = if let Some(d_buffers) = d_buffers { d_buffers } else { std::ptr::null_mut() };

        let p_setup: *mut c_void = (&setup_recursivef.p_setup).into();

        let verkey_path = setup_recursivef.verkey_file.clone();
        let mut contents = String::new();
        let mut file = File::open(verkey_path).unwrap();
        let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

        let verkey_str: String = serde_json::from_str(&contents)
            .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

        if gpu && !d_buffers_vadcop.is_null() {
            let need = 8 * (setup_recursivef.const_tree_size as u64 + setup_recursivef.prover_buffer_size);
            let total = get_unified_buffer_gpu_size_c(d_buffers_vadcop);
            if need > total {
                return Err(ProofmanError::InvalidConfiguration(format!(
                    "recursiveF needs {} of device memory but the unified buffer holds {}",
                    format_bytes(need as f64),
                    format_bytes(total as f64),
                )));
            }
        }

        // Freed if the final SNARK's prover cannot be loaded.
        let d_buffers_recursivef = RecursivefDeviceBuffers(gen_device_buffers_recursivef_c(
            p_setup as *mut u8,
            setup_recursivef.prover_buffer_size,
            d_buffers_vadcop as *mut u8,
            &verkey_str,
        ) as *mut c_void);
        if gpu && d_buffers_recursivef.as_ptr().is_null() {
            return Err(ProofmanError::InvalidConfiguration(
                "recursiveF's device buffers do not fit the unified buffer (the log says by how much)".to_string(),
            ));
        }

        timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
        let zkey_filename = setup_snark_path.display().to_string() + ".zkey";
        let snark_prover = if preload && matches!(final_snark_key, FinalSnarkKey::Zkey(_)) {
            let snark_prover = init_final_snark_prover_c(zkey_filename.as_str(), d_buffers_recursivef.as_ptr());
            if snark_prover.is_null() {
                return Err(std::io::Error::other(format!(
                    "Failed to initialize final snark prover from zkey file '{}'",
                    zkey_filename
                ))
                .into());
            }
            Some(snark_prover)
        } else {
            None
        };
        // pilfflonk's arena is in the unified buffer's aux area or in the recursivef's buffers.
        let pilfflonk_prover = match &final_snark_key {
            FinalSnarkKey::Pilfflonk(pilfflonk_key) if preload => {
                // On a shared buffer the key loads while no ProofMan job runs.
                let _shared = d_buffers.filter(|_| gpu).map(|_| PROVING.lock().unwrap_or_else(|e| e.into_inner()));
                let _aux = pilfflonk_wrap::AuxAreaUse::new(d_buffers.filter(|_| gpu));
                // SAFETY: the unified buffer or the recursivef's prover buffer, which outlive the prover
                // (the wrapper's `Drop` drops it first), and which a proof uses only for the recursivef
                // until pilfflonk proves.
                Some(unsafe {
                    pilfflonk_wrap::load_poseidon_wrap(
                        &setup_snark_path,
                        pilfflonk_key,
                        gpu,
                        d_buffers,
                        d_buffers_recursivef.as_ptr(),
                        true,
                    )
                }?)
            }
            _ => None,
        };

        timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);

        let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
            * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
            + setup_recursivef.stark_info.n_publics;

        let memory_handler_recursive_witness = Arc::new(MemoryHandlerRecursive::new(1, trace_size as usize));

        Ok(Self {
            aux_trace,
            recursivef_const_pols,
            recursivef_const_tree,
            setup_recursivef,
            setup_snark_path,
            snark_prover,
            proving_key_path: proving_key_path.to_path_buf(),
            vadcop_final_verkey,
            d_buffers,
            d_buffers_recursivef: d_buffers_recursivef.into_raw(),
            memory_handler_recursive_witness,
            reload_fixed_pols_gpu,
            gpu,
            final_snark_key,
            proving: Mutex::new(pilfflonk_prover),
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_final_snark_proof(
        &self,
        vadcop_proof: &VadcopFinalProof,
        verkey_override: Option<&[u64]>,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_info!(GENERATING_WRAPPER_SNARK_PROOF);

        if !proofman_common::hash_family::has_bn128_recursivef(&vadcop_proof.hash) {
            return Err(ProofmanError::InvalidConfiguration(format!(
                "{} proofs have no BN128 recursivef: their final SNARK is ProofMan::generate_final_snark_proof's",
                vadcop_proof.hash
            )));
        }
        if vadcop_proof.compressed {
            return Err(ProofmanError::InvalidConfiguration(
                "Compressed vadcop proofs are not supported for snark proof generation".to_string(),
            ));
        }
        let proof = vadcop_proof.proof_with_publics();

        // The RecursiveF verifier checks the proof against this verkey (stamped
        // into the leading 4 slots of the publics). Default is the vadcop_final
        // verkey; a recurser/aggregated proof committed under its own verkey, so
        // the caller passes it here — using vadcop_final's would diverge the
        // wrapper transcript (VerifyPoW aborts).
        let verkey = verkey_override.unwrap_or(&self.vadcop_final_verkey);

        // One proof at a time (`proving`), until this one is out. A proof that panicked poisons the
        // lock, and the next one takes it all the same: each proof writes its buffers afresh.
        let pilfflonk_prover = self.proving.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // On a shared buffer the caller, ProofMan::generate_final_snark_proof, holds the process-wide
        // PROVING throughout: no ProofMan job or reset runs while recursiveF and the snark write it.

        // The recursivef's const tree and prover buffer, and the snark's carve, live in the unified
        // buffer and may overwrite its const-pols regions, also in a proof that fails. The flag,
        // consumed by the next job's restore_after_snark, re-uploads them before any proof; it is set as
        // this proof ends, whatever its outcome, before the locks are let go.
        let _reload = ReloadFixedPolsOnDrop(self.d_buffers.and(self.reload_fixed_pols_gpu.as_deref()));

        let recursivef_proof = generate_recursivef_proof(
            &self.setup_recursivef,
            &self.memory_handler_recursive_witness,
            &proof,
            &self.aux_trace,
            &self.recursivef_const_pols,
            &self.recursivef_const_tree,
            verkey,
            self.setup_recursivef.prover_buffer_size as usize * std::mem::size_of::<F>(),
            self.d_buffers_recursivef,
        )?;

        if let FinalSnarkKey::Pilfflonk(pilfflonk_key) = &self.final_snark_key {
            let snark_proof = self.generate_pilfflonk_proof(
                pilfflonk_key,
                pilfflonk_prover.as_ref(),
                &proof,
                verkey,
                &recursivef_proof,
            )?;
            timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);
            return Ok(snark_proof);
        }

        timer_start_debug!(GENERATING_SNARK_PROOF);

        let snark_prover = match self.snark_prover {
            Some(prover) => prover,
            None => {
                let prover = init_final_snark_prover_c(
                    &(self.setup_snark_path.display().to_string() + ".zkey"),
                    self.d_buffers_recursivef,
                );
                if prover.is_null() {
                    return Err(std::io::Error::other(format!(
                        "Failed to initialize final snark prover from zkey file '{}'",
                        self.setup_snark_path.display().to_string() + ".zkey"
                    ))
                    .into());
                }
                prover
            }
        };

        // The zkey names the protocol, and the loaded prover read it.
        let protocol = match SnarkProtocol::from_snark_prover(snark_prover) {
            Ok(protocol) => protocol,
            Err(e) => {
                if self.snark_prover.is_none() {
                    free_final_snark_prover_c(snark_prover);
                }
                return Err(e);
            }
        };

        //  Spawn GPU pre-allocation on a separate thread so it overlaps with CPU witness computation
        let prealloc_handle = {
            let snark_prover = snark_prover as usize;
            let unified_buffer_gpu = if let Some(d_buffers) = self.d_buffers {
                get_unified_buffer_gpu_for_recursivef_c(d_buffers, self.d_buffers_recursivef)
            } else {
                std::ptr::null_mut()
            };
            let buffer = unified_buffer_gpu as usize;
            let d_buffers_recursivef = self.d_buffers_recursivef as usize;
            std::thread::spawn(move || {
                pre_allocate_final_snark_prover_c(
                    snark_prover as *mut std::ffi::c_void,
                    buffer as *mut std::ffi::c_void,
                    d_buffers_recursivef as *mut std::ffi::c_void,
                );
            })
        };

        let generated = generate_snark_proof(
            snark_prover,
            &self.setup_snark_path,
            recursivef_proof.as_ptr(),
            prealloc_handle,
            self.d_buffers_recursivef,
        );

        if self.snark_prover.is_none() {
            free_final_snark_prover_c(snark_prover);
        }
        let (snark_proof_bytes, snark_publics_bytes) = generated?;

        let publics_info = PublicsInfo::from_folder(&self.proving_key_path)?;
        let publics = &proof[1..1 + proof[0] as usize];
        check_snark_public_digest(&publics_info, publics, verkey, &snark_publics_bytes)?;
        let public_bytes = get_public_bytes_solidity(&publics_info, publics)?;
        let snark_proof = SnarkProof::new(snark_proof_bytes, public_bytes, snark_publics_bytes, protocol.protocol_id());

        timer_stop_and_log_debug!(GENERATING_SNARK_PROOF);

        timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);

        Ok(snark_proof)
    }

    /// The pilfflonk proof of the vadcop proof `proof` (with its publics), of its recursivef proof
    /// `recursivef_proof` against `verkey`, with the key at `pilfflonk_key`, whose prover is
    /// `preloaded` with `preload` (the one `proving` holds). Under the lock of `proving`. The
    /// recursivef proved in the wrapper's device buffers; pilfflonk, its key loaded now without
    /// `preload`, proves the wrap's witness, which it computes from the recursivef proof, in its
    /// arena, which the recursivef is done with.
    fn generate_pilfflonk_proof(
        &self,
        pilfflonk_key: &Path,
        preloaded: Option<&PilfflonkWrapProver>,
        proof: &[u64],
        verkey: &[u64],
        recursivef_proof: &RecursivefProof,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_debug!(GENERATING_SNARK_PROOF);

        let loaded;
        let prover = match preloaded {
            Some(prover) => prover,
            None => {
                // SAFETY: as for the preloaded prover (`new_with_preallocated_buffers`): the buffers
                // outlive this prover, which this proof drops, and the recursivef is done with them.
                loaded = unsafe {
                    pilfflonk_wrap::load_poseidon_wrap(
                        &self.setup_snark_path,
                        pilfflonk_key,
                        self.gpu,
                        self.d_buffers,
                        self.d_buffers_recursivef,
                        false,
                    )
                }?;
                &loaded
            }
        };
        // SAFETY: `recursivef_proof` is the recursivef proof, which nothing else uses.
        let (snark_proof_bytes, snark_publics_bytes) = unsafe { prover.prove(recursivef_proof.as_ptr()) }?;

        let publics_info = PublicsInfo::from_folder(&self.proving_key_path)?;
        let publics = &proof[1..1 + proof[0] as usize];
        check_snark_public_digest(&publics_info, publics, verkey, &snark_publics_bytes)?;
        let public_bytes = get_public_bytes_solidity(&publics_info, publics)?;
        let snark_proof = SnarkProof::new(
            snark_proof_bytes,
            public_bytes,
            snark_publics_bytes,
            SnarkProtocol::Pilfflonk.protocol_id(),
        );

        timer_stop_and_log_debug!(GENERATING_SNARK_PROOF);

        Ok(snark_proof)
    }
}

/// Checks that `snark_publics_bytes`, the public of a final SNARK, is the digest of the vadcop_final
/// publics `publics` and the root `verkey` ([`PublicsInfo::snark_public_digest`]): the digest
/// verifiers recompute must be the one the circuit committed to.
fn check_snark_public_digest(
    publics_info: &PublicsInfo,
    publics: &[u64],
    verkey: &[u64],
    snark_publics_bytes: &[u8],
) -> ProofmanResult<()> {
    let rootc: &[u64; 4] = verkey
        .try_into()
        .map_err(|_| ProofmanError::InvalidConfiguration(format!("verkey has {} limbs, expected 4", verkey.len())))?;
    if publics_info.snark_public_digest(publics, rootc)?.as_slice() != snark_publics_bytes {
        return Err(ProofmanError::InvalidProof(
            "the SNARK's public digest differs from PublicsInfo::snark_public_digest".to_string(),
        ));
    }
    Ok(())
}

/// The recursivef's starkinfo setup-snark puts beside the final circuit of a blake3 key, from which
/// the zkin of a recursivef proof is laid out.
pub const BLAKE3_RECURSIVEF_STARKINFO: &str = "recursivef.starkinfo.json";
/// Its verkey, beside it: the root the final circuit verifies the recursivef with.
pub const BLAKE3_RECURSIVEF_VERKEY: &str = "recursivef.verkey.json";

/// [`SnarkWrapper`] of a blake3 key: its recursivef, a Goldilocks STARK of the proving key, is proved
/// before it (`ProofMan::generate_final_snark_proof`), and this proves the final SNARK of the
/// recursivef's proofs, pilfflonk's.
///
/// **GPU memory.** With proofman's buffers (`d_buffers`), all of the key's device memory is in the aux
/// area of its unified buffer (`pilfflonk_wrap::unified_buffer`): nothing proofman keeps across proofs
/// is there but the streams' cached const pols, which each proof drops, so nothing is reloaded.
/// Preloaded, the key lives across proofs while proofman uses that buffer in between: it keeps a
/// pinned host copy of what it holds there (~4 GB, its SRS and fixed coefficients), which each
/// proof writes back while the host computes the circuit's witness. Not preloaded, each proof loads
/// it. Without buffers, the key's memory is its own.
pub(crate) struct Blake3Wrap {
    setup_snark_path: PathBuf,
    pilfflonk_key: PathBuf,
    starkinfo: serde_json::Value,
    verkey: serde_json::Value,
    publics_info: PublicsInfo,
    /// The vadcop_final's verkey, the root the recursivef verifies its proofs against unless another
    /// is given, which the SNARK's public digest hashes.
    vadcop_final_verkey: Vec<u64>,
    d_buffers: Option<*mut c_void>,
    gpu: bool,
    /// The prover, preloaded; else loaded by each proof.
    prover: Option<PilfflonkWrapProver>,
    proving: Mutex<()>,
}

impl Blake3Wrap {
    fn new(proving_key_snark: &Path, d_buffers: Option<*mut c_void>, preload: bool, gpu: bool) -> ProofmanResult<Self> {
        ensure_gpu_available(gpu)?;
        let final_dir = proving_key_snark.join("final");
        let setup_snark_path = final_dir.join("final");
        let FinalSnarkKey::Pilfflonk(pilfflonk_key) = FinalSnarkKey::find(&setup_snark_path)? else {
            return Err(ProofmanError::InvalidSetup(
                "the final SNARK of a blake3 key is pilfflonk's, and this setup has a zkey".to_string(),
            ));
        };
        fn read<T: serde::de::DeserializeOwned>(path: &Path) -> ProofmanResult<T> {
            std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))
                .map_err(|e| ProofmanError::InvalidSetup(format!("Failed to read {}: {e}", path.display())))
        }
        let starkinfo = read(&final_dir.join(BLAKE3_RECURSIVEF_STARKINFO))?;
        let verkey = read(&final_dir.join(BLAKE3_RECURSIVEF_VERKEY))?;
        let mut wrap = Self {
            setup_snark_path,
            pilfflonk_key,
            starkinfo,
            verkey,
            publics_info: PublicsInfo::from_folder(proving_key_snark)?,
            vadcop_final_verkey: read(&proving_key_snark.join("vadcop_final.verkey.json"))?,
            d_buffers: d_buffers.filter(|_| gpu),
            gpu,
            prover: None,
            proving: Mutex::new(()),
        };
        if preload {
            // The key loads into proofman's aux area while no ProofMan job runs.
            let _shared = wrap.d_buffers.map(|_| PROVING.lock().unwrap_or_else(|e| e.into_inner()));
            let _aux = pilfflonk_wrap::AuxAreaUse::new(wrap.d_buffers);
            wrap.prover = Some(wrap.load(true)?);
        }
        Ok(wrap)
    }

    /// The key's prover; `restorable` if it lives across proofs in proofman's buffer.
    fn load(&self, restorable: bool) -> ProofmanResult<PilfflonkWrapProver> {
        let device = if self.gpu { proofman_pilfflonk::Device::Gpu } else { proofman_pilfflonk::Device::Cpu };
        let buffer = self.d_buffers.map(pilfflonk_wrap::unified_buffer).transpose()?;
        timer_start_info!(INITIALIZING_FINAL_SNARK_PROVER);
        // SAFETY: SnarkWrapper::new_with_preallocated_buffers's contract: proofman's buffer is idle
        // while the key loads and while each proof runs.
        let prover = unsafe {
            PilfflonkWrapProver::load_on(&self.setup_snark_path, &self.pilfflonk_key, device, buffer, restorable)
        }?;
        timer_stop_and_log_info!(INITIALIZING_FINAL_SNARK_PROVER);
        Ok(prover)
    }

    /// The SNARK proof of `recursivef_proof`, whose publics are `[rootC(4) | the vadcop_final's]`,
    /// once its public is checked, as [`PoseidonWrap`]'s, to be the digest of the vadcop_final's
    /// publics and the root the recursivef verified the vadcop_final proof against
    /// ([`PublicsInfo::snark_public_digest`]): `verkey_override`, or the vadcop_final's verkey.
    fn generate_snark_proof(
        &self,
        recursivef_proof: &VadcopFinalProof,
        verkey_override: Option<&[u64]>,
    ) -> ProofmanResult<SnarkProof> {
        timer_start_info!(GENERATING_WRAPPER_SNARK_PROOF);
        let _proving = self.proving.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let n_publics = self.starkinfo["nPublics"].as_u64().unwrap_or(0) as usize;
        if recursivef_proof.public_values.len() != n_publics {
            return Err(ProofmanError::InvalidProof(format!(
                "a blake3 key's SNARK wraps its recursivef proof, of {n_publics} publics, and this proof has {}: \
                 a vadcop_final proof goes through ProofMan::generate_final_snark_proof, which proves its recursivef",
                recursivef_proof.public_values.len()
            )));
        }
        let zkin = pilfflonk_wrap_witness::gl_proof_zkin(
            &self.starkinfo,
            &recursivef_proof.proof,
            &recursivef_proof.public_values,
        )
        .map_err(|e| ProofmanError::InvalidProof(e.to_string()))?;
        // The caller, ProofMan::generate_final_snark_proof, holds the process-wide PROVING throughout.
        let _aux = pilfflonk_wrap::AuxAreaUse::new(self.d_buffers);
        let loaded;
        let prover = match &self.prover {
            Some(prover) => prover,
            None => {
                loaded = self.load(false)?;
                &loaded
            }
        };
        let (snark_proof_bytes, snark_publics_bytes) = prover.prove_zkin(&zkin)?;
        let vadcop_publics = recursivef_proof.public_values.get(4..).unwrap_or_default();
        let verkey = verkey_override.unwrap_or(&self.vadcop_final_verkey);
        check_snark_public_digest(&self.publics_info, vadcop_publics, verkey, &snark_publics_bytes)?;
        let public_bytes = get_public_bytes_solidity(&self.publics_info, vadcop_publics)?;
        timer_stop_and_log_info!(GENERATING_WRAPPER_SNARK_PROOF);
        Ok(SnarkProof::new(
            snark_proof_bytes,
            public_bytes,
            snark_publics_bytes,
            SnarkProtocol::Pilfflonk.protocol_id(),
        ))
    }
}

// SAFETY: as PoseidonWrap's: the key's C++ handles and proofman's device buffers are not tied to the
// thread that made them, and `proving` keeps the proofs one at a time.
unsafe impl Send for Blake3Wrap {}
unsafe impl Sync for Blake3Wrap {}

/// The final SNARK's wrapper of the key in `proving_key_path`, whose proofs are
/// `ProofMan::generate_final_snark_proof`'s, of either hash: a poseidon key's ([`PoseidonWrap`])
/// proves the recursivef and then the final SNARK of the vadcop_final proof, a blake3 key's
/// ([`Blake3Wrap`]) the final SNARK of the recursivef proof proofman proves before it. One proof at
/// a time.
pub struct SnarkWrapper<F: PrimeField64> {
    inner: Wrap<F>,
}

// One per process; boxing buys nothing.
#[allow(clippy::large_enum_variant)]
enum Wrap<F: PrimeField64> {
    Poseidon(PoseidonWrap<F>),
    Blake3(Blake3Wrap),
}

impl<F: PrimeField64> SnarkWrapper<F> {
    pub fn new(proving_key_path: &Path, verbose_mode: VerboseMode, preload: bool, gpu: bool) -> ProofmanResult<Self> {
        Self::new_with_preallocated_buffers(proving_key_path, verbose_mode, None, None, None, preload, gpu)
    }

    /// With proofman's buffers (`d_buffers`, `aux_trace`, `reload_fixed_pols_gpu`, of
    /// `ProofMan::get_preallocated_buffers`), which must outlive the wrapper, idle while it loads and
    /// while each of its proofs runs; and with `preload`, the final SNARK's prover loaded now.
    pub fn new_with_preallocated_buffers(
        proving_key_path: &Path,
        verbose_mode: VerboseMode,
        aux_trace: Option<Arc<Vec<F>>>,
        d_buffers: Option<*mut c_void>,
        reload_fixed_pols_gpu: Option<Arc<AtomicBool>>,
        preload: bool,
        gpu: bool,
    ) -> ProofmanResult<Self> {
        let inner = if Self::is_blake3_key(proving_key_path) {
            initialize_logger(verbose_mode, None);
            Wrap::Blake3(Blake3Wrap::new(proving_key_path, d_buffers, preload, gpu)?)
        } else {
            Wrap::Poseidon(PoseidonWrap::new_with_preallocated_buffers(
                proving_key_path,
                verbose_mode,
                aux_trace,
                d_buffers,
                reload_fixed_pols_gpu,
                preload,
                gpu,
            )?)
        };
        Ok(Self { inner })
    }

    /// Whether `proving_key_snark` is a blake3 key's: setup-snark puts its recursivef's starkinfo,
    /// a Goldilocks STARK's, in it.
    pub fn is_blake3_key(proving_key_snark: &Path) -> bool {
        let path = proving_key_snark.join("final").join(BLAKE3_RECURSIVEF_STARKINFO);
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            .is_some_and(|si| si["starkStruct"]["verificationHashType"].as_str() == Some("GL"))
    }

    /// Of a blake3 key, the starkinfo and the verkey of the recursivef its SNARK's circuit verifies.
    pub(crate) fn blake3_recursivef(&self) -> Option<(&serde_json::Value, &serde_json::Value)> {
        match &self.inner {
            Wrap::Blake3(wrap) => Some((&wrap.starkinfo, &wrap.verkey)),
            Wrap::Poseidon(_) => None,
        }
    }

    /// The final SNARK proof of `proof`: of a poseidon key the vadcop_final proof (checked against
    /// `verkey_override` if given, the vadcop_final's verkey otherwise), of a blake3 key the recursivef
    /// proof (`ProofMan::generate_recursivef_proof`) of the vadcop_final proof checked against that
    /// same verkey. Either SNARK's public is checked to be the digest of the vadcop_final's publics and
    /// that verkey.
    pub(crate) fn generate_final_snark_proof(
        &self,
        proof: &VadcopFinalProof,
        verkey_override: Option<&[u64]>,
    ) -> ProofmanResult<SnarkProof> {
        match &self.inner {
            Wrap::Poseidon(wrap) => wrap.generate_final_snark_proof(proof, verkey_override),
            Wrap::Blake3(wrap) => wrap.generate_snark_proof(proof, verkey_override),
        }
    }
}

/// The `publicValues` bytes a Solidity verifier takes. See
/// [`PublicsInfo::solidity_public_bytes`].
pub fn get_public_bytes_solidity(publics_info: &PublicsInfo, vadcop_public_inputs: &[u64]) -> ProofmanResult<Vec<u8>> {
    publics_info.solidity_public_bytes(vadcop_public_inputs)
}

pub fn check_setup_snark<F: PrimeField64>(
    proving_key_snark_path: &Path,
    verbose_mode: VerboseMode,
    gpu: bool,
) -> ProofmanResult<()> {
    initialize_logger(verbose_mode, None);

    ensure_gpu_available(gpu)?;

    let setup_recursivef_path =
        PathBuf::from(format!("{}/{}/{}", proving_key_snark_path.display(), "recursivef", "recursivef"));

    let setup_recursivef: Setup<F> = Setup::<F>::new(
        &setup_recursivef_path,
        0,
        0,
        &GlobalInfoAir::new("RecursiveF".to_string()),
        &ProofType::RecursiveF,
        false,
        gpu,
        None,
        &std::collections::HashMap::new(),
        false,
    )?;

    calculate_fixed_tree_snark(&setup_recursivef);

    Ok(())
}

pub fn generate_and_verify_recursivef<F: PrimeField64>(
    proving_key_path: &Path,
    vadcop_proof: &VadcopFinalProof,
    verbose_mode: VerboseMode,
    gpu: bool,
) -> ProofmanResult<bool> {
    initialize_logger(verbose_mode, None);

    ensure_gpu_available(gpu)?;

    if !proofman_common::hash_family::has_bn128_recursivef(&vadcop_proof.hash) {
        return Err(ProofmanError::InvalidConfiguration(format!(
            "{} proofs have no BN128 recursivef (setup-snark's): a blake3 key's recursivef is a STARK of its key",
            vadcop_proof.hash
        )));
    }
    if vadcop_proof.compressed {
        return Err(ProofmanError::InvalidConfiguration(
            "Compressed vadcop proofs are not supported for snark proof generation".to_string(),
        ));
    }
    let proof = vadcop_proof.proof_with_publics();

    timer_start_info!(LOADING_RECURSIVE_F_SETUP);

    let setup_recursivef_path =
        PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "recursivef", "recursivef"));

    let setup_recursivef = Setup::<F>::new(
        &setup_recursivef_path,
        0,
        0,
        &GlobalInfoAir::new("RecursiveF".to_string()),
        &ProofType::RecursiveF,
        false,
        gpu,
        None,
        &std::collections::HashMap::new(),
        false,
    )?;

    ensure_gpu_available(gpu)?;
    if gpu {
        init_gpu_setup_c(setup_recursivef.stark_info.stark_struct.merkle_tree_arity);
    }

    check_const_tree(&setup_recursivef, &None)?;

    let mut recursivef_const_pols_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_pols_size];
    load_const_pols_recursivef(&setup_recursivef, &mut recursivef_const_pols_buf);
    let recursivef_const_pols: Arc<Vec<F>> = Arc::new(recursivef_const_pols_buf);
    let mut recursivef_const_tree_buf: Vec<F> = vec![F::ZERO; setup_recursivef.const_tree_size];
    load_const_pols_tree(&setup_recursivef, &mut recursivef_const_tree_buf);
    let recursivef_const_tree: Arc<Vec<F>> = Arc::new(recursivef_const_tree_buf);

    let aux_trace =
        if gpu { Arc::new(Vec::new()) } else { Arc::new(vec![F::ZERO; setup_recursivef.prover_buffer_size as usize]) };

    timer_stop_and_log_info!(LOADING_RECURSIVE_F_SETUP);

    let vadcop_final_verkey_path = PathBuf::from(format!("{}/vadcop_final.verkey.json", proving_key_path.display()));

    let mut file = File::open(&vadcop_final_verkey_path).expect("Unable to open file");
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).expect("Unable to read file");
    let vadcop_final_verkey: Vec<u64> = serde_json::from_str(&json_str).expect("Unable to parse JSON");

    let p_setup: *mut c_void = (&setup_recursivef.p_setup).into();

    let verkey_path = setup_recursivef.verkey_file.clone();
    let mut contents = String::new();
    let mut file = File::open(verkey_path).unwrap();
    let _ = file.read_to_string(&mut contents).map_err(|err| format!("Failed to read verkey path file: {err}"));

    let verkey_str: String = serde_json::from_str(&contents)
        .map_err(|err| ProofmanError::InvalidSetup(format!("Failed to parse verkey as string: {}", err)))?;

    // Freed when this returns, whatever its outcome.
    let d_buffers_recursivef = RecursivefDeviceBuffers(gen_device_buffers_recursivef_c(
        p_setup as *mut u8,
        setup_recursivef.prover_buffer_size,
        std::ptr::null_mut(),
        &verkey_str,
    ) as *mut c_void);

    let trace_size = setup_recursivef.stark_info.map_sections_n["cm1"]
        * (1 << setup_recursivef.stark_info.stark_struct.n_bits)
        + setup_recursivef.stark_info.n_publics;

    let memory_handler_recursive_witness = Arc::new(MemoryHandlerRecursive::new(1, trace_size as usize));

    timer_start_info!(GENERATING_RECURSIVE_F_PROOF);
    let recursivef_proof = generate_recursivef_proof(
        &setup_recursivef,
        &memory_handler_recursive_witness,
        &proof,
        &aux_trace,
        &recursivef_const_pols,
        &recursivef_const_tree,
        &vadcop_final_verkey,
        setup_recursivef.prover_buffer_size as usize * std::mem::size_of::<F>(),
        d_buffers_recursivef.as_ptr(),
    )?;
    timer_stop_and_log_info!(GENERATING_RECURSIVE_F_PROOF);

    timer_start_info!(VERIFY_RECURSIVE_F_PROOF);
    let mut publics: Vec<F> = vadcop_final_verkey[0..4].iter().map(|&x| F::from_u64(x)).collect();
    publics.extend(proof[1..1 + proof[0] as usize].iter().map(|&x| F::from_u64(x)));

    let is_valid = verify_proof_bn128(recursivef_proof.as_ptr(), &setup_recursivef, Some(publics));
    timer_stop_and_log_info!(VERIFY_RECURSIVE_F_PROOF);

    let setup_snark_path = PathBuf::from(format!("{}/{}/{}", proving_key_path.display(), "final", "final"));
    if setup_snark_path.parent().is_some_and(|p| p.exists()) {
        match FinalSnarkKey::find(&setup_snark_path)? {
            FinalSnarkKey::Zkey(_) => {
                generate_witness_final_snark(recursivef_proof.as_ptr(), &setup_snark_path)?;
            }
            // SAFETY: `recursivef_proof` is the recursivef proof, which nothing else uses.
            FinalSnarkKey::Pilfflonk(pilfflonk_key) => unsafe {
                pilfflonk_wrap::check_wrap_witness(&setup_snark_path, &pilfflonk_key, recursivef_proof.as_ptr())
            }?,
        }
    }

    Ok(is_valid)
}

/// Verify a SNARK proof against a trusted `vkey`, checking snark validity only — the
/// public input the verifier is given is the digest the proof carries. That digest is the
/// committed statement only if the caller already trusts it (e.g. recomputed it from the
/// publics and a pinned rootC). On its own this does NOT bind the proof to any statement,
/// so prefer [`verify_snark_proof_with_expected`] with a pinned digest.
///
/// The verifier is `snarkjs <protocol> verify` with snarkjs's `final.verkey.json` for PLONK and
/// FFLONK, and pilfflonk's JS verifier with `pilfflonk.vkey.json`
/// (`provingKeySnark/final/provingKey/<name>/pilfflonk/`) for pilfflonk.
pub fn verify_snark_proof(snark_proof: &SnarkProof, vkey_path: &Path) -> ProofmanResult<()> {
    verify_snark_proof_with_expected(snark_proof, vkey_path, None)
}

/// As [`verify_snark_proof`], but when `expected_public_snark_bytes` is `Some`, require the
/// proof's committed public digest to equal it before verifying. That pins the statement:
/// the caller supplies the digest it recomputed from the publics and the trusted rootC, so
/// a proof carrying an attacker-chosen public input is rejected. With `None`, a warning is
/// emitted that the result attests snark self-consistency only.
pub fn verify_snark_proof_with_expected(
    snark_proof: &SnarkProof,
    vkey_path: &Path,
    expected_public_snark_bytes: Option<&[u8]>,
) -> ProofmanResult<()> {
    run_snarkjs_verify(snark_proof, vkey_path, expected_public_snark_bytes, &TempJsonFiles::new()?)
}

/// As [`verify_snark_proof_with_expected`], with the verification key given as its JSON
/// rather than a file.
pub fn verify_snark_proof_with_vkey(
    snark_proof: &SnarkProof,
    vkey_json: &str,
    expected_public_snark_bytes: Option<&[u8]>,
) -> ProofmanResult<()> {
    let temp = TempJsonFiles::new()?;
    let vkey_path = temp.write("snark_vkey", vkey_json)?;
    run_snarkjs_verify(snark_proof, &vkey_path, expected_public_snark_bytes, &temp)
}

/// JSON files snarkjs reads, in a private directory (random name, owner-only) that is
/// removed with them when dropped, on every return path.
struct TempJsonFiles(tempfile::TempDir);

impl TempJsonFiles {
    fn new() -> ProofmanResult<Self> {
        Ok(Self(tempfile::Builder::new().prefix("snarkjs_").tempdir()?))
    }

    fn write(&self, name: &str, contents: &str) -> ProofmanResult<PathBuf> {
        let path = self.path(name);
        std::fs::write(&path, contents)?;
        Ok(path)
    }

    /// The path of the file `name`, which a verifier's caller writes itself.
    fn path(&self, name: &str) -> PathBuf {
        self.0.path().join(format!("{name}.json"))
    }
}

/// The verification of [`verify_snark_proof_with_expected`]: snarkjs's for PLONK and FFLONK, and
/// pilfflonk's JS verifier for pilfflonk ([`verify_pilfflonk_snark_proof`]), with their JSON files in
/// `temp`.
fn run_snarkjs_verify(
    snark_proof: &SnarkProof,
    vkey_path: &Path,
    expected_public_snark_bytes: Option<&[u8]>,
    temp: &TempJsonFiles,
) -> ProofmanResult<()> {
    match expected_public_snark_bytes {
        Some(expected) if expected != snark_proof.public_snark_bytes.as_slice() => {
            return Err(ProofmanError::InvalidProof(
                "SNARK public digest does not match the expected (pinned) statement".to_string(),
            ));
        }
        Some(_) => {}
        None => {
            tracing::warn!(
                "verify_snark_proof: no expected public digest pinned — this attests snark \
                 validity against the proof's own public input, not a statement. A relying \
                 party must pin the digest recomputed from the publics and the trusted rootC."
            );
        }
    }

    let protocol = SnarkProtocol::from_protocol_id(snark_proof.protocol_id)?;
    if let SnarkProtocol::Pilfflonk = protocol {
        return verify_pilfflonk_snark_proof(snark_proof, vkey_path, temp);
    }

    let (proof_json, publics_json) = snark_proof
        .convert_to_json()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to convert SNARK proof to JSON: {}", e)))?;
    let to_json = |value| {
        serde_json::to_string(value)
            .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to serialize SNARK JSON: {}", e)))
    };
    let proof_path = temp.write("snark_proof", &to_json(&proof_json)?)?;
    let publics_path = temp.write("snark_publics", &to_json(&publics_json)?)?;

    let output = Command::new("snarkjs")
        .arg(protocol.protocol_name())
        .arg("verify")
        .arg(vkey_path)
        .arg(&publics_path)
        .arg(&proof_path)
        .output()
        .map_err(|e| ProofmanError::InvalidConfiguration(format!("Failed to execute snarkjs: {}", e)))?;

    // snarkjs reports everything on stdout and exits 1 both when it rejects a proof and when it
    // cannot run (unreadable vkey, missing file, ...), so only its messages tell them apart.
    let stdout = String::from_utf8_lossy(&output.stdout);
    match snarkjs_verdict(&protocol, output.status.success(), &stdout) {
        SnarkjsVerdict::Verified => {
            tracing::info!("    {}", "\u{2713} SNARK proof was verified".bright_green().bold());
            Ok(())
        }
        SnarkjsVerdict::Rejected => {
            tracing::info!("··· {}", "\u{2717} SNARK proof was not verified".bright_red().bold());
            Err(ProofmanError::InvalidProof(format!("SNARK proof was not verified: {}", stdout.trim())))
        }
        SnarkjsVerdict::Failed => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(ProofmanError::InvalidConfiguration(format!(
                "snarkjs could not verify the proof ({}): {} {}",
                output.status,
                stdout.trim(),
                stderr.trim()
            )))
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SnarkjsVerdict {
    Verified,
    /// The checks ran and the proof failed them.
    Rejected,
    /// Nothing was decided: snarkjs could not read its inputs, or the key does not fit.
    Failed,
}

/// What snarkjs 0.7's `plonk verify` / `fflonk verify` concluded from its exit status and stdout.
fn snarkjs_verdict(protocol: &SnarkProtocol, success: bool, stdout: &str) -> SnarkjsVerdict {
    // A key with another public count is misconfiguration, not a false proof.
    const REJECTED: [&str; 5] = [
        "Invalid Proof",
        "Proof commitments are not valid",
        "Proof evaluations are not valid",
        "Public inputs are not valid",
        "Invalid number of public inputs",
    ];
    if success && protocol.verified_message().is_some_and(|message| stdout.contains(message)) {
        SnarkjsVerdict::Verified
    } else if !success && REJECTED.iter().any(|m| stdout.contains(m)) {
        SnarkjsVerdict::Rejected
    } else {
        SnarkjsVerdict::Failed
    }
}

/// [`verify_snark_proof`] of a pilfflonk proof: its JSON views, named by the vkey at `vkey_path`
/// (`pilfflonk_wrap::json_views`), go to pilfflonk's JS verifier, in `temp`. As snarkjs's verdicts
/// are told apart: a proof the verifier rejects, and bytes that are not those of a proof of the vkey
/// or not as many publics as it has, are an `InvalidProof`; a vkey that cannot be read and a verifier
/// that cannot run, an `InvalidConfiguration`.
fn verify_pilfflonk_snark_proof(
    snark_proof: &SnarkProof,
    vkey_path: &Path,
    temp: &TempJsonFiles,
) -> ProofmanResult<()> {
    let vkey = pilfflonk_wrap::read_vkey(vkey_path)?;
    let rejected = match pilfflonk_wrap::json_views(&snark_proof.proof_bytes, &snark_proof.public_snark_bytes, &vkey) {
        Ok((proof_json, publics)) => {
            let (proof_path, publics_path) = (temp.path("snark_proof"), temp.path("snark_publics"));
            if pilfflonk_wrap::verify_json(vkey_path, &proof_json, &publics, &proof_path, &publics_path)? {
                tracing::info!("    {}", "\u{2713} SNARK proof was verified".bright_green().bold());
                return Ok(());
            }
            "the pilfflonk verifier rejects it".to_string()
        }
        Err(ProofmanError::InvalidProof(malformed)) => malformed,
        Err(failure) => return Err(failure),
    };
    tracing::info!("··· {}", "\u{2717} SNARK proof was not verified".bright_red().bold());
    Err(ProofmanError::InvalidProof(format!("SNARK proof was not verified: {rejected}")))
}

/// The recursivef's host prover buffer, of `words` words on the CPU: `given`, the caller's
/// (proofman's `aux_trace`, sized for its own proofs), if it holds that many, else one of its own.
/// On the GPU the recursivef proves in its device buffers, and `given` (or none) is kept as it is.
fn recursivef_prover_buffer<F: PrimeField64>(given: Option<Arc<Vec<F>>>, gpu: bool, words: usize) -> Arc<Vec<F>> {
    match given {
        Some(buffer) if gpu || buffer.len() >= words => buffer,
        None if gpu => Arc::new(Vec::new()),
        Some(short) => {
            tracing::info!(
                "the recursivef's prover buffer: {words} words of its own, the given one holds {}",
                short.len()
            );
            Arc::new(vec![F::ZERO; words])
        }
        None => Arc::new(vec![F::ZERO; words]),
    }
}

/// The recursivef's device buffers (`gen_device_buffers_recursivef_c`), freed when dropped unless
/// handed over with [`into_raw`](Self::into_raw), so that an error does not leak them. The CPU
/// backend has none: the pointer is null and nothing is freed.
struct RecursivefDeviceBuffers(*mut c_void);

impl RecursivefDeviceBuffers {
    fn as_ptr(&self) -> *mut c_void {
        self.0
    }

    /// The buffers, which the caller frees from now on.
    fn into_raw(self) -> *mut c_void {
        let buffers = self.0;
        std::mem::forget(self);
        buffers
    }
}

impl Drop for RecursivefDeviceBuffers {
    fn drop(&mut self) {
        if !self.0.is_null() {
            free_device_buffers_recursivef_c(self.0);
        }
    }
}

/// Sets proofman's `reload_fixed_pols_gpu`, if any, when it goes: on any return of a wrap's proof,
/// an error or a panic included.
struct ReloadFixedPolsOnDrop<'a>(Option<&'a AtomicBool>);

impl Drop for ReloadFixedPolsOnDrop<'_> {
    fn drop(&mut self) {
        if let Some(flag) = self.0 {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

// SAFETY: every field is `Send` but the raw pointers and pilfflonk's prover, and none of those is tied to the thread
// that made it: `snark_prover` (rapidsnark's prover of `final.zkey`) and `d_buffers_recursivef`
// (the recursivef's device buffers) are the wrapper's, which its `Drop` frees, and `d_buffers`
// (the caller's unified buffer) it never frees. They are C++ objects of the process and CUDA
// memory of its contexts, which each proof already hands to threads it spawns: the recursivef's
// loader of its fixed columns, and the GPU pre-allocation of rapidsnark's prover.
// pilfflonk's prover, in `proving`, owns its key, whose `PilFflonkProverCtx` is a C++ object any
// thread may use (pilfflonk_api.hpp), with the status of each call kept per thread and read on the
// calling thread right after it, and the final circuit's witness library, a `WrapWitness`, which
// is `Send`.
unsafe impl<F: PrimeField64> Send for PoseidonWrap<F> {}
// SAFETY: a shared wrapper proves one proof at a time: `generate_final_snark_proof` holds the lock
// of `proving` throughout, and only a proof hands the C++ side what it writes: rapidsnark's
// prover, the recursivef's and the caller's device buffers (the raw pointers), and the
// recursivef's prover buffer `aux_trace`, through a pointer of the shared `Vec`. The threads a
// proof spawns are joined before it returns, on its errors too: its own (the recursivef's loader
// of its fixed columns, the GPU pre-allocation of rapidsnark's prover) and those rapidsnark's GPU
// prover starts, which its pre-allocation and its proof join before they return. Every other
// field is `Sync`. The raw pointers are public, and handing one to the C++ side outside a proof is
// the caller's to order with the wrapper's proofs, as for any call of `proofman_starks_lib_c`
// with them.
// Only a proof, under that lock, uses pilfflonk's prover in `proving` and the final circuit's
// witness calculator, and the threads pilfflonk's proof starts are scoped.
unsafe impl<F: PrimeField64> Sync for PoseidonWrap<F> {}

#[cfg(test)]
mod snarkjs_verdict_tests {
    use super::*;

    #[test]
    fn only_snarkjs_rejections_count_as_a_rejected_proof() {
        let plonk = &SnarkProtocol::Plonk;
        assert_eq!(snarkjs_verdict(plonk, true, "[INFO]  snarkJS: OK!\n"), SnarkjsVerdict::Verified);
        assert_eq!(snarkjs_verdict(plonk, false, "[WARN]  snarkJS: Invalid Proof\n"), SnarkjsVerdict::Rejected);
        assert_eq!(
            snarkjs_verdict(plonk, false, "[ERROR] snarkJS: Proof commitments are not valid.\n"),
            SnarkjsVerdict::Rejected
        );
        // Exit 0 without OK!, a key that does not fit, an unreadable file: nothing decided.
        assert_eq!(snarkjs_verdict(plonk, true, ""), SnarkjsVerdict::Failed);
        assert_eq!(
            snarkjs_verdict(plonk, false, "[ERROR] snarkJS: Number of public signals does not match with vk\n"),
            SnarkjsVerdict::Failed
        );
        assert_eq!(
            snarkjs_verdict(plonk, false, "[ERROR] snarkJS: Error: ENOENT: no such file\n"),
            SnarkjsVerdict::Failed
        );
        assert_eq!(snarkjs_verdict(plonk, false, ""), SnarkjsVerdict::Failed);
    }

    /// `snarkjs fflonk verify` prints its own success message, not PLONK's `OK!`, and each is
    /// only the verdict of its protocol. snarkjs verifies no pilfflonk proof: no message of it is one.
    #[test]
    fn each_protocol_is_verified_by_its_own_message() {
        let fflonk = &SnarkProtocol::Fflonk;
        let fflonk_ok = "[INFO]  snarkJS: PROOF VERIFIED SUCCESSFULLY\n[INFO]  snarkJS: FFLONK VERIFIER FINISHED\n";
        assert_eq!(snarkjs_verdict(fflonk, true, fflonk_ok), SnarkjsVerdict::Verified);
        assert_eq!(snarkjs_verdict(fflonk, false, "[WARN]  snarkJS: Invalid Proof\n"), SnarkjsVerdict::Rejected);
        assert_eq!(
            snarkjs_verdict(fflonk, false, "[ERROR] snarkJS: Proof evaluations are not valid.\n"),
            SnarkjsVerdict::Rejected
        );
        assert_eq!(snarkjs_verdict(fflonk, true, "[INFO]  snarkJS: OK!\n"), SnarkjsVerdict::Failed);
        assert_eq!(snarkjs_verdict(&SnarkProtocol::Plonk, true, fflonk_ok), SnarkjsVerdict::Failed);
        for stdout in ["[INFO]  snarkJS: OK!\n", fflonk_ok] {
            assert_eq!(snarkjs_verdict(&SnarkProtocol::Pilfflonk, true, stdout), SnarkjsVerdict::Failed);
        }
    }

    /// On the CPU the recursivef proves in the given buffer only if it holds the recursivef's words,
    /// and else in one of its own of that many; on the GPU the given buffer, or none, is kept.
    #[test]
    fn the_recursivef_proves_in_the_given_buffer_only_if_it_holds_its_words() {
        use proofman_fields::{Field, Goldilocks};
        // (words of the given buffer, gpu) and (words of the buffer, whether it is the given one).
        for (words_given, gpu, expected) in [
            (Some(100), false, (100, true)),
            (Some(101), false, (101, true)),
            (Some(99), false, (100, false)),
            (Some(0), false, (100, false)),
            (None, false, (100, false)),
            (Some(99), true, (99, true)),
            (None, true, (0, false)),
        ] {
            let given = words_given.map(|words| Arc::new(vec![Goldilocks::ZERO; words]));
            let buffer = recursivef_prover_buffer(given.clone(), gpu, 100);
            let is_given = given.as_ref().is_some_and(|given| Arc::ptr_eq(&buffer, given));
            assert_eq!((buffer.len(), is_given), expected, "given {words_given:?}, gpu {gpu}");
        }
    }

    /// The reload flag is set as a wrap's proof ends, on its errors and panics too, and without one
    /// (no unified buffer) nothing is.
    #[test]
    fn the_reload_flag_is_set_whatever_the_proof_ends_with() {
        let proof = |flag: Option<&AtomicBool>, fails: bool| -> Result<(), ()> {
            let _reload = ReloadFixedPolsOnDrop(flag);
            assert!(flag.is_none_or(|f| !f.load(Ordering::SeqCst)), "set before the proof ends");
            if fails {
                return Err(());
            }
            Ok(())
        };
        for fails in [false, true] {
            let flag = AtomicBool::new(false);
            assert_eq!(proof(Some(&flag), fails).is_err(), fails);
            assert!(flag.load(Ordering::SeqCst), "fails: {fails}");
            assert_eq!(proof(None, fails).is_err(), fails);
        }
        let flag = AtomicBool::new(false);
        let panicked = std::panic::catch_unwind(|| {
            let _reload = ReloadFixedPolsOnDrop(Some(&flag));
            panic!("a proof that panics");
        });
        assert!(panicked.is_err() && flag.load(Ordering::SeqCst));
    }

    #[test]
    fn protocol_ids_map_to_protocols() {
        assert!(matches!(SnarkProtocol::from_protocol_id(2), Ok(SnarkProtocol::Plonk)));
        assert!(matches!(SnarkProtocol::from_protocol_id(10), Ok(SnarkProtocol::Fflonk)));
        assert!(matches!(SnarkProtocol::from_protocol_id(0x7066), Ok(SnarkProtocol::Pilfflonk)));
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk, SnarkProtocol::Pilfflonk] {
            let id = protocol.protocol_id();
            assert_eq!(
                SnarkProtocol::from_protocol_id(id).map(|p| p.protocol_name()).ok(),
                Some(protocol.protocol_name())
            );
        }
        assert_eq!(SnarkProtocol::Plonk.protocol_name(), "plonk");
        assert_eq!(SnarkProtocol::Fflonk.protocol_name(), "fflonk");
        assert_eq!(SnarkProtocol::Pilfflonk.protocol_name(), "pilfflonk");
        for id in [0, 1, 3, 9, 11, 0x7065, 0x7067, u64::MAX] {
            assert!(SnarkProtocol::from_protocol_id(id).is_err(), "protocol id {id}");
        }
    }

    /// The globalInfo of a pilfflonk key of one AIR, as setup-pilfflonk writes the wrap's.
    const PILFFLONK_GLOBAL_INFO: &str = r#"{
 "name": "final",
 "airs": [[{"name": "Wrap", "num_rows": 16}]],
 "air_groups": ["Wrap"],
 "aggTypes": [[]],
 "backend": "pilfflonk",
 "formatVersion": 1,
 "field": "bn128",
 "modulus": "21888242871839275222246405745257275088548364400416034343698204186575808495617",
 "transcript": "keccak256",
 "setupParams": {"maxConstraintDegree": 9, "extraMuls": 2, "maxQDegree": 0, "packing": true},
 "nPublics": 1,
 "numChallenges": [0, 2],
 "numProofValues": [],
 "proofValuesMap": [],
 "publicsMap": [{"name": "publics", "stage": 1, "lengths": [0]}]
}"#;

    /// A `provingKeySnark/final/` in `dir` with a `final.zkey` if `zkey`, and a `provingKey/` whose
    /// globalInfo is `global_info` if there is one; and the stem of its files, `final/final`.
    fn final_dir(dir: &Path, zkey: bool, global_info: Option<&str>) -> PathBuf {
        let final_dir = dir.join("final");
        std::fs::create_dir_all(&final_dir).unwrap();
        if zkey {
            std::fs::write(final_dir.join("final.zkey"), b"zkey").unwrap();
        }
        if let Some(global_info) = global_info {
            std::fs::create_dir_all(final_dir.join("provingKey")).unwrap();
            std::fs::write(final_dir.join("provingKey/pilout.globalInfo.json"), global_info).unwrap();
        }
        final_dir.join("final")
    }

    fn setup_error(result: Result<FinalSnarkKey, ProofmanError>) -> String {
        match result {
            Err(ProofmanError::InvalidSetup(message)) => message,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn the_final_snark_key_is_final_zkey_or_a_pilfflonk_proving_key() {
        let dir = tempfile::tempdir().unwrap();
        let stem = final_dir(dir.path(), true, None);
        assert_eq!(FinalSnarkKey::find(&stem).unwrap(), FinalSnarkKey::Zkey(dir.path().join("final/final.zkey")));

        let dir = tempfile::tempdir().unwrap();
        let stem = final_dir(dir.path(), false, Some(PILFFLONK_GLOBAL_INFO));
        assert_eq!(FinalSnarkKey::find(&stem).unwrap(), FinalSnarkKey::Pilfflonk(dir.path().join("final/provingKey")));
    }

    #[test]
    fn a_final_dir_with_both_keys_or_none_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let message = setup_error(FinalSnarkKey::find(&final_dir(dir.path(), true, Some(PILFFLONK_GLOBAL_INFO))));
        assert!(message.contains("two final SNARK keys") && message.contains("final.zkey"), "{message}");
        assert!(message.contains("provingKey"), "{message}");
        // What fixes the directory: setup-snark of the protocol to prove with removes the other key.
        assert!(message.contains("setup-snark writes the key of one protocol and removes the other's"), "{message}");
        assert!(message.contains("running it again with the --final-snark to prove with"), "{message}");

        let dir = tempfile::tempdir().unwrap();
        let message = setup_error(FinalSnarkKey::find(&final_dir(dir.path(), false, None)));
        assert!(message.contains("no final SNARK key") && message.contains("final.zkey"), "{message}");
    }

    /// A STARK's globalInfo, another backend's, one that is not JSON and none at all.
    #[test]
    fn a_proving_key_that_is_not_pilfflonks_is_refused() {
        let stark = r#"{"name": "t", "airs": [[{"name": "A", "num_rows": 16}]], "air_groups": ["G"],
            "aggTypes": [[]], "curve": "None", "latticeSize": 368, "transcriptArity": 4, "aggregationArity": 3,
            "hasCompressedFinal": true, "nPublics": 0, "numChallenges": [0], "numProofValues": [0],
            "proofValuesMap": [], "publicsMap": [], "hash": "Poseidon2"}"#;
        let other_backend = PILFFLONK_GLOBAL_INFO.replace("\"backend\": \"pilfflonk\"", "\"backend\": \"stark\"");
        for (name, global_info) in [("stark", stark), ("backend", &other_backend), ("text", "not JSON")] {
            let dir = tempfile::tempdir().unwrap();
            let message = setup_error(FinalSnarkKey::find(&final_dir(dir.path(), false, Some(global_info))));
            assert!(message.contains("is not a pilfflonk key"), "{name}: {message}");
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("final/provingKey")).unwrap();
        let message = setup_error(FinalSnarkKey::find(&dir.path().join("final/final")));
        assert!(message.contains("is not a pilfflonk key"), "{message}");
    }

    /// The PLONK and FFLONK zkeys of `x * x = y` in `tests/data/` report, once loaded, the protocol
    /// each names, and each, in a `provingKeySnark/final/` as setup-snark writes it, is the final
    /// SNARK's key there.
    ///
    /// They were made once with rapidsnark's setup, as setup-snark runs it:
    /// `generate_plonk_zkey_c(r1cs, ptau, "square_plonk.zkey")` and
    /// `generate_fflonk_zkey_c(r1cs, ptau, "square_fflonk.zkey")`, where the r1cs is
    /// `circom square.circom --r1cs` (circom 2.2.3) of
    /// `template Square() { signal input x; signal output y; y <== x * x; } component main = Square();`
    /// and the ptau is the snarkjs README's `powersOfTau28_hez_final_08.ptau`, whose blake2b is
    /// d6a8fb3a04feb600096c3b791f936a578c4e664d262e4aa24beed1b7a9a96aa5
    /// eb72864d628db247e9293384b74b36ffb52ca8d148d6e1b8b51e279fdf57b583.
    #[test]
    fn the_protocol_is_the_one_the_zkey_names() {
        let dir = tempfile::tempdir().unwrap();
        for expected in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            let zkey = format!("{}/tests/data/square_{}.zkey", env!("CARGO_MANIFEST_DIR"), expected.protocol_name());
            // A zkey that cannot be read aborts the C++ loader: refuse a missing one first.
            assert!(Path::new(&zkey).is_file(), "no fixture at {zkey}");

            // As setup-snark writes it: provingKeySnark/final/final.zkey, which is the final SNARK key.
            let stem = dir.path().join(expected.protocol_name()).join("final").join("final");
            std::fs::create_dir_all(stem.parent().unwrap()).unwrap();
            assert_eq!(FinalSnarkKey::find(&stem).ok(), None);
            std::fs::copy(&zkey, stem.with_extension("zkey")).unwrap();
            assert_eq!(FinalSnarkKey::find(&stem).unwrap(), FinalSnarkKey::Zkey(stem.with_extension("zkey")));

            let prover = init_final_snark_prover_c(&zkey, std::ptr::null_mut());
            assert!(!prover.is_null(), "{zkey}");
            let protocol = SnarkProtocol::from_snark_prover(prover).map(|p| p.protocol_id());
            free_final_snark_prover_c(prover);
            assert_eq!(protocol.ok(), Some(expected.protocol_id()), "{zkey}");
        }
    }

    #[test]
    fn a_prover_that_is_not_loaded_has_no_protocol() {
        assert!(SnarkProtocol::from_snark_prover(std::ptr::null_mut()).is_err());
    }

    #[test]
    fn convert_to_json_refuses_what_the_native_parser_cannot_read() {
        let proof = |proof_len: usize, digest_len: usize, protocol_id: u64| SnarkProof {
            proof_bytes: vec![0; proof_len],
            public_bytes: vec![],
            public_snark_bytes: vec![0; digest_len],
            protocol_id,
        };
        let plonk = SnarkProtocol::Plonk.protocol_id();
        assert!(proof(767, 32, plonk).convert_to_json().is_err());
        assert!(proof(768, 31, plonk).convert_to_json().is_err());
        assert!(proof(768, 0, plonk).convert_to_json().is_err());
        assert!(proof(768, 32, 99).convert_to_json().is_err());
    }

    /// Adding `r` to a word, or `q` to a coordinate, re-encodes the same proof for the native
    /// parser; the on-chain verifier rejects both, so these are refused before parsing.
    #[test]
    fn convert_to_json_refuses_a_word_at_or_above_r() {
        const R_MINUS_1: [u8; 32] = {
            let mut w = BN254_R_BE;
            w[31] -= 1;
            w
        };
        let digest = R_MINUS_1.to_vec();
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            for i in [0, 17, 18, 23] {
                for word in [BN254_R_BE, [0xff; 32]] {
                    let mut proof_bytes = vec![0; protocol.proof_bytes().unwrap()];
                    proof_bytes[32 * i..32 * (i + 1)].copy_from_slice(&word);
                    let proof = SnarkProof::new(proof_bytes, vec![], digest.clone(), protocol.protocol_id());
                    let err = proof.convert_to_json().unwrap_err().to_string();
                    assert!(err.contains(&format!("proof word {i} ")), "{err}");
                }
            }
        }

        let proof = SnarkProof::new(vec![0; 768], vec![], BN254_R_BE.to_vec(), SnarkProtocol::Plonk.protocol_id());
        assert!(proof.convert_to_json().unwrap_err().to_string().contains("public input 0 "));
    }

    #[test]
    fn first_word_not_below_r_compares_whole_words() {
        let mut below = BN254_R_BE;
        below[31] -= 1;
        let mut above_despite_a_smaller_last_byte = BN254_R_BE;
        above_despite_a_smaller_last_byte[1] += 1;
        above_despite_a_smaller_last_byte[31] = 0;

        assert_eq!(first_word_not_below_r(&below), None);
        assert_eq!(first_word_not_below_r(&[0; 64]), None);
        assert_eq!(first_word_not_below_r(&BN254_R_BE), Some(0));
        assert_eq!(first_word_not_below_r(&above_despite_a_smaller_last_byte), Some(0));
        assert_eq!(first_word_not_below_r(&[below, BN254_R_BE].concat()), Some(1));
    }

    /// 24 words 1, 2, ..., 24: 9 commitments and 6 evaluations for PLONK, 4 and 16 for FFLONK.
    fn proof_of_words(protocol: SnarkProtocol) -> SnarkProof {
        let word = |value: u8| {
            let mut word = [0u8; 32];
            word[31] = value;
            word
        };
        SnarkProof::new((1..=24).flat_map(word).collect(), vec![], word(25).to_vec(), protocol.protocol_id())
    }

    #[test]
    fn a_plonk_proof_converts_to_snarkjs_plonk_json() {
        let (proof, publics) = proof_of_words(SnarkProtocol::Plonk).convert_to_json().unwrap();
        assert_eq!(proof["protocol"], "plonk");
        assert_eq!(proof["curve"], "bn128");
        assert_eq!(proof["A"], serde_json::json!(["1", "2", "1"]));
        assert_eq!(proof["Wxiw"], serde_json::json!(["17", "18", "1"]));
        assert_eq!(proof["eval_a"], "19");
        assert_eq!(proof["eval_zw"], "24");
        assert_eq!(proof.as_object().unwrap().len(), 9 + 6 + 2);
        assert_eq!(publics, serde_json::json!(["25"]));
    }

    /// snarkjs reads an FFLONK proof's commitments under `polynomials` and its evaluations under
    /// `evaluations` (`fromObjectProof`).
    #[test]
    fn an_fflonk_proof_converts_to_snarkjs_fflonk_json() {
        let (proof, publics) = proof_of_words(SnarkProtocol::Fflonk).convert_to_json().unwrap();
        assert_eq!(proof["protocol"], "fflonk");
        assert_eq!(proof["curve"], "bn128");
        assert_eq!(proof["polynomials"]["C1"], serde_json::json!(["1", "2", "1"]));
        assert_eq!(proof["polynomials"]["W2"], serde_json::json!(["7", "8", "1"]));
        assert_eq!(proof["evaluations"]["ql"], "9");
        assert_eq!(proof["evaluations"]["inv"], "24");
        assert_eq!(proof["polynomials"].as_object().unwrap().len(), 4);
        assert_eq!(proof["evaluations"].as_object().unwrap().len(), 16);
        assert_eq!(proof.as_object().unwrap().len(), 4);
        assert_eq!(publics, serde_json::json!(["25"]));
    }

    #[test]
    fn a_proof_of_an_unknown_protocol_does_not_convert() {
        let proof = SnarkProof { protocol_id: 7, ..proof_of_words(SnarkProtocol::Plonk) };
        assert!(proof.convert_to_json().is_err());
        assert!(proof.convert_to_json_with_vkey(Path::new("vkey.json")).is_err());
    }

    /// PLONK's and FFLONK's bytes name themselves: the vkey is not read, and need not be there.
    #[test]
    fn a_rapidsnark_proof_converts_as_snarkjs_reads_it_whatever_the_vkey() {
        for protocol in [SnarkProtocol::Plonk, SnarkProtocol::Fflonk] {
            let name = protocol.protocol_name();
            let proof = proof_of_words(protocol);
            let with_vkey = proof.convert_to_json_with_vkey(Path::new("no/such/vkey.json")).unwrap();
            assert_eq!(with_vkey, proof.convert_to_json().unwrap(), "{name}");
        }
    }

    /// A pilfflonk proof's bytes need the vkey's names, and a vkey that cannot be read refuses them.
    #[test]
    fn a_pilfflonk_proof_converts_only_with_its_vkey() {
        let proof = proof_of_words(SnarkProtocol::Pilfflonk);
        let error = proof.convert_to_json().unwrap_err().to_string();
        assert!(error.contains("convert_to_json_with_vkey"), "{error}");
        let dir = tempfile::tempdir().unwrap();
        let not_a_vkey = dir.path().join("pilfflonk.vkey.json");
        std::fs::write(&not_a_vkey, "{}").unwrap();
        for vkey in [dir.path().join("none.json"), not_a_vkey] {
            assert!(proof.convert_to_json_with_vkey(&vkey).is_err(), "{}", vkey.display());
        }
    }

    /// A pilfflonk proof against a file that is not its vkey is not verified, before the JS verifier
    /// runs: the verification fails, as snarkjs's with a vkey it cannot read, and does not reject
    /// the proof.
    #[test]
    fn a_pilfflonk_proof_without_its_vkey_does_not_verify() {
        let dir = tempfile::tempdir().unwrap();
        let plonk_vkey = dir.path().join("final.verkey.json");
        std::fs::write(&plonk_vkey, r#"{"protocol": "plonk", "curve": "bn128", "nPublic": 1}"#).unwrap();
        for vkey in [dir.path().join("none.json"), plonk_vkey] {
            match verify_snark_proof(&proof_of_words(SnarkProtocol::Pilfflonk), &vkey) {
                Err(ProofmanError::InvalidConfiguration(message)) => {
                    assert!(message.starts_with("Failed to read the pilfflonk vkey"), "{message}")
                }
                other => panic!("{}: {other:?}", vkey.display()),
            }
        }
    }

    /// Each value's varint in bincode's standard configuration: one byte below 251, and 251 and
    /// the two bytes of a `u16` up to its maximum.
    fn varint(value: u64) -> Vec<u8> {
        match u16::try_from(value) {
            Ok(value) if value < 251 => vec![value as u8],
            Ok(value) => [vec![251], value.to_le_bytes().to_vec()].concat(),
            Err(_) => unreachable!("the test's values are below 2^16"),
        }
    }

    /// `snark_proof.bin` holds the fields of a [`SnarkProof`] in order, each byte vector its length
    /// and its bytes, and the protocol id last: the layout PLONK and FFLONK proofs have always had
    /// (768 bytes of proof, 24 words), which a pilfflonk proof shares with its own length and id.
    /// Every protocol's proof is read back as it was saved.
    #[test]
    fn a_snark_proof_is_saved_as_its_fields_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        for (protocol, proof_len) in
            [(SnarkProtocol::Plonk, 768), (SnarkProtocol::Fflonk, 768), (SnarkProtocol::Pilfflonk, 2048)]
        {
            let proof_bytes: Vec<u8> = (0..proof_len).map(|i| (i % 256) as u8).collect();
            let proof = SnarkProof::new(proof_bytes.clone(), vec![7; 32], vec![9; 32], protocol.protocol_id());
            let path = dir.path().join(format!("{}_snark_proof.bin", protocol.protocol_name()));
            proof.save(&path).unwrap();

            let expected = [
                varint(proof_len as u64),
                proof_bytes,
                varint(32),
                vec![7; 32],
                varint(32),
                vec![9; 32],
                varint(protocol.protocol_id()),
            ]
            .concat();
            assert_eq!(std::fs::read(&path).unwrap(), expected, "{}", protocol.protocol_name());

            let back = SnarkProof::load(&path).unwrap();
            assert_eq!(
                (back.proof_bytes, back.public_bytes, back.public_snark_bytes, back.protocol_id),
                (proof.proof_bytes, proof.public_bytes, proof.public_snark_bytes, proof.protocol_id)
            );
        }
    }

    /// A final SNARK's public is checked to be the digest of the vadcop_final's publics and the root:
    /// another digest, other publics or another root are refused as a proof whose public is not the
    /// statement's, and a root that is not 4 limbs as a misconfiguration.
    #[test]
    fn the_snark_public_must_be_the_digest_of_the_publics_and_the_root() {
        let out = proofman_common::PublicDefinition {
            name: "out".to_string(),
            initial_pos: 0,
            n_values: 3,
            chunks: [1, 64],
            verification_key: false,
        };
        let info = PublicsInfo { n_publics: 3, definitions: vec![out], has_program_vk: false };
        let (publics, rootc) = ([1, 2, 3], [11, 12, 13, 14]);
        let digest = info.snark_public_digest(&publics, &rootc).unwrap();
        assert!(check_snark_public_digest(&info, &publics, &rootc, &digest).is_ok());

        let check =
            |publics: &[u64], rootc: &[u64], digest: &[u8]| check_snark_public_digest(&info, publics, rootc, digest);
        let mut other_digest = digest;
        other_digest[31] ^= 1;
        for refused in [
            check(&publics, &rootc, &other_digest),
            check(&[1, 2, 4], &rootc, &digest),
            check(&publics, &[11, 12, 13, 15], &digest),
        ] {
            match refused {
                Err(ProofmanError::InvalidProof(message)) => {
                    assert!(message.contains("public digest differs"), "{message}")
                }
                other => panic!("{other:?}"),
            }
        }
        assert!(matches!(check(&publics, &rootc[..3], &digest), Err(ProofmanError::InvalidConfiguration(_))));
    }
}
