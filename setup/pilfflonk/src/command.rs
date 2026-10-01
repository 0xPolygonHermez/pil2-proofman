//! `proofman-setup setup-pilfflonk` (pilfflonk/docs/README.md#setup-pilfflonk): the command
//! `pil2-stark-setup` hosts, and the order of its steps. The steps are this crate's library
//! functions; this module reports their errors with `anyhow`, as the other setup commands do
//! (pilfflonk/docs/README.md#conventions).
//!
//! It writes the `provingKey/` (pilfflonk/docs/formats.md#provingkey) under the build directory,
//! for the one AIR of the pilout (pilfflonk/docs/README.md#scope):
//!
//! ```text
//! <build>/provingKey/
//! ├── pilout.globalInfo.json
//! ├── pilout.globalConstraints.json
//! └── <name>/
//!     ├── pilfflonk/{pilfflonk.srs.bin, pilfflonk.vkey.json[, pilfflonk.verifier.sol]}
//!     └── <airgroup>/airs/<air>/air/<air>.{const, pilfflonkinfo.json, expressionsinfo.json,
//!                                           verifierinfo.json, bin, verkey.json}
//! ```
//!
//! Everything that can be refused is refused before the first file is written: the pilout
//! (pilfflonk/docs/README.md#what-the-setup-refuses), what the passes return (the prover hints
//! among it), the extended domain, the names of the proof, the shape of the witness and what the
//! verifier would refuse of the vkey. The SRS is the first file, so that a ptau with too few
//! powers writes nothing else; the vkey is the last, with its digest
//! (pilfflonk/docs/formats.md#digest). With `--solidity`, `pilfflonk.verifier.sol` follows it: it
//! is made from the vkey ([`crate::solidity`]) before the vkey is written, so that a vkey it
//! cannot be made of is not written either. The files depend only on the inputs: two runs write
//! the same bytes.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pil2_pilout::pilout as pb;
use pil_info::output::global_constraints::build_global_constraints_json;
use pil_info::FieldCfg;
use prost::Message;
use proofman_pilfflonk::global_info::{GLOBAL_CONSTRAINTS_FILE, GLOBAL_INFO_FILE};
use proofman_pilfflonk::json::to_json_string;
use proofman_pilfflonk::{
    AirFile, AirVerkey, FixedCommitments, G1Affine, G2Affine, JsonFile, ProofNames, SetupParams, Vkey, WitnessShape,
};

use crate::air_info::{air_setup, AirRef, AirSetup};
use crate::bytecode::write_air_bin;
use crate::digest::seal_vkey;
use crate::error::SetupError;
use crate::fixed::FixedColumns;
use crate::global_info::global_info;
use crate::keys::{air_verkey, load_srs, write_srs, x_2};
use crate::layout::{max_degree, Packing};
use crate::passes::run_passes;
use crate::solidity::{verifier_sol, VERIFIER_SOL_FILE};
use crate::validate::{check_extended_domain, check_prover_hints, validate};

/// The directory the setup writes under the build directory.
pub const PROVING_KEY_DIR: &str = "provingKey";

/// `--max-constraint-degree` by default, as pil-stark (pilfflonk/docs/protocol.md#degree-search).
pub const DEFAULT_MAX_CONSTRAINT_DEGREE: u64 = pil_info::DEFAULT_MAX_CONSTRAINT_DEGREE as u64;

/// `--extra-muls` by default, as pil-stark.
pub const DEFAULT_EXTRA_MULS: u64 = 2;

/// `--max-q-degree` by default: `Q` is not split (pilfflonk/docs/protocol.md#q-pieces).
pub const DEFAULT_MAX_Q_DEGREE: u64 = 0;

/// The arguments of `setup-pilfflonk` (pilfflonk/docs/README.md#setup-pilfflonk).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupPilfflonkOptions {
    /// `-a`: the pilout, compiled over BN254.
    pub airout_path: PathBuf,
    /// `-b`: the build directory, where `provingKey/` goes.
    pub build_dir: PathBuf,
    /// `--powers-of-tau`: a snarkjs ptau of at least as many powers `[τ^i]₁` as the layout needs.
    pub powers_of_tau: PathBuf,
    /// `--max-constraint-degree` (pilfflonk/docs/protocol.md#degree-search).
    pub max_constraint_degree: u64,
    /// `--extra-muls` (pilfflonk/docs/protocol.md#grouping-rules, rule 3).
    pub extra_muls: u64,
    /// `--max-q-degree`: `Q` is split in pieces of this degree if its own is above it
    /// (pilfflonk/docs/protocol.md#q-pieces); 0 does not split it.
    pub max_q_degree: u64,
    /// `--no-packing`: every `f_i` packs one polynomial, `k = 1`, and `--extra-muls` is unused. For
    /// tests only.
    pub no_packing: bool,
    /// `--solidity`: also write `pilfflonk.verifier.sol`, the Solidity verifier of the vkey
    /// (pilfflonk/docs/verifier.md#solidity-verifier). It changes no other file: the globalInfo
    /// does not record it.
    pub solidity: bool,
}

impl SetupPilfflonkOptions {
    /// The parameters the globalInfo records.
    pub fn setup_params(&self) -> SetupParams {
        SetupParams {
            max_constraint_degree: self.max_constraint_degree,
            extra_muls: self.extra_muls,
            max_q_degree: self.max_q_degree,
            packing: !self.no_packing,
        }
    }

    /// How the committed polynomials go into `f_i`: grouped with `--extra-muls`
    /// (pilfflonk/docs/protocol.md#grouping-rules), or one per `f` with `--no-packing`.
    pub fn packing(&self) -> Packing {
        if self.no_packing {
            Packing::Unpacked
        } else {
            Packing::Grouped { extra_muls: self.extra_muls }
        }
    }

    /// Refuses what the setup cannot do with these arguments: a degree search below 2. Every
    /// `--max-q-degree` is one: `Q` is split only if its degree is above it
    /// (pilfflonk/docs/protocol.md#q-pieces). What `--extra-muls` can do depends on the AIR, and on
    /// the pieces of `Q` too: the grouping refuses it ([`SetupError::Grouping`]).
    pub fn check(&self) -> Result<(), SetupError> {
        if self.max_constraint_degree < 2 {
            return Err(SetupError::MaxConstraintDegree(self.max_constraint_degree));
        }
        Ok(())
    }
}

/// Reads the pilout at `path`.
fn read_pilout(path: &Path) -> Result<pb::PilOut, SetupError> {
    let bytes = fs::read(path).map_err(SetupError::io(path))?;
    pb::PilOut::decode(bytes.as_slice()).map_err(|source| SetupError::Pilout { path: path.to_path_buf(), source })
}

fn create_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))
}

/// Writes `text` at `path`, replacing any file there.
fn write_text(path: &Path, text: &str) -> Result<()> {
    fs::write(path, text).map_err(SetupError::io(path))?;
    tracing::info!("wrote {}", path.display());
    Ok(())
}

/// Runs `setup-pilfflonk`.
pub fn run_setup_pilfflonk(opts: &SetupPilfflonkOptions) -> Result<()> {
    opts.check()?;
    let refused = || format!("{} cannot be set up", opts.airout_path.display());
    let pilout = read_pilout(&opts.airout_path)?;
    let air = validate(&pilout).with_context(refused)?;
    let fixed = FixedColumns::from_air(air.air).with_context(refused)?;
    let global_info = global_info(&pilout, opts.setup_params())?;
    let (airgroup_id, air_id) = (air.airgroup_id as u64, air.air_id as u64);
    let air_ref = AirRef { name: &global_info.air(airgroup_id, air_id)?.name, airgroup_id, air_id };

    // The passes (pilfflonk/docs/README.md#setup-pilfflonk), and what follows from them: the
    // committed polynomials, their bounds and their layout, grouped unless --no-packing
    // (pilfflonk/docs/protocol.md#degrees, pilfflonk/docs/protocol.md#layout), in the
    // pilfflonkinfo.
    let result = run_passes(&pilout, air, opts.max_constraint_degree).with_context(refused)?;
    // The prover hints, as the passes process them: each column of stage 2 or above produced by one.
    check_prover_hints(&result, air_ref.name).with_context(refused)?;
    let AirSetup { info, committed } =
        air_setup(&result, air_ref, air.air, opts.max_q_degree, opts.packing()).with_context(refused)?;
    for name in &committed.unopened {
        tracing::warn!("column {name} is never opened: it is not committed (pilfflonk/docs/protocol.md#layout)");
    }
    let degrees = committed.degrees;
    check_extended_domain(degrees.n_bits_ext).with_context(refused)?;
    // What the prover will read of this key, checked now rather than when it proves: the names of
    // the proof's values must not collide, and the witness must have the shape of the pilout's.
    ProofNames::new(&global_info, &[&info]).with_context(refused)?;
    WitnessShape::from_proving_key(&global_info, &[&info]).with_context(refused)?;
    // The vkey (pilfflonk/docs/formats.md#vkey) but for its points, [τ]₂ and the fixed
    // commitments, which need the SRS: Vkey::new checks what the verifier would refuse of it (the
    // challenges, the boundaries, the qVerifier of the verifierinfo, which it can run). The points
    // and the digest are set last; until then [τ]₂ is [1]₂, a point of G2 as X_2 must be
    // (Vkey::validate).
    let pil_code = &result.pil_code;
    let q_verifier = serde_json::to_value(&pil_code.verifier_info)?
        .get("qVerifier")
        .cloned()
        .ok_or_else(|| SetupError::PassesOutput("the verifierinfo has no qVerifier".to_string()))?;
    let no_points = AirVerkey(vec![G1Affine::INFINITY; info.layout.n_fixed()]);
    let vkey = Vkey::new(
        &info,
        global_info.n_publics,
        global_info.num_challenges.clone(),
        G2Affine::generator()?,
        &no_points,
        q_verifier,
    )
    .with_context(refused)?;
    let n_g1 = max_degree(&info.layout);
    let ks: Vec<u64> = info.layout.0.iter().map(|f| f.k).collect();
    tracing::info!(
        "air {}: nBits {} | qDeg {} in {} pieces | {} im pols | {} f, k {:?} | powerW {} | |O|max {} | nBitsExt {} | \
         {} powers [τ^i]₁",
        info.name,
        info.n_bits,
        info.q_deg,
        committed.q_split.n_pieces(),
        info.cm_pols_map.iter().filter(|p| p.im_pol).count(),
        info.layout.0.len(),
        ks,
        info.layout.power_w()?,
        degrees.max_openings,
        degrees.n_bits_ext,
        n_g1
    );

    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let air_file = |file| global_info.air_file(&proving_key, airgroup_id, air_id, file);
    create_dir(&global_info.air_dir(&proving_key, airgroup_id, air_id)?)?;
    create_dir(&global_info.backend_dir(&proving_key))?;

    // The SRS first: a ptau with fewer powers than the layout's largest degree is refused
    // (pilfflonk/docs/README.md#what-the-setup-refuses) before any other file is written.
    let srs_path = global_info.srs_path(&proving_key);
    write_srs(&opts.powers_of_tau, n_g1, &srs_path)?;
    tracing::info!("wrote {} ({n_g1} powers [τ^i]₁)", srs_path.display());

    let global_info_path = proving_key.join(GLOBAL_INFO_FILE);
    global_info.write(&global_info_path)?;
    tracing::info!("wrote {}", global_info_path.display());

    let const_path = air_file(AirFile::Const)?;
    fixed.write_const(&const_path)?;
    tracing::info!("wrote {} ({} columns of {} rows)", const_path.display(), fixed.n_columns(), fixed.n_rows());

    let srs = load_srs(&srs_path)?;
    let verkey = air_verkey(&srs, &fixed, &info.layout)?;
    let verkey_path = air_file(AirFile::Verkey)?;
    verkey.write(&verkey_path)?;
    tracing::info!("wrote {} ({} fixed commitments)", verkey_path.display(), verkey.0.len());

    let info_path = air_file(AirFile::PilfflonkInfo)?;
    info.write(&info_path)?;
    tracing::info!("wrote {}", info_path.display());

    // The code, in the STARK's formats with dimension 1
    // (pilfflonk/docs/formats.md#expressionsinfo-and-verifierinfo), as pil-info serialises it for
    // the STARK setup; the verifierinfo has only the qVerifier (Opening::Shplonk).
    write_text(&air_file(AirFile::ExpressionsInfo)?, &to_json_string(&pil_code.expressions_info)?)?;
    write_text(&air_file(AirFile::VerifierInfo)?, &to_json_string(&pil_code.verifier_info)?)?;
    let bin_path = air_file(AirFile::Bin)?;
    write_air_bin(&result, &bin_path)?;
    tracing::info!("wrote {}", bin_path.display());
    // No global constraint (pilfflonk/docs/formats.md#globalconstraints): the file has none, and
    // the global hints the setup ignores.
    let global_constraints = build_global_constraints_json(&pilout, &FieldCfg::bn254()).with_context(refused)?;
    write_text(&proving_key.join(GLOBAL_CONSTRAINTS_FILE), &to_json_string(&global_constraints)?)?;

    // Last, the vkey (pilfflonk/docs/formats.md#vkey), with [τ]₂ of the SRS and the fixed
    // commitments of the verkey, sealed with its digest.
    let vkey = Vkey { x_2: x_2(&srs)?, fixed_commitments: FixedCommitments(verkey.0), ..vkey };
    let vkey = seal_vkey(vkey)?;
    // With --solidity, the verifier of this vkey, made before the vkey is written: a vkey it cannot
    // be made of is refused as the setup refuses the rest.
    let verifier = if opts.solidity { Some(verifier_sol(&vkey).with_context(refused)?) } else { None };
    let vkey_path = global_info.vkey_path(&proving_key);
    vkey.write(&vkey_path)?;
    tracing::info!("wrote {} (digest {})", vkey_path.display(), vkey.digest.to_hex());
    if let Some(verifier) = verifier {
        write_text(&global_info.backend_dir(&proving_key).join(VERIFIER_SOL_FILE), &verifier)?;
    }
    Ok(())
}
