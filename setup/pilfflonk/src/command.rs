//! `proofman-setup setup-pilfflonk` (spec §4.2): the command `pil2-stark-setup` hosts, and the
//! order of its steps. The steps are this crate's library functions; this module reports their
//! errors with `anyhow`, as the other setup commands do (spec §5.4).
//!
//! It writes the `provingKey/` of spec §4.2.6 under the build directory. Today (plan M15) that
//! is the files that do not depend on the passes: `pilout.globalInfo.json`, `<air>.const`,
//! `pilfflonk.srs.bin` and `<air>.verkey.json`. M16 adds the passes, the layout, the bytecode and
//! the vkey, where the steps below say.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pil2_pilout::pilout as pb;
use prost::Message;
use proofman_pilfflonk::global_info::GLOBAL_INFO_FILE;
use proofman_pilfflonk::{AirFile, AirVerkey, JsonFile, SetupParams};

use crate::error::SetupError;
use crate::fixed::FixedColumns;
use crate::global_info::global_info;
use crate::keys::{commit_fixed_f, load_srs, write_srs};
use crate::validate::validate;

/// The directory the setup writes under the build directory.
pub const PROVING_KEY_DIR: &str = "provingKey";

/// `--max-constraint-degree` by default (D5, as pil-stark).
pub const DEFAULT_MAX_CONSTRAINT_DEGREE: u64 = pil_info::DEFAULT_MAX_CONSTRAINT_DEGREE as u64;

/// `--extra-muls` by default, as pil-stark.
pub const DEFAULT_EXTRA_MULS: u64 = 2;

/// `--max-q-degree` by default: `Q` is not split (A.1).
pub const DEFAULT_MAX_Q_DEGREE: u64 = 0;

/// The arguments of `setup-pilfflonk` (spec §4.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupPilfflonkOptions {
    /// `-a`: the pilout, compiled over BN254.
    pub airout_path: PathBuf,
    /// `-b`: the build directory, where `provingKey/` goes.
    pub build_dir: PathBuf,
    /// `--powers-of-tau`: a snarkjs ptau of at least as many powers `[τ^i]₁` as the layout needs.
    pub powers_of_tau: PathBuf,
    /// `--max-constraint-degree` (D5).
    pub max_constraint_degree: u64,
    /// `--extra-muls` (A.2, rule 3).
    pub extra_muls: u64,
    /// `--max-q-degree`: 0 does not split `Q` (A.1).
    pub max_q_degree: u64,
    /// `--no-packing`: every `f_i` packs one polynomial, `k = 1`. For tests only.
    pub no_packing: bool,
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

    /// Refuses what the setup cannot do with these arguments: a degree search below 2, and, until
    /// they are implemented, splitting `Q` (plan R3, M33) and packing (plan R1, M22).
    pub fn check(&self) -> Result<(), SetupError> {
        if self.max_constraint_degree < 2 {
            return Err(SetupError::MaxConstraintDegree(self.max_constraint_degree));
        }
        if self.max_q_degree != 0 {
            return Err(SetupError::QSplitting(self.max_q_degree));
        }
        if !self.no_packing {
            return Err(SetupError::Packing);
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

/// Runs `setup-pilfflonk`.
pub fn run_setup_pilfflonk(opts: &SetupPilfflonkOptions) -> Result<()> {
    opts.check()?;
    let pilout = read_pilout(&opts.airout_path)?;
    let air = validate(&pilout).with_context(|| format!("{} cannot be set up", opts.airout_path.display()))?;
    let fixed = FixedColumns::from_air(air.air).with_context(|| format!("{}", opts.airout_path.display()))?;

    // M16: pil_info::run(&pilout, air.airgroup_id, air.air_id, &cfg, …) with cfg = PilInfoCfg::bn254()
    // and DegreePolicy::Search { max: max_constraint_degree }, then the layout (A.2, unpacked:
    // R1), nBitsExt (A.1, checked with validate::check_extended_domain) and the bytecode.

    let proving_key = opts.build_dir.join(PROVING_KEY_DIR);
    let global_info = global_info(&pilout, opts.setup_params())?;
    let (airgroup_id, air_id) = (air.airgroup_id as u64, air.air_id as u64);
    create_dir(&global_info.air_dir(&proving_key, airgroup_id, air_id)?)?;
    create_dir(&global_info.backend_dir(&proving_key))?;

    // The fixed f_i and the size of the SRS. M16 takes them from the layout: keys::air_verkey over
    // it, and write_srs with its largest degree, which is Q's. Until then, the fixed f_i of the
    // unpacked layout (R1): one per column, in their order, with k = 1 and N coefficients.
    let fixed_fs: Vec<Vec<u64>> = (0..fixed.n_columns() as u64).map(|id| vec![id]).collect();
    let n_g1 = fixed.n_rows() as u64;

    // The SRS first: a ptau with too few powers is refused (spec §4.2.1) before any other file is
    // written.
    let srs_path = global_info.srs_path(&proving_key);
    write_srs(&opts.powers_of_tau, n_g1, &srs_path)?;
    tracing::info!("wrote {} ({n_g1} powers [τ^i]₁)", srs_path.display());

    let global_info_path = proving_key.join(GLOBAL_INFO_FILE);
    global_info.write(&global_info_path)?;
    tracing::info!("wrote {}", global_info_path.display());

    let const_path = global_info.air_file(&proving_key, airgroup_id, air_id, AirFile::Const)?;
    fixed.write_const(&const_path)?;
    tracing::info!("wrote {} ({} columns of {} rows)", const_path.display(), fixed.n_columns(), fixed.n_rows());

    let srs = load_srs(&srs_path)?;
    let verkey =
        AirVerkey(fixed_fs.iter().map(|columns| commit_fixed_f(&srs, &fixed, columns)).collect::<Result<_, _>>()?);
    let verkey_path = global_info.air_file(&proving_key, airgroup_id, air_id, AirFile::Verkey)?;
    verkey.write(&verkey_path)?;
    tracing::info!("wrote {} ({} fixed commitments)", verkey_path.display(), verkey.0.len());

    // M16: <air>.pilfflonkinfo.json, .expressionsinfo.json, .verifierinfo.json, .bin,
    // pilout.globalConstraints.json, and last the vkey: Vkey::new(…, keys::x_2(&srs)?, &verkey, …),
    // sealed with digest::seal_vkey.
    Ok(())
}
