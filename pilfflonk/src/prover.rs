//! The prover (spec §4.4): the orchestration of a proof over the C++ core, in the order of the
//! transcript of spec A.4, which it drives through the C API (`proofman_starks_lib_c`).
//!
//! [`ProvingKey::load`] reads the `provingKey/` (spec §4.2.6): this crate's types read and validate
//! the globalInfo, the vkey (its digest checked) and each AIR's pilfflonkinfo, and check that they
//! agree; the C++ side loads the same files for its computations (the SRS, the bytecode, the fixed
//! columns) and derives the degrees of A.1, which must be [`PilfflonkInfo::degrees`]'s.
//!
//! [`prove`] runs A.4 for a proof of one instance (v1, D2; plan R4):
//!
//! 1. absorb `digest mod r`, the number of instances of each AIR of the globalInfo in canonical
//!    order (1), and the publics;
//! 2. for each stage `s = 1 … nStages`: the C++ commits the stage (its columns, im pols, blinding,
//!    packing, MSM); absorb the commitments of its f in the global order of A.5 (the layout's), then
//!    its air values, airgroup values and proof values of stage `s` (none in v1), and if
//!    `s < nStages` squeeze the `numChallenges[s]` challenges of stage `s + 1`, one per squeeze;
//! 3. squeeze `std_vc`; the C++ commits `Q`; absorb its commitments; squeeze `xiSeed`;
//! 4. the C++ evaluates every f at the roots of `ξ·ω^s`, `ξ = xiSeed^powerW`; absorb the evaluations
//!    of the proof, the fixed columns' then the others', each in the order of the evMap (no pieces of
//!    `Q`: it is not split);
//! 5. the C++ opens: squeezes `α_S`, absorbs `[W]₁`, squeezes `y`, and gives `[W']₁`, `inv` and
//!    `invZh`.
//!
//! It is `pilfflonk/js/src/challenges.js` (`computeChallenges`), the verifier's replay, step by step.
//! The proof holds the commitments of the non-fixed f (the fixed ones are the vkey's), `W`, `W'`, the
//! evaluations, `inv` and `invZh` (A.6), and [`ProofOutput::write`] writes `proof.json` and
//! `publics.json`.
//!
//! **Blinding** (decision D6) is always on: random by default, fixed by
//! [`ProveOptions::insecure_blinding_seed`] for tests and CI, never for a real proof.

use std::fs;
use std::path::{Path, PathBuf};

use proofman_starks_lib_c::{
    pilfflonk_keccak256_c, PilFflonkError, PilFflonkErrorKind, PilFflonkInstance, PilFflonkInstanceInputs,
    PilFflonkOpening, PilFflonkProverCtx, PilFflonkTranscript,
};

use crate::error::{invalid, PilfflonkError, PilfflonkResult};
use crate::field::{FrBytes, G1Affine};
use crate::global_info::{AirFile, PilfflonkGlobalInfo};
use crate::json::JsonFile;
use crate::pilfflonk_info::PilfflonkInfo;
use crate::proof::{Proof, ProofJson, ProofNames, Publics, PROOF_FILE, PUBLICS_FILE};
use crate::vkey::Vkey;
use crate::witness::{AirInstanceRef, WitnessShape, WitnessSource};

/// A call to the C++ core that failed, with what it was doing; a witness that does not satisfy the
/// constraints is [`PilfflonkError::Unsatisfied`].
fn native(context: &'static str) -> impl FnOnce(PilFflonkError) -> PilfflonkError {
    move |source| match source.kind {
        PilFflonkErrorKind::Unsatisfied => PilfflonkError::Unsatisfied(source.message),
        _ => PilfflonkError::Native { context: context.to_string(), source },
    }
}

fn keccak256(data: &[u8]) -> PilfflonkResult<[u8; 32]> {
    pilfflonk_keccak256_c(data).map_err(native("hashing the vkey"))
}

/// The `provingKey/` of a proof, loaded (see [the module](self)).
#[derive(Debug)]
pub struct ProvingKey {
    dir: PathBuf,
    global_info: PilfflonkGlobalInfo,
    vkey: Vkey,
    /// Every AIR of the globalInfo, in canonical order.
    airs: Vec<PilfflonkInfo>,
    ctx: PilFflonkProverCtx,
}

impl ProvingKey {
    /// Reads the `provingKey/` at `dir`, checks it, and loads it into the C++ core.
    ///
    /// Refuses a vkey whose digest is not that of its contents, a vkey and a pilfflonkinfo that do
    /// not describe the same AIR (the vkey of format 1 describes one), and a C++ core that derives
    /// other degrees than this crate.
    pub fn load(dir: &Path) -> PilfflonkResult<Self> {
        let global_info = PilfflonkGlobalInfo::from_proving_key(dir)?;
        let vkey_path = global_info.vkey_path(dir);
        let vkey = Vkey::read(&vkey_path)?;
        let digest = keccak256(&vkey.digest_preimage()?)?;
        if !vkey.digest_matches(|_| digest)? {
            return Err(PilfflonkError::InvalidFormat(
                "the digest of the vkey is not the digest of its contents (A.6)".into(),
            )
            .in_file(&vkey_path));
        }
        let mut airs = Vec::new();
        for (airgroup_id, group) in global_info.airs.iter().enumerate() {
            for air_id in 0..group.len() {
                let path = global_info.air_file(dir, airgroup_id as u64, air_id as u64, AirFile::PilfflonkInfo)?;
                airs.push(PilfflonkInfo::read(&path)?);
            }
        }
        let [info] = airs.as_slice() else {
            return invalid!(
                "the provingKey/ has {} AIRs, and a pilfflonk proof holds one instance of one AIR (D2)",
                airs.len()
            );
        };
        check_vkey(&vkey, info, &global_info).map_err(|e| e.in_file(&vkey_path))?;

        let ctx = PilFflonkProverCtx::load(dir).map_err(native("loading the provingKey/ into the C++ prover"))?;
        let degrees = info.degrees()?;
        let n_bits_ext = ctx
            .n_bits_ext(info.airgroup_id, info.air_id)
            .map_err(native("reading the C++ prover's extended domain"))?;
        if n_bits_ext != degrees.n_bits_ext {
            return invalid!(
                "the C++ prover extends {} to 2^{n_bits_ext} points, and A.1 says 2^{} (proofman_pilfflonk::degrees)",
                info.name,
                degrees.n_bits_ext
            );
        }
        Ok(Self { dir: dir.to_path_buf(), global_info, vkey, airs, ctx })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn global_info(&self) -> &PilfflonkGlobalInfo {
        &self.global_info
    }

    pub fn vkey(&self) -> &Vkey {
        &self.vkey
    }

    /// The pilfflonkinfo of an AIR.
    pub fn air(&self, air: AirInstanceRef) -> PilfflonkResult<&PilfflonkInfo> {
        match self.airs.iter().find(|a| a.airgroup_id == air.airgroup_id && a.air_id == air.air_id) {
            Some(info) => Ok(info),
            None => invalid!("the provingKey/ has no air {} in airgroup {}", air.air_id, air.airgroup_id),
        }
    }

    /// The shape of the witness of a proof of this key, to open a witness directory with.
    pub fn witness_shape(&self) -> PilfflonkResult<WitnessShape> {
        let infos: Vec<&PilfflonkInfo> = self.airs.iter().collect();
        WitnessShape::from_proving_key(&self.global_info, &infos)
    }
}

/// That the vkey is the one of `info` and `global_info`: what the verifier reads of it is what the
/// prover proves with.
fn check_vkey(vkey: &Vkey, info: &PilfflonkInfo, global_info: &PilfflonkGlobalInfo) -> PilfflonkResult<()> {
    let differs = |what: &str| invalid!("the vkey's {what} is not the one of the pilfflonkinfo or globalInfo");
    if vkey.n_public != global_info.n_publics {
        return differs("nPublic");
    }
    if vkey.num_challenges != global_info.num_challenges {
        return differs("numChallenges");
    }
    if vkey.power != info.n_bits {
        return differs("power");
    }
    if vkey.power_w != info.layout.power_w()? {
        return differs("powerW");
    }
    if vkey.ev_map != info.ev_map {
        return differs("evMap");
    }
    if vkey.layout != info.layout {
        return differs("layout");
    }
    if vkey.boundaries != info.boundaries {
        return differs("boundaries");
    }
    if vkey.q_deg != info.q_deg || vkey.max_q_degree != info.max_q_degree {
        return differs("qDeg or maxQDegree");
    }
    Ok(())
}

/// How to prove.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProveOptions {
    /// `None` (the default): the blinding is random, from the OS. `Some(seed)` fixes it (D6): the
    /// same seed gives the same proof, and whoever knows it can remove the blinding. For tests and
    /// CI only: a proof made with it is not zero-knowledge.
    pub insecure_blinding_seed: Option<[u8; 32]>,
}

/// The challenges of a proof and `Q(ξ)`, for tests and diagnostics: the proof does not hold them,
/// the verifier replays them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofChallenges {
    /// `stages[s - 2]`: the challenges of stage `s`, for `2 ≤ s ≤ nStages`.
    pub stages: Vec<Vec<FrBytes>>,
    pub std_vc: FrBytes,
    pub xi_seed: FrBytes,
    /// `Q(ξ)` of the instance, the value the verifier computes from the evaluations (A.1).
    pub q_at_xi: FrBytes,
}

/// A proof, what its JSON view is named by, and its publics.
#[derive(Clone, Debug)]
pub struct ProofOutput {
    pub proof: Proof,
    pub names: ProofNames,
    pub publics: Publics,
    pub challenges: ProofChallenges,
}

impl ProofOutput {
    /// `proof.json`: the JSON view of the proof (A.6).
    pub fn proof_json(&self) -> PilfflonkResult<ProofJson> {
        self.proof.to_json(&self.names)
    }

    /// Writes `proof.json` and `publics.json` to `dir`, which is created if it does not exist.
    pub fn write(&self, dir: &Path) -> PilfflonkResult<()> {
        fs::create_dir_all(dir).map_err(|source| PilfflonkError::Io { path: dir.to_path_buf(), source })?;
        self.proof_json()?.write(&dir.join(PROOF_FILE))?;
        self.publics.write(&dir.join(PUBLICS_FILE))
    }
}

fn fr(bytes: [u8; 32]) -> PilfflonkResult<FrBytes> {
    FrBytes::from_le_bytes(bytes)
}

fn le(values: &[FrBytes]) -> Vec<[u8; 32]> {
    values.iter().map(FrBytes::to_le_bytes).collect()
}

fn g1(points: Vec<[u8; 64]>) -> PilfflonkResult<Vec<G1Affine>> {
    points.iter().map(G1Affine::from_le_bytes).collect()
}

/// The transcript of A.4, with its errors in this crate's terms.
struct Transcript(PilFflonkTranscript);

impl Transcript {
    fn new() -> PilfflonkResult<Self> {
        PilFflonkTranscript::new().map(Self).map_err(native("creating the transcript"))
    }

    fn scalars(&mut self, values: &[FrBytes]) -> PilfflonkResult<()> {
        self.0.absorb_fr(&le(values)).map_err(native("absorbing scalars into the transcript"))
    }

    fn points(&mut self, points: &[G1Affine]) -> PilfflonkResult<()> {
        let bytes: Vec<[u8; 64]> = points.iter().map(G1Affine::to_le_bytes).collect();
        self.0.absorb_g1(&bytes).map_err(native("absorbing commitments into the transcript"))
    }

    fn squeeze(&mut self) -> PilfflonkResult<FrBytes> {
        fr(self.0.squeeze().map_err(native("squeezing the transcript"))?)
    }
}

/// A proof of the one instance of `witness` (see [the module](self)).
pub fn prove(pk: &ProvingKey, witness: &impl WitnessSource, options: &ProveOptions) -> PilfflonkResult<ProofOutput> {
    let instances = witness.instances();
    let [air] = instances.as_slice() else {
        return invalid!("the witness has {} instances, and a pilfflonk proof holds one (D2)", instances.len());
    };
    let info = pk.air(*air)?;
    let global_info = pk.global_info();
    let names = ProofNames::new(global_info, &[info])?;
    let stage1 = witness.stage1(0)?;
    let publics = witness.publics()?;
    let proof_values = witness.proof_values()?;
    let n_stages = info.n_stages;
    let n_f = |stage: u64| info.layout.0.iter().filter(|f| f.stage == stage).count();

    let mut instance = PilFflonkInstance::new(
        &pk.ctx,
        &PilFflonkInstanceInputs {
            airgroup_id: air.airgroup_id,
            air_id: air.air_id,
            stage1: stage1.trace_bytes(),
            air_values: &le(stage1.air_values()),
            publics: &le(&publics),
            proof_values: &le(&proof_values),
            insecure_blinding_seed: options.insecure_blinding_seed.as_ref(),
        },
    )
    .map_err(native("creating the instance"))?;

    // Step 1: the digest, the number of instances of each AIR, the publics.
    let mut transcript = Transcript::new()?;
    transcript.scalars(&[pk.vkey().digest.to_fr()])?;
    let counts: Vec<FrBytes> = global_info
        .airs
        .iter()
        .enumerate()
        .flat_map(|(airgroup_id, group)| (0..group.len()).map(move |air_id| (airgroup_id as u64, air_id as u64)))
        .map(|(airgroup_id, air_id)| {
            FrBytes::from_u64(u64::from((airgroup_id, air_id) == (air.airgroup_id, air.air_id)))
        })
        .collect();
    transcript.scalars(&counts)?;
    transcript.scalars(&publics)?;

    // Step 2: the stages. The air values, airgroup values and proof values of stage s follow its
    // commitments; v1 has none but the stage-1 ones the witness gives (none either, D2).
    let mut commitments = Vec::new();
    let mut stage_challenges: Vec<Vec<FrBytes>> = Vec::new();
    let mut challenges: Vec<FrBytes> = Vec::new();
    for stage in 1..=n_stages {
        let stage_u32 = u32::try_from(stage).map_err(|_| PilfflonkError::InvalidFormat("too many stages".into()))?;
        tracing::info!("··· Committing stage {stage}");
        let points = g1(instance
            .commit_stage(stage_u32, &le(&challenges), n_f(stage))
            .map_err(native("committing a stage"))?)?;
        transcript.points(&points)?;
        commitments.extend(points);
        if stage == 1 {
            transcript.scalars(stage1.air_values())?;
            transcript.scalars(&proof_values)?;
        }
        if stage < n_stages {
            let count = global_info.num_challenges.get(stage as usize).copied().unwrap_or(0);
            challenges = (0..count).map(|_| transcript.squeeze()).collect::<PilfflonkResult<_>>()?;
            stage_challenges.push(challenges.clone());
        }
    }

    // Step 3: Q.
    let std_vc = transcript.squeeze()?;
    tracing::info!("··· Committing Q");
    let q_points = g1(instance.commit_q(&le(&[std_vc]), n_f(info.q_stage())).map_err(native("committing Q"))?)?;
    transcript.points(&q_points)?;
    commitments.extend(q_points);
    let xi_seed = transcript.squeeze()?;

    // Step 4: the evaluations.
    tracing::info!("··· Evaluating at ξ");
    let opening =
        PilFflonkOpening::new(&[&instance], &xi_seed.to_le_bytes()).map_err(native("evaluating the polynomials"))?;
    let evaluations = opening
        .evaluations()
        .map_err(native("reading the evaluations"))?
        .into_iter()
        .map(fr)
        .collect::<Result<Vec<_>, _>>()?;
    transcript.scalars(&evaluations)?;
    let q_at_xi = fr(opening.q(0).map_err(native("reading Q(ξ)"))?)?;

    // Step 5: SHPLONK.
    tracing::info!("··· Opening with SHPLONK");
    let opened = opening.open(&mut transcript.0).map_err(native("opening the polynomials"))?;

    let proof = Proof {
        commitments,
        w: G1Affine::from_le_bytes(&opened.w)?,
        wp: G1Affine::from_le_bytes(&opened.wp)?,
        evaluations,
        air_values: stage1.air_values().to_vec(),
        airgroup_values: vec![],
        proof_values,
        inv: fr(opened.inv)?,
        inv_zh: fr(opened.inv_zh)?,
    };
    if proof.shape() != names.shape() {
        return invalid!(
            "the C++ prover gave a proof of shape {:?}, and the AIR's names are of one of shape {:?}",
            proof.shape(),
            names.shape()
        );
    }
    let challenges = ProofChallenges { stages: stage_challenges, std_vc, xi_seed, q_at_xi };
    Ok(ProofOutput { proof, names, publics: Publics(publics), challenges })
}
