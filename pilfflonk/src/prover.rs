//! The prover (pilfflonk/docs/protocol.md#proof-sequence): the orchestration of a proof over the
//! C++ core, in the order of the transcript (pilfflonk/docs/protocol.md#transcript), which it drives
//! through the C API (`proofman_starks_lib_c`).
//!
//! [`ProvingKey::load`] reads the `provingKey/` (pilfflonk/docs/formats.md#provingkey): this crate's
//! types read and validate the globalInfo, the vkey (its digest checked) and each AIR's
//! pilfflonkinfo, and check that they agree; the C++ side loads the same files for its computations
//! (the SRS, the bytecode, the fixed columns) and derives the degrees
//! (pilfflonk/docs/protocol.md#degrees), which must be [`PilfflonkInfo::degrees`]'s; its `[τ]₂` and
//! the commitments of its fixed columns must be the vkey's.
//!
//! [`prove`] follows the transcript for a proof of one instance (pilfflonk/docs/README.md#scope):
//!
//! 1. absorb `digest mod r`, the number of instances of each AIR of the globalInfo in canonical
//!    order (1), and the publics;
//! 2. for each stage `s = 1 … nStages`: the C++ commits the stage (its columns, the witness's for
//!    stage 1 and, with the stage's challenges, the std's prover hints' for the others; im pols,
//!    blinding, packing, MSM); absorb the commitments of its f in the global order (the layout's,
//!    pilfflonk/docs/protocol.md#global-order), then its air values, airgroup values and proof values
//!    of stage `s` (none in v1), and if `s < nStages` squeeze the `numChallenges[s]` challenges of
//!    stage `s + 1`, one per squeeze;
//! 3. squeeze `std_vc`; the C++ commits `Q`, or its pieces if it is split
//!    (pilfflonk/docs/protocol.md#q-pieces); absorb the commitments of its f; squeeze `xiSeed`;
//! 4. the C++ evaluates every f at the roots of `ξ·ω^s`, `ξ = xiSeed^powerW`; absorb the evaluations
//!    of the proof, the fixed columns' then the others', each in the order of the evMap, and then, if
//!    `Q` is split, its pieces' `Q_i(ξ)`, in the order of the layout;
//! 5. the C++ opens: squeezes `α_S`, absorbs `[W]₁`, squeezes `y`, and gives `[W']₁`, `inv` and
//!    `invZh`.
//!
//! It is `pilfflonk/js/src/challenges.js` (`computeChallenges`), the verifier's replay, step by step.
//! Steps 1 and 2 are `commit_stages`, which [`stage_columns`] runs too.
//! The proof holds the commitments of the non-fixed f (the fixed ones are the vkey's), `W`, `W'`, the
//! evaluations, `inv` and `invZh` (pilfflonk/docs/formats.md#proof), and [`ProofOutput::write`]
//! writes `proof.json` and `publics.json`.
//!
//! **Blinding** (pilfflonk/docs/protocol.md#blinding) is always on: random by default, fixed by
//! [`ProveOptions::insecure_blinding_seed`] for tests and CI, never for a real proof.
//!
//! **The GPU** (pilfflonk/docs/performance.md#gpu): [`ProvingKey::load_on`] with [`Device::Gpu`]
//! runs the MSMs and the NTTs of the key and of its proofs on the GPU, with the GPU entry points of
//! `pil2-stark/src/bn128/src/{msm,ntt}`, and gives the same proofs, bit for bit. It needs a library
//! built with CUDA and a GPU ([`gpu_available`]); [`ProvingKey::load`] is the CPU.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use proofman_starks_lib_c::{
    pilfflonk_gpu_available_c, PilFflonkError, PilFflonkErrorKind, PilFflonkInstance, PilFflonkInstanceInputs,
    PilFflonkOpening, PilFflonkProverCtx, PilFflonkTranscript,
};

use crate::error::{invalid, PilfflonkError, PilfflonkResult};
use crate::field::{FrBytes, G1Affine, G2Affine};
use crate::global_info::{AirFile, PilfflonkGlobalInfo};
use crate::json::JsonFile;
use crate::pilfflonk_info::PilfflonkInfo;
use crate::proof::{Proof, ProofJson, ProofNames, Publics, PROOF_FILE, PUBLICS_FILE};
use crate::vkey::Vkey;
use crate::witness::{AirInstanceRef, Stage1Witness, WitnessShape, WitnessSource};

/// Where a [`ProvingKey`] runs the MSMs and the NTTs of its proofs
/// (pilfflonk/docs/performance.md#selection-memory-and-errors).
pub use proofman_starks_lib_c::PilFflonkDevice as Device;

/// Whether [`Device::Gpu`] can be used here: this library was built with CUDA (`nvcc` found, the
/// feature `proofman-starks-lib-c/cpu-only` off) and sees a GPU. It never fails.
pub fn gpu_available() -> bool {
    pilfflonk_gpu_available_c()
}

/// A call to the C++ core that failed, with what it was doing; a witness that does not satisfy the
/// constraints is [`PilfflonkError::Unsatisfied`].
pub(crate) fn native(context: &'static str) -> impl FnOnce(PilFflonkError) -> PilfflonkError {
    move |source| match source.kind {
        PilFflonkErrorKind::Unsatisfied => PilfflonkError::Unsatisfied(source.message),
        _ => PilfflonkError::Native { context: context.to_string(), source },
    }
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
    /// not describe the same AIR (the vkey of format 1 describes one), an SRS whose `[τ]₂` is not
    /// the vkey's `X_2` and a `.const` whose fixed columns do not commit to the vkey's fixed
    /// commitments (files of other setups), and a C++ core that derives other degrees than this
    /// crate. The MSMs and NTTs run on the CPU: [`load_on`](Self::load_on) with [`Device::Cpu`].
    pub fn load(dir: &Path) -> PilfflonkResult<Self> {
        Self::load_on(dir, Device::Cpu)
    }

    /// [`load`](Self::load), with the MSMs and the NTTs of the key (the fixed columns' INTT and the
    /// commitments it checks) and of its proofs on `device`. On [`Device::Gpu`] the proofs are those
    /// of the CPU, bit for bit; without a GPU ([`gpu_available`]) it is refused before any file of
    /// the C++ core is read, saying why. [`ProvingKeyFiles::read`] and then
    /// [`ProvingKeyFiles::load_on`].
    pub fn load_on(dir: &Path, device: Device) -> PilfflonkResult<Self> {
        ProvingKeyFiles::read(dir)?.load_on(device)
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
        witness_shape(&self.global_info, &self.airs)
    }

    /// The C++ core's key.
    pub(crate) fn ctx(&self) -> &PilFflonkProverCtx {
        &self.ctx
    }
}

fn witness_shape(global_info: &PilfflonkGlobalInfo, airs: &[PilfflonkInfo]) -> PilfflonkResult<WitnessShape> {
    let infos: Vec<&PilfflonkInfo> = airs.iter().collect();
    WitnessShape::from_proving_key(global_info, &infos)
}

/// The files of a `provingKey/` that this crate reads, read and checked: the first half of
/// [`ProvingKey::load_on`], before the C++ core loads the rest (the SRS, the bytecode, the fixed
/// columns). They give the witness's shape, so that the witness can be read while the C++ core
/// loads (pilfflonk/docs/performance.md#the-start-of-a-proof).
#[derive(Debug)]
pub struct ProvingKeyFiles {
    dir: PathBuf,
    global_info: PilfflonkGlobalInfo,
    vkey: Vkey,
    /// Every AIR of the globalInfo, in canonical order: one.
    airs: Vec<PilfflonkInfo>,
}

impl ProvingKeyFiles {
    /// Reads the globalInfo, the vkey and each AIR's pilfflonkinfo of the `provingKey/` at `dir`,
    /// and refuses what [`ProvingKey::load`] refuses of them: a vkey whose digest is not that of its
    /// contents, other than one AIR, and a vkey and a pilfflonkinfo that do not describe the same
    /// AIR.
    pub fn read(dir: &Path) -> PilfflonkResult<Self> {
        timed("PILFFLONK_KEY_FILES", || {
            let global_info = PilfflonkGlobalInfo::from_proving_key(dir)?;
            let vkey_path = global_info.vkey_path(dir);
            let vkey = Vkey::read(&vkey_path)?;
            vkey.check_digest().map_err(|e| e.in_file(&vkey_path))?;
            let mut airs = Vec::new();
            for (airgroup_id, group) in global_info.airs.iter().enumerate() {
                for air_id in 0..group.len() {
                    let path = global_info.air_file(dir, airgroup_id as u64, air_id as u64, AirFile::PilfflonkInfo)?;
                    airs.push(PilfflonkInfo::read(&path)?);
                }
            }
            let [info] = airs.as_slice() else {
                return invalid!(
                    "the provingKey/ has {} AIRs, and a pilfflonk proof holds one instance of one AIR \
                     (pilfflonk/docs/README.md#scope)",
                    airs.len()
                );
            };
            check_vkey(&vkey, info, &global_info).map_err(|e| e.in_file(&vkey_path))?;
            Ok(Self { dir: dir.to_path_buf(), global_info, vkey, airs })
        })
    }

    /// The shape of the witness of a proof of this key, as [`ProvingKey::witness_shape`].
    pub fn witness_shape(&self) -> PilfflonkResult<WitnessShape> {
        witness_shape(&self.global_info, &self.airs)
    }

    /// The second half of [`ProvingKey::load_on`]: the C++ core loads the key on `device`, and its
    /// degrees, its SRS and the commitments of its fixed columns are checked against these files.
    pub fn load_on(self, device: Device) -> PilfflonkResult<ProvingKey> {
        let Self { dir, global_info, vkey, airs } = self;
        let [info] = airs.as_slice() else {
            return invalid!("the provingKey/ has {} AIRs, and read() accepts one", airs.len());
        };
        let ctx =
            PilFflonkProverCtx::load_on(&dir, device).map_err(native("loading the provingKey/ into the C++ prover"))?;
        let degrees = info.degrees()?;
        let n_bits_ext = ctx
            .n_bits_ext(info.airgroup_id, info.air_id)
            .map_err(native("reading the C++ prover's extended domain"))?;
        if n_bits_ext != degrees.n_bits_ext {
            return invalid!(
                "the C++ prover extends {} to 2^{n_bits_ext} points, and this crate to 2^{} \
                 (proofman_pilfflonk::degrees, pilfflonk/docs/protocol.md#degrees)",
                info.name,
                degrees.n_bits_ext
            );
        }
        check_srs_and_fixed(&ctx, &vkey, info, &global_info, &dir)?;
        Ok(ProvingKey { dir, global_info, vkey, airs, ctx })
    }
}

/// Runs `f` between the two lines the C++ core's timers log at -vv (TimerStart and
/// TimerStopAndLog, pil2-stark/src/utils/timer.hpp), at trace level: `--> NAME starting...` and
/// `<-- NAME done: <seconds> s`, so that the phases of the Rust side read as the C++ ones
/// (pilfflonk/docs/performance.md#method).
pub(crate) fn timed<T>(name: &str, f: impl FnOnce() -> T) -> T {
    tracing::trace!("--> {name} starting...");
    let start = Instant::now();
    let result = f();
    tracing::trace!("<-- {name} done: {:.6} s", start.elapsed().as_secs_f64());
    result
}

/// The one instance of a witness (pilfflonk/docs/README.md#scope), read: what its C++ instance is
/// made of.
pub(crate) struct WitnessInstance {
    pub(crate) air: AirInstanceRef,
    pub(crate) stage1: Stage1Witness,
    pub(crate) publics: Vec<FrBytes>,
    pub(crate) proof_values: Vec<FrBytes>,
}

impl WitnessInstance {
    /// Refuses a witness of other than one instance.
    pub(crate) fn read(witness: &impl WitnessSource) -> PilfflonkResult<Self> {
        let instances = witness.instances();
        let [air] = instances.as_slice() else {
            return invalid!(
                "the witness has {} instances, and a pilfflonk proof holds one (pilfflonk/docs/README.md#scope)",
                instances.len()
            );
        };
        Ok(Self {
            air: *air,
            stage1: witness.stage1(0)?,
            publics: witness.publics()?,
            proof_values: witness.proof_values()?,
        })
    }

    /// Its C++ instance, on `pk`, blinded as `insecure_blinding_seed` says ([`ProveOptions`]).
    pub(crate) fn instance<'pk>(
        &self,
        pk: &'pk ProvingKey,
        insecure_blinding_seed: Option<&[u8; 32]>,
    ) -> PilfflonkResult<PilFflonkInstance<'pk>> {
        PilFflonkInstance::new(
            &pk.ctx,
            &PilFflonkInstanceInputs {
                airgroup_id: self.air.airgroup_id,
                air_id: self.air.air_id,
                stage1: self.stage1.trace_bytes(),
                air_values: &le(self.stage1.air_values()),
                publics: &le(&self.publics),
                proof_values: &le(&self.proof_values),
                insecure_blinding_seed,
            },
        )
        .map_err(native("creating the instance"))
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

/// That the SRS and the `.const` the C++ core loaded are those the vkey was set up with:
/// `[τ]₂` of the SRS is the vkey's `X_2`, and the fixed columns commit to its fixed commitments. The
/// prover uses neither `X_2` nor those commitments, and the verifier takes both from the vkey: a
/// `provingKey/` whose files come from different setups would give proofs that do not verify, and
/// nothing would say why.
///
/// Always checked: the commitments cost one MSM per fixed f, of its `k·N` points, once per key
/// loaded, no more than the prover's own commitment of as many columns of stage 1.
fn check_srs_and_fixed(
    ctx: &PilFflonkProverCtx,
    vkey: &Vkey,
    info: &PilfflonkInfo,
    global_info: &PilfflonkGlobalInfo,
    dir: &Path,
) -> PilfflonkResult<()> {
    let tau_g2 = ctx.srs_tau_g2().map_err(native("reading [τ]₂ of the SRS"))?;
    if G2Affine::from_le_bytes(&tau_g2)? != vkey.x_2 {
        return Err(PilfflonkError::InvalidFormat(
            "its [τ]₂ is not the vkey's X_2: this SRS is not of the ptau the vkey was set up with".into(),
        )
        .in_file(&global_info.srs_path(dir)));
    }
    let committed = ctx
        .fixed_commitments(info.airgroup_id, info.air_id, vkey.fixed_commitments.0.len())
        .map_err(native("committing the fixed columns of the .const"))?;
    for (i, (point, expected)) in committed.iter().zip(&vkey.fixed_commitments.0).enumerate() {
        if G1Affine::from_le_bytes(point)? != *expected {
            let const_path = global_info.air_file(dir, info.airgroup_id, info.air_id, AirFile::Const)?;
            return Err(PilfflonkError::InvalidFormat(format!(
                "its fixed columns commit to another f{i} than the vkey's: this .const is not the one the vkey was \
                 set up with (or the SRS is not)"
            ))
            .in_file(&const_path));
        }
    }
    Ok(())
}

/// How to prove.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProveOptions {
    /// `None` (the default): the blinding is random, from the OS. `Some(seed)` fixes it
    /// (pilfflonk/docs/protocol.md#blinding): the same seed gives the same proof, and whoever knows
    /// it can remove the blinding. For tests and CI only: a proof made with it is not zero-knowledge.
    pub insecure_blinding_seed: Option<[u8; 32]>,
    /// How `Q` is evaluated on the extended coset of `2^nBitsExt` points
    /// (pilfflonk/docs/protocol.md#q-in-parts): in parts of `2^bits` points, one after another,
    /// `nBits <= bits <= nBitsExt`. `None` (the default) is `nBits`, one coset of `H` per part, the
    /// least memory; `nBitsExt` evaluates `Q` on the whole coset at once. The proof is the same bit
    /// for bit whatever the parts.
    pub q_part_bits: Option<u64>,
}

/// The challenges of a proof and `Q(ξ)`, for tests and diagnostics: the proof does not hold them,
/// the verifier replays them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofChallenges {
    /// `stages[s - 2]`: the challenges of stage `s`, for `2 ≤ s ≤ nStages`.
    pub stages: Vec<Vec<FrBytes>>,
    pub std_vc: FrBytes,
    pub xi_seed: FrBytes,
    /// `Q(ξ)` of the instance, the value the verifier computes from the evaluations
    /// (pilfflonk/docs/protocol.md#verifier-equations): if `Q` is split, `Σ_i ξ^(i·M·N)·Q_i(ξ)` of
    /// the proof's `Q_i(ξ)`, which the verifier checks it against.
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
    /// `proof.json`: the JSON view of the proof (pilfflonk/docs/formats.md#proof).
    pub fn proof_json(&self) -> PilfflonkResult<ProofJson> {
        self.proof.to_json(&self.names)
    }

    /// Writes `proof.json` and `publics.json` to `dir`, which is created if it does not exist.
    pub fn write(&self, dir: &Path) -> PilfflonkResult<()> {
        timed("PILFFLONK_WRITE_PROOF", || {
            fs::create_dir_all(dir).map_err(|source| PilfflonkError::Io { path: dir.to_path_buf(), source })?;
            self.proof_json()?.write(&dir.join(PROOF_FILE))?;
            self.publics.write(&dir.join(PUBLICS_FILE))
        })
    }
}

pub(crate) fn fr(bytes: [u8; 32]) -> PilfflonkResult<FrBytes> {
    FrBytes::from_le_bytes(bytes)
}

pub(crate) fn le(values: &[FrBytes]) -> Vec<[u8; 32]> {
    values.iter().map(FrBytes::to_le_bytes).collect()
}

fn g1(points: Vec<[u8; 64]>) -> PilfflonkResult<Vec<G1Affine>> {
    points.iter().map(G1Affine::from_le_bytes).collect()
}

/// The transcript (pilfflonk/docs/protocol.md#transcript), with its errors in this crate's terms.
pub(crate) struct Transcript(PilFflonkTranscript);

impl Transcript {
    pub(crate) fn new() -> PilfflonkResult<Self> {
        PilFflonkTranscript::new().map(Self).map_err(native("creating the transcript"))
    }

    pub(crate) fn scalars(&mut self, values: &[FrBytes]) -> PilfflonkResult<()> {
        self.0.absorb_fr(&le(values)).map_err(native("absorbing scalars into the transcript"))
    }

    pub(crate) fn points(&mut self, points: &[G1Affine]) -> PilfflonkResult<()> {
        let bytes: Vec<[u8; 64]> = points.iter().map(G1Affine::to_le_bytes).collect();
        self.0.absorb_g1(&bytes).map_err(native("absorbing commitments into the transcript"))
    }

    pub(crate) fn squeeze(&mut self) -> PilfflonkResult<FrBytes> {
        fr(self.0.squeeze().map_err(native("squeezing the transcript"))?)
    }
}

/// What steps 1 and 2 of the transcript leave (`commit_stages`).
struct CommittedStages {
    /// The transcript, which has absorbed everything up to the last stage's commitments and values.
    transcript: Transcript,
    /// The commitments of the non-fixed f of stages `1 … nStages`, in the order of the layout.
    commitments: Vec<G1Affine>,
    /// `challenges[s - 2]`: the challenges of stage `s`, for `2 ≤ s ≤ nStages`.
    challenges: Vec<Vec<FrBytes>>,
}

/// Steps 1 and 2 of the transcript (see [the module](self)) on `instance`, which is `witness`'s: a
/// new transcript absorbs the digest, the number of instances of each AIR and the publics; then
/// each stage `s` is committed, with the challenges the transcript gave for it (none for stage 1),
/// and the transcript absorbs its commitments and its values, and squeezes the `numChallenges[s]`
/// challenges of stage `s + 1` if `s < nStages`.
fn commit_stages(
    pk: &ProvingKey,
    witness: &WitnessInstance,
    instance: &mut PilFflonkInstance<'_>,
) -> PilfflonkResult<CommittedStages> {
    let air = witness.air;
    let info = pk.air(air)?;
    let global_info = pk.global_info();
    let n_stages = info.n_stages;
    let n_f = |stage: u64| info.layout.0.iter().filter(|f| f.stage == stage).count();

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
    transcript.scalars(&witness.publics)?;

    // Step 2: the stages. The air values, airgroup values and proof values of stage s follow its
    // commitments; v1 has none but the stage-1 ones the witness gives (none either).
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
            transcript.scalars(witness.stage1.air_values())?;
            transcript.scalars(&witness.proof_values)?;
        }
        if stage < n_stages {
            let count = global_info.num_challenges.get(stage as usize).copied().unwrap_or(0);
            challenges = (0..count).map(|_| transcript.squeeze()).collect::<PilfflonkResult<_>>()?;
            stage_challenges.push(challenges.clone());
        }
    }
    Ok(CommittedStages { transcript, commitments, challenges: stage_challenges })
}

/// The columns of every stage of an instance as the prover computes them, and the challenges it
/// computes them with ([`stage_columns`]; `check::check_columns` for the check's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageColumns {
    /// `challenges[s - 2]`: the challenges of stage `s`, for `2 ≤ s ≤ nStages`, as
    /// [`ProofChallenges::stages`].
    pub challenges: Vec<Vec<FrBytes>>,
    /// `columns[s - 1][p]`: the column of stage `s` at `stagePos` `p` of `cmPolsMap`, row by row: the
    /// witness's, a prover hint's or an im pol's.
    pub columns: Vec<Vec<Vec<FrBytes>>>,
}

/// The columns of every stage `1 … nStages` of the one instance of `witness`, as the prover computes
/// them before it commits `Q`: steps 1 and 2 of [`prove`] with the same `options`, and so, with the
/// same blinding seed, the same challenges and columns as its proof. For tests and diagnostics (the
/// oracle checks the prover hints' columns against them); a proof holds none of it.
pub fn stage_columns(
    pk: &ProvingKey,
    witness: &impl WitnessSource,
    options: &ProveOptions,
) -> PilfflonkResult<StageColumns> {
    let read = WitnessInstance::read(witness)?;
    let info = pk.air(read.air)?;
    let mut instance = read.instance(pk, options.insecure_blinding_seed.as_ref())?;
    let committed = commit_stages(pk, &read, &mut instance)?;
    let n_rows = 1usize << info.n_bits;
    let mut columns = Vec::new();
    for stage in 1..=info.n_stages {
        let width = info.map_sections_n.get(&format!("cm{stage}")).copied().unwrap_or(0);
        let stage_u32 = u32::try_from(stage).map_err(|_| PilfflonkError::InvalidFormat("too many stages".into()))?;
        let stage_columns = (0..width)
            .map(|p| {
                let values = instance.column(stage_u32, p, n_rows).map_err(native("reading a column"))?;
                values.into_iter().map(fr).collect::<PilfflonkResult<Vec<_>>>()
            })
            .collect::<PilfflonkResult<_>>()?;
        columns.push(stage_columns);
    }
    Ok(StageColumns { challenges: committed.challenges, columns })
}

/// A proof of the one instance of `witness` (see [the module](self)).
pub fn prove(pk: &ProvingKey, witness: &impl WitnessSource, options: &ProveOptions) -> PilfflonkResult<ProofOutput> {
    tracing::info!("··· Reading the witness");
    let read = WitnessInstance::read(witness)?;
    let info = pk.air(read.air)?;
    let global_info = pk.global_info();
    let names = ProofNames::new(global_info, &[info])?;
    let n_f = |stage: u64| info.layout.0.iter().filter(|f| f.stage == stage).count();

    let mut instance = read.instance(pk, options.insecure_blinding_seed.as_ref())?;
    if let Some(bits) = options.q_part_bits {
        instance.set_q_part_bits(bits).map_err(native("choosing the parts Q is evaluated in"))?;
    }
    // Steps 1 and 2.
    let CommittedStages { mut transcript, mut commitments, challenges: stage_challenges } =
        commit_stages(pk, &read, &mut instance)?;
    let WitnessInstance { stage1, publics, proof_values, .. } = read;

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
