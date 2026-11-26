//! One variant per shared step; parameters live in the variant, behaviour in `run`.
//! The step bodies live in `circom`, `plonk` and `stark`.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use pil2_stark_recurser::plonk2pil::{r1cs::types::PlonkOptions, SetupKind};
use pil2_stark_recurser::stark2circom::{
    gen_stark_verifier, gen_stark_verifier_bn128_blake3, CircuitTemplate, KeyVerifier, StarkVerifierOptions,
};
use serde_json::Value;

use super::PipelineCtx;
use super::{circom, plonk, stark};

pub(crate) use plonk::{check_plonk_result, compile_plonk_pil};

/// circom optimisation level, a per-kind builder parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircomOpt {
    O1,
    O2,
}

impl CircomOpt {
    pub(super) fn flag(self) -> &'static str {
        match self {
            CircomOpt::O1 => "--O1",
            CircomOpt::O2 => "--O2",
        }
    }
}

/// Which verifier circom generator feeds the prologue.
#[derive(Debug, Clone)]
pub enum VerifierGen {
    /// `gen_stark_verifier`: Goldilocks verifier.
    Gl(StarkVerifierOptions),
    /// `gen_stark_verifier_bn128_blake3`: the final SNARK's verifier of a blake3 recursivef.
    Bn128Blake3(StarkVerifierOptions),
}

impl VerifierGen {
    pub fn opts_mut(&mut self) -> &mut StarkVerifierOptions {
        match self {
            VerifierGen::Gl(o) | VerifierGen::Bn128Blake3(o) => o,
        }
    }

    /// The verifier of a proof of (`stark_info`, `verifier_info`) under `const_root`: its circom and its data.
    pub fn generate(&self, const_root: &[String; 4], stark_info: &Value, verifier_info: &Value) -> Result<KeyVerifier> {
        match self {
            VerifierGen::Gl(o) => {
                let root = (!o.verkey_input).then_some(const_root);
                gen_stark_verifier(root, stark_info, verifier_info, o).context("gen_stark_verifier failed")
            }
            VerifierGen::Bn128Blake3(o) => {
                let root = (!o.verkey_input).then_some(const_root);
                gen_stark_verifier_bn128_blake3(root, stark_info, verifier_info, o)
                    .context("gen_stark_verifier_bn128_blake3 failed")
            }
        }
    }
}

/// Where the `.const` file of the STARK tail comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstSource {
    /// plonk2pil's columns; `WriteFixed` writes fixed.bin/.pil/.exec, the tail the `.const`.
    Plonk,
    /// The pilout's inline fixed columns.
    Pilout,
    /// `<fixed_dir>/<name>.fixed` when present, else as `Pilout`.
    UserFixed,
}

pub type CustomFn = Box<dyn Fn(&mut PipelineCtx) -> Result<()>>;

/// The field circom compiles over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prime {
    Goldilocks,
    /// The final SNARK's circuit, which also gets its `--inspect` checks and a wasm witness.
    Bn128,
}

/// One step of a proving-key pipeline, in the order the kinds run them (`kinds.rs`):
///
/// - `CircomPrologue`: writes the verifier circoms and the circuit's main (no template: compiles
///   `ctx.circom_file` as is), compiles them with circom, copies the `.dat`.
/// - `WitnessLib`: builds the circuit's witness library in the background.
/// - `Plonk2Pil`: the r1cs as plonk rows, fixed columns and PIL (the generator; untrusted).
/// - `WriteFixed`: writes those fixed columns, PIL and exec.
/// - `CompilePil`: pil2com, PIL to pilout.
/// - `CheckPlonk`: the trusted checker, r1cs ⇔ plonk ⇔ fixed ⇔ PIL; a mismatch fails the setup.
/// - `StarkTail`: pil_info (or `ctx.reuse`'s), starkinfo/verifierinfo/expressionsinfo, `.bin`, the
///   plonk `.const`. `starkinfo_only` (recursive1): just the starkinfo the const tree reads and removes.
/// - `ConstTree`: `.const` → const tree → verkey; a missing `.const` fails when `required`.
/// - `WriteVerifierRs`: the Rust verifier.
/// - `WitnessWait`: waits for the background witness library builds.
/// - `Custom`: a kind's own step.
pub enum Step {
    CircomPrologue {
        verifier: Option<VerifierGen>,
        circom_template: Option<CircuitTemplate>,
        opt: CircomOpt,
        prime: Prime,
    },
    WitnessLib,
    Plonk2Pil {
        kind: SetupKind,
        opts: PlonkOptions,
    },
    WriteFixed {
        source: ConstSource,
    },
    CompilePil,
    CheckPlonk,
    StarkTail {
        reuse: bool,
        starkinfo_only: bool,
    },
    ConstTree {
        required: bool,
    },
    WriteVerifierRs,
    WitnessWait,
    Custom {
        name: &'static str,
        run: CustomFn,
    },
}

impl Step {
    pub fn run(&self, ctx: &mut PipelineCtx) -> Result<()> {
        match self {
            Step::CircomPrologue { verifier, circom_template, opt, prime } => {
                circom::circom_prologue(ctx, verifier.as_ref(), *circom_template, *opt, *prime)
            }
            Step::WitnessLib => circom::witness_lib(ctx),
            Step::Plonk2Pil { kind, opts } => plonk::plonk2pil_step(ctx, *kind, opts),
            Step::WriteFixed { source: ConstSource::Plonk } => plonk::write_plonk_fixed(ctx),
            Step::CheckPlonk => plonk::check_plonk(ctx),
            Step::WriteFixed { source: ConstSource::Pilout } => plonk::write_basic_const(ctx, None),
            Step::WriteFixed { source: ConstSource::UserFixed } => plonk::write_user_fixed_const(ctx),
            Step::CompilePil => plonk::compile_pil_step(ctx),
            Step::StarkTail { reuse, starkinfo_only } => stark::stark_tail(ctx, *reuse, *starkinfo_only),
            Step::ConstTree { required } => stark::const_tree(ctx, *required),
            Step::WriteVerifierRs => stark::write_verifier_rs(ctx),
            Step::WitnessWait => ctx.witness.as_ref().map_or(Ok(()), |w| w.await_all()),
            Step::Custom { run, .. } => run(ctx),
        }
    }
}

pub(super) fn path_str(p: &Path) -> Result<&str> {
    p.to_str().with_context(|| format!("non-UTF-8 path {}", p.display()))
}

/// `words` little-endian, as the exec and verkey.bin files hold them.
pub(crate) fn write_exec(path: &Path, words: &[u64]) -> Result<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}
