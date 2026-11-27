# `proofman-setup`

CLI for the PIL2 STARK proving-key setup pipeline. Compiles a `.pilout` into the
artifacts the prover needs at runtime: stark info, expressions, verifier code,
const trees, recursive aggregation circuits, and witness libraries.

```
proofman-setup <subcommand> [options]
```

## Subcommands at a glance

| Subcommand | Purpose | Inputs | Outputs |
|---|---|---|---|
| [`setup`](#setup) | Run the main setup pipeline (per-AIR + optional recursive) | `.pilout`, fixed cols, optional starkstructs | `provingKey/` populated end-to-end |
| [`stats`](#stats) | Per-AIR statistics report | `.pilout` | text stats file |
| [`setup-snark`](#setup-snark) | Final SNARK setup on top of `vadcop_final` | `provingKey/` from a prior `setup --recursive` | `recursivef`, `final` SNARK artifacts |
| [`setup-recursive-test`](#setup-recursive-test) | Setup a single user-provided recursive circuit | a `.circom` file | per-circuit setup artifacts |
| [`rebuild-witness-libs`](#rebuild-witness-libs) | Rebuild every witness `.so`/`.dylib` from a `provingKey/` directory | existing `provingKey/` | refreshed witness libraries in-place |
| [`setup-pilfflonk`](../../pilfflonk/docs/README.md#setup-pilfflonk) | Set up a BN128 pilout for the pilfflonk backend (one AIR, one instance) | `.pilout` compiled over BN128, `.ptau` | pilfflonk `provingKey/` |
| [`pilfflonk-solidity`](../../pilfflonk/docs/README.md#pilfflonk-solidity) | Write the Solidity verifier of an existing pilfflonk vkey | `pilfflonk.vkey.json` | `pilfflonk.verifier.sol` |

## `setup`

Run the full per-AIR setup, and (with `--recursive`) the compressor / recursive1
/ recursive2 / vadcop_final / vadcop_final_compressed pipeline.

```bash
proofman-setup setup \
    --airout pil/zisk.pilout \
    --build-dir build \
    [--fixed-dir <dir>] \
    [--stark-structs starkstructs.json] \
    [--hash blake3] \
    [--recursive] \
    [--recursive-jobs N] [--setup-jobs N] \
    [--output stats.txt]
```

| Flag | Purpose |
|---|---|
| `-a, --airout` | Compiled `.pilout` file |
| `-b, --build-dir` | Output directory; will contain `provingKey/`, `circom/`, `build/`, `pil/` |
| `-u, --fixed-dir` | Directory with `.fixed` columns (optional; falls back to inline pilout values) |
| `-s, --stark-structs` | `starkstructs.json` overriding default stark settings per AIR |
| `--hash` | Hash family of the key: `blake3` (default), `Poseidon1` or `Poseidon2`. [`setup-snark`](#setup-snark) wraps a blake3 key in pilfflonk only |
| `-r, --recursive` | Also run compressor → recursive1 → recursive2 → vadcop_final → vadcop_final_compressed |
| `--recursive-jobs` | Parallel recursive1 air pipelines (env: `RECURSIVE_JOBS`, default 1). Sized by RAM |
| `--setup-jobs` | Parallel AIRs for the non-recursive setup (env: `SETUP_JOBS`, default 1) |
| `-o, --output` | Optional path to write per-AIR stats |

Without `--recursive`, only the per-AIR artifacts are produced
(`<airgroup>/airs/<air>/air/<air>.{starkinfo,verifierinfo,expressionsinfo,verkey}.json`,
plus `.const`, `.verkey.bin` and `.bin`).

With `--recursive`, the recursive layers are added under
`<airgroup>/airs/<air>/{compressor,recursive1}/`,
`<airgroup>/recursive2/`, `vadcop_final/`, and `vadcop_final_compressed/` —
each with its own const tree, witness `.so`/`.dylib`, and verifier artifacts.

### Where each AIR's info comes from

Every STARK circuit (basic AIRs, compressor, recursive1/2, final, recursivef, the recursive test)
goes through one entry, `pil::stark_setup::stark_setup`, over the thin core in `pil-info`:

| Module | Rule |
|---|---|
| `core/from_pilout` | strict pilout -> typed `ConstraintSystem` |
| `search/im` (untrusted) -> `core/im` | picks the im columns; `check_im` derives `qDeg` |
| `core/dag` | the constraint DAG (`constraintDag`) and its code lowering |
| `core/layout`, `core/openings`, `core/project` | columns, evMap, the starkinfo projection |
| `emit/prover` | today's arena and generator: expressionsinfo, hints (pilfflonk's `qVerifier` from the DAG) |
| `output/stark_info` | starkinfo JSON from the projection and the solved low-degree test |
| `output/dag_eval` | the DAG's resolver context (`lower_ctx`) and a Goldilocks evaluator for tests |

pilfflonk runs the same core (`pil_info::shplonk_info`, SHPLONK).

## `stats`

Print per-AIR statistics: constraints, intermediate polynomials per stage,
column counts, etc. Useful for sizing the build before running it.

```bash
proofman-setup stats \
    --airout pil/zisk.pilout \
    [--output stats.txt] \
    [--starkstructs starkstructs.json] \
    [--airgroups Group1 Group2 ...] \
    [--airs Air1 Air2 ...] \
    [--impols]
```

| Flag | Purpose |
|---|---|
| `-a, --airout` | Compiled `.pilout` file |
| `-o, --output` | Output file (default: `tmp/stats.txt`) |
| `-s, --starkstructs` | Same starkstructs.json used by `setup`, for accurate sizing |
| `-g, --airgroups` | Filter to specific airgroup names |
| `-i, --airs` | Filter to specific air names |
| `-m, --impols` | Include intermediate polynomial details per stage |

Read-only — does not write to `provingKey/`.

## `setup-snark`

Continues from a `provingKey/` produced by `setup --recursive`, generating the
final SNARK layer (`recursivef` + `pilfflonk` final by default, or `plonk` or
`fflonk`).

```bash
proofman-setup setup-snark \
    --build-dir build \
    [--powers-of-tau ptau.ptau] \
    [--final-snark pilfflonk] \
    [--publics-info publics.json] \
    [--only-recursive-final]
```

| Flag | Purpose |
|---|---|
| `-b, --build-dir` | Build directory containing `provingKey/` from a prior `setup --recursive` |
| `--powers-of-tau` | `.ptau` file consumed by the final SNARK's setup |
| `--final-snark` | `pilfflonk` (default), `plonk` or `fflonk`. PLONK and FFLONK take a key of a poseidon family only (`setup --hash Poseidon1` or `Poseidon2`) |
| `--publics-info` | Optional JSON describing the publics-hash layout |
| `--only-recursive-final` | Stop after `recursivef`; skip the final SNARK step |

Requires `vadcop_final/vadcop_final.{starkinfo,verifierinfo,verkey}.json` to
already exist.

The `recursivef`, in `provingKeySnark/recursivef/`, depends on the key's hash
family. A poseidon key's is a BN128 STARK that `setup-snark` sets up, for any final
SNARK. A blake3 key's, the default, is one more Goldilocks STARK after the
vadcop_final, whose verifier only the pilfflonk wrap has:
`setup-snark` reuses it if there is one, and otherwise sets it up first, as
`setup-recursivef -b <build dir>` does with its defaults. Run `setup-recursivef`
first for another geometry (`--blowup`, `--lanes`, `--grinding`, ...).

With `pilfflonk`, the final circuit is laid out as a PIL2 AIR over BN128 by
plonk2pil, compiled with `--field bn128` (by a pil2-compiler that has `--field`, as
this crate's does, see
[pilfflonk/docs/README.md](../../pilfflonk/docs/README.md#compile-pil)), and set
up with `setup-pilfflonk`'s steps at the wrap family's knobs (plonk2pil's
`wrap::MAX_CONSTRAINT_DEGREE` and `wrap::EXTRA_MULS`): `provingKeySnark/final/`
gets the circuit's witness calculator and `.exec`, the pilfflonk `provingKey/`
and the Solidity verifiers. The ptau needs at least as many powers `[τ^i]₁` as
the layout's largest degree, which the setup logs with the SRS it writes
(`pilfflonk.srs.bin`). fibonacci-square's wrap is an AIR of `N = 2^19` rows
either way: its blake3 key's needs 15,204,438 powers (the official ptau of power
23), and its Poseidon2 key's 6,815,756, `13·N + 12` with range checks (power
22). `proofman-cli prove-snark` and `verify-snark` prove and verify with that key
([pilfflonk/docs/README.md](../../pilfflonk/docs/README.md#prove-snark-and-verify-snark)).

## `setup-recursive-test`

Run a single recursive setup over a user-provided `.circom` file. Used by CI to
exercise compressor / aggregation / final-vadcop / light templates without
booting the full pipeline.

```bash
proofman-setup setup-recursive-test \
    --build-dir build \
    --circom path/to/test.circom \
    --name test \
    [--type aggregation]
```

| Flag | Purpose |
|---|---|
| `-b, --build-dir` | Build directory |
| `-c, --circom` | Path to the `.circom` source file |
| `-n, --name` | Circuit name (used as the artifact basename) |
| `-t, --type` | One of `compressor`, `aggregation` (default) |

## `rebuild-witness-libs`

Rebuild **only** the witness libraries (`.so` / `.dylib`) inside a
`provingKey/` directory, without re-running the expensive setup steps
(`pil_info`, `plonk2pil`, const-tree, etc.).

```bash
proofman-setup rebuild-witness-libs --proving-key build/provingKey
```

The command takes the `provingKey/` path directly — no other folders need to
exist alongside it. Useful when you've downloaded or copied just the proving
key and want fresh witness libraries (e.g. macOS `.dylib`s on top of a
Linux-built proving key).

| Flag | Purpose |
|---|---|
| `-p, --proving-key` | Path to the `provingKey/` directory |
| `-s, --proving-key-snark` | Optional `provingKeySnark/` directory; its `recursivef` and `final` libraries are rebuilt too |
| `-j, --jobs` | Libraries to build at once (env: `REBUILD_JOBS`, default: the CPU count). Each `make` is g++-bound, ~1–2 GB RAM |

### What it does

For every circuit under `provingKey/` (and `provingKeySnark/`) whose setup stored the circom C++
(`<circuit>.cpp`), it rebuilds the witness library with the same `make` recipe the setup uses; no
circom runs, and no other file is touched. A library without its `.cpp` is an error (the key
predates stored C++: rerun the setup). `-j` builds that many libraries at once, with
`[i/N]`-prefixed logs.

### Output extension is host-OS controlled

| Host | Extension produced |
|---|---|
| Linux | `<circuit>.so` |
| macOS | `<circuit>.dylib` |

Output goes **into the same per-circuit `provingKey/build/...` directories** as
the existing libraries. Running this on a Mac against a Linux-built proving
key deposits a `.dylib` next to the existing `.so` (the prior `.so` is left
alone). Running it on Linux when a `.so` is already present overwrites that
`.so`.

### When to use it

- You want macOS-compatible witness libs from a proving key originally built on
  Linux — run this on a Mac (or in a `macos-latest` CI runner) against the
  proving key, and you get `.dylib`s in place.
- You're iterating on circom templates and want to rebuild `.so`s without
  rerunning the full setup.

## Proving-key pipeline (module map)

Every proving-key circuit is a list of steps run by one executor
(`src/proving_key/pipeline/`): `mod.rs` holds `PipelineCtx` and `run`, `steps.rs` the shared
steps, `kinds.rs` one pure builder per kind. The `gen_*` entry points fill the context and run
their kind's list.

| Kind (entry) | Steps |
|---|---|
| basic (`commands/setup.rs`) | WriteFixed{UserFixed\|Pilout} -> StarkTail -> ConstTree{optional} |
| recursive test (`recursive_test.rs`) | CircomPrologue{raw circom, O2} -> WitnessLib -> Plonk2Pil -> WriteFixed{Plonk} -> CompilePil -> StarkTail -> ConstTree -> *witness_wait* -> *global_info* |
| compressor / recursive1 / recursive2 (`recursive.rs`) | [*batch_fit* (r1) \| *arity_fit* (r2)] -> CircomPrologue{O1} -> Plonk2Pil -> [*batch_retry* (r1)] -> *compressor_check* -> WitnessLib -> WriteFixed -> CompilePil -> StarkTail{reuse; r1: starkinfo only} -> ConstTree -> [*drop_starkinfo* (r1) \| *vks_json* + WriteVerifierRs (r2)] |
| vadcop_final (`final_setup.rs`) | CircomPrologue{O1} -> WitnessLib -> Plonk2Pil -> *final_pin* -> WriteFixed -> CompilePil -> StarkTail -> ConstTree -> WriteVerifierRs -> *witness_wait* |
| compressed final (`compressed_final.rs`) | CircomPrologue{O2} -> WitnessLib -> Plonk2Pil -> WriteFixed -> CompilePil -> *compressed_settings* -> StarkTail -> ConstTree -> WriteVerifierRs -> *witness_wait* |
| recursivef (`snark_setup.rs::gen_recursivef_setup`) | CircomPrologue{O2} -> WitnessLib -> Plonk2Pil -> WriteFixed -> CompilePil -> StarkTail -> ConstTree |

*Italic* steps are `Custom` (kind-specific control flow, named in the list). The final SNARK
back-end (`snark_setup.rs`: rapidsnark / pilfflonk keys, BN128 final circuit) is not a pipeline
kind; `gen_snark_setup` calls it after the recursivef.

## Environment variables

These override the auto-detected tooling paths used by `setup`, `setup-snark`,
`setup-recursive-test`, and `rebuild-witness-libs`. All have sensible defaults
that resolve relative to the repo or the executable's parent directories.

| Variable | Default | What it points to |
|---|---|---|
| `CIRCUITS_GL_PATH` | `setup/stark-recurser/stark2circom/circom_verifier/circuits.gl` | Goldilocks circom helpers |
| `RECURSER_CIRCUITS_PATH` | `setup/stark-recurser/stark2circom/circom_verifier/helper_circuits` | vadcop circom templates |
| `RECURSER_CIRCUITS_COMPRESSED_FINAL_PATH` | `setup/stark-recurser/stark2circom/circom_verifier/helper_circuits` | `vadcop_final_compressed` circom helpers (goldilocks-side) |
| `CIRCUITS_BN128_PATH` | `setup/stark-recurser/stark2circom/circom_verifier/circuits.bn128` | BN128 circom helpers (snark setup) |
| `STD_PIL_PATH` | `pil2-components/lib/std/pil` | PIL standard library |
| `RECURSER_PIL_PATH` | `setup/stark-recurser/plonk2pil/pil` | `plonk2pil` PIL templates |
| `CIRCOM_HELPERS_DIR` | `setup/circom` | C++ witness helpers + Makefile |
| `FINAL_SNARK_CIRCOM_HELPERS_DIR` | `setup/final_snark_circom` | C++ witness helpers for the final SNARK |
| `GOLDILOCKS_SRC_DIR` | `pil2-stark/src/goldilocks/src` | Goldilocks C++ sources copied into the witness build |
| `RECURSIVE_JOBS` | `1` | Parallelism for recursive1 air pipelines |
| `SETUP_JOBS` | `1` | Parallelism for non-recursive AIR setup |
| `PIL2C_EXEC` | `pil2com` from `node_modules/.bin`, or on `PATH` | The pil2-compiler's `pil.js`. Compiling over BN128 (`compile-pil --field bn128`, `setup-snark` with pilfflonk, its default) needs one that has `--field`, as the branch `develop-0.15.0` that `package.json` pins does ([pilfflonk/docs/README.md](../../pilfflonk/docs/README.md#compile-pil)) |
| `REBUILD_JOBS` | `1` | Parallelism for `rebuild-witness-libs` circom compiles |

Path resolution checks (in order): the env var, the path relative to CWD, then
walks up from the executable's directory looking for the same suffix. Set the
env var explicitly when running outside the repo.

## Typical end-to-end flow

```bash
# 1. Per-AIR + recursive setup
proofman-setup setup \
    -a pil/zisk.pilout -b build \
    -s starkstructs.json -r --recursive-jobs 4 --setup-jobs 4

# 2. Final SNARK on top: by default the recursivef of the blake3 key and the pilfflonk wrap
proofman-setup setup-snark -b build --powers-of-tau ptau/pot24.ptau

# 3. (Cross-platform) regenerate witness libs for the other OS
#    Run this on a Mac to get .dylibs into the same provingKey:
proofman-setup rebuild-witness-libs -p build/provingKey
```
