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

## `setup`

Run the full per-AIR setup, and (with `--recursive`) the compressor / recursive1
/ recursive2 / vadcop_final / vadcop_final_compressed pipeline.

```bash
proofman-setup setup \
    --airout pil/zisk.pilout \
    --build-dir build \
    [--fixed-dir <dir>] \
    [--stark-structs starkstructs.json] \
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
| `-r, --recursive` | Also run compressor → recursive1 → recursive2 → vadcop_final → vadcop_final_compressed |
| `--recursive-jobs` | Parallel recursive1 air pipelines (env: `RECURSIVE_JOBS`, default 1). Sized by RAM |
| `--setup-jobs` | Parallel AIRs for the non-recursive setup (env: `SETUP_JOBS`, default 1) |
| `-o, --output` | Optional path to write per-AIR stats |

Without `--recursive`, only the per-AIR artifacts are produced
(`<airgroup>/airs/<air>/air/<air>.{starkinfo,verifierinfo,expressionsinfo,verkey}.json`,
plus `.const`, `.consttree`, `.bin`, `.verifier.bin`, `.verifier.rs`).

With `--recursive`, the recursive layers are added under
`<airgroup>/airs/<air>/{compressor,recursive1}/`,
`<airgroup>/recursive2/`, `vadcop_final/`, and `vadcop_final_compressed/` —
each with its own const tree, witness `.so`/`.dylib`, and verifier artifacts.

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
final SNARK layer (`recursivef` + `fflonk` or `plonk` final).

```bash
proofman-setup setup-snark \
    --build-dir build \
    [--powers-of-tau ptau.ptau] \
    [--final-snark fflonk] \
    [--publics-info publics.json] \
    [--only-recursive-final]
```

| Flag | Purpose |
|---|---|
| `-b, --build-dir` | Build directory containing `provingKey/` from a prior `setup --recursive` |
| `--powers-of-tau` | `.ptau` file consumed by the snarkjs setup |
| `--final-snark` | `fflonk` (default) or `plonk` |
| `--publics-info` | Optional JSON describing the publics-hash layout |
| `--only-recursive-final` | Stop after `recursivef`; skip the final SNARK step |

Requires `vadcop_final/vadcop_final.{starkinfo,verifierinfo,verkey}.json` to
already exist.

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
| `-b, --build-dir` | Optional directory for intermediate `.circom`/`.cpp` files. Defaults to a tempdir that is removed when the command finishes |
| `-j, --jobs` | Number of circom compiles to run in parallel (env: `REBUILD_JOBS`, default 1 = serial). Each circom invocation is single-threaded but RAM-hungry — size by available memory rather than CPU count. ~10–20 GB peak per large circuit |

### What it does

For every circuit in `provingKey/<global_name>/...` that already has a witness
library (`compressor`, `recursive1`, `recursive2`, `vadcop_final`,
`vadcop_final_compressed`):

1. Loads the persisted `*.starkinfo.json` / `*.verifierinfo.json` /
   `*.verkey.json` from the proving key.
2. Regenerates the verifier circom via `pil2circom` and the wrapper circom
   via `gen_circom`.
3. Runs `circom <name>.circom --c -O2 --prime goldilocks` to emit C++.
4. Compiles the C++ into a shared library via the same `make witness` recipe
   the original setup uses.

No `.dat`, `.const`, `.consttree`, `.exec`, `.bin`, `.verifier.rs` files are
touched.

### Progress logging

The command prints a discovery summary, then `[i/N]` progress lines for each
circuit. Sample output for fibonacci-square (with `--jobs 1`):

```
Discovered 6 witness library(ies) to rebuild:
  [1/6] compressor / FiboCPU.FibonacciSquare
  [2/6] recursive1 / FiboCPU.FibonacciSquare (with compressor)
  [3/6] recursive1 / FiboCPU.Module
  [4/6] recursive1 / FiboCPU.SpecifiedRanges
  [5/6] recursive2 / FiboCPU
  [6/6] vadcop_final
Running circom compiles with 1 job(s) in parallel
[1/6] compressor / FiboCPU.FibonacciSquare: loading starkinfo / verifierinfo / verkey
[1/6] compressor / FiboCPU.FibonacciSquare: generating circom sources (pil2circom + gen_circom)
[1/6] compressor / FiboCPU.FibonacciSquare: invoking circom (this is the slow step)
[1/6] compressor / FiboCPU.FibonacciSquare: circom done in 28.4s
[1/6] compressor / FiboCPU.FibonacciSquare: spawning witness library build (.so/.dylib)
...
All circom compiles complete; 6 witness library build(s) running in background
```

With `--jobs > 1`, lines from different circuits will be interleaved — the
`[i/N]` prefix lets you trace which one each log belongs to.

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

# 2. Final SNARK on top
proofman-setup setup-snark -b build --powers-of-tau ptau/pot24.ptau

# 3. (Cross-platform) regenerate witness libs for the other OS
#    Run this on a Mac to get .dylibs into the same provingKey:
proofman-setup rebuild-witness-libs -p build/provingKey
```
