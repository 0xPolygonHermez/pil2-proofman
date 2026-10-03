# Golden STARK setup

`manifest.sha256` freezes what the non-recursive STARK setup
(`proofman-setup setup`) produces for the 10 programs CI builds:
`examples/fibonacci-square` and the nine `pil2-components/test` programs of the
`test-std` job. A refactor of the setup passes is correct only if it leaves all
of it byte-identical, and `check.sh` is how to show that.

`plonk2pil/` holds a second golden, of what plonk2pil returns for the recursive
test circuits, with its own manifest and scripts (see its README). Both use the
helpers of `lib.sh`.

## Usage

```bash
(cd setup/pil2-stark && npm install)   # once: the pinned pil2-compiler
setup/golden/check.sh                  # build, regenerate, compare
```

`check.sh` builds `proofman-setup`, runs `generate.sh` into `target/golden-setup`
and compares every file with the manifest. It exits non-zero, naming each file,
when one is:

- `MISSING`: in the manifest but not produced;
- `DIFFERS`: produced with a different sha256;
- `UNEXPECTED`: produced but not in the manifest.

`check.sh <dir>` compares an existing `generate.sh` output directory instead of
regenerating. `generate.sh [dir]` only regenerates; it wipes `dir` first (and
refuses to wipe a non-empty directory it did not create) and writes
`dir/manifest.sha256`. Both honour:

| Variable | Effect |
|---|---|
| `GOLDEN_OUT` | Output directory (default `target/golden-setup`) |
| `PROOFMAN_SETUP` | Use this `proofman-setup` binary instead of building one |
| `SETUP_JOBS` | Passed to `proofman-setup setup`; the outputs do not depend on it |

A full run takes about 30 s on a large machine, plus the first build. The build
is CPU-only (`proofman-starks-lib-c/cpu-only`): the setup runs on the CPU, and
this keeps CUDA and the GPU submodules out of the way on machines with `nvcc`.

## What is covered

For each program, everything under `target/golden-setup/programs/<name>/`:

- inputs: the `.pilout` and, for `fibonacci-square`, `fixed/*.fixed`;
- per air: `starkinfo.json`, `expressionsinfo.json`, `verifierinfo.json`,
  `.bin`, `.verifier.bin`, `.const`, `.verkey.json` and `.verkey.bin`;
- per program: `pilout.globalInfo.json` and `pilout.globalConstraints.{json,bin}`.

The commands are those of `.github/workflows/ci.yaml`, with one exception:
`fibonacci-square` is set up without `-r`. The recursive setup needs circom and
is slow, so it is out of scope, and this program's `pilout.globalInfo.json` is
the non-recursive one.

## When it fails

- A `.pilout` or `.fixed` entry differs: the input changed (the `.pil` sources
  or pil2-compiler), and the setup outputs that differ with it say nothing about
  the setup. `package.json` pins pil2-compiler to a branch, not a commit, so a
  new commit there can change the `.pilout`s. `generate.sh` prints the commit it
  used, and overrides any `PIL2C_EXEC` in the environment.
- Only setup outputs differ: the setup's behaviour changed. The CI job
  `golden-setup` uploads the outputs as an artifact to diff against a local run.

Update the manifest only when a change of output is intended, and say why in the
pull request:

```bash
setup/golden/generate.sh && cp target/golden-setup/manifest.sha256 setup/golden/
```

## Baseline

Generated at `ff0ff959` with pil2-compiler `503862ca` (the head of
`develop-0.14.0` on 2026-09-29). The manifest came out identical with
`SETUP_JOBS=1`, 4 (the default) and 64, and with Node 18 (as in CI) and Node 22.
