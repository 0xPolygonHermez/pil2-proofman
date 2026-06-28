# GPU Q-Expression Codegen

Generates a straight-line CUDA kernel for each AIR's **Q** (quotient / `cExpId`) expression and compiles
it into a per-AIR shared library, replacing the bytecode interpreter for that one expression. It's
optional and drop-in: an AIR with no generated `.so` falls back to the interpreter, unchanged.

## Use it

One-time per provingKey — auto-tunes, builds the kernels, and installs them:

```bash
./build_exps.sh <provingKey> [archspec] [keep-dir]
```

- **`<provingKey>`** — the provingKey build dir (globbed for each AIR's `*.starkinfo.json` + `*.expressionsinfo.json`).
- **`archspec`** — CUDA arch(s): `auto` (default; detect the host GPU) · `major` · `89,120` / `sm_120`.
- **`keep-dir`** — optional dir to retain the generated `.cu`/`.o` (default: a temp dir, deleted on exit).
  ⚠️ **Must be outside `pil2-stark/src/starkpil/`** — otherwise the pil2-stark Makefile will pick up the
  generated kernels and try to compile them into the main lib, breaking `cargo build`.

This writes one **`<base>.exps.so` next to each AIR's `.bin`** (e.g. `airs/Main/air/Main.exps.so`).

Then **just prove — nothing to configure.** Each AIR's `ExpressionsGPU` constructor `dlopen`s its sibling
`<base>.exps.so` by convention; if it's absent, the interpreter runs. No environment variables.

To remove the kernels, delete the `*.exps.so` files (proving reverts to the interpreter).

## What `build_exps.sh` does

1. Runs `gen_exps.py` (below), which emits the kernels + a `gen.log` manifest into a work dir.
2. `nvcc`-compiles them and links one self-contained `<base>.exps.so` per AIR (kernel(s) + the fixed
   C-ABI exports `exps_launch` / `exps_min_scratch`, no prover symbols). Each `.so` is per-GPU-arch.

## `gen_exps.py` (called by `build_exps.sh`; run standalone to inspect)

```bash
python3 gen_exps.py <provingKey> [out-dir] [--cap N] [--nvcc-cmd "<nvcc …>"] [--chunk N]
```

Omit `out-dir` for a preview run (writes to a temp dir, deleted on exit — the summary still prints).
Run `python3 gen_exps.py -h` for the full list.

| Flag | Default | Meaning |
|---|---|---|
| `--cap N` | 40000 | skip an AIR whose Q has more than N ops (→ interpreter) |
| `--nvcc-cmd "<nvcc …>"` | "" | nvcc command that turns on the no-spill autotuner; empty (or `--chunk`) ⇒ off |
| `--chunk N` | autotune | fixed ops/chunk for every AIR; setting it forces the autotuner **off** |

With `--nvcc-cmd` set (which `build_exps.sh` always does), the autotuner picks, per AIR, the largest
chunk size that compiles with **zero register spill** (verified via `cuobjdump`); if even the smallest
chunk still spills, that AIR is skipped (→ interpreter). Rarely-touched tuning constants
(`CHUNK_MAX/MIN`, `SLOTS_CAP`, `DEFAULT_CHUNK`, `CHUNKS_PER_TU`, `GEN_BLK`) are at the top of `gen_exps.py`.
