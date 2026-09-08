# Fibonacci Square Example Proofman Setup Guide

This guide provides step-by-step instructions for setting up the repository and executing the Fibonacci square example with pil2-proofman, from PIL compilation to a fully aggregated proof.

## 0. Platform Compatibility

Detect your platform and set the appropriate library extension:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
```

## 1. Set Up the Repository

### 1.1 Clone the Repository

```bash
git clone https://github.com/0xPolygonHermez/pil2-proofman.git
cd pil2-proofman
```

All commands below run from the repository root.

### 1.2 Install System Packages

```bash
sudo apt update
sudo apt install -y build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev nasm libsodium-dev cmake
```

### 1.3 Node Dependencies (PIL Compiler)

`pil2-compiler` is an npm dependency of the setup crate (`setup/pil2-stark/package.json`). 
`proofman-setup` installs it on first use by itself, so this step is optional; run it
by hand to see any npm error up front:

```bash
(cd setup/pil2-stark && npm install)
```

To use a local checkout of the compiler instead, point `PIL2C_EXEC` at its `pil2com` executable.

### 1.4 Compile the PIL2 Stark C++ Library

The Cargo build script compiles the library on demand as well; running it once up front surfaces missing
system packages early:

```bash
(cd pil2-stark && make clean && make -j starks_lib && make -j bctree)
```

---


## 2. Execute the Fibonacci Square Example

### 2.1 Compile PIL

To begin, compile the PIL files:

```bash
cargo run --bin proofman-setup -- compile-pil \
     -p ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     --fixed-to-file -u ./examples/fibonacci-square/pil/fixed \
     -o ./examples/fibonacci-square/pil/build.pilout
```

### 2.2 Generate Setup

After compiling the PIL files, generate the setup:

```bash
cargo run --bin proofman-setup -- setup \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -b ./examples/fibonacci-square/build -r -u ./examples/fibonacci-square/build/fixed \
     --hash Poseidon2
```

`--hash Poseidon2` is required for the snark steps below and only for those: the BN128 wrap is
built for the poseidon families only, and `setup-snark` refuses a key built with the default
family. Drop the flag if you are not going on to a snark proof.

Additionally, to run the snark setup:

```bash
cargo run --bin proofman-setup -- setup-snark \
     -b ./examples/fibonacci-square/build \
     --final-snark plonk --publics-info examples/fibonacci-square/src/publics_info.json --powers-of-tau <powers_of_tau>
```

If only wants to generate the recursive final for debugging purposes, run:

```bash
cargo run --bin proofman-setup -- setup-snark \
     -b ./examples/fibonacci-square/build --only-recursive-final
```

Both `setup-snark` forms need a proving key generated with `-r`.

### 2.3 Generate PIL Helpers

Generate the corresponding PIL helpers by running the following command:

```bash
cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o
```

### 2.4 Generate Custom Commits

To generate the custom commits, run the following command:

```bash
cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin
```


### 2.5 Build the Project

Build the project with the following command:

```bash
cargo build --workspace
```

### 2.6 Verify Constraints

Verify the constraints by executing this command:

```bash
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin
```

### 2.7 Generate Basic Proofs

Finally, generate the basic proofs using the following command:

```bash
cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
     --output-dir examples/fibonacci-square/build/proofs \
     --verify-proofs
```


### 2.8 Generate Full Aggregated Proof

This will only work if setup is generated with `-r` flag.
Generate the final proof using the following command:

```bash
cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
     --output-dir examples/fibonacci-square/build/proofs \
     --aggregation --verify-proofs
```

### 2.9 Generating GPU proof

In order to generate a proof in the GPU, the following commands need to be executed after generating the setup and pil-helpers.

Note that `gen-custom-commits-fixed` must be invoked with `--gpu` so the resulting `rom_gpu.bin` is laid out for the GPU Merkle hasher; the file produced without `--gpu` is the CPU layout and is not interchangeable.

```bash
cargo build --workspace \
&& cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin \
     --gpu \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin -y -f --gpu -vv
```
### 2.9 All at once

**Without recursion:**

```bash
export PIL2_PROOFMAN_EXT=$(if [[  "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- compile-pil --pil ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/fibonacci-square/pil/build.pilout \
&& cargo run --bin proofman-setup -- setup \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -b ./examples/fibonacci-square/build \
&& cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o \
&& cargo build --workspace \
&& cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin -d \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs_cpu \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin -y
```

**With recursion (Poseidon2):**

```bash
export PIL2_PROOFMAN_EXT=$(if [[  "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- compile-pil --pil ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/fibonacci-square/pil/build.pilout \
&& cargo run --bin proofman-setup -- setup \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -b ./examples/fibonacci-square/build -r \
     --hash Poseidon2 \
&& cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o \
&& cargo build --workspace \
&& cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
&& cargo run --bin proofman-cli stats \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
     --verify-proofs \
     --aggregation \
     --output-dir examples/fibonacci-square/build/proofs \
&& cargo run --bin proofman-cli verify-stark \
     --proof ./examples/fibonacci-square/build/proofs/vadcop_final_proof.bin \
     --verkey ./examples/fibonacci-square/build/provingKey/build/vadcop_final/vadcop_final.verkey.bin
```

**With recursion (BLAKE3), on GPU, in one shot:**

The hash family is chosen at setup time and read back at runtime, so only `setup` takes
`--hash`; nothing else in the flow changes and no rebuild is needed. Swap `blake3` for
`Poseidon1` or `Poseidon2` and the same block works.

Three things are GPU-specific, and each one bites differently if you skip it:

- **`gen-custom-commits-fixed --gpu`** writes the ROM in the GPU Merkle hasher's layout. Keep it
  in its own file (`rom_gpu.bin`): the CPU layout is *not* interchangeable, so reusing `rom.bin`
  for both silently leaves whichever ran last, and the other path then fails or, worse, proves
  against the wrong commitment.
- **`prove --gpu`** runs the prover on the device. `-vv` is worth having: it reports the per-stage
  timings you need to tell a real regression from noise.
- **`gen-exps`** compiles each AIR's Q-expression into a CUDA kernel (`.exps.so`). Optional — an
  AIR without one falls back to the interpreter — but it is where a large share of the GPU speedup
  comes from. It is a **no-op without `nvcc` on PATH**, and silently so, which is the one failure
  mode to watch for. `setup --gen-exps` does it inline; the standalone `gen-exps` subcommand does
  it on an existing `provingKey/` without re-running setup, which is what you want when iterating.

```bash
export PIL2_PROOFMAN_EXT=$(if [[  "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- compile-pil --pil ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/fibonacci-square/pil/build.pilout \
&& cargo run --bin proofman-setup -- setup \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -b ./examples/fibonacci-square/build -r \
     --hash blake3 --gen-exps \
&& cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o \
&& cargo build --workspace \
&& cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin --gpu \
&& cargo run --bin proofman-cli stats \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin \
     --verify-proofs \
     --aggregation \
     --output-dir examples/fibonacci-square/build/proofs -vv --gpu \
&& cargo run --bin proofman-cli verify-stark \
     --proof ./examples/fibonacci-square/build/proofs/vadcop_final_proof.bin \
     --verkey ./examples/fibonacci-square/build/provingKey/build/vadcop_final/vadcop_final.verkey.bin
```

`--verify-proofs` checks every proof as it is produced, and the closing `verify-stark` checks the
aggregated one against its verification key. That last step is what exercises the **native Rust
verifier** in `verifier/src/blake3/`, which is generated from the AIR's constraints by `setup -r`
and lives in the repo: change a recursion AIR and it has to be regenerated, or it will reject
proofs a correct prover produced.

### Regenerating only the CUDA kernels

`gen-exps` works on an existing proving key, so iterating on the kernels costs a few seconds
instead of a whole setup:

```bash
cargo run --release --bin proofman-setup -- gen-exps \
     -p ./examples/fibonacci-square/build/provingKey
```

`--exps-arch` pins the CUDA arch (`auto` by default, or e.g. `sm_120` / `"89,120"`),
`--exps-cap` skips an AIR whose Q exceeds N ops (default 60000, leaving it on the interpreter),
and `--exps-chunk` fixes ops per chunk instead of auto-tuning the largest no-spill size.
