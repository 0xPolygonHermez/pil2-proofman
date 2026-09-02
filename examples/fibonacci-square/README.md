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
     --fixed-to-file -u ./examples/fibonacci-square/build/fixed \
     -o ./examples/fibonacci-square/pil/build.pilout
```

### 2.2 Generate Setup

After compiling the PIL files, generate the setup:

```bash
cargo run --bin proofman-setup -- setup \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -u ./examples/fibonacci-square/build/fixed \
     --hash Poseidon1 \
     --recursive-jobs 4 --setup-jobs 4 \
     -b ./examples/fibonacci-square/build \
     -r
```

Optionally, you can get the as-built stats of every circuit in the key:

```bash
cargo run --bin proofman-setup -- stats \
     --proving-key examples/fibonacci-square/build/provingKey \
     --output tmp/stats_fibosq.txt \
     --aggregation
```

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
