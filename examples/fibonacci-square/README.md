# Fibonacci Square Example Proofman Setup Guide

This guide provides step-by-step instructions for setting up the necessary repositories and executing the Fibonacci square example using the Polygon Hermez zkEVM prover.

## 0. Platform Compatibility

Detect your platform and set the appropriate library extension:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
```

## 1. Download and Set Up Required Repositories

### 1.2 Install `pil2-compiler`

Next, clone the `pil2-compiler` repository and install its dependencies:

```bash
git clone https://github.com/0xPolygonHermez/pil2-compiler.git
cd pil2-compiler
npm install
cd ..
```

### 1.3 Install `pil2-proofman-js`

Clone the `pil2-proofman-js` repository, switch to the `develop` branch, and install the dependencies:

```bash
git clone https://github.com/0xPolygonHermez/pil2-proofman-js
cd pil2-proofman-js
git checkout develop

# TODO: Verify if the Stark Recurser raises any issues during this process

npm install
cd ..
```
# Update package lists and install required system packages
sudo apt update
sudo apt install -y build-essential libbenchmark-dev libomp-dev libgmp-dev nlohmann-json3-dev nasm libsodium-dev cmake

### 1.4 Compile the PIL2 Stark C++ Library

Compile the PIL2 Stark C++ Library (run only once):

```bash
(cd ../pil2-proofman/pil2-stark && make clean && make -j starks_lib && make -j bctree)
```

### 1.5 Install `pil2-proofman`

Finally, clone the `pil2-proofman` repository:

```bash
git clone https://github.com/0xPolygonHermez/pil2-proofman.git
cd pil2-proofman
```

---


## 2. Execute the Fibonacci Square Example

### 2.1 Compile PIL

To begin, compile the PIL files:

```bash
node ../pil2-compiler/src/pil.js ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/fibonacci-square/pil/build.pilout -u ./examples/fibonacci-square/build/fixed -O fixed-to-file
```

### 2.2 Generate Setup

After compiling the PIL files, generate the setup:

```bash
node ../pil2-proofman-js/src/main_setup.js \
     -a ./examples/fibonacci-square/pil/build.pilout -t ./pil2-components/lib/std/pil \
     -b ./examples/fibonacci-square/build -r -u ./examples/fibonacci-square/build/fixed
```

To run the aggregated proof, need to add -r to the previous command

### 2.3 Generate PIL Helpers

Generate the corresponding PIL helpers by running the following command:

```bash
cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o
```


### 2.4 Build the Project

Build the project with the following command:

```bash
cargo build
```

### 2.5 Verify Constraints

Verify the constraints by executing this command:

```bash
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json
```

### 2.6 Generate Proof

Finally, generate the proof using the following command:

```bash
cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs -y 
```


### 2.7 Generate VadcopFinal Proof

This will only work if setup is generated with `-r` flag.
Generate the final proof using the following command:

```bash
cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs \
     -a
```

### 2.8 Generating GPU proof

In order to generate a proof in the GPU, the following commands needs to be executed after generating the setup and pil-helpers

```bash
cargo build --features gpu \
&& cargo run --features gpu --bin proofman-cli check-setup --proving-key \
     examples/fibonacci-square/build/provingKey/ \
&& cargo run --features gpu --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin \
&& cargo run --features gpu --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs \
     --custom-commits rom=examples/fibonacci-square/build/rom_gpu.bin -y
```
### 2.9 All at once

**Without recursion:**

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& node --max-old-space-size=65536 ../pil2-compiler/src/pil.js ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/fibonacci-square/pil/build.pilout \
&& node --max-old-space-size=65536 ../pil2-proofman-js/src/main_setup.js \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -b ./examples/fibonacci-square/build -t pil2-components/lib/std/pil \
&& cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o \
&& cargo build \
&& cargo run --bin proofman-cli check-setup --proving-key examples/fibonacci-square/build/provingKey/ \
&& cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libfibonacci_square${PIL2_PROOFMAN_EXT} \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     --public-inputs examples/fibonacci-square/src/inputs.json \
     --output-dir examples/fibonacci-square/build/proofs_cpu \
     --custom-commits rom=examples/fibonacci-square/build/rom.bin -y
```

**With recursion:**

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& node --max-old-space-size=65536 ../pil2-compiler/src/pil.js ./examples/fibonacci-square/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/fibonacci-square/pil/build.pilout \
&& node --max-old-space-size=65536 ../pil2-proofman-js/src/main_setup.js \
     -a ./examples/fibonacci-square/pil/build.pilout \
     -b ./examples/fibonacci-square/build -t pil2-components/lib/std/pil \
     -r \
&& cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-square/pil/build.pilout \
     --path ./examples/fibonacci-square/src -o \
&& cargo build \
&& cargo run --bin proofman-cli check-setup \
     --proving-key examples/fibonacci-square/build/provingKey/ \
     -a \
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
     --proof ./examples/fibonacci-square/build/proofs/proofs/vadcop_final_proof.bin \
     --verkey ./examples/fibonacci-square/build/provingKey/build/vadcop_final/vadcop_final.verkey.bin
```
