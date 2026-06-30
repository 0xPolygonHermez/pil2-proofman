# Hashes Example Proofman Setup Guide

This guide provides step-by-step instructions for setting up the necessary repositories and executing the hashes example.

## Execute the Hashes Example

### 1 Generate the connections

To generate the connections, run the following command:

```bash
cargo run --package hashes --bin blake3-connection -- \
  build -o ./examples/hashes/src/blake3_connection_fixed.bin
```

### 2 Compile PIL

To begin, compile the PIL files:

```bash
cargo run --bin proofman-setup -- compile-pil --pil ./examples/hashes/pil/main.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/hashes/pil/main.pilout -u ./examples/hashes/build/fixed --fixed-to-file
```

### 3 Generate PIL Helpers

Generate the corresponding PIL helpers by running the following command:

```bash
cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/hashes/pil/main.pilout \
     --path ./examples/hashes/src -o
```

### 4 Generate Setup

After compiling the PIL files, generate the setup:

```bash
cargo run --bin proofman-setup -- setup \
     -a ./examples/hashes/pil/main.pilout \
     -b ./examples/hashes/build -u ./examples/hashes/build/fixed
```

### 5 Build the Project

Build the project with the following command:

```bash
cargo build --workspace
```

### 6 Verify Constraints

Verify the constraints by executing this command:

```bash
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libhashes.so \
     --proving-key examples/hashes/build/provingKey/
```

## Hash Throughput Comparison

Cost is measured in clocks per column; lower **cost / byte** is better.

| Hash       | Full-op cost            | Msg bytes/block | Cost / byte | Relative |  BF |
| :--------- | :---------------------- | :-------------- | ----------: | -------: | --: |
| Poseidon2  | 14 × 392 = 5,488        | 96 (*)          |        57.2 |     1.0× |   1 |
| Blake3     | 56 × 121 = 6,776        | 64              |       105.9 |     1.8× |   1 |
| SHA2-256   | 72 × 122 = 8,784        | 64              |       137.3 |     2.4× |   1 |
| Blake2b    | 12 × 24 × 209 = 60,192  | 128             |       470.3 |     8.2× |   1 |
| Keccak-256 | 25 × 3,023 = 75,575     | 136             |       555.7 |     9.7× |   1 |

(*) Poseidon2 bytes are nominal (12 Goldilocks elements × 8 bytes); a Goldilocks element holds ~63.99 bits, so the truly absorbable payload is slightly under 96 bytes.