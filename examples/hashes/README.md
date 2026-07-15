# Hashes Example Proofman Setup Guide

This guide provides step-by-step instructions for setting up the necessary repositories and executing the hashes example.

## Execute the Hashes Example

### 1 Compile PIL

To begin, compile the PIL files:

```bash
cargo run --bin proofman-setup -- compile-pil --pil ./examples/hashes/pil/main.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/hashes/pil/main.pilout -u ./examples/hashes/build/fixed --fixed-to-file
```

### 2 Generate Setup

After compiling the PIL files, generate the setup:

```bash
cargo run --bin proofman-setup -- setup \
     -a ./examples/hashes/pil/main.pilout \
     -b ./examples/hashes/build -u ./examples/hashes/build/fixed
```

Additionally, you can generate some stats about the setup by running:

```bash
cargo run --bin proofman-setup -- stats \
     -a ./examples/hashes/pil/main.pilout \
     -o ./examples/hashes/build/stats.txt
```

### 3 Generate PIL Helpers

Generate the corresponding PIL helpers by running the following command:

```bash
cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/hashes/pil/main.pilout \
     --path ./examples/hashes/src -o
```

### 4 Build the Project

Build the project with the following command:

```bash
cargo build --workspace
```

### 5 Verify Constraints

Verify the constraints by executing this command:

```bash
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libhashes.so \
     --proving-key examples/hashes/build/provingKey/
```

## Hash Throughput Comparison

**Size / Rate** is the committed witness bytes per hashed input byte; lower is better.

| Hash       | Field      | Full-Op Cells           | Witness Size (bytes) | Rate (bytes)    | Size / Rate | Relative  |
| :--------- | :--------- | :---------------------- | -------------------: | :-------------- | ----------: | --------: |
| Blake3     | Binary     | 56 × 381 = 21.336       |                2.667 | 64              |        41,7 |     1,00× |
| Poseidon2  | Goldilocks | 14 × 392 = 5.488        |               43.904 | 96 (*)          |       457,3 |    10,97× |
| Blake3     | Goldilocks | 56 × 108 = 6.048        |               48.384 | 64              |       756,0 |    18,14× |
| Blake2b    | Goldilocks | 64 × 190 = 12.160       |               97.280 | 128             |       760,0 |    18,24× |
| SHA2-256   | Goldilocks | 72 × 115 = 8.280        |               66.240 | 64              |     1.035,0 |    24,84× |

(*) Poseidon2 bytes are nominal (12 Goldilocks elements × 8 bytes); a Goldilocks element holds ~63.99 bits, so the truly absorbable payload is slightly under 96 bytes.