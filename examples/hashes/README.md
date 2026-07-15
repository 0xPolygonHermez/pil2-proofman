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

Cost is measured in clocks per column; lower **cost / byte** is better.

| Hash       | Full-op cost            | Msg bytes/block | Cost / byte | Relative  | BF  |
| :--------- | :---------------------- | :-------------- | ----------: | --------: | --: |
| Poseidon2  | 14 × 392 = 5.488        | 96 (*)          |        57,2 |     1,00× |   1 |
| Blake3     | 56 × 108 = 6.048        | 64              |        94,5 |     1,65× |   1 |
| Blake2b    | 64 × 190 = 12.160       | 128             |        95,0 |     1,66× |   1 |
| SHA2-256   | 72 × 115 = 8.280        | 64              |       129,3 |     2,26× |   1 |

(*) Poseidon2 bytes are nominal (12 Goldilocks elements × 8 bytes); a Goldilocks element holds ~63.99 bits, so the truly absorbable payload is slightly under 96 bytes.