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
     -b ./examples/hashes/build -u ./examples/hashes/build/fixed --gen-exps --hash blake3 -s ./examples/hashes/pil/config.json
```

Additionally, you can generate some stats about the setup by running:

```bash
cargo run --bin proofman-setup -- stats \
     -a ./examples/hashes/pil/main.pilout \
     -s ./examples/hashes/pil/config.json \
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

### 6 Prove a Single Air

`prove-air` proves one air on its own: there is no contributions phase and therefore no
global challenge — the transcript is seeded from that air's own verkey and publics, so the
proof verifies standalone against `<Air>.verkey.json`. Only the requested air's witness is
computed; the other airs of the pilout are planned but left untouched.

```bash
cargo run --release --bin proofman-cli -- prove-air \
     --witness-lib ./target/release/libhashes.so \
     --proving-key ./examples/hashes/build/provingKey \
     --air Blake3 --gpu -vv
```

Swap `--air` for `Sha2` or `Blake2b` to prove those instead. Add `--no-verify` to skip the
verification step when timing, and drop `--gpu` to run on the CPU.

This library exports `packed_info`, so `--gpu` packs the trace by default (a ~5x smaller H2D
copy); `--packed` packs on the CPU too, and `--no-packed` turns it off either way.

### 7 All at once

Steps 1-6 chained. This one builds in release throughout, so the same library serves both
`verify-constraints` and the GPU prove; swap `--hash blake3` for `Poseidon1` or `Poseidon2` to set
up a different family, and `--air Blake3` for `Sha2` or `Blake2b`.

```bash
cargo run --release --bin proofman-setup -- compile-pil --pil ./examples/hashes/pil/main.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/hashes/pil/main.pilout -u ./examples/hashes/build/fixed --fixed-to-file \
&& cargo run --release --bin proofman-setup -- setup \
     -a ./examples/hashes/pil/main.pilout \
     -b ./examples/hashes/build -u ./examples/hashes/build/fixed --gen-exps --hash blake3 \
     -s ./examples/hashes/pil/config.json \
&& cargo run --release --bin proofman-cli pil-helpers \
     --pilout ./examples/hashes/pil/main.pilout \
     --path ./examples/hashes/src -o \
&& cargo build --release --workspace \
&& cargo run --release --bin proofman-cli verify-constraints \
     --witness-lib ./target/release/libhashes.so \
     --proving-key examples/hashes/build/provingKey/ \
&& cargo run --release --bin proofman-cli -- prove-air \
     --witness-lib ./target/release/libhashes.so \
     --proving-key ./examples/hashes/build/provingKey \
     --air Blake3 --gpu -vv
```

Changing the PIL (including `LANES` below) invalidates the proving key, the pil-helpers and the
GPU const cache, so re-run the whole chain rather than a suffix of it. If a run trips over a stale
`.const_gpu`/`.consttree_gpu`, delete them under `examples/hashes/build/provingKey` and retry.

### Blake3 Lanes

`Blake3` takes a second parameter, `LANES` (default 1): the number of independent Blake3
permutations that share every row.

```
Blake3(2**20, 2);   // 2 lanes -> 37 446 hashes per 2^20-row proof instead of 18 723
```

Each lane gets its own copy of the 52 G-function columns; the fixed lookup tables (the
XOR-rotate table, the 16-bit range checker and their two multiplicity columns) are shared.
Since `N` already has a floor of `2**19 + 2**18` rows just to hold the XOR table, lanes are
how you buy throughput without growing `N`. The witness computation reads `LANES` back from
the proving key's cm1 width, so the PIL stays the single source of truth — but the pilout,
the pil-helpers and the setup must all be regenerated after changing it (steps 1–3).

## Hash Throughput Comparison

Cost is measured in clocks per column; lower **cost / byte** is better.

|                               | **Blake3**          | **SHA2-256**                | **Blake2b**                   |
| ----------------------------- | ------------------- | --------------------------- | ----------------------------- |
| Clock length                  | 56 rows             | 72 rows                     | 96 rows                       |
| Fixed                         | 8                   | 5                           | 7                             |
| Stage1                        | 55                  | 103                         | 100                           |
| Stage2                        | 48                  | 6                           | 84                            |
| **Total cols**                | **109**             | **115**                     | **190**                       |
| Constraints                   | 26                  | 105                         | 47                            |
| Max degree                    | 3                   | 3                           | 3                             |
| Opening points                | 57                  | 73                          | 97                            |
| nEvals                        | 186                 | 633                         | 330                           |
| Expressions                   | 1,085               | 3,589                       | 1,992                         |
| **Min prover mem / instance** | **(N=2¹⁷) 0.30 GB** | **(N=2¹⁶) 0.18 GB**         | **(N=2¹⁶) 0.23 GB**           |
| **Cells / hash**              | 109x56 = **6,104**  | 115x72 = **8,280** (+36%)   | 190x96 = **18,240** (+201%)   |
| **Cost / byte**               | 6,104/64 = **95.4** | 8,280/64 = **129.3** (+37%) | 18,240/128 = **142.5** (+51%) |
| **Throughput / instance**     | 2¹⁸/56 = **4,681**  | 2¹⁸/72 = **3,640**          | 2¹⁸/145 = **1,807**           |
