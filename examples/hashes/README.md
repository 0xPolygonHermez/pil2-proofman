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

## Hash Throughput Comparison

Cost is measured in clocks per column; lower **cost / byte** is better.

|                               | **Blake3**          | **SHA2-256**                | **Blake2b**                   |
| ----------------------------- | ------------------- | --------------------------- | ----------------------------- |
| Clock length                  | 56 rows             | 72 rows                     | 96 rows                       |
| Fixed                         | 8                  | 5                           | 7                             |
| Stage1                        | 55                  | 100                         | 100                           |
| Stage2                        | 48                  | 9                           | 84                            |
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
