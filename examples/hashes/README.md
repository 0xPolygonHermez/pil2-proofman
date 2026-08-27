# Hashes Example Proofman Setup Guide

This guide provides step-by-step instructions for setting up the necessary repositories and executing the hashes example.

## Execute the Hashes Example

### 1 Compile PIL

To begin, compile the PIL files:

```bash
cargo run --bin proofman-setup -- compile-pil --pil ./examples/hashes/pil/main.pil \
     -I ./pil2-components/lib/std/pil -I ./setup/stark-recurser/plonk2pil/pil \
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
     -I ./pil2-components/lib/std/pil -I ./setup/stark-recurser/plonk2pil/pil \
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

### Blowup Factor

`config.json` sets `blowupFactor` per air as the **log2** of the LDE expansion — Blake3's `2`
means the trace is extended to `4 * N` rows, Sha2 and Blake2b's `1` means `2 * N`.

It is coupled to the constraint degree: a blowup of `2**k` admits constraints of degree at most
`2**k + 1`, so Blake3's `blowupFactor: 2` is what lets `main.pil` ask for
`set_max_constraint_degree(5)`. Raising the blowup buys degree — and degree usually buys columns
back, since a higher-degree constraint needs fewer intermediate witness columns — but every
committed column is then NTT'd and Merkle-ised over `2**k * N` rows, so the prover pays for it on
each of the three commit stages. Lower it and the max degree must come down with it.

The degree headroom is paid back almost entirely out of stage2, which is why a marginal Blake3 lane
costs 95 columns at `blowupFactor: 1`, 74 at `2` and 65 at `3` (stage1 is structural at 53/lane and
moves with neither; stageQ goes 6 -> 12 -> 21). So the knobs are not independent in the way they
look: `LANES` amortises the shared columns over more hashes per row, while `blowupFactor` sets both
how many extended rows each column costs and how many columns there are to begin with.

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

Widths are affine in `LANES`, not proportional: stage1 is always `53k + 2`. At this example's
`blowupFactor: 2` a lane costs 53 stage-1 + 21 stage-2 columns, while 17 committed columns
(2 stage-1 + 3 stage-2 + 12 stageQ) and the 8 fixed are paid once however many lanes there are.
Amortising those is the whole gain:

| LANES | stage1 | stage2 | stageQ | total | cells / hash      | vs 1 lane  |
| ----- | ------ | ------ | ------ | ----- | ----------------- | ---------- |
| 1     | 55     | 24     | 12     | 91    | 91x56 = 5,096     | —          |
| 2     | 108    | 45     | 12     | 165   | 165x56/2 = 4,620  | **-9.3%**  |
| 3     | 161    | 66     | 12     | 239   | 239x56/3 = 4,461  | **-12.5%** |
| 4     | 214    | 87     | 12     | 313   | 313x56/4 = 4,382  | **-14.0%** |

Only those 17 columns are being amortised, so the returns taper fast: cost/hash approaches
`74x56 = 4,144` (-18.7%) and no lane count beats that. Any `LANES >= 1` works — there is no
power-of-two requirement.

Read the table at a **fixed** blowup. stage2 shrinks sharply as blowup buys degree (48 -> 24 -> 15
at `LANES=1`), so comparing lane counts measured at different blowups conflates the two knobs. All
twelve `LANES 1-4 x blowup 1-3` points come from `proofman-setup stats` and are recorded in
`setup/stark-recurser/docs-research/recursion-cell-memory-model.html`.

## Hash Throughput Comparison

All from `proofman-setup stats` at `LANES=1`, Blake3 at `N=2**17`, SHA2/Blake2b at `N=2**16`.
`total` is the committed width (stage1+stage2+stageQ); fixed is listed separately. Clock length is
rows per hash per lane and does not move with blowup. Lower **cost / byte** is better.

Compared at the same `blowupFactor: 1` — raising it only helps Blake3, see below:

|                       | **Blake3**          | **SHA2-256**                | **Blake2b**                   |
| --------------------- | ------------------- | --------------------------- | ----------------------------- |
| Clock length          | 56 rows             | 72 rows                     | 96 rows                       |
| Fixed                 | 8                   | 5                           | 6                             |
| Stage1                | 55                  | 103                         | 100                           |
| Stage2                | 48                  | 6                           | 84                            |
| StageQ                | 6                   | 6                           | 6                             |
| **Total cols**        | **109**             | **115**                     | **190**                       |
| Constraints           | 26                  | 105                         | 47                            |
| Max degree            | 3                   | 3                           | 3                             |
| Opening points        | 57                  | 73                          | 97                            |
| nEvals                | 186                 | 633                         | 329                           |
| Expressions           | 1,143               | 3,808                       | 2,088                         |
| Prover mem / instance | 0.32 GB             | 0.19 GB                     | 0.24 GB                       |
| Verifier hashes       | 23,326              | 22,160                      | 24,042                        |
| **Cells / hash**      | 109x56 = **6,104**  | 115x72 = **8,280** (+36%)   | 190x96 = **18,240** (+199%)   |
| **Cost / byte**       | 6,104/64 = **95.4** | 8,280/64 = **129.4** (+36%) | 18,240/128 = **142.5** (+49%) |

### What blowup buys Blake3

`config.json` raises only Blake3 to `blowupFactor: 2`, because Blake3 is the one air here with
degree headroom to spend. SHA2-256 never exceeds degree 3, so its `maxConstraintDegree` stays 3 and
its width is unchanged at any blowup — it would pay the memory for nothing.

| Blake3 `blowupFactor`  | **1**               | **2**               | **3**               |
| ---------------------- | ------------------- | ------------------- | ------------------- |
| Max degree             | 3                   | 5                   | 8 (PIL ceiling)     |
| Stage1 / Stage2/ StageQ| 55 / 48 / 6         | 55 / 24 / 12        | 55 / 15 / 21        |
| **Total cols**         | **109**             | **91**              | **91**              |
| Constraints            | 26                  | 18                  | 15                  |
| Prover mem / instance  | 0.32 GB             | 0.56 GB             | 1.12 GB             |
| Verifier hashes        | 23,326              | 12,818              | 9,199               |
| **Cost / byte**        | 6,104/64 = **95.4** | 5,096/64 = **79.6** | 5,096/64 = **79.6** |

The degree headroom is paid back almost entirely out of stage2 (48 -> 24 -> 15) while stageQ grows
(6 -> 12 -> 21), so the width win saturates: **2 and 3 have the same 91 committed columns**, but 3
costs twice the prover memory. Blowup 2 is the sweet spot for width; go to 3 only to buy verifier
hashes (-28%), which is what recursion pays for.
