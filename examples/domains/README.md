# Domains Example

Nine AIRs in one airgroup `Domains`, all at `N = 2^10`, each exercising one shape of domain
constraint (`expr === expr domain NAME;`, `domain !NAME`, `domain NAME { ... }`). Each AIR also carries a
self-balancing std permutation (`self_permutation` in `pil/domains.pil`), only to give it stage 2; it
touches no domain constraint of the AIR, so a failure still points at one domain shape.

The std's grand sum is a cyclic accumulator, one everyRow constraint with no `__L1__` fixed column and no
domain (on BN128 it keeps `__L1__`). No AIR declares a fixed column, so all nine have no constants at all.

`DomMixed` also range-checks `x` in `[0, 255]`. That brings in the std range-check table AIR
`SpecifiedRanges`, whose table is fixed columns: it is the one AIR with constants in this pilout, the
control next to the zero-fixed ones, and `DomMixed` feeds a table the prover owns.

`DomFirstTwin` is an exact copy of `DomFirst`: two AIRs with identical PIL, told apart only by their
`circuitType`.

| AIR | Domain shape | Constraint |
|---|---|---|
| `DomFirst` | direct, full cycle, 1 offset | `a === 1 domain FIRST` |
| `DomFirstTwin` | `DomFirst`'s twin | `a === 1 domain FIRST` |
| `DomLast` | direct, full cycle, offset N−1 | `b === 0 domain LAST` |
| `DomNotFirst` | complement, full cycle, degree 2 (zerofier adds no degree) | `(a - 'a - 1) * (a - 'a - 1) === 0 domain !FIRST` |
| `DomNotLast` | complement, wrap row excluded | `a' === a + 1 domain !LAST` |
| `DomOdd` | direct, cycle 2 | `t === 0 domain ODD` |
| `DomLimits16` | direct, cycle 16, 2 offsets (degree 4 on 128 rows, so an im pol) | `s * s * (1 - s) === 0 domain LIMITS_16` |
| `DomNotQuarter` | complement of a periodic domain | `u === 7 domain !Q1` (`Q1 = [1,0,0,0]`) |
| `DomMixed` | several domains, dedup, degree 3 on FIRST and !FIRST, one everyRow, a `domain EVEN4 { ... }` block, a range check | see `pil/domains.pil` |

The witness of each AIR breaks its constraint outside the domain, so a proof that verifies shows
that the rows outside the domain are not constrained. `DomNotFirst` and `DomNotLast` do this on the
wrap row.

`src/inputs.json` lists the AIRs that get an instance. A file that lists fewer AIRs proves only
those AIRs, for example `{"airs": ["DomOdd"]}`.

`DOMAINS_TAMPER="<Air>:<col>:<row>:<value>"` overwrites one cell after the witness is filled.

`cli/tests/domains_e2e.rs` runs the commands below. Its stage-3 tests run `setup -r`, which takes
minutes, so they are `#[ignore]`d. Its GPU tests skip without a GPU, a CUDA build or nvcc;
`DOMAINS_GPU=1` turns the skip into a failure, and `DOMAINS_CUDA_ARCH` (e.g. `sm_120`) sets `gen-exps`'s arch:

```bash
DOMAINS_GPU=1 DOMAINS_CUDA_ARCH=sm_120 cargo test --release -p proofman-cli --test domains_e2e -- --include-ignored
```

The native Rust verifier refuses an AIR with no fixed columns, so it is tested on `examples/domain-native`
instead. Its one AIR has a fixed column and no std bus.

All commands run from the repository root, with
`export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)`.

## 1. Compile PIL

This step needs a pil2-compiler with domains. Point `PIL2C_EXEC` at its `src/pil.js`.
`pil/build.pilout` is checked in, so you only need to compile again after you change the PIL.

```bash
node $PIL2C_EXEC examples/domains/pil/build.pil \
     -I ./pil2-components/lib/std/pil \
     -o ./examples/domains/pil/build.pilout
cargo run --release --bin proofman-cli pil-helpers \
     --pilout ./examples/domains/pil/build.pilout \
     --path ./examples/domains/src -o
```

## 2. Stage 0: setup

```bash
cargo run --release --bin proofman-setup -- setup \
     -a ./examples/domains/pil/build.pilout \
     -b ./examples/domains/build
cargo build --release -p domains
```

## 3. Stage 1: CPU, no aggregation

```bash
cargo run --release --bin proofman-cli verify-constraints \
     --witness-lib ./target/release/libdomains${PIL2_PROOFMAN_EXT} \
     --proving-key ./examples/domains/build/provingKey \
     --public-inputs ./examples/domains/src/inputs.json
cargo run --release --bin proofman-cli prove \
     --witness-lib ./target/release/libdomains${PIL2_PROOFMAN_EXT} \
     --proving-key ./examples/domains/build/provingKey \
     --public-inputs ./examples/domains/src/inputs.json \
     --output-dir ./examples/domains/build/proofs --verify-proofs
```

A cell inside a domain makes both commands fail. A cell outside every domain still verifies:

```bash
DOMAINS_TAMPER="DomOdd:t:1:5" cargo run --release --bin proofman-cli verify-constraints ...  # fails, row 1
DOMAINS_TAMPER="DomOdd:t:2:5" cargo run --release --bin proofman-cli verify-constraints ...  # passes
```

## 4. Stage 2: GPU, no aggregation

```bash
cargo run --release --bin proofman-cli gen-exps --proving-key ./examples/domains/build/provingKey
cargo run --release --bin proofman-cli prove \
     --witness-lib ./target/release/libdomains${PIL2_PROOFMAN_EXT} \
     --proving-key ./examples/domains/build/provingKey \
     --public-inputs ./examples/domains/src/inputs.json \
     --output-dir ./examples/domains/build/proofs --verify-proofs --gpu
```

GPU proofs must match CPU proofs byte for byte. To compare them, prove one AIR on each device:

```bash
cargo run --release --bin proofman-cli prove-air \
     --witness-lib ./target/release/libdomains${PIL2_PROOFMAN_EXT} --air DomOdd \
     --proving-key ./examples/domains/build/provingKey \
     --public-inputs ./examples/domains/src/inputs.json \
     --save-proof ./examples/domains/build/DomOdd_cpu.bin   # add --gpu for the GPU proof
```

## 5. Stage 3: aggregation

```bash
cargo run --release --bin proofman-setup -- setup \
     -a ./examples/domains/pil/build.pilout \
     -b ./examples/domains/build-recursive -r
cargo run --release --bin proofman-cli prove \
     --witness-lib ./target/release/libdomains${PIL2_PROOFMAN_EXT} \
     --proving-key ./examples/domains/build-recursive/provingKey \
     --public-inputs ./examples/domains/src/inputs.json \
     --output-dir ./examples/domains/build-recursive/proofs \
     --aggregation --verify-proofs            # and again with --gpu, after gen-exps
cargo run --release --bin proofman-cli verify-stark \
     --proof ./examples/domains/build-recursive/proofs/vadcop_final_proof.bin \
     --verkey ./examples/domains/build-recursive/provingKey/build/vadcop_final/vadcop_final.verkey.bin
```

## Known gaps

- **Compressor:** no end-to-end test aggregates a zero-fixed AIR through a compressor.
- **GPU aggregation is not byte-reproducible:** recursive2 pairs its inputs in completion order
  (the recursive workers in `proofman.rs`). The CPU/GPU byte checks cover the basic proofs and the recursive1
  inputs only.
- **Tiny AIRs on the GPU:** FRI aborts at launch when the extended domain has fewer than 256 rows
  (`starks_gpu.cu`). Every AIR here is large enough.
- **BN128:** the std keeps `__L1__` there, so pilfflonk AIRs keep their fixed column.
- **Keys:** the std's grand sum changed, so every proving key must be set up again, including its
  `.exps.so` (`gen-exps`) and the emitted verifiers. The recursion AIRs changed too. A key that has a
  verkey for an AIR without fixed columns fails to load with "stale verkey".
