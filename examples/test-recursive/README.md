## Execute the Recursive Example

This example recursively verifies a previously generated regular (inner) STARK
proof. The inner-proof fixtures live in per-hash-family subfolders and are
selected automatically by `--hash`:

```
examples/test-recursive/
├── poseidon1/   test.circom  test.verifier.circom  ag0_air0_tCompressor.bin   (Poseidon1 / Hades)
└── poseidon2/   test.circom  test.verifier.circom  ag0_air0_tCompressor.bin   (Poseidon2, default)
```

`setup-recursive-test -c .../test.circom --hash <H>` resolves the fixture to
`.../<h>/test.circom` (lowercased family). The witness library reads the matching
`<h>/ag0_air0_tCompressor.bin` at run time (from `pctx.global_info.hash`), so a
single `libtest_recursive` works for both families.

There are two ways to prove:

- **`prove --witness-lib`** — the full pipeline (contributions → basic proof →
  recursion). This is the path that supports `verify-constraints`.
- **`prove-air`** — proves the inner proof **directly**: it loads the zkin
  proof, generates its recursion witness, and calls `gen_recursive_proof_c`,
  bypassing contributions and the basic proof entirely. The target recursive AIR
  is parsed from the proof file **name** via `ag<airgroup>_air<air>_t<ProofType>`
  (hence `ag0_air0_tCompressor.bin` — airgroup 0 / air 0 / `Compressor`). Runs on
  CPU or `--gpu`. The const tree is regenerated automatically if missing or stale.
  Add `--emit-witness-only` to stop after the recursion witness (the legacy
  `gen-witness` behaviour).

## The fixture

Both families' inner proofs are **recursive2** proofs, so `-t aggregation` reproduces the production
aggregator geometry: 2^17 rows, blowupFactor 3, arity 4, 73 queries, cm 48/12/21. Measured `NUsed` is
120196 (Poseidon1) and 119594 (Poseidon2) of the 131072 available -- ~91% occupancy, and within 0.5%
of each other, since both verify the same shape of proof.

The fixtures were generated from the fibonacci-square example, one setup per family: `setup -r`
emits `build/circom/FiboCPU_recursive2.{circom,verifier.circom}`, and a `prove --aggregation` run
with `PIL2_DUMP_ZKIN=recursive2` captures the proof blob (see `proofman/src/recursion.rs`). The dump
is named `zkin_ag<N>_air<M>_t<proof_type>.bin`, which is what `prove-air --proof` parses to resolve
the setup path; the committed fixtures keep the older `ag0_air0_tCompressor.bin` spelling, which the
same parser accepts, because the harness names every recursive-test AIR `Compressor`.

Note the harness always uses blowupFactor 3, while a production **compressor** uses 2 -- so these
numbers are faithful for the aggregator, not for the compressor.

## Platform Compatibility

Detect your platform and set the appropriate library extension (needed for the
`prove --witness-lib` / `verify-constraints` flow):

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
```

## Poseidon2 (default)

Full pipeline (with constraint verification):

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- setup-recursive-test \
     -b ./examples/test-recursive/build2 -c ./examples/test-recursive/test.circom -n test -t aggregation \
     --hash Poseidon2 --gen-exps \
&& cargo build --workspace \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build2/provingKey/ \
&& cargo run --bin proofman-cli -- prove-air \
     --proof examples/test-recursive/poseidon2/ag0_air0_tCompressor.bin \
     --proving-key examples/test-recursive/build2/provingKey/ \
     --gpu -vv
```

Use `-t compressor` instead of `-t aggregation` for the compressor variant.

## Poseidon1 (Hades)

Same flow, but pass `--hash Poseidon1` to the setup. The hash family is selected
at runtime from the setup, so no feature flag or rebuild is needed — the prover
picks up the correct hash family automatically:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- setup-recursive-test \
     -b ./examples/test-recursive/build -c ./examples/test-recursive/test.circom -n test -t aggregation \
     --hash Poseidon1 --gen-exps \
&& cargo build --workspace \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build/provingKey/ \
&& cargo run --bin proofman-cli -- prove-air \
     --proof examples/test-recursive/poseidon1/ag0_air0_tCompressor.bin \
     --proving-key examples/test-recursive/build/provingKey/ \
     --gpu -vv
```