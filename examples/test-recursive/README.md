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
- **`prove-recursive`** — proves the inner proof **directly**: it loads the zkin
  proof, generates its recursion witness, and calls `gen_recursive_proof_c`,
  bypassing contributions and the basic proof entirely. The target recursive AIR
  is parsed from the proof file **name** via `ag<airgroup>_air<air>_t<ProofType>`
  (hence `ag0_air0_tCompressor.bin` — airgroup 0 / air 0 / `Compressor`). Runs on
  CPU or `--gpu`. The const tree is regenerated automatically if missing or stale.
  Add `--emit-witness-only` to stop after the recursion witness (the legacy
  `gen-witness` behaviour).

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
     --hash Poseidon2 \
&& cargo build --workspace \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build2/provingKey/ \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build2/provingKey/ \
     --output-dir examples/test-recursive/build2/proofs -y -vv
```

Direct recursive prove (bypasses contributions/basic):

```bash
cargo run --bin proofman-cli -- prove-recursive \
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
     -b ./examples/test-recursive/build -c ./examples/test-recursive/test.circom -n test -t compressor \
     --hash Poseidon1 \
&& cargo build --workspace \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build/provingKey/ \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build/provingKey/ \
     --output-dir examples/test-recursive/build/proofs -y -vv --gpu
```

Direct recursive prove (bypasses contributions/basic):

```bash
cargo run --bin proofman-cli -- prove-recursive \
     --proof examples/test-recursive/poseidon1/ag0_air0_tCompressor.bin \
     --proving-key examples/test-recursive/build/provingKey/ \
     --gpu -vv
```
