## Execute the Recursive Example

This example recursively verifies a previously generated regular (inner) STARK
proof. The inner-proof fixtures live in per-hash-family subfolders and are
selected automatically by `--hash`:

```
examples/test-recursive/
├── poseidon1/   test.circom  test.verifier.circom  proof.bin   (Poseidon1 / Hades)
└── poseidon2/   test.circom  test.verifier.circom  proof.bin   (Poseidon2, default)
```

`setup-recursive-test -c .../test.circom --hash <H>` resolves the fixture to
`.../<h>/test.circom` (lowercased family). The witness library reads the matching
`<h>/proof.bin` at run time (from `pctx.global_info.hash`), so a single
`libtest_recursive` works for both families.

## Platform Compatibility

Detect your platform and set the appropriate library extension:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
```

## Poseidon2 (default)

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

Use `-t compressor` instead of `-t aggregation` for the compressor variant.

## Poseidon1 (Hades)

Same flow, but pass `--hash Poseidon1` to the setup. The hash family is selected
at runtime from the setup, so no feature flag or rebuild is needed — the prover
picks up the correct hash family automatically:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- setup-recursive-test \
     -b ./examples/test-recursive/build -c ./examples/test-recursive/test.circom -n test -t aggregation \
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