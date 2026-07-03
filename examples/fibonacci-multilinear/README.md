# Fibonacci Multilinear

A plain Fibonacci AIR proven with the **multilinear prover**.

## Full flow

```bash
# 1. Compile the PIL to a pilout
cargo run --bin proofman-setup -- compile-pil \
    --pil ./examples/fibonacci-multilinear/pil/fibonacci.pil \
    -I ./pil2-components/lib/std/pil \
    -o ./examples/fibonacci-multilinear/pil/fibonacci.pilout

# 2. Generate the proving key (also emits FibonacciML.mlinfo.bin, the
#    constraint IR consumed by the multilinear prover)
cargo run --bin proofman-setup -- setup \
    -a ./examples/fibonacci-multilinear/pil/fibonacci.pilout \
    -b ./examples/fibonacci-multilinear/build

# 3. Build the witness library
cargo build -p fibonacci-multilinear

# 4. Verify the constraints
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libfibonacci_multilinear.so \
     --proving-key examples/fibonacci-multilinear/build/provingKey/ \
     --public-inputs examples/fibonacci-multilinear/src/inputs.json

# 5. Prove (one .mlproof.bin per AIR instance)
cargo run --bin proofman-cli -- prove --multilinear \
    --witness-lib ./target/debug/libfibonacci_multilinear.so \
    --proving-key examples/fibonacci-multilinear/build/provingKey \
    --public-inputs examples/fibonacci-multilinear/src/inputs.json \
    --output-dir examples/fibonacci-multilinear/build/proofs

# 6. Verify
cargo run --bin proofman-cli -- verify-multilinear \
    --proof examples/fibonacci-multilinear/build/proofs/FibonacciML_0.mlproof.bin \
    --proving-key examples/fibonacci-multilinear/build/provingKey
```

With the proving key generated, `cargo test -p proofman-multilinear --test setup_artifact` runs an end-to-end prove+verify against the real setup artifacts (it skips when the proving key is absent).
