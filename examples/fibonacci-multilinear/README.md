# Fibonacci Multilinear

A plain Fibonacci AIR proven with the **multilinear prover**.

## Full flow

```bash
# 1. Compile the PIL to a pilout
cargo run --bin proofman-setup -- compile-pil \
    --pil ./examples/fibonacci-multilinear/pil/build.pil \
    -I ./pil2-components/lib/std/pil \
    -o ./examples/fibonacci-multilinear/pil/build.pilout

# 2. Run the setup to generate the proving key and verification key
cargo run --bin proofman-setup -- setup \
    -a ./examples/fibonacci-multilinear/pil/build.pilout \
    -b ./examples/fibonacci-multilinear/build

# 2.1 Optionally, you can generate the stats
cargo run --bin proofman-setup -- stats --multilinear \
    -a ./examples/fibonacci-multilinear/pil/build.pilout \
    -o ./examples/fibonacci-multilinear/build/stats.txt

# 3. Generate the PIL Helpers
cargo run --bin proofman-cli pil-helpers \
     --pilout ./examples/fibonacci-multilinear/pil/build.pilout \
     --path ./examples/fibonacci-multilinear/src -o

# 4. Build the witness library
cargo build -p fibonacci-multilinear

# 5. Generate custom commits
cargo run --bin proofman-cli gen-custom-commits-fixed \
     --witness-lib ./target/debug/libfibonacci_multilinear.so \
     --proving-key examples/fibonacci-multilinear/build/provingKey/ \
     --custom-commits rom=examples/fibonacci-multilinear/build/rom.bin

# 6. Verify the constraints
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libfibonacci_multilinear.so \
     --proving-key examples/fibonacci-multilinear/build/provingKey/ \
     --public-inputs examples/fibonacci-multilinear/src/inputs.json \
     --custom-commits rom=examples/fibonacci-multilinear/build/rom.bin

# 7. Prove the base proofs
cargo run --bin proofman-cli -- prove-multilinear \
    --witness-lib ./target/debug/libfibonacci_multilinear.so \
    --proving-key examples/fibonacci-multilinear/build/provingKey \
    --public-inputs examples/fibonacci-multilinear/src/inputs.json \
    --custom-commits rom=examples/fibonacci-multilinear/build/rom.bin \
    --output-dir examples/fibonacci-multilinear/build/proofs -y
```

With the proving key generated, `cargo test -p proofman-multilinear --test setup_artifact` runs an end-to-end prove+verify against the real setup artifacts (it skips when the proving key is absent).
