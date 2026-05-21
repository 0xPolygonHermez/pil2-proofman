## Execute the Recursive Example


## Platform Compatibility

Detect your platform and set the appropriate library extension:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
```
### Generate Setup

After compiling the PIL files, generate the setup:

```bash
cargo run --bin proofman-setup -- setup-recursive-test \
     -b ./examples/test-recursive/build -c ./examples/test-recursive/test.circom -n test
```

To run the aggregated proof, need to add `-t aggregation` to the previous command

### Build the Project

Build the project with the following command:

```bash
cargo build --workspace
```

### Verify Constraints

Verify the constraints by executing this command:

```bash
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build/provingKey/
```

### Generate Proof

Finally, generate the proof using the following command:

```bash
     cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT}\
     --proving-key examples/test-recursive/build/provingKey/ \
     --output-dir examples/test-recursive/build/proofs -y -vv
```

### All at once

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- setup-recursive-test \
     -b ./examples/test-recursive/build -c ./examples/test-recursive/test.circom -n test \
&& cargo build --workspace \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build/provingKey/ \
&& cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT}\
     --proving-key examples/test-recursive/build/provingKey/ \
     --output-dir examples/test-recursive/build/proofs -y -vv
```