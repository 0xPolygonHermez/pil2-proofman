## Execute the Recursive C36 Example


## Platform Compatibility

Detect your platform and set the appropriate library extension:

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi)
```
### Generate Setup

After compiling the PIL files, generate the setup:

```bash
node ../pil2-proofman-js/src/main_setup_recursive.js \
     -b ./examples/test-recursive-c12/build -c ./examples/test-recursive-c12/test.circom -n test -t pil2-components/lib/std/pil -l 12
```

To run the aggregated proof, need to add -r to the previous command

### Build the Project

Build the project with the following command:

```bash
cargo build
```

### Verify Constraints

Verify the constraints by executing this command:

```bash
cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_c12${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive-c12/build/provingKey/
```

### Check setup

```bash
cargo run --bin proofman-cli check-setup --proving-key examples/test-recursive-c12/build/provingKey
```

### Generate Proof

Finally, generate the proof using the following command:

```bash
     cargo run --bin proofman-cli prove \
     --witness-lib ./target/debug/libtest_c12${PIL2_PROOFMAN_EXT}\
     --proving-key examples/test-recursive-c12/build/provingKey/ \
     --output-dir examples/test-recursive-c12/build/proofs -y -vv
```