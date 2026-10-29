# Domain Native Example

One AIR, `DomNative` (`N = 2^10`), for the native Rust verifier (`verifier/`) on domain constraints.
`cli/tests/domain_native_e2e.rs` runs the steps below, CPU only.

| Constraint | Domain |
|---|---|
| `(a - PHASE) * (a + alpha) * (b + beta) === 0` | `!Q1`, the complement of a periodic domain (`Q1 = [1,0,0,0]`) |
| `b === 0` | `ODD = [0,1]`, a direct periodic domain |

`PHASE = [0..3]...` is a fixed column. `alpha` and `beta` are stage-2 challenges. The first constraint has
degree 4 on its domain, so setup splits it with im pols; those are the stage-2 columns, and the prover
computes them from the challenges.

The witness breaks each constraint outside its domain: `a` is `100 + i` on the `Q1` rows, and `b` is 5 on
the even rows.

## Why it is not in `examples/domains`

The native verifier replays a transcript with exactly two stage-2 challenges and reads no airgroupvalues
(`verifier/src/replay.rs`, `prefix_transcript`). Challenges are global to a pilout. In `examples/domains`
every AIR uses a std permutation, which gives each AIR `std_alpha`, `std_gamma` and the
`Domains.gsum_result` airgroupvalue. A `DomNative` there would have four stage-2 challenges, and its
`alpha`/`beta` would also change the challenge map of the other AIRs. And every AIR there has no fixed
columns, which the native verifier refuses; `PHASE` gives `DomNative` a constants group.

## Commands

From the repository root, CPU only. `PIL2C_EXEC` must point at a pil2-compiler with domains;
`pil/build.pilout` is checked in.

```bash
node $PIL2C_EXEC examples/domain-native/pil/build.pil -o examples/domain-native/pil/build.pilout
cargo run --release --bin proofman-cli -- pil-helpers \
     --pilout examples/domain-native/pil/build.pilout --path examples/domain-native/src -o
cargo run --release --bin proofman-setup -- setup \
     -a examples/domain-native/pil/build.pilout -b examples/domain-native/build
cargo build --release -p domain_native
cargo run --release --bin proofman-cli -- prove-air \
     --witness-lib target/release/libdomain_native.so --air DomNative \
     --proving-key examples/domain-native/build/provingKey --save-proof DomNative.bin
```

`--save-proof` writes the proof as `[nPublics, publics…, proof…]`, the native verifier's input. To render
the AIR's native verifier, run `STARKINFO=… VERIFIERINFO=… OUT=… HASH=blake3 cargo test -p pil2-stark-setup
--test render_rust_verifier -- --ignored`.

`DOMAIN_NATIVE_TAMPER="<col>:<row>:<value>"` overwrites one cell after the witness is filled. Add
`--no-verify` to save a tampered proof. `a:1:9` and `b:3:5` each break a constraint inside its domain.
`a:4:9` falls outside both domains.
