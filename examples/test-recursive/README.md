## Execute the Recursive Example

This example recursively verifies a previously generated regular (inner) STARK
proof. The inner-proof fixtures live in per-hash-family subfolders and are
selected automatically by `--hash`:

```
examples/test-recursive/
├── blake3/      test.circom  test.verifier.circom  ag0_air0_tCompressor.bin   (BLAKE3)
├── poseidon1/   test.circom  test.verifier.circom  ag0_air0_tCompressor.bin   (Poseidon1 / Hades)
└── poseidon2/   test.circom  test.verifier.circom  ag0_air0_tCompressor.bin   (Poseidon2, default)
```

`setup-recursive-test -c .../test.circom --hash <H>` resolves the fixture to
`.../<h>/test.circom` (lowercased family). The witness library reads the matching
`<h>/ag0_air0_tCompressor.bin` at run time (from `pctx.global_info.hash`), so a
single `libtest_recursive` works for every family.

**The three files of a fixture must come from one setup.** The `.bin` is nothing but the circom's
input vector, so a `test.circom` from a different circuit -- or a different recursion level -- is not
a weaker fixture, it is a broken one: witness generation dies on an assert and every constraint that
reads the witness then reports as failing, which points at the AIR rather than at the pairing. The
cheap check is to count them, since the two numbers have to be equal:

```bash
python3 - <<'EOF'
import re, os
fam = "blake3"
src = open(f"examples/test-recursive/{fam}/test.circom").read()
main = re.search(r'component main.*?= *(\w+)\(', src).group(1)
body = src[src.index(f"template {main}()"):]
n = sum(eval("*".join(re.findall(r'\[(\d+)\]', m.group(2)) + ["1"]))
        for m in re.finditer(r'signal input (\w+)((?:\[\d+\])*);', body))
print("circom wants", n, "| bin has",
      os.path.getsize(f"examples/test-recursive/{fam}/ag0_air0_tCompressor.bin") // 8)
EOF
```

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

## The fixtures

All three were generated from the fibonacci-square example, one setup per family. `setup -r` emits
the circuits under `build/circom/`, and a `prove --aggregation` run with `PIL2_DUMP_ZKIN=<level>`
captures the matching input blob (see `proofman/src/recursion.rs`). The dump is named
`zkin_ag<N>_air<M>_t<proof_type>.bin`, which is what `prove-air --proof` parses to resolve the setup
path; the committed fixtures keep the older `ag0_air0_tCompressor.bin` spelling, which the same
parser accepts, because the harness names every recursive-test AIR `Compressor`.

### Poseidon1 and Poseidon2 -- recursive2

Their inner proofs are **recursive2** proofs (`FiboCPU_recursive2.circom`, captured with
`PIL2_DUMP_ZKIN=recursive2`), so `-t aggregation` reproduces the production aggregator geometry:
2^17 rows, blowupFactor 3, arity 4, 73 queries, cm 48/12/21. Measured `NUsed` is 120196 (Poseidon1)
and 119594 (Poseidon2) of the 131072 available -- ~91% occupancy, and within 0.5% of each other,
since both verify the same shape of proof.

Note the harness always uses blowupFactor 3 here, while a production **compressor** uses 2 -- so
these numbers are faithful for the aggregator, not for the compressor.

### BLAKE3 -- recursive1

BLAKE3's fixture is a **recursive1** proof (`FibonacciSquare_recursive1.circom`, captured with
`PIL2_DUMP_ZKIN=recursive1`) rather than a recursive2 one. A recursive1 circuit verifies a single
basic proof, so the fixture depends on nothing but the basic prover; a recursive2 one needs two
recursive1 proofs to already be sound, which is exactly what this fixture exists to test. Both build
the same aggregator AIR, so nothing about the geometry is lost.

That AIR is the blake3 aggregator at its production settings -- 2^19 rows, blowupFactor 2, maxDeg 5,
LANES 4 -- and this fixture fills it as:

```
28214 Blake3Node + 6424 Blake3Compress(chunk) + 0 parent  ->  8661 blocks x 56 rows = 485016 NUsed
band: 61477 gate + 22927 plonk rows, in 1761 of the 8661 block interiors
committed: fixed 42 | stage1 264 | stage2 117 | stageQ 12  =  393
```

The chunk uses carry four distinct `flags` values (0, 1, 10, 11). `flags` reaches the AIR as a fixed
column filled per whole 56-row block, so the setup groups the uses by value and gives each group
whole blocks -- one block of rounding, and the only thing that makes a block's flags well defined.

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

## BLAKE3

Same flow, `--hash blake3`. The AIR is 2^19 rows against Poseidon's 2^17, so the setup, the
constraint check and the proof all take noticeably longer, and the prover wants ~7.7 GB.

Full pipeline (with constraint verification):

```bash
export PIL2_PROOFMAN_EXT=$(if [[ "$(uname -s)" == "Darwin" ]]; then echo ".dylib"; else echo ".so"; fi) \
&& cargo run --bin proofman-setup -- setup-recursive-test \
     -b ./examples/test-recursive/build3 -c ./examples/test-recursive/test.circom -n test -t aggregation \
     --hash blake3 --gen-exps \
&& cargo build --workspace \
&& cargo run --bin proofman-cli verify-constraints \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build3/provingKey/ \
&& cargo run --bin proofman-cli -- prove-air \
     --proof examples/test-recursive/blake3/ag0_air0_tCompressor.bin \
     --proving-key examples/test-recursive/build3/provingKey/ \
     --gpu -vv
```

Drop `--gpu` to prove on CPU. `prove --witness-lib` runs the other path -- contributions, basic
proof, then recursion -- over the same proving key:

```bash
cargo run --bin proofman-cli -- prove \
     --witness-lib ./target/debug/libtest_recursive${PIL2_PROOFMAN_EXT} \
     --proving-key examples/test-recursive/build3/provingKey/ \
     --output-dir ./examples/test-recursive/build3/proofs -vv
```

`verify-constraints --debug` additionally reports the lookup buses by opid, which is the only thing
that catches a multiplicity landing on the wrong table row: the AIR's algebraic constraints all pass
in that case, because the tuple involved is bound by its lookup alone.

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