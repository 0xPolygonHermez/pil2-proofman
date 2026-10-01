# Golden plonk2pil

`manifest.sha256` freezes what plonk2pil (`setup/stark-recurser/plonk2pil`) returns
for the circuits of the `test-recursive` CI job, in every configuration that job
sets up, and for the blake3 circuit beside them. A change to plonk2pil that must
not change its outputs (making it generic over the field, for one) is correct
only if it leaves all of them byte-identical, and `check.sh` is how to show that.

## Usage

```bash
setup/golden/plonk2pil/check.sh    # build the driver, regenerate, compare
```

`check.sh` builds the driver, runs `generate.sh` into `target/golden-plonk2pil`
and compares every file with the manifest. It exits non-zero, naming each file,
when one is:

- `MISSING`: in the manifest but not produced;
- `DIFFERS`: produced with a different sha256;
- `UNEXPECTED`: produced but not in the manifest.

`check.sh <dir>` compares an existing `generate.sh` output directory instead of
regenerating. `generate.sh [dir]` only regenerates; it wipes `dir` first (and
refuses to wipe a non-empty directory it did not create) and writes
`dir/manifest.sha256`. Both honour:

| Variable | Effect |
|---|---|
| `PLONK2PIL_GOLDEN_OUT` | Output directory (default `target/golden-plonk2pil`) |
| `PLONK2PIL_GOLDEN_BIN` | Use this `plonk2pil_golden` driver instead of building one |

A full run takes about 140 s on a large machine, plus the first build. Most of
it is circom: it compiles the three circuits in parallel, and the blake3 one
alone takes about two minutes. plonk2pil runs all six configurations in about
12 s, which is why the driver is built in release: unoptimized, it takes about
seven times longer. The run writes 1.2 GB, 1 GB of it hashed. Neither Node.js
nor pil2-compiler is needed; circom is the committed `setup/circom/circom`.

The scripts share `../lib.sh` (the output directory, the manifest format and the
comparison) with the STARK golden of `setup/golden`.

## What is covered

The inputs are the three fixtures of `examples/test-recursive`,
`<family>/test.circom` with its `test.verifier.circom`, which
`setup-recursive-test -c examples/test-recursive/test.circom --hash <H>`
resolves to. `generate.sh` compiles them with `setup/circom/circom` (2.2.3) and
the flags and libraries of `gen_recursive_test_setup`
(`setup/pil2-stark/src/proving_key/recursive_test.rs`):

```bash
circom --O2 --r1cs --prime goldilocks --c --verbose \
    -l setup/stark-recurser/stark2circom/circom_verifier/helper_circuits \
    -l setup/stark-recurser/stark2circom/circom_verifier/circuits.gl \
    examples/test-recursive/<family>/test.circom -o <dir>
```

circom writes the same constraint system in a different order from run to run,
so its r1cs cannot be hashed as it is. The driver rewrites it in a canonical
form: the sections in type order, the terms of each linear combination by wire,
and the constraints and the custom-gate uses sorted by their bytes. plonk2pil
reads both forms to the same system, since it keys each combination by wire and
sorts the constraints and the uses itself, so the canonical r1cs is the one it
is run on and the one that is hashed (`r1cs/<family>.r1cs`). The driver parses
the file with its own code, not with plonk2pil's reader, so a change to the
reader does not move the digest of an input.

Each configuration runs plonk2pil with the options `gen_recursive_test_setup`
passes: airgroup `Compressor`, `merge_copies`, and the `max_constraint_degree`
that `recursive_blowup` gives for the template and the family.

| Configuration | Input | Setup type | maxDeg | In CI's `test-recursive` |
|---|---|---|---|---|
| `poseidon1-compressor` | `poseidon1` | compressor | 5 | yes |
| `poseidon1-aggregation` | `poseidon1` | aggregation | 8 | yes |
| `poseidon2-compressor` | `poseidon2` | compressor | 5 | yes |
| `poseidon2-aggregation` | `poseidon2` | aggregation | 8 | yes |
| `blake3-compressor` | `blake3` | compressor | 3 | no |
| `blake3-aggregation` | `blake3` | aggregation | 5 | no |

The driver, `setup/stark-recurser/examples/plonk2pil_golden.rs`, writes for each
one, under `<configuration>/`:

- `air.pil`: `pil_str`, as it is;
- `air.exec`: `exec` as u64 little-endian, the bytes the setups write to
  `<air>.exec`;
- `fixed/<name>.<index>.bin`: the values of one `fixed_pols` entry, as u64
  little-endian;
- `result.json`: `nBits`, `nBitsNatural`, `nUsed`, the airgroup and air names,
  and the `fixed_pols` entries in their order.

The row placement shows up in the exec map and the fixed columns, and with it
the two orders that are easiest to move by accident: the sort of the
constraints by the canonical values of their coefficients in the r1cs reader,
and blake3's sort on the hex spelling of `ckey`.

When the manifest was generated, the driver's `air.exec` and `air.pil` were
checked, for all six configurations, to be byte-identical to the
`Compressor.exec` and `Compressor.pil` that `proofman-setup setup-recursive-test`
writes for the same setup type and family, and its fixed columns to the values
of that run's `Compressor.fixed.bin`.

## What is not covered

- The options of the other callers of plonk2pil: the production recursion
  (`recursive.rs`, with its own airgroup names, no `max_constraint_degree` for
  the aggregation templates and a `min_n_bits` floor when a setup is reused),
  `vadcop_final` (`final_setup.rs`), `vadcop_final_compressed`
  (`compressed_final.rs`) and recursivef (`snark_setup.rs`).
- Their circuits. Those are generated from a setup's starkinfo by pil2circom and
  gen_circom rather than committed, and recursivef needs a `vadcop_final`
  verifier circuit.
- Everything after plonk2pil: pil2com, pil_info and the constant tree.

### Adding an r1cs

`generate.sh` takes its inputs from the `inputs` table, one `name|source` line
each. A `.circom` source is compiled as above; a `.r1cs` source, absolute or from
the repo root, is an r1cs built elsewhere, used as it is. That is the way in for
recursivef: point a line at the `build/recursivef.r1cs` that
`proofman-setup setup-snark` leaves in its build directory, and add a `configs`
line with the options `snark_setup.rs` passes, which are airgroup `Recursivef`,
no `max_constraint_degree` and the family of that setup:

```bash
inputs+=("recursivef|<build_dir>/build/recursivef.r1cs")
configs+=("recursivef-aggregation|recursivef|aggregation|--hash Poseidon2 --airgroup Recursivef")
```

The driver also takes `--min-n-bits` and `--blake3-lanes`, the two options the
current configurations leave unset. An r1cs that is not in the repo has to be
built by a step of its own before CI can check it.

## When it fails

- An `r1cs/*.r1cs` entry differs: an input changed (a fixture, the circom
  libraries under `setup/stark-recurser/stark2circom/circom_verifier`, or
  circom), and the outputs that differ with it say nothing about plonk2pil.
- Only outputs differ: plonk2pil's behaviour changed. `result.json` says whether
  the size or the column list moved; the CI job `golden-setup` uploads every
  `air.pil` and `result.json`, with the manifest and the logs, as the artifact
  `golden-plonk2pil`. The execs and the fixed columns run to hundreds of MB, so
  diff those against a local run.

Update the manifest only when a change of output is intended, and say why in the
pull request:

```bash
setup/golden/plonk2pil/generate.sh && cp target/golden-plonk2pil/manifest.sha256 setup/golden/plonk2pil/
```

## Baseline

Generated at `98d0444b` with the committed circom 2.2.3. Two generations, each
compiling the circuits afresh, gave the same manifest, and the canonical r1cs
came out the same with and without circom's `--c`.
