# pilfflonk

pilfflonk proves PIL2 programs with fflonk over BN254. The committed polynomials are packed into a
few polynomials `f_i`, each is committed with KZG, and one SHPLONK opening of all of them is checked
with one pairing. The proof is a handful of points and scalars, and the verifier runs in JS (the
reference) or as a generated Solidity contract.

pilfflonk is a path beside the STARK one, not a generalisation of it. It shares the pilout, the
symbolic setup passes (`pil-info`), the binaries `proofman-setup` and `proofman-cli`, and `libstarks`.
The STARK outputs are byte for byte what they were without it.

## Documents

| Document | What it covers |
|---|---|
| [protocol.md](protocol.md) | The constraint polynomial and its degrees, the layout of the `f_i`, blinding, the transcript, the pieces of `Q`, the SHPLONK opening, the verifier's equations, and the prover's sequence. |
| [formats.md](formats.md) | The `provingKey/` and its files, the `<air>.bin` bytecode, the vkey and its digest, the proof, the witness directory and the calldata. |
| [verifier.md](verifier.md) | The JS verifier, the Solidity verifier, the calldata encoder, the gas, and how the two verifiers are tested against each other. |
| [performance.md](performance.md) | Time and memory on the CPU, the GPU path and its speed-up, and how to reproduce both. |

## Scope

A proof has **one instance of one AIR**. It may have any number of stages and publics, and the
std's buses in `STD_MODE_ONE_INSTANCE`, which closes each bus inside the AIR.

Out of scope, and refused by the setup ([What the setup refuses](#what-the-setup-refuses)):

- more than one AIR, or more than one instance of it;
- air values, airgroup values, proof values and global constraints;
- custom commits, periodic columns and public tables;
- packed trace rows: witness columns declared with `bits(n)`, which the compiler marks with
  `witness_bits` hints;
- the constraint domains of pilout v2. The four of pilout v1 are supported: `everyRow`, `firstRow`,
  `lastRow` and `everyFrame`.

The formats leave room for several instances (the transcript absorbs the number of instances, and
the proof names have scope prefixes), but nothing implements them. There is no native verifier, as
there is none for the existing FFLONK wrap, and no recursion or aggregation with STARK proofs.

The criterion of success: the pil-fflonk examples ported to PIL2 ([Fixtures](#fixtures)) prove with
`proofman-cli pilfflonk prove`, the JS verifier and the Solidity verifier accept the proofs, and any
change to a proof, its publics or the witness is rejected.

## Design

1. **A path beside the STARK one.** The STARK runtime is untouched. `proofman-pilfflonk` depends on
   neither `proofman-common` nor `proofman-witness`, and the STARK runtime stays `F: PrimeField64`.
2. **Rust orchestrates, C++ computes.** The BN254 arithmetic of the prover (MSMs, NTTs, polynomials)
   is C++: ffiasm and rapidsnark's polynomials on the CPU, `pil2-stark/src/bn128/src/{msm,ntt}` on the
   GPU. Rust reaches it through a hand-written FFI, as the STARK path does. No curve library is added:
   Rust only has big integers in the setup (`num-bigint`) and the field type `Bn254` to compute
   witnesses in ([Witness](#witness)).
3. **The setup decides, the prover executes.** The degrees, the layout and the bytecode are fixed by
   the setup; the prover makes no choice.
4. **Explicit errors.** No new global state, no `panic!` and no `exit()` in library code. What is not
   supported fails at setup, not when proving.
5. **An independent verifier.** The JS verifier shares no code with the prover: it replays the
   transcript, computes `Q(ξ)` with the vkey's `qVerifier` and checks the pairing with
   `ffjavascript`. The only code it shares with the prover is the setup's codegen, and the tests check
   that code against a direct walk of the pilout (the oracle, [Tests](#tests)).

## Commands

```
program.pil
  │  proofman-setup compile-pil -P <config with "prime">     compile over BN254
  ▼
program.pilout
  │  proofman-setup setup-pilfflonk --powers-of-tau <ptau>    setup
  ▼
provingKey/
  │  proofman-cli pilfflonk prove --witness <dir> | -w <lib>   prove
  ▼
proof.json, publics.json
  │  proofman-cli pilfflonk verify                             JS verifier
  │  proofman-cli pilfflonk calldata  →  verifyProof            Solidity verifier
  ▼
accepted or rejected
```

### compile-pil

```
proofman-setup compile-pil -p <program.pil> -o <program.pilout> -I <std dir> -P <config.json>
```

`-P` passes a pil2com configuration file through as it is. Its `prime` field, a decimal or
hexadecimal **string**, selects the base field; without it the compiler uses Goldilocks. For pilfflonk
it is BN254's `r`:

```json
{"prime": "21888242871839275222246405745257275088548364400416034343698204186575808495617"}
```

The compiler must honour `prime` (`PIL2C_EXEC` points at it): it is `../pil2-compiler` on its branch
`develop-0.14.0-pil2-fflonk`, which also encodes field values of 64 bits and more correctly. The
pilout's `baseField` is then `r`. Its writers `fixed-to-file` and `extern_fixed_file` work with `u64`
and refuse a field wider than 64 bits, so the fixed columns stay in the pilout, but for those it
declares `#pragma fixed_external`: the compiler writes them without values, and the caller of the
setup gives these ([formats.md#fixed-columns](formats.md#fixed-columns)).

The std works over BN254 through `pil2-components/lib/std/pil/bn254.pil`, selected when `PRIME` is
`r`: `Bn254_Gen[i] = 5^((r−1)/2^i)` for `i ≤ 28` and `Bn254_k = 5^(2^28)`, the root of unity and the
coset shift of the connections. The rest of the std works with `PRIME` already.

### setup-pilfflonk

```
proofman-setup setup-pilfflonk -a <pilout> -b <build dir> --powers-of-tau <ptau>
    [--max-constraint-degree D]   default 9: the degree search tries 2 … D
    [--extra-muls E]              default 2, as pil-stark: f_i the grouping may add
    [--max-q-degree M]            default 0: Q is not split
    [--solidity]                  also write pilfflonk.verifier.sol
    [--no-packing]                for tests: one polynomial per f_i, k = 1
```

The ptau is a snarkjs powers-of-tau file. Only its sections 2 (`[τ^i]₁`) and 3 (`[τ^i]₂`) are read, so
it need not be prepared for phase 2, and it needs at least as many powers as the largest `degree` of
the layout. The largest ptau there is has `2^28` powers (`powersOfTau28_hez_final.ptau`), which bounds
the degree of every `f_i` and `N·2^extendBits` by `2^28`: with `qDeg` up to 8, `N ≤ 2^24`.

The steps, each a pure function but for the file I/O and the fixed commitments:

1. Read the pilout and validate it ([What the setup refuses](#what-the-setup-refuses)), and its
   fixed columns, with the values of the external ones if it has any
   ([formats.md#fixed-columns](formats.md#fixed-columns)).
2. Run the symbolic passes of `pil-info`, the STARK's, with `PilInfoCfg::bn254()`: field `r`,
   extension dimension 1, the degree search of
   [protocol.md#degree-search](protocol.md#degree-search), and no FRI.
3. Derive the committed polynomials, their bounds and `nBitsExt`, and group them into the layout
   ([protocol.md#layout](protocol.md#layout)).
4. Write the bytecode `<air>.bin` and the `qVerifier`
   ([formats.md#bytecode](formats.md#bytecode)).
5. Read the SRS from the ptau, commit the fixed `f_i` in C++
   (`pilfflonk_commit_fixed`), and write the vkey with its digest last
   ([formats.md#vkey](formats.md#vkey)).

A caller in the same process gives the values of the fixed columns that the pilout declares
`#pragma fixed_external` to `run_setup_pilfflonk_with_external_fixed`, the library entry of the
command ([formats.md#fixed-columns](formats.md#fixed-columns)); the command line has no option for
them, and `run_setup_pilfflonk`, which it calls, gives none.

It writes the `provingKey/` of [formats.md#provingkey](formats.md#provingkey), in a deterministic
order, after validating everything. `--solidity` writes the Solidity verifier next to the vkey and
changes no other file: the globalInfo does not record it.

### What the setup refuses

The setup stops with an error, before it writes any file, when:

- the pilout's `baseField` is not BN254's `r` (a Goldilocks pilout says how to compile one over BN254);
- it has custom commits, periodic columns or public tables;
- it has more than one AIR, air values, airgroup values, proof values or global constraints
  ([Scope](#scope)). The hints are checked before the values, so that an `im_airval` is refused by its
  name rather than for the air value it brings;
- it has a prover hint other than `im_col`, `gsum_col` and `gprod_col`. `im_airval` computes an air
  value, and an unknown name is refused. The witness and debug hints (`gsum_debug_data`,
  `gprod_debug_data` and their `_global` forms, `range_def`, `specified_ranges`,
  `specified_ranges_data`, `virtual_table_data`, `virtual_table_data_global`, `std_sum_users`,
  `std_prod_users`, `std_rc_users`) are ignored, and are not written to `<air>.bin`;
- it has a `witness_bits` hint, which asks for packed trace rows (`SetupError::PackedTrace`);
- a column of stage 2 or above is not given by exactly one `im_col`, `gsum_col` or `gprod_col`. This
  is checked after the passes, on the hints `<air>.bin` holds (`validate::check_prover_hints`):
  - the `reference` is a column of a stage ≥ 2, read at its row, and not an im pol;
  - the numerator and the denominator (`numerator` and `denominator` of an `im_col`, `numerator_air`
    and `denominator_air` of a `gsum_col` or `gprod_col`) are an expression, a column at an opening
    point or a number;
  - they read only what the prover computes before the hint: the fixed columns, the earlier stages,
    and of the hint's own stage the columns of the earlier hints in the prover's order
    ([protocol.md#hint-columns](protocol.md#hint-columns)). An `im_col` may read earlier `im_col`, a
    `gsum_col` the `im_col` of its stage; none reads a later `im_col`, its own column or an im pol of
    the stage. The STARK does not check this: its `multiplyHintFields` would read whatever the
    buffer held;
  - an `im_col` belongs to an AIR with a `gsum_col` or a `gprod_col`, as the STARK's
    `calculateImHints` computes none in an AIR without;
  - the `result` of a `gsum_col` or `gprod_col`, if any, is a number, as the std writes it in
    `STD_MODE_ONE_INSTANCE`; anything else would be an airgroup value;
- two columns would have the same name in the proof and the layout, once the setup has indexed the
  im pols and the columns that share a name and have no `lengths`
  ([formats.md#proof-names](formats.md#proof-names)); the error names them;
- a fixed value is not below `r`. A pilout over BN254 has none, and reducing one would hide a
  compiler bug;
- a fixed column has no values, in the pilout or external; or an external fixed column is not one
  the pilout has without values, is given twice, or has not one value per row
  ([formats.md#fixed-columns](formats.md#fixed-columns));
- the extended domain does not fit in BN254's 2-adicity: `nBitsExt > 28`
  ([protocol.md#degrees](protocol.md#degrees));
- the grouping has no valid partition, or `--extra-muls` is too large or makes the search too large
  ([protocol.md#grouping-errors](protocol.md#grouping-errors));
- the ptau has fewer powers `[τ^i]₁` than the largest `degree` of the layout;
- the ptau's `[τ]₂` is not a point of G2: the point at infinity (`τ = 0`), off the twist or outside
  its `r`-torsion group. It would be the vkey's `X_2`, which the JS verifier refuses, and with the
  point at infinity the Solidity verifier's pairing would accept a forged proof
  ([verifier.md#refused-vkeys](verifier.md#refused-vkeys)).

A column that no constraint opens is not committed, and the setup warns about it.

### pilfflonk-solidity

```
proofman-setup pilfflonk-solidity -k <pilfflonk.vkey.json> -o <pilfflonk.verifier.sol>
```

Writes the Solidity verifier of an existing vkey: the same bytes as `setup-pilfflonk --solidity`. It
is how the repository derives an output from a finished key (`gen-exps`), and how snarkjs exports its
verifier (`zkey export solidityverifier`). See [verifier.md#solidity-verifier](verifier.md#solidity-verifier).

### pilfflonk prove

```
proofman-cli pilfflonk prove -k <provingKey> -o <dir> (--witness <dir> | -w <lib> [-i <json>])
    [--insecure-blinding-seed <64 hex digits>] [-g/--gpu] [-v…]
```

Loads the `provingKey/`, takes the witness from a directory or a witness library
([Witness](#witness)), proves, and writes `proof.json` and `publics.json` to `-o`. The sequence is
[protocol.md#proof-sequence](protocol.md#proof-sequence).

- **Blinding** is always on, random by default (libsodium). `--insecure-blinding-seed` fixes it, so
  that the same seed gives the same proof: for tests and CI only, as whoever knows the seed can
  remove the blinding ([protocol.md#blinding](protocol.md#blinding)).
- **`-g/--gpu`** runs the MSMs and the NTTs on the GPU and gives the same proof, bit for bit
  ([performance.md#gpu](performance.md#gpu)). It needs a build that found `nvcc` (without the feature
  `proofman-starks-lib-c/cpu-only`) and a GPU; otherwise it is refused, saying why, before the SRS is
  read. The CPU is the default, and a GPU build without `--gpu` proves on the CPU.
- A witness that does not satisfy the constraints is refused (`UnsatisfiedError`): no proof is
  written that the verifier would have to reject.

### pilfflonk check

```
proofman-cli pilfflonk check -k <provingKey> (--witness <dir> | -w <lib> [-i <json>]) [--max-rows N]
```

Checks the witness row by row against every constraint, without proving, and says which constraint
fails on which rows (the first `--max-rows` of each; all are counted). It is the counterpart of the
STARK's `verify-constraints`. The constraints are those of section 2 of `<air>.bin`, the pilout's
and the ones the im pols add.

The columns of stage 2 and above depend on challenges, which `check` takes as `verify-constraints`
does: from a transcript of fixed elements, with no commitment, no MSM and no blinding. A
[transcript](protocol.md#transcript) absorbs the STARK's `dummy_element`, `[0, 1, 2, r − 1]`, as
four `Fr`; then, for each `s = 1 … nStages − 1`, it squeezes the `numChallenges[s]` challenges of
stage `s + 1`, one per call, and absorbs `[0, 1, 2, r − 1]` again. The C++ core computes the hint
columns and the im pols of each stage with them, as the prover does, in buffers of their own: nothing
is committed, and the same witness always gives the same report. A zero denominator is the prover's
error.

### pilfflonk verify

```
proofman-cli pilfflonk verify <pilfflonk.vkey.json> <publics.json> <proof.json>
```

Runs the JS verifier with Node, as `verify-snark` runs `snarkjs fflonk verify`, with the same three
files. It exits with 0 only if the proof verifies. See [verifier.md#js-verifier](verifier.md#js-verifier).

### pilfflonk calldata

```
proofman-cli pilfflonk calldata -k <pilfflonk.vkey.json> -p <proof.json | proof.bin> --publics <publics.json>
    [--format solidity|hex] [-o <file>]
```

Gives the arguments of the Solidity verifier's `verifyProof` for a proof, as snarkjs's
`zkey export soliditycalldata` gives those of its `FflonkVerifier`. See
[verifier.md#calldata-encoder](verifier.md#calldata-encoder) and the layout in
[formats.md#calldata](formats.md#calldata).

## Witness

The prover takes from outside the stage-1 columns of the instance, its stage-1 air values, the
publics and the stage-1 proof values (none of the values in this version). The columns of stage 2
and above, and the im pols, are the prover's to compute. The values are canonical `Fr`, 32 bytes
little-endian, row after row; the conversion to ffiasm's Montgomery form happens in C++.

The witness comes from one of two sources, chosen with exactly one of `--witness` and `--witness-lib`:

- **A witness directory**, read by `FileWitnessSource` and written by `Witness::write`, which the
  fixtures' generators call ([formats.md#witness-directory](formats.md#witness-directory)).
- **A witness library**: a dynamic library that computes the witness over BN254's `Fr`, as the
  STARK's witness libraries do over Goldilocks, but on a path of its own: no `ProofCtx`, no
  `WitnessManager`.
  - It implements `PilfflonkWitnessLibrary::witness(&mut self, shape, public_inputs) -> Witness` and
    is exported with `pilfflonk_witness_library!(Name)`, which defines the symbol
    `pilfflonk_init_library`. The STARK's loader looks for `init_library`, so neither loader takes
    the other's libraries; `load_witness_library` refuses a STARK library and says so.
  - Its rows are the typed rows `pil-helpers` generates for a BN254 pilout: unpacked rows over `F`
    (`trace_row!` asks `PrimeField64` only of the typed accessors), no `FieldExtension`, and publics of
    type `Bn254`.
  - `-i/--public-inputs <json>` names the public inputs the library reads (only with
    `--witness-lib`). Their values are decimal strings (`"5"`, not `5`): a JSON number is refused.
  - `compute_witness` checks the witness against the shape of the key, as `FileWitnessSource::open`
    checks a directory. The library is never unloaded.

`proofman_fields::Bn254` is BN254's scalar field `Fr`, of order `r` (not the base field `Fq`), in
pure Rust: Montgomery form in four 64-bit limbs, always reduced, the same as ffiasm's `RawFr::Element`.
It implements `Field` and `PrimeField` but not `PrimeField64`; integers convert with `Bn254::from_int`
(a negative `x` is `r − |x|`). `GENERATOR = 5`, `TWO_ADICITY = 28` and `W[i] = 5^((r−1)/2^i)`.
`to_le_bytes`/`from_le_bytes` use the 32-byte canonical little-endian encoding, and serde the
canonical decimal string ([formats.md#json-encoding](formats.md#json-encoding)). It is for computing
witnesses only: the prover's arithmetic is ffiasm's.

## Code map

| Component | Where | Language | What it does |
|---|---|---|---|
| Compiler | `../pil2-compiler`, branch `develop-0.14.0-pil2-fflonk` | JS | the pilout over BN254 |
| Std constants | `pil2-components/lib/std/pil/bn254.pil` | PIL | `GEN` and `k_coset` over BN254 |
| Symbolic passes | `pil-info`, `setup/pil-info/` | Rust | the STARK's passes, parameterised by the field (`PilInfoCfg`); the `"chps"` container; temporaries; `globalConstraints.json` |
| Setup | `pilfflonk-setup`, `setup/pilfflonk/` | Rust | validation, layout, bytecode, keys, `provingKey/`, digest, Solidity |
| Setup commands | `pil2-stark-setup` (`proofman-setup`) | Rust | `setup-pilfflonk` and `pilfflonk-solidity`, which call `pilfflonk-setup` |
| Types and orchestration | `proofman-pilfflonk`, `pilfflonk/` | Rust | the types of every pilfflonk file, the proving key, the stage loop, `WitnessSource`, witness libraries, `check`, the calldata |
| BN254 core | `pil2-stark/src/pilfflonk/` | C++ | the `Fr` interpreter, the LDE on a coset, blinding, packing, MSMs, `Q`, evaluations, the prover side of SHPLONK, the GPU path |
| C API | `pil2-stark/src/api/pilfflonk_api.{hpp,cpp}` | C++ | the surface Rust sees: status codes, never `exitProcess` |
| Bindings | `provers/starks-lib-c/bindings_pilfflonk.rs`, `src/ffi_pilfflonk.rs` | Rust | the hand-written FFI, as `bindings_starks.rs` and `ffi_starks.rs` |
| CLI | `cli/src/commands/pilfflonk/` | Rust | `pilfflonk prove`, `check`, `verify` and `calldata` |
| JS verifier | `pilfflonk/js/` | JS (Node) | transcript, `Q(ξ)`, SHPLONK and the pairing |
| Solidity template | `setup/pilfflonk/src/tera/verifier_pilfflonk.sol.tera` | Tera, Solidity | the contract `setup/pilfflonk/src/solidity.rs` renders |
| Foundry project | `pilfflonk/solidity/` | Solidity | the contract's tests and the fuzzer's harness |
| Benchmarks | `pilfflonk/bench/` | shell, PIL, Rust | `bench.sh` and `gpu_check.sh` ([performance.md](performance.md)) |

**Dependencies.** There are no cycles:

```
proofman-cli ──────────────► proofman-pilfflonk ──────────► proofman-starks-lib-c ──► libstarks (C++, ffiasm)
                                    ▲                               ▲
pil2-stark-setup ─► pilfflonk-setup ┘───────────────────────────────┘  (fixed commitments)
 (proofman-setup)     │     │
       │              │     └──► pil2-pilout
       └──────────────┴────────► pil-info ──► pil2-pilout
```

`pilfflonk-setup` depends on `pil-info`, `proofman-pilfflonk` and `proofman-starks-lib-c`, but not
on `pil2-stark-setup`, which depends on it to host its commands. Only the setup binary gains
dependencies; the STARK runtime gains none.

**File ownership.** Every pilfflonk file has one owner, `proofman-pilfflonk`: the setup writes it and
the prover reads it, both through its types. The C++ core reads `<air>.pilfflonkinfo.json` with
`nlohmann/json`, as it reads `starkinfo.json`, and a round-trip test makes both sides read the same
fixture. `pilout.globalConstraints.json` is `pil-info`'s, for both backends. The JS verifier depends
on no crate: it reads the vkey, the proof and the publics, and an end-to-end test checks that it reads
them as Rust writes them.

## Conventions

- **Errors.** Libraries (`pil-info`, `proofman-pilfflonk`, the library part of `pilfflonk-setup`) use
  `thiserror`, following `common/src/error_manager.rs`; the setup commands use `anyhow`; the CLI
  `Box<dyn Error + Send + Sync>`. No `panic!` or `exit()` in library code.
- **C++.** No global state; RAII; errors go out through the C API. Template bodies go in `.c.hpp`
  files, since the Makefile compiles every `.cpp` of the directories it lists. File names start with
  `pilfflonk_` and the namespace is `PilFflonk`, since every directory is on the include path.
- **Build.** `./src/api/pilfflonk_api.*` and `./src/pilfflonk` are in the Makefile's source lists of
  the CPU and GPU libraries. The `*_gpu.cpp` files (compiled with g++) and `pilfflonk_kernels.cu`
  (with nvcc) only go into `libstarksgpu.a`.
- **Logs and timers.** `tracing`, and the C++ timers `TimerStart`/`TimerStopAndLog` with names
  `PILFFLONK_*`, logged at trace level (`-vv`).
- **Format and lints.** `rustfmt` (`max_width = 120`) and `clippy -D warnings`.

### C API

- **Encoding.** A scalar is 32 bytes, canonical, little-endian (`FrBytes`); a G1 point is 64 bytes,
  `x‖y`, each coordinate little-endian. The transcript encodes big-endian internally
  ([protocol.md#transcript](protocol.md#transcript)).
- **Handles.** Every object is opaque and has a `_free` function.
- **Errors.** A function that creates an object returns `NULL` on error; the others return an `int`
  status. After a failure, `pilfflonk_last_error` describes it and `pilfflonk_last_status` returns
  its status, for the calling thread. Nothing calls `exitProcess`, with one exception: a CUDA failure
  inside the GPU helpers the GPU path reuses aborts the process, as in the PLONK GPU prover
  ([performance.md#gpu](performance.md#gpu)).
- **Transcript.** Rust decides what is absorbed and when, and absorbs the evaluations itself; the
  transcript object is C++'s. `pilfflonk_open` squeezes `α_S`, absorbs `W` and squeezes `y`.

## Tests

- **Powers of tau.** No ptau is downloaded and no JS runs to make one: the tests write small ones,
  with only sections 1 to 3, which is all pilfflonk reads. `pilfflonk_setup::test_ptau` (feature
  `test-ptau`) writes one with `τ = 1`, whose commitments a test knows in advance but with which
  anyone can open anything, and one with a fixed full-width `τ` (`TEST_TAU`), the C++ helper's
  (`pil2-stark/test/pilfflonk/pilfflonk_test_ptau.hpp`), with which a proof verifies only if it is
  sound. Outside the tests the ptau is an input of the setup.
- **The oracle.** `proofman_pilfflonk::oracle` (feature `oracle`, tests only) evaluates a pilout's
  constraints, the std's hint columns and `Q` from the pilout alone, with `num-bigint`. The prover's
  bytecode, the `qVerifier` and `pilfflonk check` are checked against it.
- **Cross-checks with the JS verifier.** The C++ SHPLONK tests write their openings as fixtures, and
  `pilfflonk/js/test/fixtures.sh` runs the JS tests on them: the JS verifier accepts every opening and
  refuses it with any bit changed, signed offsets and shared roots included. The C++, Rust and JS
  transcripts are pinned to the same challenges. `pilfflonk/js/test/setup-fixtures.sh` sets up the
  Fibonacci (with the `τ = 1` ptau) and runs the JS tests on its key and on the oracle's `Q(ξ)`.
- **The compiler.** The end-to-end tests compile their programs with `PIL2C_EXEC`, and are
  `#[ignore]` without it.
- **Foundry.** The Solidity tests need `PILFFLONK_FORGE` and `PILFFLONK_SOLC`
  ([verifier.md#tools](verifier.md#tools)), and are `#[ignore]` without them.
- **Determinism.** The tests fix the blinding with a seed, so that a proof is the same on every run.
- **Threads.** Each test thread that calls the C++ core gets an OpenMP team of its own, and libomp 14
  (Ubuntu 22.04's) can crash once the teams outgrow its first table of threads (4 per CPU): it
  replaces the table while workers may still read the old one. The tests that run the C++ core's
  OpenMP code hold a lock (`cpp_core`) so that they run one at a time; run the suites with
  `--test-threads 2`.
- **C++.** `make -C pil2-stark pilfflonk_test` runs the C++ tests against the CPU library, and
  `make -C pil2-stark pilfflonk_gpu_test` the same tests against the GPU library
  ([performance.md#gpu](performance.md#gpu)).

## Fixtures

The fixtures are in `pilfflonk/tests/fixtures/`, and their witness generators in
`pilfflonk/tests/data/`.

**The pil-fflonk examples, ported to PIL2.** pil-fflonk's `config/` is the output of pil-stark's
example `all` (`test/state_machines/sm_all/all_main.pil`, `N = 2^8`, inputs `[1, 2]`, `extraMuls: 2`,
`maxQDegree: 0`): the Fibonacci, Connection, Permutation and Plookup state machines in one trace, with
the publics `[1, 2, out]` of its `runtime/public.json`. Each is a fixture here, and `all` too, each in
one AIR. Each has a file with the state machine and two programs to compile, `<example>_sum.pil` and
`<example>_prod.pil`, which fix its bus type and use the std in `STD_MODE_ONE_INSTANCE`. The fixed
columns are computed in PIL, as the PIL1 generators computed them.

| PIL1 | Fixture | `N` | What changes |
|---|---|---|---|
| `sm_fibonacci/fibonacci.pil` | `fibonacci/fibonacci.pil` | 2^8 | The publics are inputs, tied to the trace with the fixed columns `L1` and `LLAST` (the compiler emits only `everyRow` constraints). |
| `sel {a, b', a*b'} in SEL {A, B, cc}` | `plookup/plookup.pil` | 2^8 | Sum bus: `lookup_assumes(1, [a, b', a·b'], sel)` and `lookup_proves(1, [A, B, cc], mul)`, with `mul` a new column, the times each row of the table is looked up, and `(1 − SEL)·mul = 0`. Product bus: the std's lookup has none, as a product cannot take multiplicities, so the tuples go to the std's permutation with `mul` as the table's selector. PIL1's plookup (`h1`/`h2` and `Z`) is the std's bus. |
| `selC {c, c} is selD {d, d}` | `permutation/permutation.pil` | 2^8 | `permutation_assumes(2, [c, c], selC)` and `permutation_proves(2, [d, d], selD)`. `a` and `b` are read by no constraint, so they are not committed, as in PIL1, and the setup says so. |
| `{a, b, c} connect {S1, S2, S3}` | `connection/connection.pil` | 2^10 (2^8 in `all`) | `connection(3, [a, b, c], [S1, S2, S3])`, the `S_i` from the identity `k^(i−1)·ω^row` with BN254's `k_coset` and `GEN[BITS]`, and the swaps of `sm_connection.js` in the same order. |
| — | `range_check/range_check.pil` | 2^6 | New: `v ∈ [0, 16)` against a fixed table of the same AIR. Sum bus: a lookup. Product bus: as plookup, `h1` then `h2` are the values of `v` and of the table sorted, which the std's permutation checks. The std's own range check cannot be used ([Upstream issues](#std-range-check)). |
| `sm_all/all_main.pil` | `all/all.pil` | 2^8 | One AIR, `All`, which calls the four state machines as PIL2 functions. A PIL2 AIR has one scope and the proof names each evaluation by its column, so the witness columns take their state machine's name as a prefix (`connection_a`, `permutation_a`, `plookup_a`). PIL1's `Global.L1` is the std's `__L1__`. |

The 9 fixed columns of `all`, compiled by pil2com, are those of `pil-fflonk/config/pilfflonk.const`
value for value, the 15 PIL1 witness columns `tests/data/all.rs` writes are those of
`pilfflonk.commit`, and its publics those of `runtime/public.json`.
The columns of stage 2 are the std's (`gsum` with `im_single`/`im_cluster` on the sum bus, `gprod`
with `im_low` on the product bus), each with its hint.

**Synthetic fixtures.**

- `packed/`: enough columns of each kind for the grouping to pack them (`k = 3, 4, 4`, `powerW = 12`).
- `signed/`: columns read at the rows `−1` to `2`, and a constraint of degree ≥ 4 that needs an im pol.
- `sum_bus/`, `prod_bus/`, `prod_bus_im/`: one bus of the std each, of two stages; `prod_bus_im` with
  `im_col` columns.
- `pilfflonk/tests/data/domains.rs`: pilouts built in code with `prost`, for the domains the compiler
  does not emit (`firstRow`, `lastRow` and `everyFrame`).
- `std_bn254/`: the std's connections over BN254, against `bn254.pil`.
- `packed/packed_external.pil` and `connection/connection_external.pil`: `packed.pil` and
  `connection_prod.pil` with fixed columns declared `#pragma fixed_external` (`K[4]` and `S`;
  `S1`, `S2` and `S3`), whose values the setup takes from its caller
  ([formats.md#fixed-columns](formats.md#fixed-columns)). Their pilouts are the others' without
  those values, source lines aside, and give the same keys and proofs
  (`setup/pilfflonk/tests/setup/external_fixed.rs`).

**Witness libraries.** `fibonacci/rs`, `connection/rs` and `all/rs` are the witness libraries of
those fixtures (crates `pilfflonk-fibonacci`, `pilfflonk-connection` and `pilfflonk-all`, `dylib`,
workspace members that are not default members). Their `pil_helpers` are versioned, as the BN254
pilout needs `PIL2C_EXEC`, and a test checks that they are what `pil-helpers` writes. One library
serves both buses of a fixture: the two pilouts have the same stage-1 columns and publics. Their
witness is the generator's, byte for byte.

## Upstream issues

Defects found in code pilfflonk uses but does not own. None is fixed here unless it says so;
pilfflonk works around them.

### STARK lastRow zerofier

`pil2-stark/src/starkpil/setup_ctx.hpp:104` calls `buildOneRowZerofierInv(..., N)`, which would use
the root `ω^N = 1`, where `:47-56` uses `ω^(N−1)`; and `:57` seems to test `everyRow` where it means
`everyFrame`. Not verified: the compiler only emits `everyRow`, so no real AIR reaches it. pilfflonk's
`lastRow` zerofier is `X − ω^(N−1)`.

### rapidsnark and ffiasm

- `CPolynomial::getPolynomial` has undefined behaviour below degree 2 (`std::log2(0)`), returns one
  coefficient short when the packed degree is a power of two, and clears a power-of-two prefix of the
  buffer. `PilFflonk::pack` calls `CPolynomial::getCoefficients`, added to rapidsnark next to it,
  which packs the same coefficients without these.
- `Polynomial` leaked memory in `divByMonic`, `lagrangePolynomialInterpolation`, `byXSubValue`,
  `fastDivByVanishing` and when it took ownership of a reserved buffer. **Fixed in rapidsnark**,
  without changing any result; the wrap's `FflonkProver` benefits too.
- `divByMonic` writes before its buffer when the degree is below `2m − 1`, does not check the
  remainder, and runs on `m` threads; `PilFflonk::divideExactly` calls
  `Polynomial::divByMonicInPlace`, added to rapidsnark next to it, which gives the same quotient at
  every degree, in place and in parallel, and whether the remainder is zero
  ([performance.md#the-shplonk-division](performance.md#the-shplonk-division)).
- `BinFile`'s direct-read mode dereferences a null pointer, and `readSectionToParallel` throws inside a
  `std::thread`; pilfflonk only uses `readSectionTo`.
- `multiexp.c.hpp:30` reads 8 unaligned bytes on every MSM (UBSan).
- `Keccak256Transcript` encodes the point at infinity wrongly, and ffiasm's `RawFq::toRprBE` writes a
  coordinate below `2^192` wrongly; pilfflonk never absorbs such points
  ([protocol.md#transcript](protocol.md#transcript)).
- ffjavascript 0.3.1: `G1.sub(a, b)` with `a` affine and `b` Jacobian returns `b − a`, so the JS
  verifier works in Jacobian coordinates; `Fr.e(v)` does not reduce `v = p`, so its decoders check
  `< r` and `< q` themselves.

### std product bus

In `std_prod.pil` (`piop_gprod_air`), a term of degree above `MAX_CONSTRAINT_DEGREE` is reduced with
an `im_high`, and `gprod_e[term] = im_high[idx]` reassigns an element of a `const expr` array: the
compiler refuses it, so no pilout has an `im_high`. The fixture `prod_bus_im` takes the path of the
low-degree terms (`im_low`), which compiles and chains the `im_col`.

When the last group of low-degree terms reaches `MAX_CONSTRAINT_DEGREE` exactly in both numerator
and denominator, a second `im_col` comes out as the inverse of the first, and `gprod` accumulates the
square of each row's ratio. The bus still balances, at 1, and stays sound, but costs a column and a
degree. `all` on the product bus has it (`im_low[3] = 1/im_low[2]`).

### std sum bus names

`piop_gsum_air` (`std_sum.pil`) declares `im_single` and `im_cluster` inside a loop, each with that
name and no index, so an AIR with two of them has two columns of the same name. The STARK does not
mind, but the proof names each evaluation by its column: the setup gives the columns of a map that
share a name and have no `lengths` an index in the pilout's order, `im_cluster[0]`, `im_cluster[1]`,
… ([formats.md#proof-names](formats.md#proof-names)). The std, the STARK and its golden files do not
change.

### std range check

The std's `range_check` declares its table in an AIR of its own, so its pilout has two AIRs, which
pilfflonk refuses; and in `STD_MODE_ONE_INSTANCE` each AIR closes its bus alone, so the AIR that only
looks up would not balance. The std's lookup has no product bus either: a multiplicity would be an
exponent. The range check fixture looks up a fixed table of its own AIR.
