# pilfflonk verifiers

Two verifiers check a pilfflonk proof against its vkey and publics: the JS verifier, which is the
reference, and a Solidity contract generated from the vkey, which accepts exactly the proofs the JS
verifier accepts. Both compute what [protocol.md#verifier-equations](protocol.md#verifier-equations)
says; the files they read are in [formats.md](formats.md).

## JS verifier

`pilfflonk/js/` (Node 18 or later), `verify(vkey, publics, proof, logger) → true | false` in
`src/verify.js`, and the command:

```
node pilfflonk/js/bin/verify.js <pilfflonk.vkey.json> <publics.json> <proof.json>
```

It exits with 0 if the proof verifies, 1 if it does not (a malformed input included) and 2 if the
arguments are wrong or a file cannot be read as JSON. pil-stark's old `main_verifier.js` exited with 0
either way; this one does not. `proofman-cli pilfflonk verify` runs it with Node, as `verify-snark`
runs snarkjs (`proofman_pilfflonk::js_verifier`). The verifier is the crate's `js/` directory, or the
copy `PILFFLONK_JS` names. Its dependencies, `ffjavascript` (BN254 and the pairing) and
`@noble/hashes` (Keccak-256), those of snarkjs 0.7.6, are looked for as Node looks for them, in a
`node_modules/` of that directory or of an ancestor; if one is missing, `npm install` runs there.

**Template.** snarkjs's FFLONK verifier (`src/fflonk_verify.js`), the existing FFLONK's: its
interface, its steps 1–3 (commitments in G1, evaluations and publics in `Fr`), its transcript pattern,
its computation of `F`, `E` and `J`, and its final `pairingEq`. It departs from snarkjs only where
snarkjs is specific to circom's PLONK circuit:

- snarkjs has three fixed polynomials (`C0`, `C1`, `C2`, with `k` = 8, 4 and 3) and their opening
  points; pilfflonk reads the list of `f_i`, with their `k` and offsets, from the vkey, and generalises
  `computeR0/1/2` as shplonkjs's `verifyOpenings` does;
- snarkjs writes the PLONK identity in its code; pilfflonk computes `Q(ξ)` with the vkey's
  `qVerifier`, as pil-stark's fflonk verifier runs its `verifierCode`;
- snarkjs fixes its challenges `β`, `γ`, `α` and `y`; pilfflonk squeezes those of each stage from
  `numChallenges`.

snarkjs only exports its public API, so the transcript and the helpers are copied. The transcript
(`src/transcript.js`) reproduces rapidsnark's `Keccak256Transcript`, as snarkjs's
`Keccak256Transcript.js` does for FFLONK. The verifier works in Jacobian coordinates
([README.md#rapidsnark-and-ffiasm](README.md#rapidsnark-and-ffiasm)).

## Steps

Both verifiers take these steps, in this order, and name them the same way. Each refuses the proof
at the first check that fails.

- **1–3. Input** (JS: `vkey.js`, `proof.js`, `elements.js`; Solidity: `checkInput`). The vkey
  decodes, its `X_2` is a point of G2 in its `r`-torsion, and its digest is that of its content
  ([formats.md#digest](formats.md#digest)); in Solidity the vkey is the contract's constants. The
  commitments, `W` and `W'` are affine points of G1 with coordinates below `q`, not `(0, 0)`, on
  `y² = x³ + 3`; the points the transcript absorbs (the commitments and `W`) have no coordinate below
  `2^192`; the evaluations, the pieces of `Q`, `inv`, `invZh`, the auxiliary inverses and the
  publics are below `r`; the proof holds exactly the values the vkey names.
- **4. Transcript** (`challenges.js`, `computeChallenges`). The
  [transcript](protocol.md#transcript), replayed. In Solidity it is a buffer in memory: each squeeze
  is `keccak256` of the buffer mod `r`, and the buffer becomes the challenge. A stage's commitments
  and the evaluations (with the pieces of `Q`) are consecutive in the calldata and absorbed with one
  `calldatacopy`, from the first scalar: the first evaluation or, if there is none, the first piece
  of `Q` in the order of the layout, which is not always `Q0`.
- **5. `Q(ξ)`** (`qverifier.js`; Solidity: `computeZh`, `computeZi`, `computeQ`, `checkQPieces`).
  `ξ = xiSeed^powerW`, `Z_H(ξ)·invZh = 1`, the `Zi` of each boundary (with the auxiliary inverses in
  Solidity), `Q(ξ)` from the `qVerifier` (in Solidity, unrolled: one Yul instruction per entry, the
  temporaries in memory), and, if `Q` is split, `Σ_i ξ^(i·M·N)·Q_i(ξ) = Q(ξ)` (`joinQPieces`).
- **6. Inverses** (`shplonk.js`; Solidity: `computeRoots`, `computeInversions`). `xiSeed ≠ 0`; the
  roots of each distinct `(k, offsets)`; `Z_{T_i}(y) ≠ 0`; the denominators `inv` inverts, in the
  order of [protocol.md#inverses](protocol.md#inverses), and `inv·Π = 1` (`isValidInverse`; in
  Solidity `inverseArray` with the proof's `inv`, checked before it is used).
- **7. Pairing** (`shplonk.js`; Solidity: `computeR`, `computeFEJ`, `checkPairing`). `r_i(y)`, `F`,
  `E` and `J`, with the vkey's fixed commitments, and the pairing with `X_2`
  ([protocol.md#pairing-check](protocol.md#pairing-check)).

## Solidity verifier

`proofman-setup setup-pilfflonk --solidity` writes `pilfflonk.verifier.sol` next to the vkey, and
`proofman-setup pilfflonk-solidity -k <vkey> -o <file>` writes the same bytes from a vkey alone.

**Generation.** A Tera template included with `include_str!`,
`setup/pilfflonk/src/tera/verifier_pilfflonk.sol.tera`, and one `render`, as
`setup/stark-recurser/stark2circom/circuit_templates/templates.rs` does. `setup/pilfflonk/src/solidity.rs`
computes the context from the vkey: the constants (`[τ]₂`, the fixed commitments, `digest mod r`, the
roots of unity), where each value is in the calldata and in memory, and two fragments of straight-line
Yul, the transcript and the unrolled `qVerifier`. The generator refuses a vkey it should not turn into
a contract ([Refused vkeys](#refused-vkeys)). The setup generates the contract before it writes the
vkey, and an error stops it.

**Template.** snarkjs 0.7.6's FFLONK verifier (`templates/verifier_fflonk.sol.ejs`): one contract,
`PilfflonkVerifier`, with one `assembly` block of Yul functions; snarkjs's `checkField`,
`checkPointBelongsToBN128Curve`, `inverseArray`, `g1_acc`, `g1_mulAcc`, `g1_mulAccC` and
`checkPairing`; the Montgomery batch inversion with the proof's `inv`, checked before it is used; the
closed forms of the Lagrange denominators ([Closed forms](#closed-forms)); and the precompiles `0x06`,
`0x07` and `0x08`. It departs from snarkjs where the JS verifier does: a list of `f_i` with their `k`
and offsets, the `qVerifier`, and the challenges of `numChallenges`. Every step names the JS function
it does, so the two can be read side by side. The old system's two contracts, `PilFflonkVerifier` and
shplonkjs's `ShPlonkVerifier`, called with `staticcall`, are the structural reference for `computeQ`
and for the roots shared between `f` of the same shape; here there is one contract, as in snarkjs.

**Interface**, `FflonkVerifier`'s: `verifyProof(bytes32[W] calldata proof, uint256[P] calldata
pubSignals) public view returns (bool)`, with `P = nPublic` and `W` fixed by the vkey
([formats.md#calldata](formats.md#calldata)). With no publics there is no `pubSignals`. A proof that
does not verify, or malformed values, returns `false`, as snarkjs; only calldata shorter than the
arguments reverts, in Solidity's ABI decoder.

**Compilation.** With the optimizer at 200 runs (the tests' `foundry.toml`), solc 0.8.37 compiles it
with no warning. The contract uses `shr` (Constantinople) and the Byzantium precompiles, with
Istanbul's costs (EIP-1108). solc 0.8.37 targets a recent EVM by default, with `PUSH0` (Shanghai); a
chain without it needs `--evm-version`. The fixtures' contracts take 4.0 to 15.5 kB, below EIP-170's
24,576 bytes; without the optimizer `all_sum`'s reaches 21.8 kB. The unrolled `qVerifier` grows with
the constraints, about 50 bytes per entry: an AIR much larger than `all` could pass EIP-170, and would
need two contracts or a loop over the code.

**Licence.** The generated contract and the Foundry tests are `SPDX-License-Identifier: MIT OR
Apache-2.0`, the repository's.

### Closed forms

snarkjs's closed forms, which the JS does not use. For the roots `x_j` of an offset `s`, those of
`X^k − z_s` with `z_s = ξ·ω^s`: `Z_T(y) = Π_s (y^k − z_s)`, and the Lagrange denominator
`(y − x_j)·Π_{x'≠x_j}(x_j − x')` is `(y − x_j)·k·x_j^(k−1)·Π_{s'≠s}(z_s − z_{s'})`, since
`k·X^(k−1)` is the derivative of `X^k − z_s` and `x_j^k − z_{s'} = z_s − z_{s'}`. They are the values
the JS computes (polynomial identities), and `inv·Π = 1` on every proof confirms it: one wrong
denominator would make the contract refuse a good proof. The denominators are counted per `f_i`,
repeated when two `f` share roots, as [protocol.md#inverses](protocol.md#inverses) lists them.

### Refused vkeys

The generator refuses a vkey the verifier would not read (`Vkey::validate`), one whose digest is not
that of its content (the JS accepts no proof of it), and one with a fixed commitment off the curve (the
point at infinity, that of a fixed column that vanishes at `τ`, is allowed). `Vkey::validate`, which
the setup runs when it writes a vkey and the prover when it reads one, also checks, as the JS does:

- **`X_2` is a point of G2 other than the point at infinity**: on the twist and in its `r`-torsion
  (`elements.js`, `g2FromObject`), checked by the C++ core's SRS functions (`checkG2`) through
  `pilfflonk_g2_check`, as Rust has no G2 arithmetic. The precompile `0x08` takes `(0, 0, 0, 0)` as the
  point at infinity (EIP-197), where `e(A, [τ]₂) = 1`: with such an `X_2` and its digest recomputed,
  anyone could forge a proof of any statement with `W' = y⁻¹·(E + J − F)`. The setup refuses a ptau
  whose `[τ]₂` is the point at infinity for the same reason.
- **The offsets of each `f`**, as `checkLayout` (`shplonk.js`): `|s| < N`, and no two offsets the same
  row modulo `N` (`Layout::check`).

### Differences from snarkjs

Found in snarkjs 0.7.6's `verifier_fflonk.sol.ejs`, which is not changed; pilfflonk's verifier does
otherwise:

- **Coordinates against `q`.** `checkProofData` checks the coordinates of `C1`, `C2`, `W` and `W'`,
  which are of the base field, against the group order: a valid point with a coordinate in `[r, q)`
  is refused (probability about `2^−127` per coordinate). pilfflonk compares them with `q`, as the JS.
- **Publics.** snarkjs does not check them: `p` and `p + r` are the same statement with two
  transcripts. pilfflonk refuses a public not below `r`, as the JS (`fromObjectPublics`).
- **No publics.** snarkjs declares `uint256[1] pubSignals` all the same; pilfflonk declares none.
- **What the precompiles return.** snarkjs only checks that the calls to `0x06` and `0x07` succeeded,
  and takes any non-zero word from `0x08`. On a chain without those precompiles, a call to an empty
  address succeeds and returns nothing, and the verifier reads stale memory. pilfflonk requires
  `returndatasize() = 64` from `0x06` and `0x07` (`checkPointResult`), and 32 bytes equal to 1 from
  `0x08`, for 587 to 907 gas per proof.
- **solc warning 5667.** snarkjs's contract reads the proof at fixed calldata offsets and leaves its
  `proof` parameter unused; pilfflonk's uses it (`checkInput(proof, …)`) and compiles without a warning.

## Calldata encoder

```
proofman-cli pilfflonk calldata -k <pilfflonk.vkey.json> -p <proof.json | proof.bin> --publics <publics.json>
    [--format solidity|hex] [-o <file>]
```

It gives the arguments of `verifyProof` for a proof ([formats.md#calldata](formats.md#calldata)), as
snarkjs's `zkey export soliditycalldata` gives those of its `FflonkVerifier`; it lives in
`proofman_pilfflonk::calldata`, with the command in `cli/src/commands/pilfflonk/pilfflonk_calldata.rs`.

- **Inputs.** The three files of `pilfflonk verify`, read as the JS verifier reads them. The vkey is
  validated and its digest checked. The proof has exactly the values the vkey names
  (`ProofNames::of_vkey`); it is the JSON view or, if the file name ends in `.bin`, the bytes. There
  are `nPublic` publics, below `r`. Unlike snarkjs, which reads only the proof and the publics, it
  needs the vkey: the shape of the proof and the auxiliary inverses depend on it.
- **What it does.** It replays the transcript on the proof, step by step as `computeChallenges`, with
  the C++ core's `Keccak256Transcript`, the prover's (`verifier_challenges`), takes
  `ξ = xiSeed^powerW`, and computes the auxiliary inverses `1/(ξ − ω^j)` with `ω = 5^((r−1)/N)` in
  `Bn254`. It replays the transcript even without inverses, so it refuses the same for every key.
- **What it refuses**, with exit status 1, a message and no file written: a vkey that does not read or
  whose digest is not its content's; a proof of another key, or bytes of another length; publics of
  another number or not below `r`; a proof with a commitment or a `W` the transcript does not absorb
  (off the curve, the point at infinity, or a coordinate below `2^192`), which it names. Such a proof
  has no `ξ`, and both verifiers refuse it. `W'` is not absorbed, so the encoder does not check it: that
  is the verifier's job. With inverses, it also refuses a `ξ` that is a row of the domain, where
  `Z_H(ξ) = 0`.
- **Output.** `--format solidity`, the default, writes the list of arguments as snarkjs prints it,
  `[0x…,0x…],[0x…]`, each word `0x` and 64 hexadecimal digits, `proof`'s first and `pubSignals`'
  after; with no publics only `[…]`; no spaces. `--format hex` writes the ABI-encoded call: the
  selector (the first 4 bytes of `keccak256("verifyProof(bytes32[W],uint256[P])")`, or of
  `verifyProof(bytes32[W])` with no publics) and the two arguments in place, fixed in size, as `0x` and
  hexadecimal digits: the `data` of an `eth_call`, or of `cast call <address> --data <hex>`. With `-o`
  the file holds the calldata and a newline; without it the calldata is printed on a line of its own,
  after the header `proofman-cli` always prints.

`Calldata::read` reads the files, `Calldata::encode` works from the values, `Calldata::to_solidity` and
`Calldata::to_hex` write the result, and `Calldata::with_auxiliary_inverses` makes a calldata with the
inverses it is given, for tests.

## Gas

**Measurement.** Foundry v1.8.3 and solc 0.8.37, optimizer at 200 runs, solc's default EVM (with
`PUSH0`). The precompiles cost Istanbul's: `0x06` 150, `0x07` 6,000, `0x08` 45,000 plus 34,000 per
pair. The gas of `verifyProof` is that of the call, measured with `gasleft()` around it: the contract's
execution and the call (100, the address warm), not the transaction's 21,000 nor the calldata. Foundry
isolates each call in a transaction of its own by default, and every call after a test's first one
then costs 2,500 more, the address cold (EIP-2929): the end-to-end test measures the first call of
each key, and the fuzzer runs Foundry without isolation; both give the same gas. The calldata costs 16
per non-zero byte and 4 per zero byte (EIP-2028), and depends on the proof's values.

**Per key.** The 73 keys of the fixtures, by family (`foundry_accepts_the_proof_of_every_fixture`
prints each key's):

| Family | Keys | `f` | Words of `proof` | Code (bytes) | Gas of `verifyProof` | Gas of the calldata |
|---|---|---|---|---|---|---|
| Fibonacci | 3 | 3–6 | 18–22 | 5,124–5,290 | 170,869 (`--extra-muls 0`) – 185,168 (`--no-packing`) | 10,036–12,108 |
| `packed` | 2 | 6–19 | 44–57 | 8,787–10,094 | 229,280 – 301,425 (`--no-packing`) | 22,920–29,528 |
| `signed` | 9 | 6–20 | 40–61 | 9,820–12,662 | 236,264 – 313,881 (`--no-packing`, degree 2) | 21,264–32,016 |
| `domains.rs` | 21 | 5–15 | 19–54 | 4,030–9,303 | 174,065 (`firstRow`) – 269,241 (`Frames`, degree 2, `--no-packing`) | 10,060–27,640 |
| The std's buses | 9 | 6–11 | 26–38 | 6,064–8,153 | 191,791 (`prod_bus`) – 232,387 (`prod_bus_im`, `--no-packing`) | 13,480–19,600 |
| The pil-fflonk examples | 29 | 5–31 | 24–82 | 5,902–15,474 | 186,580 (`permutation_prod`) – 406,897 (`all_sum`, `--no-packing`) | 12,316–42,300 |

A transaction that calls `verifyProof` directly costs about 21,000, plus the calldata, plus the gas of
`verifyProof`, less the call's 100: from 201,805 (Fibonacci, `--extra-muls 0`) to 470,097 (`all_sum`,
`--no-packing`). A contract that calls it first in a transaction adds the cold address's 2,500.

**Where it goes.** The precompiles take 66 % to 89 % of the gas. Their minimum is
`113,000 + 6,150·(f + 2) + 100·(2f + 5)`: `f + 2` multiplications (`q_i·[f_i]` for `f − 1` of the `f`,
`E`, `J` and `y·[W']`), `f + 2` additions, a pairing of two pairs (121,206 on every key), and 100 per
call. `computeFEJ` is `12,716 + 6,866·(f − 1)`. The rest is Yul and grows with the packing: the
inversions (6,058 to 33,417) and `r_i(y)` (1,125 to 21,837) grow with the roots and `k`, so keys
grouped with large `k` (`packed`, `signed`, `all`) spend a third of their gas there. The input checks
cost about 160–210 per word of calldata, `Q(ξ)` about 75 per `qVerifier` entry, the transcript 1,175
to 3,548 (with the growth of memory), the call and the ABI 596 to 756.

**Model.** Least squares over the 73 keys:

```
gas of verifyProof ≈ 132,900 + 6,710·f + 1,300·roots + 100·Horner + 106·(qVerifier entries)
```

with `roots = Σ_i k_i·|O_i|` and `Horner = Σ_i k_i²·|O_i|` (the products of the Horner rule that gives
each `f_i` at each root). It errs by −4,737 to +4,236 (at most 1.7 %, RMS 1,597). Fewer `f` is
usually cheaper, though they bring more roots: the Fibonacci spends 170,869 with `--extra-muls 0` (3
`f`), 180,790 by default (5) and 185,168 with `--no-packing` (6); `all_sum` 275,771 grouped (9) and
406,897 unpacked (31). The calldata grows by about 500 gas per word of the proof.

**Against snarkjs's `FflonkVerifier`** (snarkjs 0.7.6, circom 2.2.0, measured as pilfflonk's):

| Verifier | Publics | `f` | Words of `proof` | Code (bytes) | Gas of `verifyProof` | Gas of the calldata |
|---|---|---|---|---|---|---|
| snarkjs, `c = a·b` | 1 | 3 | 24 | 13,657 | 181,639 | 11,700 |
| snarkjs, three signals | 3 | 3 | 24 | 14,287 | 183,200 | 12,376 |
| pilfflonk, Fibonacci, `--extra-muls 0` | 3 | 3 | 18 | 5,124 | 170,869 | 10,036 |
| pilfflonk, Fibonacci | 3 | 5 | 22 | 5,290 | 180,790 | 12,108 |
| pilfflonk, `all_sum` | 3 | 9 | 55 | 12,545 | 275,771 | 28,572 |

snarkjs's costs the same for every circuit but for about 780 per public: its proof is always 24 words
and its `F` always two multiplications. pilfflonk's depends on the AIR; with three `f` the Fibonacci
spends 7 % less than snarkjs with the same publics (fewer evaluations, and a `qVerifier` shorter than
PLONK's identity). pilfflonk's contracts are smaller, as snarkjs writes all its code straight-line.

**Not implemented.** More optimizer runs are the deployer's choice: at 1,000,000 runs, −3.2 %
(Fibonacci) and −5.5 % (`all_sum`, `--no-packing`), for 2 to 8 kB more code. `--extra-muls 0` saves
5.5 % on the Fibonacci: each `f` the grouping adds costs about 6,700 gas per verification. Unrolling the
loops and the powers of known exponent would save 1 % to 2 %, for more code.

## Tests

**Foundry** (`pilfflonk/solidity/`, run by `pilfflonk/tests/data/foundry.rs`). For each key a test
copies `foundry.toml` and `test/` to a directory of its own under `target/`, writes the generated
verifier as `src/PilfflonkVerifier.sol` and the cases as `cases/cases.json`, and runs `forge test`
there. Nothing is built in the repository.

- `setup/pilfflonk/tests/solidity.rs`: the contracts of 11 keys accept their proofs and refuse them
  mutated (an evaluation, a commitment, a public or `W'` changed; a point off the curve or one the
  transcript does not absorb; a coordinate `x + q`, a scalar `e + r`, a wrong auxiliary inverse), with
  the same verdict as the JS verifier on each. Mutations are repeated *fixed up*, with `invZh`, `inv`
  and the auxiliary inverses recomputed for the new transcript (`pilfflonk/tests/data/mutations.rs`), so
  that they reach the pairing or `checkQPieces`.
- `cli/tests/pilfflonk_prove.rs`, `foundry_accepts_the_proof_of_every_fixture`: 73 keys, every fixture
  with the setups of its end-to-end test (grouped, `--no-packing`, its degrees and `--extra-muls`, and
  `Q` split where `qDeg ≥ 2` allows). For each, `pilfflonk-solidity` writes the verifier, solc compiles
  it with no warning and below EIP-170, `pilfflonk prove` proves (from the witness library where the
  fixture has one), `pilfflonk verify` accepts, `pilfflonk calldata` encodes, and Foundry accepts the
  proof and refuses it with an evaluation changed, a commitment replaced by `W` and a public changed:
  73 proofs accepted and 192 refused.
- Without the tools (in CI): `setup/pilfflonk/tests/setup/solidity.rs` and
  `setup/pil2-stark/tests/setup_pilfflonk.rs` check that `--solidity` changes no other file, that
  `pilfflonk-solidity` writes the same bytes, what is refused and the shape of the calldata;
  `cli/tests/pilfflonk_calldata.rs` checks the calldata of keys with and without auxiliary inverses and
  publics.

### Differential fuzzer

`foundry_and_the_js_verifier_agree_on_mutated_proofs` (`cli/tests/pilfflonk_prove.rs`,
`pilfflonk/tests/data/fuzz.rs`) mutates honest proofs and checks that the JS verifier and the contract
agree on every case.

- **Keys.** 13 of the 73 (`FUZZ_KEYS`): the Fibonacci, `packed` and `signed` with `Q` in three pieces,
  each grouped and unpacked; `Domains` (the auxiliary inverses of `firstRow` and `lastRow`), grouped
  and, split, unpacked; `sum_bus` grouped and `prod_bus` unpacked; `all` split on the sum bus (two
  pieces) and the product bus (three), and `all_sum` unpacked, the most expensive.
- **Cases.** A few honest proofs per key (two, and one more per 400 cases of the key). Each mutation
  takes one at random, with a fixed seed (`FUZZ_SEED`; `PILFFLONK_FUZZ_SEED` changes it). 25 families:
  any word changed (a bit, a random value, a boundary value of the fields: 0, 1, `r − 1`, `r`, `r + 1`,
  `q − 1`, `q`, `2^256 − 1`); a point replaced by another point of G1, by one off the curve, by
  `(0, 0)` or by `(1, ±2)`; two scalars or two points swapped; a public (`+ 1`, random, `p + r`, `r`,
  any 256 bits); each value with a check of its own (`W`, `W'`, `inv`, `invZh`, an auxiliary inverse, a
  piece of `Q`); the length of the calldata; and the fixed-up mutations, including the pieces of `Q`
  rebalanced and the forgery `W' = y⁻¹·(E + J − F)`.
- **What is compared.** The JS verdict (`verify()` of `verify.js`, in batches, one Node process per
  round: `pilfflonk/tests/data/js_batch.mjs`), the checks only the calldata has (each auxiliary inverse
  `1/(ξ − ω^j)` and below `r`; short calldata reverts), what `verifyProof` does on Foundry (`true`,
  `false`, or a revert only for short calldata), and **the check that refuses each case**, which the JS
  message and a probe of the contract must name alike. The probe is an instrumented copy of the
  generated contract that the test writes (never the generator): each `fail` says where it is, and the
  body records the gas left after each step. Each family must reach the checks it aims at, and a case
  refused by the pairing must cost at least 90 % of the honest proof's gas.
- **Runs.** 403 cases in CI (31 per key) in 37 s with the setups; 10,400 (`PILFFLONK_FUZZ_CASES=10400`)
  in 94 s. Foundry runs in rounds of 250 cases, without isolation. **No discrepancy**: every verdict
  and every refusing check agree, the contract only reverts on short calldata, and it accepts no
  mutated proof (the accepted cases of the length family are honest proofs with trailing bytes). Three
  checks are out of reach: `xiSeed = 0` and `Z_T(y) = 0` need the transcript to give one value
  (probability about `2^−254`), and the precompiles' answers only fail on a chain without EIP-196 and
  EIP-197.

### Tools

Foundry v1.8.3 and solc 0.8.37, pinned and installed outside the repository. The tests find them at
`PILFFLONK_FORGE` and `PILFFLONK_SOLC`, as the compiler at `PIL2C_EXEC`, and are `#[ignore]` without
them, so CI without Foundry passes. The Foundry project has no `forge-std`: its tests declare the
cheatcodes they use. `foundry.toml` sets `offline = true` and `solc = "0.8.37"`, and the tests give the
compiler's path with `FOUNDRY_SOLC=$PILFFLONK_SOLC`, so Foundry downloads nothing.

```sh
export PILFFLONK_FORGE=<forge> PILFFLONK_SOLC=<solc> PIL2C_EXEC=<pil2-compiler>/src/pil.js
cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only --test pilfflonk_prove \
    foundry_accepts_the_proof_of_every_fixture -- --ignored --nocapture
PILFFLONK_FUZZ_CASES=10400 cargo test -p proofman-cli --features proofman-starks-lib-c/cpu-only \
    --test pilfflonk_prove foundry_and_the_js_verifier_agree_on_mutated_proofs -- --ignored --nocapture
```

Measure a contract's size with solc (`--bin-runtime`), as the tests do: `forge build --sizes` writes a
cache to `~/.foundry`. Keep Foundry's version: the gas it reports depends on its isolation mode.
