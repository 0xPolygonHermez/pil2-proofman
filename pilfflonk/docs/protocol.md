# The pilfflonk protocol

What the setup, the prover and the verifiers compute, normatively. The file formats are in
[formats.md](formats.md), and the verifiers' implementations in [verifier.md](verifier.md).

## Notation

- `r` is the order of BN254's groups and `Fr` its scalar field; `q` is the modulus of the base field
  `Fq`. Every value of a polynomial is in `Fr`.
- An AIR has `N = 2^nBits` rows and `H = ⟨ω_N⟩`. The roots of unity are ffiasm's and ffjavascript's:
  `ω_k = 5^((r−1)/k)` for every `k` that divides `r − 1`. 5 is the smallest quadratic non-residue, and
  `r − 1 = 2^28 · odd`, so no domain of roots of unity, and no trace, has more than `2^28` points.
- `[x]₁ = x·G1` and `[x]₂ = x·G2`. The SRS is `[τ^i]₁`, `[1]₂` and `[τ]₂`, from a snarkjs ptau.
- A polynomial is opened at a set `O` of signed row offsets: at `ξ·ω^s` for each `s ∈ O`.
- The canonical order: AIRs by `(airgroupId, airId)`, instances by `(airgroupId, airId, index)`.
- A stage is a round of commitments: the challenges of stage `s + 1` depend on everything committed up
  to stage `s`. The setup adds two challenges to the pilout's: `std_vc`, which folds the constraints,
  and `std_xi`, the challenge of the evaluation point, here `xiSeed`.

## Constraint polynomial

For each AIR, with its `n` constraints `c_i` in the pilout's order:

```
Q(X) = Σ_{i=0..n−1} std_vc^(n−1−i) · c_i(X) / Z_{D_i}(X)
```

This is the STARK's Horner fold (`setup/pil2-stark/src/pil/constraint_poly.rs`): `acc = acc·std_vc +
c_i·(Z_H/Z_{D_i})`, divided by `Z_H` at the end. The first constraint gets the highest power.

| Domain | `Z_D(X)` |
|---|---|
| `everyRow` | `X^N − 1` |
| `firstRow` | `X − 1` |
| `lastRow` | `X − ω^(N−1)` |
| `everyFrame{min, max}` | `X^N − 1` divided by the factors `X − ω^j` of the rows it leaves out |

The code of `Q` and the `qVerifier` read the zerofiers as `Zi` operands, one per entry of the AIR's
`boundaries` (`everyRow` first): `Zi(everyRow) = 1/Z_H(X)`, and `Zi(D) = Z_H(X)/Z_D(X)` for any other
boundary `D`. `Q`'s code ends by multiplying by `Zi(everyRow)`.

The compiler emits only `everyRow` constraints: it parses `when` but does not execute it. Programs
express their boundaries with fixed columns such as `L1`, and the other three domains are exercised
with pilouts built in code.

Without pieces the verifier never receives `Q(ξ)`: it computes it from the evaluations and the
`Z_D(ξ)`, with the `qVerifier`.

## Degree search

`qDeg = max_i(deg c_i + δ_i) − 1`, in units of `N`, after the intermediate polynomials (im pols) are
introduced, where `δ_i` is 1 if constraint `i` is not `everyRow` and 0 if it is: the factor
`Z_H/Z_{D_i}` adds almost `N` to its degree. If no constraint depends on a column, `qDeg` would be
negative, and the setup refuses the pilout.

The setup chooses the im pols with the search of `pil-info` (`PilInfoCfg::bn254()`, degree policy
`Search { max: D }`): it tries the constraint degrees `2 … D`, with `D` from
`--max-constraint-degree` (9 by default, as pil-stark), and keeps the one that minimises
`nImPols + qDeg`. On a tie it keeps the lowest degree, as pil-stark's `cp_prover.js`, so that the
result is the old system's: the Fibonacci gets one im pol and `qDeg = 1` (degree 2), not `qDeg = 2`
without im pols (degree 3), which costs the same.

The im pols go to the last stage of the AIR, as in the STARK (`im_polynomials.rs`). The prover
computes them with the bytecode before it commits that stage, after the hint columns of the stage,
which they may read. They are not the std's `im_col` columns, which come from hints.

## Degrees

Every bound is a number of coefficients:

- a fixed column has `N`: it has no blinding;
- a committed column opened at the offsets `O` of its `f` has `N + |O| + 1`, `|O| + 1` of them its
  blinding's ([Blinding](#blinding));
- `Q` has `qDeg·N + (qDeg + 1)·|O|_max + 1`, where `|O|_max` is the most offsets a column with
  blinding is opened at, after the fusions of the layout ([Bounds after fusion](#bounds-after-fusion)).

The **extended domain** is the smallest power of two that is at least `Q`'s coefficients and at least
`N + |O|_max + 1`: the columns with blinding are extended to it too, and with a small `qDeg` the first
condition alone could give a domain too small for them. It has `2^nBitsExt` points, and
`nBitsExt ≤ 28`. It is that of the whole `Q` also when `Q` is split: the prover computes `Q` whole
and then splits it.

The **degree of an `f_i`**, the `degree` of its layout entry, is its bound in coefficients,
`max_j(deg_j·k + j)` with `deg_j` the bound of `p_j` ([Layout](#layout)). The SRS must hold at least
as many powers `[τ^i]₁` as the largest `degree` of the layout. The largest ptau there is has `2^28`,
so the degree of every `f_i` and `N·2^extendBits` are at most `2^28`: with `qDeg` up to 8,
`N ≤ 2^24`.

## Q pieces

`Q` is whole by default (`--max-q-degree 0`). With `M = maxQDegree > 0` and `qDeg > M`, it is split:

- **Pieces.** `m = ⌈qDeg/M⌉` pieces `Q_0 … Q_{m−1}` of `M·N` coefficients: piece `i` holds the
  coefficients `i·M·N … (i+1)·M·N − 1` of `Q`, and the last one those left up to `Q`'s bound, so that
  `Q(X) = Σ_i X^(i·M·N)·Q_i(X)`. They form a group of the layout of their own, which `extraMuls` may
  split into several `f`, as in the old system.
- **Blinding**, as PLONK's. Each boundary between pieces `i` and `i + 1` gets two random coefficients
  `b_0, b_1` that cancel: `b_0·X^(M·N) + b_1·X^(M·N+1)` is added to piece `i`, and `b_0 + b_1·X`
  subtracted from piece `i + 1`. So each piece but the last has `M·N + 2` coefficients, and the last
  one `Q`'s bound minus `(m − 1)·M·N`, which is at least `N + 1`. The factors come from the same source
  as the columns', after them, boundary by boundary, `b_0` before `b_1`.
- **In the keys.** The pilfflonkinfo and the vkey hold `maxQDegree = M` if `Q` is split and 0 if it is
  not (`M = 0` or `qDeg ≤ M`), as the old system's setup: `maxQDegree > 0` means split, and the
  globalInfo keeps the option. The pieces are the `cmPolsMap` entries `Q0 … Q<m−1>` of stage
  `nStages + 1`, piece `i` at `stageId` and `stagePos` `i`. Whole, `Q` is the one entry `Q0`.
- **Evaluations.** The `Q_i(ξ)` go into the proof and the transcript
  ([Transcript](#transcript), step 4).
- **Check.** The verifier checks `Σ_i ξ^(i·M·N)·Q_i(ξ) = Q(ξ)`, with `Q(ξ)` from the `qVerifier`.

## Layout

The layout of an AIR is its list of `f_i`. fflonk packs `k` polynomials into one:

```
f_i(X) = Σ_{j<k} p_j(X^k)·X^j
```

Each `f_i` is of one stage and one opening set `O`, and packs its polynomials in a fixed order with a
valid `k`. The layout goes by ascending stage: the fixed `f_i` (stage 0) first, `Q`'s (stage
`nStages + 1`) last. `setup/pilfflonk/src/grouping.rs` computes it, a pure function
`group(polynomials, parameters) → Layout`.

**Input.** The committed polynomials, each with its name, stage, bound in coefficients
([Degrees](#degrees)) and offsets `O`. A column no constraint opens is not committed, and the setup
warns about it (pil-stark dropped such columns silently). The old system counted `Q`'s degree
inclusively, one less than its coefficients; that does not change the grouping, as `Q` is always on
its own, but it changes the size of the SRS.

### Grouping rules

These are the old system's rules (pil-stark's `fflonk_shkey.js` and shplonkjs's `setup.js`),
generalised to signed offsets:

1. **Classes and fusion** (`fixFIndex`, `minPols = 3`). The polynomials are classified by
   `(stage, O)`. For each stage, let `U` be the union of the `O` of its classes. Every class with fewer
   than `minPols` polynomials and `O ≠ U` moves to `U`, and merges with the class `U` if there is one.
   A class with `O = U` never moves. For `O ⊆ {0, 1}` this is the old rule exactly: `{0}:4, {0,1}:2`
   stays two classes, and `{0}:5, {1}:1` gives `{0}×5` and `{0,1}×1`.
2. **Q.** `Q`, or its pieces, forms a group of its own.
3. **Distribution of `extraMuls`** (`applyExtraScalarMuls`).
   - The layout has `#groups + extraMuls` polynomials `f`: each group `g` is split into `c_g + 1`
     consecutive chunks, with `Σ c_g = extraMuls`.
   - Each chunk has a size `k` with `k | r − 1` and `v₂(k) + nBits ≤ 28`, that is `kN | r − 1`.
     shplonkjs does not check it; the setup does, and never enumerates an invalid size.
   - Only non-decreasing sequences of sizes are enumerated.
   - The cost of a chunk is `max_j(deg_j·k + j)`, and that of a group the largest of its chunks'.
   - Within a group, for each number of chunks, the partition of least cost; on a tie, the first
     enumerated.
   - Between groups, each combination gives the vector of its groups' costs, sorted from largest to
     smallest and compared lexicographically. A combination replaces the best one only if it is
     strictly better.
4. **Composition.** `f_i(X) = Σ_j p_j(X^k)·X^j`. Within a group the polynomials go in the reverse
   order of insertion; the pieces of `Q` too, `Q_{m−1}` first.
5. **Roots**, derived when a key is loaded, not stored ([Roots](#roots)).
6. **Compatibility.** For `O ⊆ {0, 1}` the classes, partitions, order and roots are the old system's
   exactly; a golden test checks it on pil-fflonk's example `all`.

Where the old JS and this grouping differ: the JS numbers the `f` by their first appearance in its
lists of offsets, which is not always by stage, and the global order needs them by stage, so
`group()` sorts them stably by stage but enumerates the tie-breaks in the old order of the groups,
which keeps the old partitions (for `all` both orders agree). The JS needs at least two `f`; `group()`
takes one. With signed offsets, the lists go by increasing offset (`−1` first), and the moves of rule 1
by stage, by lexicographic `O` and by input order. Where the JS throws, the setup gives an error
([Grouping errors](#grouping-errors)).

### Roots

- `w_k = 5^((r−1)/k)`.
- `powerW` is the least common multiple of the `k` of every `f_i`, and `ξ = xiSeed^powerW`.
- For an offset `s`, the `k` roots of `f_i` are the `x` with `x^k = ξ·ω_N^s`:
  `x_j = xiSeed^(powerW/k) · ω_{kN}^s · w_k^j` for `j < k`. Negative `s` included.
- `T_i` is the set of roots of `f_i`, offset-major: `x_{m·k+j}` for the `m`-th offset of `O`.

### Unpacked layout

With `--no-packing` (for tests) there is no fusion: each polynomial is alone in an `f` of `k = 1`, at
its own offsets, and `--extra-muls` is not used. The globalInfo records `"packing": false`.

### Bounds after fusion

The setup fuses the classes before it fixes `Q`'s bound: `|O|_max`, `Q`'s bound and `nBitsExt` come
from the fused offsets, the ones the prover reads from the layout. A fused column has the bound of its
`f`'s offsets, `N + |O_f| + 1`. The old system computed `maxPolsOpenings` before the fusions: when a
fusion raised the largest `|O|` from 1 to 2, the highest coefficient of `Q` could be lost silently.

### Evaluation map

A fusion opens a column at offsets the passes did not open it at. The evMap is `pil-info`'s, as it
is, followed by the `(column, offset)` pairs the layout opens that it lacks, ordered among themselves
as `pil-info` orders its own (by opening point, the fixed columns first, by id). So the indices of
`pil-info`'s entries, which are the `eval` operands of the `qVerifier`, do not change. The proof, the
transcript and the verifier list the evaluations in the order of the evMap, the fixed columns'
first: an added pair of a fixed column goes after the other fixed ones, one of a committed column
after the other committed ones.

### Grouping errors

Both enumerations of rule 3 are exhaustive, as in the old system, so that the tie-breaks are its
own, and they grow fast with `extraMuls`. Before enumerating, the setup counts the steps it would take
(the count itself, the partitions it evaluates, each at the cost of the group's length, and the nodes
and complete combinations of the walk between groups); above `2^26`, under a second in release, it
fails with `SearchTooLarge`, which says to lower `--extra-muls`. It prunes nothing. With the default,
a group of 500 columns fits easily.

The errors, all before any file is written: `TooManyExtraMuls` (more than `#pols − #groups`),
`SearchTooLarge`, and `NoValidPartition`, which says how far `--extra-muls` can go. A class of 5, 7,
10, 11, … columns cannot be one chunk (`k` must divide `r − 1`), and with `extraMuls = 2` three such
classes have no valid partition.

## Blinding

Every committed column `p` that is not fixed, of an `f` with opening set `O`, becomes:

```
p'(X) = p(X) + (X^N − 1)·b(X)
```

- `b(X)` has `|O| + 1` random coefficients, added in coefficient form after the INTT: `(X^N − 1)·b`
  vanishes on `H`, so the column's values do not change.
- Fixed columns have no blinding, and neither has `Q` whole: its bound already counts the columns'.
- The pieces of a split `Q` are blinded as PLONK's ([Q pieces](#q-pieces)).
- The prover draws the factors in a fixed order: `f` by `f` in the order of the layout, column by
  column within an `f`, and then the pieces' boundaries. So a deterministic source gives the same
  proof twice.

Blinding is always on, in tests and CI too, so that every run has the same degrees and layout.

- **Random**, the default and the only choice for a real proof: libsodium's `randombytes_buf`.
- **Seeded**, for tests and CI only (`--insecure-blinding-seed`, 32 bytes as 64 hexadecimal digits):
  a stream of blocks of 4096 bytes, block `b` being `randombytes_buf_deterministic(4096, s_b)` with
  `s_b` the BLAKE2b-256 of the 8 bytes of `b` little-endian, keyed with the seed. Whoever knows the
  seed can remove the blinding, so a seeded proof is not zero-knowledge, and the CLI warns.

Each element is uniform in `Fr`: 32 bytes read little-endian, the top two bits cleared, drawn again
while not below `r`.

## Transcript

The transcript is rapidsnark's `Keccak256Transcript` (`pil2-stark/src/rapidsnark/`), unmodified, the
one the existing FFLONK prover uses; the JS verifier reproduces it, as snarkjs does for FFLONK.

**Encoding.**

- An `Fr` is 32 bytes, big-endian, canonical. An integer is absorbed as an `Fr`.
- A G1 point is `x‖y`, 64 bytes, big-endian. The point at infinity and points with a coordinate below
  `2^192` are never absorbed, as `Keccak256Transcript` does not encode them as `x‖y`: for the point at
  infinity it erases the first 64 bytes of its buffer and writes none, and ffiasm's `RawFq::toRprBE`,
  which it calls, writes a coordinate below `2^192` as 8-byte words without right-aligning them. The
  prover refuses such points with `PILFFLONK_ERR_INVALID_POINT`, and so do the JS verifier and the
  calldata encoder. Every point absorbed is a blinded commitment or `W`, so this does not happen in
  practice: a random point has a coordinate below `2^192` with probability about `2^−61`.

**`squeeze`.** `h = keccak256(buffer) mod r`, and the buffer becomes `enc(h)`: the old system's
`reset()` followed by `addScalar(h)`.

**The sequence**, one transcript per proof. The global order of the `f_i` is that of
[Global order](#global-order).

1. Absorb the vkey's `digest mod r` ([formats.md#digest](formats.md#digest)), read big-endian, as an
   `Fr`; the number of instances of each AIR, in canonical order of the AIRs; the publics.
2. For each stage `s = 1 … nStages`:
   1. absorb the commitments of the non-fixed `f_i` of stage `s`, in the global order;
   2. absorb the air values of stage `s` of each instance in canonical order, then the airgroup values
      of stage `s`, then the proof values of stage `s` (none in this version);
   3. if `s < nStages`, squeeze the `numChallenges[s]` challenges of stage `s + 1`, one per squeeze.
3. Squeeze `std_vc` (stage `nStages + 1`). Absorb the commitments of `Q`'s `f_i` in the global order,
   and squeeze `xiSeed`, the `std_xi` of stage `nStages + 2`. `ξ = xiSeed^powerW`.
4. Absorb the evaluations:
   1. for each AIR in canonical order, those of its fixed columns, in the order of the evMap;
   2. for each instance in canonical order, those of its other columns, in the order of the evMap;
   3. if `Q` is split, then the `Q_i(ξ)` of each instance, in the order of its layout: the `f` of `Q`'s
      stage in order, and the pieces of each in order.
5. SHPLONK: squeeze `α_S`, absorb `W`, squeeze `y`.

`α_S` is SHPLONK's challenge, unrelated to PIL1's `α`.

With the std's buses, `nStages = 2` and `numChallenges = [0, 2]`: after stage 1 the transcript gives
`std_alpha` and `std_gamma`, in the order of their `stageId`, and the prover computes stage 2 with
them. A test checks that the prover's challenges, `std_vc` and `xiSeed` are those the JS verifier
replays on the same proof (`cli/tests/pilfflonk_prove.rs`, `the_transcripts_agree`).

## SHPLONK opening

### Global order

1. The fixed `f_i` of each AIR that has an instance, in canonical order of the AIRs and in the order
   of its layout. They are common to every instance of the AIR, and opened once.
2. Then, for each instance in canonical order, its non-fixed `f_i` in the order of its layout: by
   ascending stage, `Q`'s last.

`f_0` is the first of this list. In a proof of one AIR, the global index of an `f` is its index in
the layout.

### Pairing check

```
e(F − E − J + y·W', [1]₂) = e(W', [τ]₂)
```

where

- `F = [f_0] + Σ_{i≥1} q_i·[f_i]`;
- `E = (r_0(y) + Σ_{i≥1} q_i·r_i(y))·[1]₁`;
- `J = q_0·[W]`;
- `q_0 = Z_{T_0}(y)`, and `q_i = α_S^i·Z_{T_0}(y)/Z_{T_i}(y)` for `i ≥ 1`, with `Z_{T_i}(y) =
  Π_{x∈T_i}(y − x)`;
- `r_i` interpolates the values of `f_i` on `T_i`. For each root `x` of the offset `s`, the value is
  `f_i(x) = Σ_j p_j(ξ·ω^s)·x^j`, from the proof's evaluations.

The commitments of the fixed `f` **always** come from the vkey, never from the proof. (pil-fflonk's
JS verifier took them from the proof for the pairing while its transcript absorbed the vkey's.)

The prover uses pil-fflonk's convention (`shplonk.cpp`), with `α = α_S`:

```
W(X)  = Σ_i α^i·(f_i(X) − r_i(X)) / Z_{T_i}(X)
L(X)  = Σ_i α^i·Z_{T∖T_i}(y)·(f_i(X) − r_i(y)) − Z_T(y)·W(X)
W'(X) = L(X) / (Z_{T∖T_0}(y)·(X − y))
```

where `Z_T = Π_i Z_{T_i}`, repetitions included (a root two `f_i` share counts twice), and
`Z_{T∖T_i} = Z_T/Z_{T_i}`.

### Inverses

The proof carries two inverses, which the verifier checks instead of inverting:

- `invZh = 1/Z_H(ξ)`, checked as `Z_H(ξ)·invZh = 1`;
- `inv = 1/Π`, where `Π` is the product of this list, in this order (`n` the number of `f_i`, in the
  global order):
  1. `Z_{T_i}(y)` for `i = 1 … n − 1`, the denominators of the `q_i`;
  2. for each `f_i`, `i = 0 … n − 1`, and each root `x_m` of `T_i` in offset-major order:
     `(y − x_m)·Π_{l≠m}(x_m − x_l)`, the Lagrange denominators of `r_i(y)`.

The verifier recomputes `Π` and refuses the proof if `inv·Π ≠ 1`, so `inv` is not malleable (snarkjs's
verifier does not check its own). The Fibonacci has 13 factors.

## Verifier equations

Given the vkey, the publics and the proof, the verifier:

1. checks the shape of its input: points on the curve (G1's cofactor is 1) with coordinates below
   `q`, scalars below `r`, the lengths the vkey says; and recomputes the vkey's digest and compares it
   with the one it holds;
2. replays the [transcript](#transcript);
3. computes `Q(ξ)` from the evaluations with the vkey's `qVerifier`, as the STARK verifier does, and,
   if `Q` is split, checks `Σ_i ξ^(i·M·N)·Q_i(ξ) = Q(ξ)` ([Q pieces](#q-pieces));
4. checks `invZh`, `inv`, and the pairing ([SHPLONK opening](#shplonk-opening)), with the fixed
   commitments of the vkey.

Global constraints and airgroup values are out of scope, so the verifier neither evaluates
`pilout.globalConstraints.json` (which has no constraint) nor aggregates anything.

## Prover

### Proof sequence

Rust (`proofman_pilfflonk::prover`) loads the `provingKey/`, decides the order of the operations and
what is absorbed, and writes the proof. C++ (`pil2-stark/src/pilfflonk/`) holds the polynomials and
the transcript object, and computes. The steps follow the [transcript](#transcript):

1. **Load.** The globalInfo, the vkey (its digest checked), the SRS, and for each AIR the
   pilfflonkinfo, `<air>.bin` and `<air>.const`. The C++ side derives the [degrees](#degrees) again,
   which must be Rust's, and recomputes the fixed commitments, which must be the vkey's. The instance
   arrives in canonical order. The transcript absorbs the digest, the number of instances and the
   publics.
2. **Stages.** For each `s = 1 … nStages`, the C++ side:
   - computes the stage's columns: the witness's at stage 1; from stage 2 on, the
     [hint columns](#hint-columns) with the stage's challenges;
   - at the last stage, computes the im pols too, after the hint columns;
   - interpolates each column (`Polynomial::fromEvaluations`, with room for its blinding), adds the
     [blinding](#blinding) in coefficient form, packs each `f_i` (as `CPolynomial` does) and commits it
     ([Commitments](#commitments)).

   Rust absorbs the commitments and squeezes the next stage's challenges. The transcript is one per
   proof, so every instance shares the challenges of stage 2, and the buses balance with no global
   challenge.
3. **Quotient.** Squeeze `std_vc`. The C++ side evaluates `Q` with the bytecode on the
   [extended coset](#extended-coset), [in parts](#q-in-parts), interpolates it, checks that it fits
   its bound, splits it if it must, with the pieces' blinding, and commits its `f_i`. Rust absorbs the
   commitments and squeezes `xiSeed`.
4. **Evaluations.** The C++ side evaluates every opened polynomial at its points `ξ·ω^s`: the fixed
   columns once per AIR, the others per instance, and the pieces of `Q` if it is split. Rust absorbs
   them.
5. **Opening.** One SHPLONK opening of every `f_i` in the global order: `pilfflonk_open` squeezes
   `α_S`, computes `W` and absorbs `[W]₁`, squeezes `y`, and computes `W'`, `inv` and `invZh`. It is
   pil-fflonk's `ShPlonkProver` orchestration, generalised to signed offsets and several instances, on
   rapidsnark's `Polynomial`, with a division by `X^m − β` of its own
   ([performance.md#the-shplonk-division](performance.md#the-shplonk-division)) in place of
   `divByXSubValue` and `divByMonic`.
6. **Output.** Rust writes `proof.json` and `publics.json` ([formats.md#proof](formats.md#proof)).

`Q` has a coefficient beyond its bound exactly when some constraint fails on some row: the prover
then refuses the witness (`UnsatisfiedError`, `PILFFLONK_ERR_UNSATISFIED`) rather than write a proof
that does not verify.

### Hint columns

From stage 2 on, the columns of a stage are computed from the std's prover hints, in the STARK's
order (`pil2-stark/src/starkpil/gen_proof.hpp`), whatever their order in `<air>.bin`. The key fixes
that order when it is loaded (`AirKey::stdHints`).

1. Every `im_col`, in the order of `<air>.bin` (that of `getHintIdsByName`), as the STARK's
   `calculateImHints` with `multiplyHintFields`: the `reference` column is `numerator/denominator` on
   every row of `H`, with one batch inversion. As in the STARK, only in an AIR with a `gsum_col` or a
   `gprod_col`. The STARK leaves out a denominator that is the number 1 (`addHintField`); here it is
   inverted all the same, with the same result.
2. Then every `gprod_col`, and then every `gsum_col`, as `calculateWitnessSTD`, each as
   `accMulHintFields`: `numerator_air/denominator_air` on every row, with one batch inversion,
   accumulated row by row into the `reference` column, as a product (`gprod_col`) or as a sum
   (`gsum_col`). The std writes one of each per AIR; if there were more, every one is computed, where
   the STARK computes only the first.

A hint reads only the columns of the hints before it, which are in the buffers already (a `gsum_col`
those of the `im_col`, an `im_col` those of the earlier `im_col`); the key checks it. With no airgroup
values, `result`, `numerator_direct` and `denominator_direct` are not read, as `calculateWitnessSTD`
does when `hintFieldNameAirgroupVal` is empty. A denominator of 0 on some row is an error
(`UnsatisfiedError`) that names the hint, the column and the row: the column has no value there.

### Extended coset

pil-fflonk divides by `Z_H` in coefficient form. The zerofiers of the domains need a pointwise
division instead, so `Q` is evaluated on an extended coset `g·H'`, `H' = ⟨ω_{N'}⟩`, `N' = 2^nBitsExt`.
The shift is `g = 5`, the smallest quadratic non-residue, which is ffiasm's FFT `nqr` and the
generator of the roots: it is in no subgroup of order `2^k`, so `g·H'` does not meet `H`. ffiasm's FFT
has no coset API, so the LDE multiplies the coefficients by the powers of `g` before the FFT. The shift
is the prover's own: the verifier never sees it.

### Q in parts

`g·H'` is the union of `N'/S` parts of `S = 2^partBits` points, `N ≤ S ≤ N'`: part `p` is the points
`g·ω_{N'}^(p + (N'/S)·i)`, `i < S`, that is `c·ω_S^i` with `c = g·ω_{N'}^p`. By default each part is a
coset of `H` (`S = N`). For each part the prover:

1. extends every column `Q` reads to the part (`Lde::extendCosetPart`): coefficient `j` is multiplied
   by `c^j` and folded onto `j mod S`, then an FFT of `S` points;
2. builds the part's domain (`ExpressionsDomain::cosetPart`): its points and `Zi`, those of the whole
   coset at the same points;
3. evaluates `Q` there with the interpreter, in blocks of 128 rows as the STARK's
   (`NROWS_PACK`); a column at offset `o` is read `o·S/N` points further within the part;
4. stores the values at the positions `p + (N'/S)·i` of `Q` on the coset.

The INTT of `Q` and the rest do not change. The extended columns take `32·S` bytes each instead of
`32·N'`, and `Q` and the proof are the same bit for bit whatever `S`. The option is
`ProveOptions::q_part_bits` (Rust), `pilfflonk_instance_set_q_part_bits` (C) and
`Instance::setQPartBits` (C++); it changes neither the format nor the protocol.
[performance.md#cpu](performance.md#cpu) has what it saves.

### Commitments

A commitment is `[f(τ)]₁`, the MSM of `f`'s coefficients over the SRS's powers `[τ^i]₁`
(`Srs::commit`), and leaves the prover in affine coordinates. ffiasm's MSM (`multiMulByScalar`) wants
canonical little-endian scalars, so the coefficients are converted from Montgomery form right before
each MSM; Montgomery limbs would commit to `p·2^256` instead. The GPU's MSM takes them in Montgomery
form ([performance.md#gpu](performance.md#gpu)).
