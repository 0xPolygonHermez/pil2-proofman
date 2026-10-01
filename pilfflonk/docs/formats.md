# pilfflonk file formats

Format version 1. `proofman-pilfflonk` owns every type: the setup writes the files and the prover
reads them through the same types ([README.md#code-map](README.md#code-map)). The protocol they
encode is [protocol.md](protocol.md).

## provingKey

`setup-pilfflonk` writes a `provingKey/` with the hierarchy of the STARK setup's. A file keeps the
STARK's name where its content has the same role; the files that depend on the proof system change.

```
<build>/provingKey/
├── pilout.globalInfo.json             the common part of the STARK's schema, and "backend": "pilfflonk"
├── pilout.globalConstraints.json      the STARK's format (pil-info writes it)
└── <name>/                            <name>: the pilout's
    ├── pilfflonk/                     the backend's global files (get_setup_path's convention)
    │   ├── pilfflonk.srs.bin          the ptau's powers the layout needs, and [1]₂, [τ]₂
    │   ├── pilfflonk.vkey.json        the self-contained vkey, with its digest
    │   └── pilfflonk.verifier.sol     the Solidity verifier, with --solidity
    └── <airgroup>/airs/<air>/air/
        ├── <air>.const                the fixed columns, 32-byte canonical Fr little-endian (8 bytes in the STARK)
        ├── <air>.pilfflonkinfo.json   starkinfo.json's role: the maps and the layout of the f_i
        ├── <air>.expressionsinfo.json the STARK's format, of dimension 1
        ├── <air>.verifierinfo.json    the STARK's format, only the qVerifier
        ├── <air>.bin                  the prover's bytecode and hints ("chps", pilfflonk's version)
        └── <air>.verkey.json          the commitments of the fixed f_i (in the STARK, the Merkle root of .const)
```

The coefficients and the extended evaluations of the fixed columns, the roots and `powerW` are derived
when the key is loaded, as the STARK derives `.consttree` from `.const`. There is no `<air>.verkey.bin`
and no `.verifier.bin`: there is no native verifier.

The pilfflonk runtime reads the globalInfo with a type of its own, not `common::GlobalInfo`, which
validates `hash`, `curve` and `transcriptArity` and sets global C++ state. `curve` is mandatory for
the STARK, so the STARK tools refuse this file by themselves and cannot load it by mistake.

## globalInfo

`pilout.globalInfo.json`:

- **The STARK's common fields**, with their names and meaning: `name`, `airs` (per airgroup, each
  `{name, num_rows}`), `air_groups`, `aggTypes`, `nPublics`, `numChallenges`, `numProofValues`,
  `proofValuesMap`, `publicsMap`.
- **pilfflonk's**: `"backend": "pilfflonk"`, `"formatVersion": 1`, `"field": "bn254"`, `"modulus"`
  (`r` in decimal), `"transcript": "keccak256"`, and `setupParams`, the arguments that fix the layout
  and the degrees: `maxConstraintDegree`, `extraMuls`, `maxQDegree` (the option, whether or not it
  splits `Q`) and `packing` (`false` with `--no-packing`).
- **Not there**: `hash`, `curve`, `transcriptArity`, `aggregationArity`, `latticeSize`,
  `hasCompressedFinal`.

## globalConstraints

`pilout.globalConstraints.json` is the STARK's format, written by `pil-info`. A pilfflonk pilout has
no global constraints, so it has none; the global hints are ignored.

## pilfflonkinfo

`<air>.pilfflonkinfo.json` is the part of the STARK's `starkinfo.json` pilfflonk keeps, every
dimension 1, and the layout. In order:

- `name`, `airgroupId`, `airId`; `nBits` (in `starkStruct` in the STARK); `nStages`, `nConstants`;
- `cmPolsMap`: the committed columns of stages `1 … nStages`, then the pieces of `Q` at stage
  `nStages + 1`, `Q0 … Q<m−1>` (just `Q0` whole), piece `i` at `stageId` and `stagePos` `i`. An im pol
  has `imPol: true` and its `expId`. Each entry: `stage`, `name`, `dim` (1), `polsMapId`, `stageId`,
  `lengths` (if an array element), `imPol`, `expId`, `stagePos`;
- `constPolsMap`, the fixed columns, stage 0;
- `challengesMap`, `airValuesMap`, `airgroupValuesMap`;
- `mapSectionsN`: the width of `const` and of `cm1 … cm<nStages+1>`;
- `openingPoints`: every offset some column is opened at, increasing;
- `boundaries`: `everyRow` first, then each other domain; the `Zi` operands index them;
- `evMap`: the evaluations of the proof, in its order: each `(column, offset)` the layout opens,
  but `Q`'s ([protocol.md#evaluation-map](protocol.md#evaluation-map));
- `qDeg`, `qDim` (1), `maxQDegree` (0 unless `Q` is split, [protocol.md#q-pieces](protocol.md#q-pieces)),
  `cExpId` (the expression of `Q`);
- `layout`: the `f_i`, each `{stage, pols: [{id, name}], k, offsets, degree}`, by ascending stage
  ([protocol.md#layout](protocol.md#layout)). `id` is the index in `constPolsMap` at stage 0 and in
  `cmPolsMap` otherwise; `name` is the column's name in the proof; `degree` is the `f_i`'s bound in
  coefficients, and the SRS holds the largest `degree` powers, not one more.

There is no `starkStruct`, nothing of FRI, no custom commits, and no publics or proof values (they are
the globalInfo's). `pilfflonk/src/pilfflonk_info.rs` fixes the order of the fields and the shape of
the entries, and `pil2-stark/src/pilfflonk/pilfflonk_info.{hpp,cpp}` reads it for the prover.

## expressionsinfo and verifierinfo

`<air>.expressionsinfo.json` is the STARK's format, with dimension 1 for every operand and the
constants in decimal. `<air>.verifierinfo.json` is the STARK's format with only the `qVerifier`, the
code that computes `Q(ξ)` from the evaluations, and no `queryVerifier`. The setup copies the
`qVerifier` into the vkey, which is where the verifiers read it.

## Bytecode

`<air>.bin`, revision 3: the code the prover runs over `Fr` (the im pols, `Q`, and the expressions the
hints refer to), the constraints for `pilfflonk check`, and the prover hints. It is the STARK's prover
`.bin` field by field (`setup/pil2-stark/src/io/bin_file.rs`, read by `expressions_bin.cpp` and run by
`expressions_pack.hpp`), every value of dimension 1, and it departs from it only where BN254 and
dimension 1 force it to:

- no dimension fields: no `destDim`, `nTemp3`, `maxTmp3` or temporaries of the extension field, and no
  `dim` of an expression in the hints;
- args are `u32`, not `u16`, which would truncate an index above 65535 silently;
- numbers are 32-byte canonical `Fr`, little-endian, not `u64`, in the code and in the hints;
- section 1 starts with a prefix: the version, `n8`, `r` and `nStages`;
- a copy is written as `add(a, 0)`; the STARK writes an `add` without its second operand, which its
  own interpreter, reading 8 args per op, cannot run.

`setup/pilfflonk/src/bytecode.rs` writes it and documents it field by field; `pilfflonk_expressions_bin`
reads it. Every integer is little-endian.

```
"chps"       4 bytes
version      u32   0x7066_0003: "pf" in the high half, revision 3 in the low half
nSections    u32   3
3 × { id u32, size u64, payload }, in the order 1, 2, 3
```

The STARK's reader expects version 1 and refuses this file. rapidsnark's `BinFile` only checks a
maximum version and does not expose it, so section 1 repeats the version for pilfflonk's reader to
check for equality, which refuses a STARK `.bin`. A key of revision 2, which had no hints
(`nHints = 0`), must be set up again. `im_col` needed no new revision: hints go by name.

**Section 1, expressions**: the prefix (`version`, `n8 = 32`, `r` in 32 bytes, `nStages`); `maxTmp`,
`maxArgs` and `maxOps` (the largest of sections 1 and 2), `nOps`, `nArgs`, `nNumbers`,
`nExpressions`; for each expression `expId`, `destId`, `stage`, `nTemp`, `nOps`, `opsOffset`, `nArgs`,
`argsOffset` (`u32`) and `line` (the PIL it comes from, UTF-8, NUL-terminated); then `ops` (`u8`),
`args` (`u32`) and `numbers` (32 bytes each). The prover finds an im pol's code by the `expId` of its
`cmPolsMap` entry, and `Q`'s by `cExpId`.

**Section 2, constraints** (for `pilfflonk check`): `nOps`, `nArgs`, `nNumbers`, `nConstraints`; for
each `stage`, `destId`, `firstRow`, `lastRow`, `nTemp`, `nOps`, `opsOffset`, `nArgs`, `argsOffset`,
`imPol` and `line`; then `ops`, `args` and `numbers`. A constraint holds on the rows
`firstRow ≤ i < lastRow`, and its code's value at a row is its numerator. When the degree search
promotes a constraint's whole expression to an im pol (a domain other than `everyRow` adds 1 to its
degree), `pil-info` leaves its debug code empty; the encoder writes it as a copy of the im pol's
column at the row.

**Section 3, hints**: `nHints u32` and, for each prover hint the setup supports (`im_col`,
`gprod_col`, `gsum_col`) in the pilout's order, as the STARK's `write_hints_section`: `name`; `nFields`
and for each field `name` and `nValues` (one, or the elements of an array field); for each value `op`
(the STARK's name: `cm`, `const`, `tmp`, `number`, `string`, `public`, `challenge`, `airvalue`,
`airgroupvalue`, `proofvalue`), the value (`number`: a 32-byte canonical `Fr`; `string`: a string;
the others: `id u32`), `rowOffsetIndex u32` for `cm` and `const`, and `nPos u32` followed by the
positions (`u32`) in the field's array, none for a single value. The `id` is the STARK's: the
`cmPolsMap` index of a `cm` (not its `stagePos`), the `constPolsMap` index of a `const`, the `expId` of
a `tmp` (an expression of section 1), and the index in its map of the rest. No `commitId`: there are
no custom commits. The prover looks hints up by name and checks their operands against the
pilfflonkinfo when it loads the key.

**Ops and args.** One op byte per operation, always 0 (`dim1 = dim1 ∘ dim1`). Eight args per op:
`opType dest aType aArg1 aArg2 bType bArg1 bArg2`, with the STARK's `opType` (0 add, 1 sub, 2 mul,
3 sub_swap) and its order of operands (by rank of type; a sub whose operands are swapped becomes a
sub_swap). `pil-info`'s `get_id_maps` allocates the temporaries.

**Operands** `(type, arg1, arg2)`. The type is the STARK's buffer index, with `bs = nStages + 4` (no
custom commits). Where the STARK multiplies an index by 3, the dimension, here it is the index.

| type | operand | arg1 | arg2 |
|---|---|---|---|
| 0 | fixed column | its column in `<air>.const` | `openingPoints` index |
| 1 … nStages + 1 | committed column of that stage | its `stagePos` in `cmPolsMap` | `openingPoints` index |
| nStages + 2 | `Zi` | 1 + its index in `boundaries` | 0 |
| bs | tmp | the temporary | 0 |
| bs + 2 | public | `publicsMap` id | 0 |
| bs + 3 | number | index in the section's numbers | 0 |
| bs + 4 | air value | `airValuesMap` id | 0 |
| bs + 5 | proof value | `proofValuesMap` id | 0 |
| bs + 6 | airgroup value | `airgroupValuesMap` id | 0 |
| bs + 7 | challenge | `challengesMap` id | 0 |
| bs + 8 | evaluation | `evMap` id (only in code evaluated at `ξ`) | 0 |

**Semantics.** `Q`'s code (`cExpId`) runs point by point on the extended coset; every other
expression, and every constraint, row by row on `H`. On a domain of `M = 2^e·N` points, a column at
`o = openingPoints[arg2]` is read at point `(i + 2^e·o) mod M`. `Zi` of boundary 0 (`everyRow`) is
`1/Z_H(X)`, and of any other boundary `D`, `Z_H(X)/Z_D(X)` ([protocol.md#constraint-polynomial](protocol.md#constraint-polynomial));
`Q`'s code ends by multiplying by `Zi(everyRow)`. The last op of an im pol's code or of `Q`'s writes a
new temporary (`tmpUsed`), as in the STARK. The encoding is deterministic.

## Fixed columns

`<air>.const`: the fixed columns, row after row, each value a canonical `Fr` of 32 bytes,
little-endian. A pilout value that is not below `r` is refused.

## Verkey

`<air>.verkey.json`: the commitments `[f_i(τ)]₁` of the fixed `f_i` of the AIR, in the order of its
layout, as decimal strings: `[["x", "y"], …]`. The vkey holds the same points.

## SRS

`pilfflonk.srs.bin`, version 1: a binfile container as rapidsnark's `BinFile` and snarkjs read them
(4-byte type, `u32` version, `u32` number of sections, then each section as `u32` id, `u64` size and
contents; integers little-endian), of type `"pfsr"`, with three sections, which take the ids of
their counterparts in the ptau:

1. header, 88 bytes: `u32 n8q = 32`, `q` (32 bytes, little-endian), `u32 n8r = 32`, `r` (32 bytes,
   little-endian), `u64 nG1` (`1 ≤ nG1 ≤ 2^32 − 1`, the MSM's limit), `u64 nG2 = 2`;
2. `[τ^i]₁` for `i < nG1`, 64 bytes each;
3. `[1]₂` and `[τ]₂`, 128 bytes each (`Fq2` as `c0‖c1`).

The points are affine `x‖y`, each coordinate in Montgomery form, little-endian: sections 2 and 3 of
the ptau, copied byte for byte. `nG1` is the largest `degree` of the layout. Every point is checked
when it is read: coordinates below `q`, on the curve, `[1]₁` and `[1]₂` the generators, and `[τ]₂` in
G2. That catches a corrupt file; whether the points are powers of one `τ` takes pairings
(`snarkjs powersoftau verify`).

## Vkey

`pilfflonk.vkey.json` is self-contained, as snarkjs's `verification_key.json`: everything the
verifiers need, and nothing else. The verifiers read no other file of the key. Its fields, in the
order of `pilfflonk/src/vkey.rs`:

- `protocol` (`"pilfflonk"`), `curve` (`"bn128"`), `formatVersion` (1), `nPublic`;
- `power` (`nBits`) and `powerW` (the lcm of the layout's `k`);
- `X_2`, `[τ]₂` as `[[x.c0, x.c1], [y.c0, y.c1]]`: a point of G2, never the point at infinity. While
  the setup has not read the SRS it is `[1]₂`;
- `numChallenges`, one entry per stage (none of stage 1);
- `evMap`, `layout` (the `f_i` with `stage`, `pols: [{id, name}]`, `k`, `offsets`, `degree`) and
  `boundaries` (the `qVerifier` refers to them by index);
- the fixed commitments, `f0`, `f1`, … at the top level, `f<i>` for layout entry `i`, as in snarkjs's
  and pil-fflonk's vkeys;
- `qDeg`, `maxQDegree`, the `qVerifier` (as `pil-info` writes it, keys sorted);
- `digest`: `0x` and 64 hexadecimal digits ([Digest](#digest)).

Format version 1 holds one AIR. Big integers and points are decimal strings. The setup, the prover
and the Solidity generator all check a vkey the same way (`Vkey::validate`): the shape the verifier
reads, `X_2` in G2 and not the point at infinity (checked by the C++ core, `pilfflonk_g2_check`), and
the layout's offsets (`|s| < N`, no two the same row modulo `N`).

## Digest

```
digest = keccak256( "pilfflonk-v1" ‖ canonical(vkey without its digest field) )
```

- **`canonical(·)`**: JSON with no whitespace, object keys sorted by UTF-16 code units (JavaScript's
  default `sort()`, which is not the order of Rust's `str`), strings escaped as `JSON.stringify`
  escapes them, big integers and points as decimal strings, and numbers only as integers of at most
  `2^53 − 1` in absolute value. `proofman_pilfflonk::json::canonical_json` sorts explicitly: it cannot
  rely on `serde_json::Value`, as `setup/stark-recurser` turns on `preserve_order` for the whole
  workspace. The JS must write the keys in this order itself, as engines list integer-like keys
  first.
- **The hash** is Keccak-256, not SHA3-256: rapidsnark's `keccak_wrapper`, through the C API
  (`pilfflonk_keccak256`), the transcript's hash. The workspace has no Keccak crate.
- **Why the vkey alone.** It holds everything the verification depends on, and the verifier gets no
  other file. `<air>.bin`, `<air>.const` and `pilfflonk.srs.bin` do not take part; the fixed columns
  are bound through their commitments.
- **In the transcript** it enters as `digest mod r`, read big-endian, as an `Fr`.

## JSON encoding

- Every field element, coordinate and big integer is a **decimal string** in one spelling: no sign,
  spaces or leading zeros, below its modulus. A JSON number is refused: many readers cannot read a
  254-bit number. Integers that are counts or indices are JSON numbers.
- Every file is written as `JSON.stringify(value, null, 1)` lays it out (one space of indentation, no
  final newline), as the STARK setup writes its `provingKey/`, and the bytes depend only on the value.

## Proof

The proof's format is that of the existing FFLONK wrap: bytes, as `gen_final_snark_proof` writes, and
a JSON view in snarkjs's style, as `snark_proof_to_json` writes.

**Bytes**, in order, each point `x‖y` and each coordinate or scalar 32 bytes big-endian:

1. the commitments of the non-fixed `f`, in the global order
   ([protocol.md#global-order](protocol.md#global-order)); the fixed ones are the vkey's, never the
   proof's;
2. `W` and `W'`;
3. the evaluations: the fixed columns' of each AIR, then the others' of each instance, each in the
   order of its evMap, then the `Q_i(ξ)` if `Q` is split, in the order of the layout;
4. the air values, the airgroup values and the proof values (none in this version);
5. `inv` and `invZh`, as pil-fflonk.

The bytes alone do not say where a part ends; the vkey's names do. A file whose name ends in `.bin`
holds these bytes (`Proof::read`).

**`proof.json`**: `{"protocol": "pilfflonk", "curve": "bn128", "polynomials": {name: [x, y, "1"]},
"evaluations": {name: value}}`, keys sorted.

### Proof names

The names are pil-fflonk's, extended to signed offsets, to array columns and to several instances.

- `polynomials`: `f<g>`, the commitment of the `f` at position `g` of the global order (only the
  non-fixed ones), and `W` and `Wp`.
- `evaluations`: `<column><suffix>`, the evaluation of a column at `ξ·ω^s`:
  - `<column>` is the column's name in its map, with `[i]` for each entry of its `lengths`:
    `Fibonacci.L1`, `l1`, `Main.a[0]`. The compiler names witness columns without the AIR's prefix and
    fixed ones with it;
  - `<suffix>` is empty for `s = 0`, `w` for `s = 1`, and `w` followed by `s` in decimal, sign
    included, otherwise: `w2`, `w-1`;
  - `Q<i>`: the piece `i` of a split `Q`, the one the verifier multiplies by `ξ^(i·M·N)`, after the
    other evaluations, in the order of the layout. Whole, `Q` has no evaluation;
  - air values, airgroup values and proof values by their names; `inv` and `invZh`.
- **Indexed names.** `pil-info` names every im pol `<air>.ImPol`; the setup gives the `k`-th
  `lengths: [k]`, so it is `<air>.ImPol[k]`. The same way, the columns of a map that share a name and
  have no `lengths`, as the `im_cluster` the std declares in a loop, are the vector of that name: the
  `k`-th, in the map's order (the pilout's), gets `lengths: [k]`: `im_cluster[0]`, `im_cluster[1]`, …
  A name some column has with `lengths` does not change. If two names still collide, the setup
  refuses the pilout and names them.
- **Scopes**, for proofs of several instances (not in this version): `<ag>.<a>:` before the fixed
  evaluations of AIR `a` of airgroup `ag`, `<ag>.<a>.<t>:` before those of its instance `t`, `<ag>:`
  before airgroup values. A proof of one instance uses the names as they are.

## Publics

`publics.json`: an array of decimal strings, in the order of the globalInfo's `publicsMap`, as
pil-fflonk and the FFLONK wrap write them.

## Witness directory

The prover's input, not part of the `provingKey/` (`pilfflonk/src/witness.rs`). Exactly these files,
and no other:

- `instances.json`: a non-empty array of `{"airgroupId", "airId", "airValues": [...]}` in canonical
  order; the instances of an AIR are consecutive, and the `t`-th of them is its instance `t`.
  `airValues` are those of stage 1, in the order of the `airValuesMap` (empty in this version);
- `instance_<ag>_<a>_<t>.bin`: the stage-1 columns of one instance, with no header, row after row,
  each value 32 bytes canonical little-endian. Column `c` of row `i` is at byte `(i·C + c)·32`, and the
  file has exactly `N·C·32` bytes. `C` is the AIR's `stageWidths[0]`: the `cmPolsMap` entries of stage
  1 that are not im pols, column `c` the one with `stageId` `c`;
- `publics.json`, as the proof's, and `proof_values.json`, those of stage 1 (empty in this version).

Every JSON value is a canonical decimal string below `r`. The reader refuses extra files, wrong sizes
and values not below `r`.

## Calldata

The arguments of the Solidity verifier's
`verifyProof(bytes32[W] calldata proof, uint256[P] calldata pubSignals)`, with `P = nPublic` (no
`pubSignals` if it is 0, as Solidity has no `uint256[0]`):

- `proof`: the proof's bytes as 32-byte words, `2·(n_f + 2) + n_evals + n_pieces + 2` of them: the
  commitments of the non-fixed `f` (`x`, `y`), `W`, `W'`, the evaluations, the `Q_i(ξ)` if `Q` is
  split, `inv` and `invZh`;
- then one **auxiliary inverse** per `firstRow` or `lastRow` boundary, in the order of the boundaries:
  `1/(ξ − ω^j)`, with `j = 0` or `j = N − 1`. The `Zi` of those boundaries is `Z_H(ξ)/(ξ − ω^j)`, the
  only division of the verifier that neither `inv` nor `invZh` covers. The contract checks
  `(ξ − ω^j)·aux = 1` and refuses `aux ≥ r`, as it does `inv`, so a proof has one calldata. `invZh`
  gives `Zi(everyRow)`, and `everyFrame` is a product;
- `pubSignals`: the publics, in the order of `publics.json`.

The auxiliary inverses are of the calldata only: the proof does not change, and a vkey without those
boundaries, as that of every compiled PIL2 program, has none, and its calldata is the proof.
`proofman-cli pilfflonk calldata` computes them ([verifier.md#calldata-encoder](verifier.md#calldata-encoder)).
