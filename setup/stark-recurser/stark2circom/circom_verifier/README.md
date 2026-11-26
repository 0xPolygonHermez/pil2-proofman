# circom_verifier

The circom side of the recursion: the STARK verifier, the vadcop folding and the recursion circuits.
It is a static library; Rust (`lib_gl.rs`, `lib_bn128.rs`, `circuit_templates/`) emits only the
per-key **data** (`key_*(k)` functions, `vadcop_*` functions) and the constraint DAG, plus a
`component main` line.

```
circom_verifier/
├── helper_circuits/     what the circuits are made of (a -l root)
│   ├── verifier/        the STARK verifier: transcript, FRI, STIR, DEEP, Merkle openings, key shape,
│   │                    and one entry file per hash family (family_*.circom, bn128_open_*.circom)
│   ├── vadcop/          what a proof carries for aggregation: values, folding, keys, hashes, a
│   │                    verified child proof
│   └── mains/           one file per recursion circuit: recursive1 (and the compressor), recursive2,
│                        final_compressed, recursivef, vadcop_final, final_snark
├── circuits.gl/         Goldilocks: field gates, Fp5 and the EC, hash families (hash/), tree selectors
└── circuits.bn128/      BN128: Goldilocks emulated over BN128, circomlib-based hashes, custom gates
```

## How includes resolve

`circuits.gl/` and `circuits.bn128/` are **two implementations of one interface**: they define the same
names in files of the same name (`fft.circom`, `evalpol.circom`, `bitify.circom`, `utils.circom`, ...).
The verifier files in `helper_circuits/verifier/` include those by bare name, and the compiler's `-l`
decides which implementation a circuit gets:

| circuit | `-l` roots, in order |
|---|---|
| Goldilocks recursion (compressor, recursive1/2, vadcop_final, recursivef) | `helper_circuits`, `circuits.gl` |
| BN128 final | `helper_circuits`, `circuits.bn128`, circomlib |

So a file in `helper_circuits/` never includes `circuits.gl/...` by path, except where a BN128 circuit
needs Goldilocks gates (the blake3 gates), which it spells relatively.

A path under `helper_circuits/` is written from that root: `include "verifier/proof_shape.circom"`,
`include "vadcop/values.circom"`. An include without a directory is first looked up next to the
including file, then through `-l`.

The roots are set in `setup/pil2-stark/src/proving_key/pipeline/mod.rs` (Goldilocks, overridable by
`RECURSER_CIRCUITS_PATH` and `CIRCUITS_GL_PATH`) and in `snark_setup.rs` (BN128).

## A circuit's generated main

```
pragma ...;
include "<verifier>.circom";          // generated per key: its family's entry and ConstraintRoots<sfx> (the DAG)
include "mains/<circuit>.circom";     // the circuit and, through its own includes, the library
<keys' data: key_<name>(k), the value of key k>
<keys' constraints: ConstraintRootsAt(k)>
<vadcop data: vadcop_*()>
component main {public [...]} = <Template>(...);
```

Key `k` is the k-th key the circuit verifies: there is one, except in `vadcop_final` (one per airgroup).
With one key the data reads `function key_nQueries(k) { return 72; }`; with more, one branch per key.
The library reads only these names (`lib_params.rs` `FIELDS`, checked by a test).

Inputs keep the order the witness fill copies the zkin into: a proof is one flat array, a slot (a proof
and the vadcop values before it) one row.
