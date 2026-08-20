# Blake3 Fiat-Shamir transcript: use BLAKE3 as specified

Date: 2026-08-20
Status: approved, implementation not started

Companion to `2026-08-20-blake3-circom-gates-design.md`, which is **paused**
pending this work: its `Blake3Sponge` template and the rate/width branches it
added to `transcript.rs` and `calculate_hashes.rs` are all provisional on the
outcome here.

## Problem

`blake3` currently reuses the Poseidon transcript wholesale. `TranscriptGL`
builds `inputs = [pending(4), state(4)]` and calls
`Blake3Goldilocks::permuteTranscript`, which is `permute_xof` at width 8 — a
complete BLAKE3 over one 64-byte block with `CHUNK_START|CHUNK_END|ROOT` and
counter 0, squeezing 64 bytes back. Two things are wrong with that:

1. **BLAKE3-the-hash is being used as a fixed-width permutation.** The
   construction is not BLAKE3 in any sense a reader of the BLAKE3 spec would
   recognise, so none of BLAKE3's analysis transfers, and the transcript hash
   cannot be checked against any reference implementation.
2. **The capacity eats half of every block.** BLAKE3's block is 64 bytes = 8
   Goldilocks words, and the sponge spends 4 of them carrying the state, so it
   absorbs 4 field elements per compression.

The second point is not fixable within the sponge shape. Carrying a 32-byte
state *as block content* leaves exactly 4 words of a 64-byte block, so rate 4 is
forced. Rate 8 requires the state to live in the chaining value instead — which
is what BLAKE3 actually does, and is not a sponge. Raising `transcript_arity`
does not help either: arity 3 gives pending 8 / width 12, two blocks for eight
absorbed words, the same 4 per compression.

This traces to a comment in `hash_family.rs` — *"the transcript squeezes through
the same sponge/compression as the trees, so it shares the Merkle arity."* True
for Poseidon. For blake3 the arity-2 trees are forced by the GPU kernels, and
inheriting rate 4 from that is what halves throughput.

## Why now, and what it is worth

Measured on a real generated verifier
(`examples/test-recursive/poseidon2/test.verifier.circom`, 6165 lines), counting
every transcript round and classifying it by whether its output is consumed or
drained:

```
transcript rounds                : 85
  pure absorption (out discarded): 81
  genuine squeeze points         :  4   consuming 23 challenge words total
field elements absorbed          : 924  = 7392 B = 7.2 BLAKE3 chunks
```

The transcript is overwhelmingly absorb-dominated: 924 words in, 23 out, at four
finalize points. That is what makes "as specified" cheap — the worry that
BLAKE3's non-incremental ROOT finalization would be punitive does not survive
contact with four squeeze points.

| | compressions |
| --- | --- |
| current sponge, rate 4 | **231** = ceil(924/4); squeezing is free |
| BLAKE3 as specified, rate 8 | **~134** = 116 absorb + ~6 chunk-tree parents + 4 finalizes x ~3 |

**Circuit-wide this is only ~2%**, because the arity-2 query side dominates the
recursion circuit. The case for doing it is not size. It is that the
construction becomes reviewable against the BLAKE3 spec, testable against the
reference implementation, and that it is a protocol decision which is far cheaper
to make before blake3 recursion ships than after.

## Section 1 — The construction

**Absorbed stream.** Every field element contributes `to_canonical(x)` as 8
little-endian bytes — the encoding `blake3_core.hpp` already uses everywhere.
The transcript's absorbed stream is exactly that concatenation, so the transcript
hash is literally `b3sum` of the stream.

**Absorb.** `put(xs)` appends to the stream. No padding, no block alignment
visible to the caller.

**Squeeze.** At squeeze point *k*, `XOF_k = BLAKE3_XOF(stream_k)`. Challenge
words are read 8 bytes at a time and reduced with the same single-conditional-
subtract `to_canonical` the codebase already uses. A read offset advances; the
next `put` resets it to 0, because the stream changed and the XOF stream is
entirely new. This preserves today's semantics exactly, where `_add1` sets
`out_cursor = 0`.

**`getState()`** returns `XOF_k[0..4]` — the standard 32-byte BLAKE3 digest of
the transcript. More meaningful than the sponge state it replaces. It does
**not** advance the read offset, matching today's behaviour where `getState`
reads `state` rather than draining `out`; only `getFields1` (and therefore
`getField` / `getPermutations`) consumes.

**Read order** is forward from offset 0. Today's `getFields1` reads
`out[(transcriptOutSize - out_cursor) % transcriptOutSize]` with `out_cursor`
counting down, i.e. index 0 upward, which is the XOF stream order — so the
mapping is direct.

**Incremental implementation.** Hold BLAKE3's chunk state: the cv chain, a
64-byte block buffer, the chunk counter, and the cv stack (up to `CV_STACK` = 24
entries). Absorption fills blocks; `CHUNK_START` on block 0 of a chunk,
`CHUNK_END` on block 15, chunk counter = chunk index; completed chunks merge into
the stack by BLAKE3's "merge while the completed count is even" rule.

A squeeze finalizes a **copy**, leaving the absorb chain un-rooted so absorption
continues:

- single chunk so far: the root node is the current chunk's final partial block,
  with `CHUNK_END|ROOT`;
- multiple chunks: merge a copy of the stack right to left; the root node is the
  final parent, with `PARENT|ROOT`.

Either way the root node's `(cv, block, block_len, flags)` is then fed to
`compress_xof` with the output-block counter, which yields 64 bytes per call.
Reading past 64 bytes costs one more compression and **no re-finalize** — only
the output-block counter changes.

## Section 2 — Where it lives

**A family branch inside the existing classes.** The public API is unchanged, so
all 34 `TranscriptGL` references outside the transcript directory, and the whole
`TranscriptDyn` surface, stay exactly as they are. Callers construct
`TranscriptGL` as stack objects in `gen_proof.hpp` and friends; a separate class
would have forced a factory on every one of them.

`arity` stays in the C++ constructor for Poseidon and is ignored for blake3.
`transcript_arity` / `transcript_pending_size` / `transcript_out_size` in
`hash_family.rs` become Poseidon-only concepts.

### The structural decision that makes this tractable

The naive reading of this design is "four implementations that must agree" — CPU,
GPU, Rust, circom. Two moves cut that down:

1. **The incremental hasher goes in `blake3_core.hpp`.** That header is already
   the shared `B3_HD` (`__host__ __device__`) core used by *both* the CPU
   (`Blake3Goldilocks`, g++) and GPU (`blake3_goldilocks.cu`, nvcc) paths,
   precisely so they are bit-identical by construction. Putting the absorb/
   finalize state machine there means `TranscriptGL` and the CUDA kernels are
   thin wrappers over one implementation, not two.
2. **The Rust verifier uses the official `blake3` crate.** It is already a
   workspace dependency (`Cargo.toml:115`, resolved to 1.8.6, used by `proofman`
   and `proofman-cli`), so the verifier side becomes definitionally correct
   rather than a reimplementation to be validated.

That leaves **two** hand-written implementations — the shared C++/CUDA core and
the circom circuits — plus the reference crate, which doubles as the oracle.

## Section 3 — Work items

1. **`blake3_core.hpp`** — add an incremental hasher: a small `B3_HD` struct
   holding cv chain, block buffer, chunk counter and cv stack, with `absorb`,
   and `finalize_xof(ob, out[8])` operating on a copy. Existing `hash_le64` and
   `permute_xof` stay for the trees and grinding.
2. **CPU** — `transcriptGL.{hpp,cpp}`: blake3 state alongside the sponge one;
   branch in `_add1`, `_updateState`, `getFields1`, `getState`.
3. **GPU** — `transcriptGL.{cuh,cu}`: state moves to device memory, and the
   sponge logic is duplicated across `_updateState` *and* `_updateStateWarp`.
   blake3's path is scalar in both, so what changes is the state layout and
   cursors, in two places. Largest and least testable piece.
4. **Rust verifier** — `Blake3Transcript<F>` over the `blake3` crate, wired into
   `TranscriptDyn::Blake3` in place of `Transcript<F, Blake3_8>`.
   `Transcript<F, H: Hash<F>>` is a sponge abstraction (`WIDTH`/`RATE`/
   `CAPACITY`/`State`) and cannot express this; `Blake3_8: Hash` stays for
   `hash_state`. Needs `default-features = false` since `fields` is
   `#![no_std] + alloc`; the `zisk` target vendor cfg means the portable
   fallback must be confirmed, not assumed.
5. **Recurser emitter** — `transcript.rs`'s `update_state()` fuses absorb and
   squeeze into one operation. Split it into "emit absorb blocks as data
   accumulates" and "emit a finalize at a squeeze point". Same for
   `calculate_hashes.rs`. This is the core state-machine rewrite and the main
   implementation risk on the verifier side.
6. **Circom circuits** — delete `Blake3Sponge`; add absorb/finalize templates
   reusing the already-tested `Blake3AbsorbBlock` and `Blake3Parent`, plus an
   XOF finalize. The emitter stops needing rate/width family branches at all.

### Staging

The six items are one coherent protocol change and the clean break means they
must all land before blake3 is usable, but they are separable for review:

- **A**: item 1 (shared core) + item 2 (CPU) + item 4 (Rust) + the oracle tests.
  Self-contained and fully testable on any machine.
- **B**: item 3 (GPU). Depends on A's core; needs a GPU to validate.
- **C**: items 5 and 6 (emitter rewrite + circom). Depends on A for the
  reference values; unblocks the paused companion spec.

A is the right first step: it establishes the construction and the oracle, so B
and C are validated against something real rather than against each other.

## Section 4 — Testing

The point of "as specified" is that it admits a **true external oracle**, which
the sponge never could.

1. **Rust against the crate** — the `Blake3Transcript` wrapper is thin, so the
   test that matters is that the byte encoding and squeeze offsets are right:
   drive a scripted put/get sequence and compare against `blake3::Hasher` fed
   the same byte stream directly.
2. **C++ against `b3sum`** — dump the absorbed byte stream from a scripted
   sequence, digest it with `b3sum` (installed), and compare to
   `getState()`. Covers the shared core, hence CPU and GPU.
3. **Cross-implementation** — one scripted put/get sequence driven through CPU,
   GPU, Rust and circom, asserting identical challenge words.
4. **Circom** — extend `tests/blake3/run_tests.sh` with transcript cases against
   the same oracle, in the differential style already established there.
5. **Boundary cases**, chosen where BLAKE3's structure changes behaviour: stream
   lengths straddling 64 bytes (block), 1024 bytes (chunk), and 2048/8192 bytes
   (stack merges); squeezing more than 64 bytes at one point (output-block
   counter); squeeze / absorb / squeeze interleaving; and a squeeze with an empty
   stream.

Every implementation is checked against the reference BLAKE3, not merely against
the others, so "all of them agree on the wrong thing" stops being a failure mode.

## Costs, risks and deliberate choices

- **The GPU path is the largest piece and needs a GPU to validate at all.** The
  shared core mitigates the divergence risk but not the validation gap.
- **Clean break, no compat path.** Every existing blake3 proof, setup and
  committed verifier artifact becomes invalid. Poseidon1 and Poseidon2 are
  untouched throughout. This was an explicit decision, taken because blake3 is
  not yet in production.
- **`to_canonical` on XOF output keeps a ~2^-32 bias per challenge word.** This
  is the same bias the current transcript has, it is harmless for Fiat-Shamir,
  and keeping it avoids a rejection-sampling loop that would be expensive
  in-circuit. A deliberate choice, not an oversight.
- **~2% circuit-wide.** Restated because the effort is disproportionate to the
  size win, and the justification has to rest on correctness and irreversibility.
- **The emitter rewrite (item 5) is the least mechanical part.** Absorb and
  squeeze are currently one operation, and every call site of the emitter assumes
  that.

## Out of scope

- The Poseidon transcripts, which keep the sponge.
- `fields/src/merkle.rs`'s generic sponge linear hash, which is also wrong for
  blake3 (same "blake3 is not a sponge" mismatch) but is a separate defect.
- The blake3 gate registry and PIL arithmetization, still deferred per the
  companion spec.
