# Prompt — Validate Poseidon v1 AVX512 paths on an AVX512-capable server

Paste this to Claude Code on a machine that has an AVX512-capable CPU.

---

## Context

The branch `feature/add_poseidon1` in this repo contains a port of the old
Poseidon v1 algorithm (from pil2-stark 0.14.0) into the current
templated/mode-dispatched interface. The port is complete and the CPU
(Scalar + AVX2) + GPU (CUDA) paths are fully verified.

What is **not yet verified** is the **AVX512 batch path**: it compiles cleanly
but has never been executed, because the machine where it was written does
not have AVX512. All AVX512 code is gated by `#ifdef __AVX512__`, which the
Makefile turns on only when the host CPU advertises `avx512f` in
`/proc/cpuinfo`.

Your job: on this AVX512-capable host, build and run the Poseidon v1 AVX512
test + bench, confirm byte-exact goldens, and report numbers. Then either
report all-green, or, if a test fails, localise the bug.

---

## What to do

### 1. Confirm you're on the right machine and branch

```bash
grep -m1 avx512f /proc/cpuinfo && echo "AVX512F present"
cd /path/to/pil2-proofman
git checkout feature/add_poseidon1
git log -1 --oneline
```

`avx512f` must appear in /proc/cpuinfo. If it doesn't, STOP and tell the user
this machine doesn't have AVX512 either.

### 2. Build testscpu (the Poseidon v1 AVX512 code lives in this binary)

```bash
cd pil2-stark/src/goldilocks
make clean
make testscpu 2>&1 | tail -20
```

Expected in the compiler flags: `-mavx512f -D__AVX512__`. If that's missing,
the Makefile auto-detect failed — check line ~35 of
[pil2-stark/src/goldilocks/Makefile](pil2-stark/src/goldilocks/Makefile).

### 3. Run the full Poseidon v1 test suite

```bash
./testscpu --gtest_filter='PoseidonV1*'
```

**Expected result: 17 tests run, 17 pass.**

On this AVX512 host the existing 17 tests already cover AVX512 because one of
them (`merkletree_binary_fib128x64_avx512batch`) is `#ifdef __AVX512__`-gated
— it compiles out on non-AVX512 machines, in on AVX512 machines.

Additionally, re-verify the cross-mode fuzz still holds:

```bash
./testscpu --gtest_filter='PoseidonV1.mode_parity_*'
```

These fuzz tests assert `Scalar ≡ Avx == AvxBatch == Avx512Batch` on 4096 random
inputs. With AVX512 active, the fourth backend is really tested for the first
time.

### 4. Run the bench and record timings

```bash
./benchscpu --benchmark_filter='Poseidon' \
            --benchmark_repetitions=3 \
            --benchmark_display_aggregates_only=true
```

Record the `_mean` numbers for:
- `PERMUTE_W12_AVX_CPU_BENCH`
- `MERKLETREE_W12_AVX_CPU_BENCH`
- `MERKLETREE_W12_AVXBATCH_CPU_BENCH`
- `MERKLETREE_W12_AVX512BATCH_CPU_BENCH`  **(this is the new one — compare to AvxBatch; it should be noticeably faster, typically 1.3–1.8×)**

### 5. Report back

Write a short summary with:

- The exact CPU model (`cat /proc/cpuinfo | grep 'model name' | head -1`).
- "17/17 tests passed" **or** the name of any failing test + the assertion
  diff.
- The four bench numbers above, plus the ratio `AvxBatch / Avx512Batch`.

---

## If something fails — where to look

The AVX512 code path consists of exactly three files — these are the only
places a bug can live:

1. [pil2-stark/src/goldilocks/src/poseidon_goldilocks_avx512.hpp](pil2-stark/src/goldilocks/src/poseidon_goldilocks_avx512.hpp)
   — templated AVX512 primitives (`pow7_avx512`, `add_avx512`,
   `add_avx512_small`, `add_avx512_a`). Port of pil2-stark 0.14.0's primitives
   with the class templated to `<W>`.

2. [pil2-stark/src/goldilocks/src/poseidon_goldilocks.cpp](pil2-stark/src/goldilocks/src/poseidon_goldilocks.cpp)
   under `#ifdef __AVX512__` — `permute_batch_avx512`, `compress_batch_avx512`,
   `linear_hash_batch_avx512`, `merkletree_batch_avx512`. The batch reduction
   uses the already-fixed stride: `const uint64_t STRIDE = arity * CAPACITY;`
   (line ~711).

3. [pil2-stark/src/goldilocks/src/poseidon_goldilocks.hpp](pil2-stark/src/goldilocks/src/poseidon_goldilocks.hpp)
   — dispatcher at the bottom (`merkletree` / `compress` / `permute` / `linearHash`
   switch on `PoseidonMode`, routing `Avx512Batch` to the right implementation).

If the failure is in a scalar-vs-Avx512Batch parity test, the math in one of
the primitives is wrong — most likely `pow7_avx512` (port of 0.14.0) since
the other three are just `add` variants.

If the failure is specifically in the merkletree assertion (wrong root),
check the reduction stride in `merkletree_batch_avx512` — it must be
`arity * CAPACITY`, not `SPONGE_WIDTH`. This mismatch is the exact bug I hit
during the port and fixed for Scalar/AVX/AvxBatch; re-verify it's consistent
in the AVX512 block.

If the bench produces sensible numbers but the test `merkletree_*_avx512batch`
fails, don't "fix" the test by weakening the assertion — the Scalar output is
the ground truth. The AVX512 implementation must match scalar byte-for-byte.

---

## Ground truth for the Poseidon v1 W=12 golden vectors

These are committed inline in the test file. Any regression should show as a
divergence from one of these:

- **permute Fibonacci** `[0,1,1,2,3,5,...,89]`:
  `{0x3095570037F4605D, 0x3D561B5EF1BC8B58, 0x8129DB5EC75C3226, 0x8EC2B67AFB6B87ED,
    0xFC591F17D0FAB161, 0x1D2B045CC2FEA1AD, 0x8A4E3B0CB12D4527, 0x0FF217A756AE2211,
    0x78F6E79CFC407293, 0x3DE827E086AE61C9, 0x921456F6D2D11E27, 0xF58A41D4028C66A5}`

- **permute zero input**:
  `{0x3C18A9786CB0B359, 0xC4055E3364A246C3, 0x7953DB0AB48808F4, 0xC71603F33A1144CA,
    0xD7709673896996DC, 0x46A84E87642F44ED, 0xD032648251EE0B3C, 0x1C687363B207DF62,
    0xDF8565563E8045FE, 0x40F5B37FF4254DAE, 0xD070F637B431067C, 0x1792B1C4342109D7}`

- **linear_hash on 128 Fibonacci elements**:
  `{0xB214FEA22C79AE3C, 0x49DA61DEED54466A, 0x7338CC9DBA8256FD, 0xC1043293021620CE}`

- **merkletree binary, arity=2, 128 cols × 64 rows of Fibonacci**:
  root = `{0x918F7CD0C3E8701F, 0x83A130E00F961B02, 0x6921497B364123F8, 0xBD2B98A57B748BF4}`

---

## Stop conditions

- **All green**: report the bench numbers and declare validation complete.
  Update the project memory file
  `.claude/projects/-home-rick-pil2-proofman/memory/project_poseidon_v1_port.md`
  to mark AVX512 validation as done.
- **Any test fails**: do NOT attempt a fix without confirming with the user
  first. Report the failure + exact diff and ask how to proceed.
