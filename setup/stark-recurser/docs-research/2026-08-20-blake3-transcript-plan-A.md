# Blake3 Transcript — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sponge-shaped blake3 Fiat-Shamir transcript with a genuine BLAKE3 over the absorbed byte stream, on the CPU prover and the Rust verifier, anchored to the reference BLAKE3 implementation as an oracle.

**Architecture:** One incremental hasher added to `blake3_core.hpp` — already the shared `__host__ __device__` core used by both the CPU and CUDA paths — so `TranscriptGL` becomes a thin wrapper and the later GPU stage reuses the same state machine rather than duplicating it. The Rust verifier wraps the official `blake3` crate, so that side is correct by construction and doubles as the test oracle.

**Tech Stack:** C++17, gtest (`/usr/include/gtest`), Rust (`#![no_std]` + `alloc`), the `blake3` crate 1.8.6 (already a workspace dependency), `b3sum` 1.8.5 CLI.

**Spec:** `setup/stark-recurser/docs-research/2026-08-20-blake3-transcript-design.md`

> Plan location deviates from the skill's `docs/superpowers/plans/` default because
> `docs/` is excluded by this user's global gitignore; plans live beside the specs.

## Global Constraints

- Absorbed stream encoding: each field element is `to_canonical(x)` as **8 little-endian bytes**. Never any other encoding.
- Challenge words are read from the XOF stream **8 bytes at a time, forward from offset 0**, each reduced with `to_canonical` (one conditional subtract). The ~2⁻³² bias is deliberate — do not add rejection sampling.
- `getState()` must **not** advance the read offset. Only `getFields1` (hence `getField` / `getPermutations`) consumes.
- Any `put` after a read resets the read offset to 0.
- Poseidon1 and Poseidon2 behaviour must be **bit-identical** before and after. Every change is inside a blake3 branch.
- Clean break: no compatibility path, no versioning, no dual implementation.
- `fields` is `#![no_std]` with `extern crate alloc`. The `blake3` dependency must be `default-features = false`.
- `blake3core` additions must compile under **both** g++ and nvcc: use the existing `B3_HD` macro, no `std::`, no heap, no exceptions.

**Verified facts this plan relies on** (re-confirm if anything looks off):
- `blake3core::hash_le64` is genuine BLAKE3. `b3sum` of the canonical-LE stream of words `1..8` yields exactly `{16432952784711837466, 12565756115161032165, 6915939387221618258, 11123773279136987111}`, matching `hash_le64`. So `hash_le64` is a valid in-repo oracle for the first four XOF words.
- In BLAKE3 the **root node's counter field is the output-block index**, not a chunk index. `permute_xof` already relies on this (`compress_xof(cv, last_block, last_len, ob, last_flags)`).
- C++ tests build with `make testscpu` from `pil2-stark/src/goldilocks/` and produce a `./testscpu` gtest binary.

---

### Task 1: Incremental BLAKE3 hasher in the shared core

**Files:**
- Modify: `pil2-stark/src/goldilocks/src/blake3_core.hpp` (append before the closing `}  // namespace blake3core`)
- Test: `pil2-stark/src/goldilocks/tests/test_blake3_transcript_cpu.cpp` (create)

**Interfaces:**
- Consumes: existing `blake3core` internals — `compress_in_place`, `compress_xof`, `parent_cv`, `b3_iv`, `to_canonical`, `hash_le64`, and the constants `FLAG_CHUNK_START`, `FLAG_CHUNK_END`, `FLAG_PARENT`, `FLAG_ROOT`, `BLOCK_U64`, `CHUNK_U64`, `CV_STACK`.
- Produces:
  - `struct blake3core::Hasher` with fields as written in Step 3.
  - `B3_HD void Hasher::init()`
  - `B3_HD void Hasher::absorb(const uint64_t *in, uint32_t n)`
  - `B3_HD void Hasher::finalize_xof(uint32_t ob, uint64_t out[8]) const` — const; operates on a copy so absorption can continue.

- [ ] **Step 1: Write the failing test**

Create `pil2-stark/src/goldilocks/tests/test_blake3_transcript_cpu.cpp`:

```cpp
#include "test_helpers.hpp"
#include "blake3_core.hpp"

// Golden vectors: first 64 bytes of the BLAKE3 XOF over the canonical-LE byte
// stream of words {i*7+3 : i < n}, read as 8 little-endian u64s and reduced by
// to_canonical. Produced with:
//   b3sum --no-names --raw --length 64 <stream>
// These come from the reference implementation, not from this code.
struct GoldenXof { uint32_t n; uint64_t out[8]; };

static const GoldenXof kGolden[] = {
    {0,   {12007152915317330863ULL, 5317022963504857248ULL, 13191819210669804443ULL,
           7075753032064146124ULL, 7778449616772730848ULL, 5778175280161008255ULL,
           9713341381474225459ULL, 4189482813856917916ULL}},
    {1,   {7175107962703238627ULL, 10266939839236670080ULL, 4991015526828484636ULL,
           8987319093039342568ULL, 1351182096268293405ULL, 7939036329090870593ULL,
           164747584250988928ULL, 4740858698442839413ULL}},
    {8,   {4730537481712038625ULL, 11017037621828616572ULL, 8170974584552723967ULL,
           7517513685320260256ULL, 6657018348538791ULL, 1565292996946479837ULL,
           7314032597147226577ULL, 13026651682203648936ULL}},
    {9,   {2424636365142760339ULL, 15165381830123158802ULL, 9487485792073438855ULL,
           5920058426812994410ULL, 16462720151111991777ULL, 7237086037464224556ULL,
           14801379881922525855ULL, 18396241790501459263ULL}},
    {128, {17371188378716342344ULL, 16531850910111656179ULL, 1014328584800827036ULL,
           3601941703461256790ULL, 1751875036402092858ULL, 4808593708865557967ULL,
           15152517445808520735ULL, 2997300546274182758ULL}},
    {129, {5720223199198177089ULL, 1616328176700693306ULL, 11607354963061503359ULL,
           124068739580767596ULL, 10168400764208780594ULL, 2177400346631034771ULL,
           6479388027346566143ULL, 3331708523207561586ULL}},
    {300, {8904515693256777727ULL, 8781243969420736812ULL, 5279824308682382935ULL,
           17909270760646641756ULL, 7942935639058460579ULL, 1402413059402118237ULL,
           14886447596766557147ULL, 5094709958704476073ULL}},
};

static void makeStream(uint32_t n, std::vector<uint64_t> &v)
{
    v.resize(n);
    for (uint32_t i = 0; i < n; ++i) v[i] = (uint64_t)i * 7 + 3;
}

// The XOF's first four words are the BLAKE3 digest, which hash_le64 already
// computes. Cross-checking against it covers every stream length cheaply.
TEST(Blake3Transcript, FinalizeMatchesHashLe64)
{
    for (uint32_t n : {0u, 1u, 7u, 8u, 9u, 127u, 128u, 129u, 200u, 256u, 257u, 300u, 641u}) {
        std::vector<uint64_t> in;
        makeStream(n, in);

        blake3core::Hasher h;
        h.init();
        h.absorb(in.data(), n);
        uint64_t xof[8];
        h.finalize_xof(0, xof);

        uint64_t expect[4];
        blake3core::hash_le64(in.data(), n, expect);
        for (int i = 0; i < 4; ++i)
            ASSERT_EQ(xof[i], expect[i]) << "n=" << n << " word=" << i;
    }
}

// Full 64-byte XOF against the reference implementation.
TEST(Blake3Transcript, FinalizeMatchesReferenceBlake3)
{
    for (const auto &g : kGolden) {
        std::vector<uint64_t> in;
        makeStream(g.n, in);

        blake3core::Hasher h;
        h.init();
        h.absorb(in.data(), g.n);
        uint64_t xof[8];
        h.finalize_xof(0, xof);

        for (int i = 0; i < 8; ++i)
            ASSERT_EQ(xof[i], g.out[i]) << "n=" << g.n << " word=" << i;
    }
}

// Absorbing in arbitrary pieces must equal absorbing in one go.
TEST(Blake3Transcript, AbsorbIsChunkingInvariant)
{
    std::vector<uint64_t> in;
    makeStream(300, in);

    uint64_t one[8];
    {
        blake3core::Hasher h; h.init();
        h.absorb(in.data(), 300);
        h.finalize_xof(0, one);
    }
    for (uint32_t split : {1u, 3u, 7u, 8u, 9u, 63u, 128u, 129u}) {
        blake3core::Hasher h; h.init();
        uint32_t off = 0;
        while (off < 300) {
            uint32_t take = split;
            if (off + take > 300) take = 300 - off;
            h.absorb(in.data() + off, take);
            off += take;
        }
        uint64_t got[8];
        h.finalize_xof(0, got);
        for (int i = 0; i < 8; ++i)
            ASSERT_EQ(got[i], one[i]) << "split=" << split << " word=" << i;
    }
}

// finalize_xof is const: absorbing after a squeeze must continue the same
// stream, not restart it.
TEST(Blake3Transcript, FinalizeDoesNotDisturbTheChain)
{
    std::vector<uint64_t> in;
    makeStream(300, in);

    blake3core::Hasher h; h.init();
    h.absorb(in.data(), 150);
    uint64_t mid[8];
    h.finalize_xof(0, mid);          // squeeze in the middle
    h.absorb(in.data() + 150, 150);
    uint64_t got[8];
    h.finalize_xof(0, got);

    blake3core::Hasher clean; clean.init();
    clean.absorb(in.data(), 300);
    uint64_t want[8];
    clean.finalize_xof(0, want);

    for (int i = 0; i < 8; ++i) ASSERT_EQ(got[i], want[i]) << "word=" << i;
}

// Output-block counter: ob=1 must give the next 64 bytes, and must not equal ob=0.
TEST(Blake3Transcript, XofOutputBlockCounterAdvances)
{
    std::vector<uint64_t> in;
    makeStream(9, in);
    blake3core::Hasher h; h.init();
    h.absorb(in.data(), 9);

    uint64_t b0[8], b1[8];
    h.finalize_xof(0, b0);
    h.finalize_xof(1, b1);

    bool differs = false;
    for (int i = 0; i < 8; ++i) if (b0[i] != b1[i]) differs = true;
    ASSERT_TRUE(differs);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd pil2-stark/src/goldilocks && make testscpu -j
```
Expected: **compile failure**, `'Hasher' is not a member of 'blake3core'`.

- [ ] **Step 3: Write minimal implementation**

Append to `pil2-stark/src/goldilocks/src/blake3_core.hpp`, immediately before `}  // namespace blake3core`:

```cpp
// ---------------------------------------------------------------------------
// Incremental hasher, for the Fiat-Shamir transcript.
//
// Absorbs whole Goldilocks words and produces challenge material from BLAKE3's
// XOF, so the transcript hash is genuinely blake3(canonical-LE byte stream) and
// can be checked against any reference implementation.
//
// finalize_xof is const: it roots a *copy*, because BLAKE3's ROOT flag is
// terminal but a Fiat-Shamir transcript keeps absorbing after each challenge.
// This is what the reference implementation's finalize_xof does too.
// ---------------------------------------------------------------------------
struct Hasher
{
    uint32_t cv[8];            // chaining value of the chunk in progress
    uint64_t buf[BLOCK_U64];   // buffered words, never compressed until more arrive
    uint32_t buf_len;          // words buffered, 0..8
    uint32_t chunk_blocks;     // blocks already compressed into cv, 0..15
    uint64_t chunk_counter;    // index of the chunk in progress
    uint32_t stack[CV_STACK * 8];
    int32_t slen;              // completed subtree cvs on the stack
    uint64_t chunks_done;      // completed chunks, drives the merge rule

    B3_HD void init()
    {
        for (int i = 0; i < 8; ++i) cv[i] = b3_iv(i);
        buf_len = 0;
        chunk_blocks = 0;
        chunk_counter = 0;
        slen = 0;
        chunks_done = 0;
    }

    // Pack the buffered words into a 16-word block, zero-padding the tail.
    B3_HD void fill_block(uint32_t block[16]) const
    {
        for (uint32_t k = 0; k < BLOCK_U64; ++k)
        {
            const uint64_t v = (k < buf_len) ? to_canonical(buf[k]) : 0ull;
            block[2 * k]     = (uint32_t)v;
            block[2 * k + 1] = (uint32_t)(v >> 32);
        }
    }

    // Compress the buffered block as a NON-final block of the current chunk.
    B3_HD void compress_buffered()
    {
        uint32_t block[16];
        fill_block(block);
        uint8_t flags = 0;
        if (chunk_blocks == 0) flags |= FLAG_CHUNK_START;
        compress_in_place(cv, block, (uint8_t)(buf_len * 8u), chunk_counter, flags);
        ++chunk_blocks;
        buf_len = 0;

        if (chunk_blocks == CHUNK_U64 / BLOCK_U64)   // 16 blocks: chunk complete
        {
            // Push this chunk's cv, merging while the completed count is even,
            // exactly as hash_le64 does.
            uint32_t node[8];
            for (int i = 0; i < 8; ++i) node[i] = cv[i];
            uint64_t total = chunks_done + 1;
            while ((total & 1ull) == 0)
            {
                uint32_t merged[8];
                parent_cv(&stack[(slen - 1) * 8], node, false, merged);
                for (int i = 0; i < 8; ++i) node[i] = merged[i];
                --slen;
                total >>= 1;
            }
            for (int i = 0; i < 8; ++i) stack[slen * 8 + i] = node[i];
            ++slen;

            ++chunks_done;
            ++chunk_counter;
            chunk_blocks = 0;
            for (int i = 0; i < 8; ++i) cv[i] = b3_iv(i);
        }
    }

    B3_HD void absorb(const uint64_t *in, uint32_t n)
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            // Only compress once we know more input follows, so the final block
            // is always still buffered when finalize_xof runs.
            if (buf_len == BLOCK_U64) compress_buffered();
            buf[buf_len++] = in[i];
        }
    }

    // 64 bytes of XOF output as 8 canonical Goldilocks words. `ob` is the
    // output-block index: in BLAKE3 the root node's counter field carries it.
    B3_HD void finalize_xof(uint32_t ob, uint64_t out[8]) const
    {
        uint32_t root_cv[8];
        uint32_t root_block[16];
        uint8_t root_len;
        uint8_t root_flags;

        uint8_t last_flags = FLAG_CHUNK_END;
        if (chunk_blocks == 0) last_flags |= FLAG_CHUNK_START;

        if (slen == 0)
        {
            // Single chunk: the buffered block is the root node.
            for (int i = 0; i < 8; ++i) root_cv[i] = cv[i];
            fill_block(root_block);
            root_len = (uint8_t)(buf_len * 8u);
            root_flags = (uint8_t)(last_flags | FLAG_ROOT);
        }
        else
        {
            // Close this chunk (no ROOT), then merge a copy of the stack. The
            // final merge is the root parent.
            uint32_t node[8];
            for (int i = 0; i < 8; ++i) node[i] = cv[i];
            uint32_t block[16];
            fill_block(block);
            compress_in_place(node, block, (uint8_t)(buf_len * 8u), chunk_counter, last_flags);

            int32_t s = slen;
            while (s > 1)
            {
                uint32_t merged[8];
                parent_cv(&stack[(s - 1) * 8], node, false, merged);
                for (int i = 0; i < 8; ++i) node[i] = merged[i];
                --s;
            }
            for (int i = 0; i < 8; ++i) root_cv[i] = b3_iv(i);
            for (int i = 0; i < 8; ++i)
            {
                root_block[i]     = stack[i];
                root_block[8 + i] = node[i];
            }
            root_len = 64;
            root_flags = (uint8_t)(FLAG_PARENT | FLAG_ROOT);
        }

        uint32_t xof[16];
        compress_xof(root_cv, root_block, root_len, (uint64_t)ob, root_flags, xof);
        for (int k = 0; k < 8; ++k)
            out[k] = to_canonical((uint64_t)xof[2 * k] | ((uint64_t)xof[2 * k + 1] << 32));
    }
};
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd pil2-stark/src/goldilocks && make testscpu -j && ./testscpu --gtest_filter='Blake3Transcript.*'
```
Expected: 5 tests PASS.

If `FinalizeMatchesHashLe64` fails only at `n=0`, the empty-stream flags are wrong: with nothing absorbed, `chunk_blocks == 0` and `buf_len == 0`, so the root must still carry `CHUNK_START|CHUNK_END|ROOT` with `block_len = 0`.

- [ ] **Step 5: Confirm Poseidon is untouched**

```bash
cd pil2-stark/src/goldilocks && ./testscpu
```
Expected: the whole suite passes, no pre-existing test changes result.

- [ ] **Step 6: Commit**

```bash
git add pil2-stark/src/goldilocks/src/blake3_core.hpp \
        pil2-stark/src/goldilocks/tests/test_blake3_transcript_cpu.cpp
git commit -m "feat(blake3): incremental hasher with XOF finalize in the shared core

Absorbs whole Goldilocks words and finalizes a copy, so a Fiat-Shamir
transcript can keep absorbing after each challenge. Lives in blake3_core.hpp,
the shared host/device core, so the CPU and CUDA transcripts will be one
implementation rather than two.

Validated against hash_le64 for the digest at 13 stream lengths, and against
b3sum-derived golden vectors for the full 64-byte XOF."
```

---

### Task 2: CPU transcript uses the hasher

**Files:**
- Modify: `pil2-stark/src/starkpil/transcript/transcriptGL.hpp`
- Modify: `pil2-stark/src/starkpil/transcript/transcriptGL.cpp`
- Test: `pil2-stark/src/goldilocks/tests/test_blake3_transcript_cpu.cpp` (extend)

**Interfaces:**
- Consumes: `blake3core::Hasher` from Task 1 (`init`, `absorb`, `finalize_xof`).
- Produces: no API change. `TranscriptGL::put`, `getField`, `getState`, `getState(n)`, `getPermutations` keep their exact signatures, so all 34 call sites outside the transcript directory are untouched.

> `TranscriptGL` lives in `starkpil`, not `goldilocks`, so it is not linked into
> `testscpu`. Test it through `blake3core::Hasher` plus a small local reimplementation
> of the cursor logic — or, if `starkpil` is reachable from the test target, prefer
> linking it directly. Check with:
> `grep -n "starkpil" pil2-stark/src/goldilocks/Makefile`
> If it is not reachable, keep Task 2's assertions in Task 4's cross-implementation
> harness, which drives the real `TranscriptGL` through the C API.

- [ ] **Step 1: Add the blake3 state to the header**

In `transcriptGL.hpp`, add the include and the member. Keep every existing member — Poseidon still needs them.

```cpp
#include "blake3_core.hpp"
```

Inside `class TranscriptGL`, in the `private:` section:

```cpp
    // blake3 transcript state. Unused by the Poseidon families, which keep the
    // sponge members above. `b3_offset` counts words already consumed from the
    // current XOF stream; `b3_xof_valid` says whether `b3_xof` is live.
    blake3core::Hasher b3;
    uint64_t b3_xof[8];
    uint32_t b3_offset = 0;    // words consumed from the current XOF block
    uint32_t b3_ob = 0;        // index of the XOF output block in b3_xof
    bool b3_xof_valid = false;
```

In the constructor body, after the existing `memset` calls:

```cpp
        b3.init();
        b3_offset = 0;
        b3_ob = 0;
        b3_xof_valid = false;
```

- [ ] **Step 2: Branch `_add1` in the .cpp**

Replace `TranscriptGL::_add1` with:

```cpp
void TranscriptGL::_add1(Goldilocks::Element input)
{
    if (get_hash_family() == HashFamily::Blake3)
    {
        const uint64_t w = Goldilocks::toU64(input);
        b3.absorb(&w, 1);
        // The stream changed, so any XOF material derived from it is stale.
        b3_xof_valid = false;
        b3_offset = 0;
        b3_ob = 0;
        return;
    }

    pending[pending_cursor] = input;
    pending_cursor++;
    out_cursor = 0;
    if (pending_cursor == transcriptPendingSize)
    {
        _updateState();
    }
}
```

- [ ] **Step 3: Branch `getFields1` and `getState`**

Replace `TranscriptGL::getFields1` with:

```cpp
Goldilocks::Element TranscriptGL::getFields1()
{
    if (get_hash_family() == HashFamily::Blake3)
    {
        // Refill when the stream changed or the current output block is drained.
        // A refill only advances the output-block counter -- it does not
        // re-finalize, because the root node is unchanged.
        if (!b3_xof_valid)
        {
            b3_ob = 0;
            b3.finalize_xof(b3_ob, b3_xof);
            b3_offset = 0;
            b3_xof_valid = true;
        }
        else if (b3_offset == 8)
        {
            ++b3_ob;
            b3.finalize_xof(b3_ob, b3_xof);
            b3_offset = 0;
        }
        return Goldilocks::fromU64(b3_xof[b3_offset++]);
    }

    if (out_cursor == 0)
    {
        _updateState();
    }
    Goldilocks::Element res = out[(transcriptOutSize - out_cursor) % transcriptOutSize];
    out_cursor--;
    return res;
}
```

The output-block index must live in its own field rather than be derived from
`b3_offset`, which is reset on every refill.

Then in both `getState` overloads, branch before the `pending_cursor` check:

```cpp
void TranscriptGL::getState(Goldilocks::Element* output) {
    if (get_hash_family() == HashFamily::Blake3)
    {
        // Does not consume: getState reads the digest, it does not drain output.
        uint64_t xof[8];
        b3.finalize_xof(0, xof);
        for (uint32_t i = 0; i < transcriptStateSize; ++i)
            output[i] = Goldilocks::fromU64(xof[i]);
        return;
    }
    if(pending_cursor > 0) {
        _updateState();
    }
    std::memcpy(output, state, transcriptStateSize * sizeof(Goldilocks::Element));
}
```

Apply the same branch to `getState(Goldilocks::Element* output, uint64_t nOutputs)`, copying `nOutputs` words and asserting `nOutputs <= 8`.

`put`, `getField` and `getPermutations` need no change — they route through `_add1` and `getFields1`.

- [ ] **Step 4: Make `_updateState` unreachable for blake3**

The blake3 branch in `_updateState` is now dead. Replace the `case HashFamily::Blake3:` body with a hard error so a missed branch fails loudly instead of silently using the old construction:

```cpp
    case HashFamily::Blake3:
        zklog.error("TranscriptGL::_updateState: unreachable for blake3; "
                    "the blake3 transcript absorbs via blake3core::Hasher");
        exitProcess();
        exit(-1);
```

- [ ] **Step 5: Build**

```bash
cd pil2-stark && make -j 2>&1 | tail -20
```
Expected: builds clean. If `blake3_core.hpp` is not on the include path for `starkpil`, add `-I$(GOLDILOCKS_SRC)` — check with `grep -n "goldilocks/src" pil2-stark/Makefile`.

- [ ] **Step 6: Commit**

```bash
git add pil2-stark/src/starkpil/transcript/transcriptGL.hpp \
        pil2-stark/src/starkpil/transcript/transcriptGL.cpp
git commit -m "feat(blake3): CPU transcript absorbs into a real BLAKE3 hasher

put() absorbs canonical-LE words; challenges come from the XOF, refilling by
advancing the output-block counter rather than re-finalizing. getState returns
the BLAKE3 digest and does not consume. Poseidon paths untouched; the old
blake3 branch in _updateState is now a hard error."
```

---

### Task 3: Rust verifier transcript over the official crate

**Files:**
- Modify: `fields/Cargo.toml`
- Create: `fields/src/blake3_transcript.rs`
- Modify: `fields/src/lib.rs`
- Modify: `fields/src/transcript_api.rs`

**Interfaces:**
- Consumes: `blake3` crate (`Hasher::new`, `update`, `finalize_xof`, `OutputReader::fill`).
- Produces:
  - `pub struct Blake3Transcript<F: PrimeField64>`
  - `pub fn Blake3Transcript::new() -> Self`
  - `pub fn put(&mut self, inputs: &[F])`
  - `pub fn get_state(&mut self) -> Vec<F>` — 4 words
  - `pub fn get_field(&mut self, value: &mut [F])` — fills 3
  - `pub fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64>`

- [ ] **Step 1: Write the failing test**

Create `fields/src/blake3_transcript.rs` containing only the test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Goldilocks;

    fn words(n: usize) -> Vec<Goldilocks> {
        (0..n).map(|i| Goldilocks::from_u64((i as u64) * 7 + 3)).collect()
    }

    /// The transcript must be blake3(canonical-LE stream), so the reference
    /// crate fed the same bytes must agree word for word.
    #[test]
    fn matches_reference_blake3() {
        for n in [0usize, 1, 7, 8, 9, 127, 128, 129, 300] {
            let xs = words(n);

            let mut t = Blake3Transcript::<Goldilocks>::new();
            t.put(&xs);
            let got = t.get_state();

            let mut h = blake3::Hasher::new();
            for x in &xs {
                h.update(&x.as_canonical_u64().to_le_bytes());
            }
            let mut buf = [0u8; 32];
            h.finalize_xof().fill(&mut buf);

            for i in 0..4 {
                let raw = u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
                let expect = Goldilocks::from_u64(canon(raw));
                assert_eq!(got[i], expect, "n={n} word={i}");
            }
        }
    }

    /// put() after a read must restart the XOF stream from the new prefix.
    #[test]
    fn put_invalidates_the_xof_stream() {
        let mut t = Blake3Transcript::<Goldilocks>::new();
        t.put(&words(4));
        let mut a = [Goldilocks::ZERO; 3];
        t.get_field(&mut a);

        t.put(&words(1));
        let mut b = [Goldilocks::ZERO; 3];
        t.get_field(&mut b);
        assert_ne!(a, b);
    }

    /// get_state must not consume, so a following get_field is unaffected.
    #[test]
    fn get_state_does_not_consume() {
        let mut t1 = Blake3Transcript::<Goldilocks>::new();
        t1.put(&words(5));
        let _ = t1.get_state();
        let mut with = [Goldilocks::ZERO; 3];
        t1.get_field(&mut with);

        let mut t2 = Blake3Transcript::<Goldilocks>::new();
        t2.put(&words(5));
        let mut without = [Goldilocks::ZERO; 3];
        t2.get_field(&mut without);

        assert_eq!(with, without);
    }

    /// Reading more than 8 words must advance into the next XOF block, not repeat.
    #[test]
    fn reads_past_one_block_advance() {
        let mut t = Blake3Transcript::<Goldilocks>::new();
        t.put(&words(5));
        let mut seen = alloc::vec::Vec::new();
        for _ in 0..6 {
            let mut v = [Goldilocks::ZERO; 3];
            t.get_field(&mut v);
            seen.extend_from_slice(&v);
        }
        // 18 words spans three 8-word XOF blocks; no duplicates expected.
        for i in 0..seen.len() {
            for j in i + 1..seen.len() {
                assert_ne!(seen[i], seen[j], "duplicate at {i},{j}");
            }
        }
    }
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test -p proofman-fields blake3_transcript
```
Expected: compile error — `Blake3Transcript` not found, `blake3` crate not a dependency of `proofman-fields`.

- [ ] **Step 3: Add the dependency**

In `fields/Cargo.toml`, under `[dependencies]`:

```toml
blake3 = { workspace = true, default-features = false }
```

`default-features = false` is required because `fields` is `#![no_std]`. Confirm the portable build works for the `zisk` target vendor too:

```bash
cargo build -p proofman-fields
```

If `blake3` pulls in `std` transitively, check its features with
`cargo tree -p proofman-fields -i blake3 -e features` and disable the offender.

- [ ] **Step 4: Write the implementation**

Prepend to `fields/src/blake3_transcript.rs`, above the test module:

```rust
//! Fiat-Shamir transcript that is a genuine BLAKE3 over the absorbed byte
//! stream, with challenges from its XOF.
//!
//! Wraps the reference `blake3` crate rather than reimplementing, so this side
//! is correct by construction and serves as the oracle for the C++ prover.

use alloc::vec::Vec;

use crate::PrimeField64;

const GL_P: u64 = 0xFFFFFFFF00000001;

/// One conditional subtract, matching `blake3core::to_canonical`.
fn canon(x: u64) -> u64 {
    if x >= GL_P { x - GL_P } else { x }
}

pub struct Blake3Transcript<F: PrimeField64> {
    hasher: blake3::Hasher,
    xof: [u64; 8],
    /// Words consumed from the current XOF block.
    offset: usize,
    /// Index of the XOF output block currently in `xof`.
    block: u64,
    valid: bool,
    _marker: core::marker::PhantomData<F>,
}

impl<F: PrimeField64> Default for Blake3Transcript<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField64> Blake3Transcript<F> {
    pub fn new() -> Self {
        Blake3Transcript {
            hasher: blake3::Hasher::new(),
            xof: [0u64; 8],
            offset: 0,
            block: 0,
            valid: false,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn put(&mut self, inputs: &[F]) {
        for x in inputs {
            self.hasher.update(&x.as_canonical_u64().to_le_bytes());
        }
        // Stream changed: the XOF derived from the old prefix is stale.
        self.valid = false;
        self.offset = 0;
        self.block = 0;
    }

    /// Load XOF output block `self.block` into `self.xof`.
    fn load_block(&mut self) {
        let mut reader = self.hasher.finalize_xof();
        // Seek to the requested 64-byte block, then read it.
        let mut skip = [0u8; 64];
        for _ in 0..self.block {
            reader.fill(&mut skip);
        }
        let mut buf = [0u8; 64];
        reader.fill(&mut buf);
        for i in 0..8 {
            self.xof[i] = canon(u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap()));
        }
        self.offset = 0;
        self.valid = true;
    }

    fn get_fields1(&mut self) -> F {
        if !self.valid {
            self.block = 0;
            self.load_block();
        } else if self.offset == 8 {
            self.block += 1;
            self.load_block();
        }
        let v = self.xof[self.offset];
        self.offset += 1;
        F::from_u64(v)
    }

    /// The BLAKE3 digest of the transcript so far. Does not consume.
    pub fn get_state(&mut self) -> Vec<F> {
        let mut reader = self.hasher.finalize_xof();
        let mut buf = [0u8; 32];
        reader.fill(&mut buf);
        (0..4)
            .map(|i| F::from_u64(canon(u64::from_le_bytes(buf[8 * i..8 * i + 8].try_into().unwrap()))))
            .collect()
    }

    pub fn get_field(&mut self, value: &mut [F]) {
        for v in value.iter_mut().take(3) {
            *v = self.get_fields1();
        }
    }

    pub fn get_permutations(&mut self, n: u64, n_bits: u64) -> Vec<u64> {
        let total_bits = n * n_bits;
        let n_fields = ((total_bits - 1) / 63) + 1;
        let mut fields = Vec::with_capacity(n_fields as usize);
        for _ in 0..n_fields {
            fields.push(self.get_fields1());
        }

        let mut cur_field = 0usize;
        let mut cur_bit = 0u64;
        let mut permutations = alloc::vec![0u64; n as usize];
        for i in 0..n as usize {
            let mut a = 0u64;
            for j in 0..n_bits {
                let bit = (fields[cur_field].as_canonical_u64() >> cur_bit) & 1;
                if bit == 1 {
                    a += 1 << j;
                }
                cur_bit += 1;
                if cur_bit == 63 {
                    cur_bit = 0;
                    cur_field += 1;
                }
            }
            permutations[i] = a;
        }
        permutations
    }
}
```

> `load_block` re-reads from the start each time, which is O(block²) in the
> number of 64-byte blocks consumed. With at most a few blocks per squeeze point
> that is irrelevant, and it keeps the reader stateless across `put`
> invalidation. If `blake3::OutputReader::set_position` is available in 1.8.6,
> use it instead and delete the skip loop.

`get_permutations` is a verbatim port of `Transcript::get_permutations` so the
bit-packing stays identical; do not "improve" it.

- [ ] **Step 5: Register the module**

In `fields/src/lib.rs`, beside `mod poseidon2;` etc.:

```rust
mod blake3_transcript;
```

and in the re-export block alongside the other `pub use` lines:

```rust
pub use blake3_transcript::*;
```

- [ ] **Step 6: Run the tests**

```bash
cargo test -p proofman-fields blake3_transcript
```
Expected: 4 tests PASS.

- [ ] **Step 7: Wire it into `TranscriptDyn`**

In `fields/src/transcript_api.rs`, change the variant and both uses:

```rust
use crate::{Blake3Transcript, Blake3_8, Hash, Poseidon1_16, Poseidon2_16, PrimeField64, Transcript};

pub enum TranscriptDyn<F: PrimeField64> {
    Poseidon1(Transcript<F, Poseidon1_16>),
    Poseidon2(Transcript<F, Poseidon2_16>),
    Blake3(Blake3Transcript<F>),
}
```

and in `new_transcript`:

```rust
        "blake3" => TranscriptDyn::Blake3(Blake3Transcript::<F>::new()),
```

The three `match` arms in `put` / `get_state` / `get_field` need no change — the
method names and signatures are identical. `Blake3_8` stays imported because
`hash_state` still uses it.

- [ ] **Step 8: Full test run**

```bash
cargo test -p proofman-fields && cargo build
```
Expected: all pass, workspace builds.

- [ ] **Step 9: Commit**

```bash
git add fields/Cargo.toml fields/src/blake3_transcript.rs fields/src/lib.rs \
        fields/src/transcript_api.rs
git commit -m "feat(blake3): Rust verifier transcript over the reference blake3 crate

TranscriptDyn::Blake3 now wraps Blake3Transcript instead of
Transcript<F, Blake3_8>: the sponge Hash trait (WIDTH/RATE/CAPACITY) cannot
express a BLAKE3-as-specified transcript. Uses the official crate, already a
workspace dependency, so the verifier is correct by construction and can serve
as the oracle for the C++ side."
```

---

### Task 4: Cross-implementation oracle harness

**Files:**
- Create: `setup/stark-recurser/stark2circom/tests/blake3/transcript/run_transcript_tests.sh`
- Create: `setup/stark-recurser/stark2circom/tests/blake3/transcript/cpp_driver.cpp`
- Create: `setup/stark-recurser/stark2circom/tests/blake3/transcript/rust_driver.rs` (as a `#[test]` in `fields`)
- Modify: `setup/stark-recurser/stark2circom/tests/blake3/README.md`

**Interfaces:**
- Consumes: `blake3core::Hasher` (Task 1), `TranscriptGL` (Task 2), `Blake3Transcript` (Task 3), `b3sum`.
- Produces: a script exiting non-zero on any mismatch.

- [ ] **Step 1: Define the scripted sequence**

One fixed put/get script, exercising the boundaries the spec names. Write it to
`transcript/script.txt` in the test dir; both drivers read it:

```
put 3
get_field
put 5
get_field
put 120
get_field
put 8
get_state
put 1024
get_field
get_field
get_field
put 1
get_permutations 64 5
get_state
```

`put N` absorbs words `i*7+3` for the next `N` values of a running counter, so
both drivers absorb identical streams. Cumulative absorb crosses 64 B, 1024 B
and 8192 B, and the three consecutive `get_field`s span more than one 64-byte
XOF block.

- [ ] **Step 2: Write the C++ driver**

`cpp_driver.cpp` reads `script.txt`, drives a real `TranscriptGL` with
`set_hash_family(HashFamily::Blake3)`, prints every produced word one per line,
and separately writes the absorbed byte stream to `stream.bin` so `b3sum` can
check it. Model the build on `tests/blake3/run_tests.sh`'s `ref_merkle`
compilation, which already links `blake3_goldilocks.cpp` and
`goldilocks_base_field.cpp` against `-I$GL`; `TranscriptGL` additionally needs
`starkpil` on the include path.

- [ ] **Step 3: Write the Rust driver as a test**

Add to `fields/src/blake3_transcript.rs`'s test module a `#[test]` that runs the
same script through `Blake3Transcript` and writes the produced words to a path
given by the `BLAKE3_TRANSCRIPT_OUT` env var, skipping if unset. That keeps it a
normal unit test while letting the shell harness drive it.

- [ ] **Step 4: Write the comparison script**

`run_transcript_tests.sh` must:
1. build and run the C++ driver, capturing its words;
2. run the Rust driver with `BLAKE3_TRANSCRIPT_OUT` set, capturing its words;
3. `diff` the two, failing loudly on any difference;
4. independently check the final `get_state` against
   `b3sum --no-names --raw --length 32 stream.bin`, reduced by `to_canonical`,
   so both implementations are anchored to the reference rather than to each other.

- [ ] **Step 5: Run it**

```bash
cd setup/stark-recurser/stark2circom/tests/blake3/transcript && ./run_transcript_tests.sh
```
Expected: all comparisons pass.

- [ ] **Step 6: Document and commit**

Add a "Transcript" section to `tests/blake3/README.md` describing the script, the
three-way comparison, and why the `b3sum` anchor matters (so the failure mode
"all implementations agree on the wrong thing" is excluded).

```bash
git add setup/stark-recurser/stark2circom/tests/blake3/
git commit -m "test(blake3): three-way transcript oracle harness

One scripted put/get sequence driven through the C++ TranscriptGL and the Rust
Blake3Transcript, compared against each other and independently anchored to
b3sum, so agreement on a wrong construction is not a passing state."
```

---

## Self-Review

**Spec coverage.** Spec Section 1 (construction) → Task 1 + Task 2. Section 2
(where it lives, family branch, API unchanged) → Task 2 Step 3/Interfaces, Task 3
Step 7. Section 3 items 1, 2, 4 → Tasks 1, 2, 3. Section 4 (testing, oracle) →
Task 1 Steps 1/4 and Task 4. Spec items 3 (GPU), 5 (emitter) and 6 (circom) are
**deliberately out of this plan** — they are Stages B and C.

**Gap found and accepted:** the spec says `hash_family.rs`'s
`transcript_arity` / `transcript_pending_size` / `transcript_out_size` become
Poseidon-only. No task changes them, because nothing in Stage A reads them for
blake3 — they are consumed by the recurser emitter, which is Stage C. Left to
Stage C deliberately rather than churning them twice.

**Placeholder scan.** No TBD/TODO. Task 4 Steps 2–4 describe drivers in prose
rather than full listings — they are shell and glue whose exact form depends on
include paths that must be discovered (noted inline in Task 2's callout). Every
step that produces library or circuit code has literal code.

**Type consistency.** `Hasher::init` / `absorb` / `finalize_xof(ob, out[8])` are
used with those exact names and signatures in Tasks 1, 2 and 4.
`Blake3Transcript::{new, put, get_state, get_field, get_permutations}` match the
`TranscriptDyn` arms in Task 3 Step 7, which are unchanged from the current
`transcript_api.rs`. `canon` is defined once in `blake3_transcript.rs` and used
by both the implementation and its tests.
