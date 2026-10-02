# SHA-256 with instance-sliced cells: soundness argument

**Purpose.** We arithmetize SHA-256 so that one trace cell carries the same bit position of `n`
independent SHA-256 instances, and the non-linear functions run through batched lookup tables. This
note states the encoding and the constraints, and argues that every accepted trace computes, in
every slot, the SHA-256 round function and message schedule. Section 9 lists what is not covered.

The argument covers two designs (the circuit uses `sa/se/sw` for the `va/ve/vw` below):

| | `n` (instances per cell) | slot spacing `2^K` | add chunk `CH` | file |
|---|---|---|---|---|
| D5 | 5 | `2^12` | 8 bits | `setup/stark-recurser/plonk2pil/pil/circuits/sha2.pil` |
| D4 | 4 | `2^15` | 8 bits | stats prototype (not in the repo) |

Both target `blowupFactor 2` (max constraint degree 5); every constraint below has degree ≤ 2.

---

## 1. Notation

`p = 2^64 − 2^32 + 1`. Integers are identified with their residues when they lie in `[0, p)`.

A **sliced value** with digits `x_0..x_{n-1}` is `x = Σ_k x_k · 2^{Kk}`. `ONE = Σ_k 2^{Kk}` is the
sliced value with every digit 1. A sliced value is **boolean** if every digit is 0 or 1.

The trace has one row per round, `CLOCKS = 68` rows per block (4 load-state rows, 64 rounds). Row
`r` holds three words, each as 32 sliced cells, one per bit position:

```
va[0..31]   a_r      ve[0..31]   e_r      vw[0..31]   w_r
```

`'x` is `x` one row back; `(j)'x` is `j` rows back. So `b, c, d = 'a, 2'a, 3'a` shifted once more:
round `r` reads `a_{r-1..r-4}` as `(1..4)'va`, `e_{r-1..r-4}` as `(1..4)'ve`, and `w_{r-2}`,
`w_{r-7}`, `w_{r-15}`, `w_{r-16}`.

---

## 2. Lemma 1 (slot equations)

> Let `x = Σ_k x_k 2^{Kk}` with integer digits `|x_k| < 2^{K-1}`. If `x ≡ 0 (mod p)` then every
> `x_k = 0`.

*Proof.* `|x| < 2^{K-1} · Σ_{k<n} 2^{Kk} < 2^{Kn} ≤ 2^{60} < p/2`, so `x ≡ 0` forces `x = 0` in
`ℤ`. The balanced base-`2^K` representation with digits in `(−2^{K-1}, 2^{K-1})` is unique, so
every digit is 0. ∎

Numerically, `2^{K-1}·Σ 2^{Kk} < 2^{60}` for both D5 and D4.

---

## 3. Range coverage

Every lookup below is **unconditional** (multiplicity 1 on every row, padding included), and row
shifts are cyclic, so a statement about "every cell at its own row" covers every cell any
constraint reads at any shift.

| table | entries | D5 cells covered | D4 cells covered |
|---|---|---|---|
| range | tuples of 4 boolean sliced values | `ve`, `vw` (all 32 each) | `vw[16..31]` |
| ch | `r` component, boolean sliced value | `va` (all 32) | — |
| parity/maj | `r` component, boolean sliced value | — | `va`, `ve`, `vw[0..15]` |
| carry | sliced value, digits in `[0,6]` | all carries | all carries |

**Lemma 2.** In an accepted trace every state cell is boolean and every carry has digits in
`[0,6]`.

*Proof.* Lookup soundness (the logUp bus of `std_sum`) gives that each assumed tuple equals some
table tuple **component-wise**. Each listed cell is one whole component of an assumed tuple, and the
matching table component is boolean (resp. digits in `[0,6]`) by construction. ∎

**Correction recorded during design.** An earlier revision of D5 omitted the range check on `e`,
arguing that the ch lookup's input `u = e1 + 2e2 + 4e3 ∈ [0,7]` pins `e`. It does not: `u` bounds
only the combination of three consecutive `e` values, and non-boolean values can be chained through
the recurrence (`e_r` slot = 2 with `e_{r-1}`, `e_{r-2}` = 0 gives `u = 2`, which is valid). D5 now
range-checks `e` explicitly.

---

## 4. Lemma 3 (bounded inputs)

The table inputs are sums of boolean cells, computed linearly in the field:

| input | expression per bit position `i` | digits |
|---|---|---|
| Σ0 | `'a[i+2] + 'a[i+13] + 'a[i+22]` (indices mod 32) | `[0,3]` |
| Σ1 | `'e[i+6] + 'e[i+11] + 'e[i+25]` | `[0,3]` |
| σ0 | `w15[i+7] + w15[i+18] + w15[i+3]·[i+3<32]` | `[0,3]` |
| σ1 | `w2[i+17] + w2[i+19] + w2[i+10]·[i+10<32]` | `[0,3]` |
| maj | `'a[i] + 2'a[i] + 3'a[i]` | `[0,3]` |
| ch, D5 | `u = 'e[i] + 2·2'e[i] + 4·3'e[i]` | `[0,7]` |
| ch, D4 | `u = 2·'e[i] + 2'e[i] − 3'e[i]` | `[−1,3]` |

By Lemma 2 every digit is a small integer as listed, and the field value equals the integer
`Σ_k digit_k 2^{Kk}` (it is below `p` in absolute value). ∎

---

## 5. Lemma 4 (keyed lookups)

Every non-linear lookup has the shape **(key, [range cell], output)**, where the key packs *only*
bounded inputs and the output is its own component.

| lookup | key, per slot digit | output |
|---|---|---|
| parity (Σ0, Σ1, σ0, σ1), positions `i, i+1` | `s_i + 4·s_{i+1}` | `o = par(s_i) + 2·par(s_{i+1})` |
| maj, positions `i, i+1` | `s_i + 4·s_{i+1} + 64` (tag) | `o = maj(s_i) + 2·maj(s_{i+1})` |
| ch, D5, position `i` | `u_i` | `o = ch(e,f,g)` |
| ch, D4, positions `i, i+1` | `(u_i + 1) + 8·(u_{i+1} + 1)` | `o = F(u_i) + 2·F(u_{i+1})` |

**Claim.** If an assumed `(key, …, o)` matches a table row, then `o` equals the table's function of
the true inputs, slot by slot.

*Proof.* Per-slot key digits are bounded by 79 (parity/maj, D5 and D4) and 36 (ch, D4), all below
`2^K`. Both the assumed key and the row's key are sliced values with such digits, so equality in
`F_p` is equality in `ℤ` and then digit by digit; the per-slot encoding
`(s, s', t) ↦ s + 4s' + 64t` is injective on `[0,3]² × {0,1}`, as is
`(u, u') ↦ (u+1) + 8(u'+1)` on `[−1,3]²`. So the row's inputs are the true inputs, and component
equality on `o` gives `o` = the row's output. ∎

For D4's ch, the output is `F`, not ch itself. The identity used by the adds is

```
ch(e, f, g) = (g − e) + F(2e + f − g),    F(−1) = F(0) = F(1) = 0, F(2) = 1, F(3) = 2
```

checked exhaustively over `{0,1}^3`. `(g − e)` enters the adds linearly.

**Rejected construction.** Packing the output into the key, `[s + 4s' + 16·o + 64·t]`, saves one
fixed column per block but is **unsound**. With `t` at weight 64 the output absorbs a tag flip: a
parity query whose inputs have every digit equal to 2 matches the **maj** row if it claims
`o` = digits 7 instead of the honest 0. Without the tag, a key mismatch still turns `o` into
`o_R + Δ/16` with `Δ ≠ 0`, and across a chunk's groups (weights `1, 4, 16, 64`) those shifts can
combine into a small integer that the adds absorb. Both prototypes keep `o` as its own component.

---

## 6. Lemma 5 (additions)

Each 32-bit addition is checked per 8-bit chunk `h = 0..3`. With `P(x)_h = Σ_{i<8} 2^i·x[8h+i]`,
the `a` update is

```
MIX · ( P(a)_h + 2^8·ca_h − P(h)_h − S1_h − CH_h − KC_h·ONE − P(w)_h − S0_h − MAJ_h − ca_{h−1} ) = 0
```

where `S0_h = Σ_g 4^j·oS0_g` over the chunk's four groups, and similarly for the other table
contributions; `ca_{−1} = 0`. The `e` update replaces `S0 + MAJ` by `P(d)_h`, and the schedule
uses `σ1, w_{r−7}, σ0, w_{r−16}`.

*Per-slot bounds.* With Lemmas 2–4 every term is a sliced value with integer digits: state packs
and table contributions in `[0, 255]`, carries in `[0, 6]`, `KC` in `[0,255]`, and for D4 the ch
split `(g − e)` in `[−255, 255]` and `F` in `[0, 510]`. The per-slot residual of each equation lies
in:

| equation | residual range | `2^{K−1}` |
|---|---|---|
| D5 `a` (7 terms) | `[−1791, 1791]` | 2048 |
| D5 `e` (6 terms) | `[−1536, 1791]` | 2048 |
| D5 `w` (4 terms) | `[−1026, 1791]` | 2048 |
| D4 `a`, ch split | `[−2301, 2046]` | 16384 |
| D4 `e`, ch split | `[−2046, 2046]` | 16384 |

*Proof.* Each equation is a sliced value with those digits that vanishes mod `p`; by Lemma 1 every
slot's residual is 0 in `ℤ`. Per slot and chunk this reads
`chunk_h + 2^8·c_h = (sum of the chunk's terms) + c_{h−1}` with `chunk_h ∈ [0, 255]`, which fixes
`chunk_h` and `c_h` uniquely as the chunk and the carry of the true sum. Chaining over `h` gives
`new = (sum of terms) mod 2^32` exactly, the final carry being discarded. For D4 the ch term is
`(g − e) + F = ch` by the identity of §5. ∎

---

## 7. Theorem

Fix a slot `k`. In an accepted trace, on every round row the slot-`k` bits of `a_r`, `e_r`
(`MIX = 1`) and `w_r` (`WCOMP = 1`) are those of the SHA-256 round function and message schedule
applied to the slot-`k` bits of the previous rows.

*Proof.* Lemma 2 makes every cell boolean; Lemma 3 makes each table input the integer the
specification needs; Lemma 4 makes each table output the right function of it; and Lemma 5 makes
each new word the 32-bit sum of the right terms. Every step is slot-wise: lookups enumerate all
slot combinations independently, and Lemma 1 splits the additions slot by slot. The instances do
not interact. ∎

---

## 8. Degrees and what the prover pays

All constraints are degree 2 (`MIX` × linear). All lookup tuples are linear in witness columns, so
`std_sum` groups 4 terms per intermediate column at max degree 5. The tables are integrated in the
same air and split into blocks of `N` rows, one multiplicity and one provide term per block.

| D5 tables | rows | blocks at `N = 2^19` |
|---|---|---|
| parity/maj (key, out) | `2·1024²` = 2^21 | 4 |
| ch (u, r, out) | `8^5·2^5` = 2^20 | 2 |
| range (4 cells) | `32^4` = 2^20 | 2 |
| carry | `7^5` = 16,807 | 1 |

---

## 9. Not covered, and questions

1. **Boundary binding.** How the load-state rows, the message words and the outputs (including the
   feed-forward) are tied to the rest of the recursion air is not modelled. Each needs its own
   argument; in particular, any bus must keep slot `k` of every word in slot `k`.
2. **Padding rows.** Lookups run on every row, so the witness must fill padding with boolean cells
   and valid table outputs. Nothing here depends on the padding values, because `MIX = 0` and
   `WCOMP = 0` disable the additions there.
3. **Lookup soundness.** We rely on the logUp argument of `std_sum` (extension-field challenges)
   giving component-wise multiset inclusion. Since every tuple element is a full field element,
   tuples are compared exactly.
4. **Witness and `verify-constraints`.** D5 now has a real witness and honest tables
   (`examples/hashes`); `verify-constraints` passes and rejects a forged table output and a bumped
   state cell.
5. **Questions for a reviewer.** Is Lemma 4's "bounded key, separate output" sufficient for every
   lookup in §5, with no other cross-table aliasing (the tag separates parity from maj; the tables
   have distinct bus ids)? Is anything lost by Lemma 1 using the balanced bound rather than the
   exact residual intervals?

---

## 10. Composition in the recursion air

The `examples/hashes` air runs the 64 rounds over an arbitrary initial state, with no feed-forward
and no binding, exactly like the standalone Blake3 air (`BIND_BOUNDARY = 0`). Sections 2-7 prove
only that. A recursion air must add the following, and each needs its own argument. They follow the
Blake3 recursion circuit, and the bug classes ZisK's history keeps fixing (partial bindings,
miscounted carries, forgeable selectors, unpinned padding rows).

**R1. Selectors are fixed and block-wide.** `SEL`, `MIX` and `WCOMP` must be fixed columns that
are constant over a whole block, as Blake3's `BLAKE3_NODE/CHUNK/PARENT` are. A constraint on row
`r` reads cells up to 16 rows back (68 with the feed-forward below), and those cells are bounded
only by the lookups on their own rows. So `SEL = 1` must hold on every row of an active block, and
`MIX = WCOMP = 0` wherever `SEL = 0`. A witness selector, or one that can switch off inside a block,
breaks Lemma 2 and with it Lemma 1. The air must also leave `CLOCKS - 1` rows after the last block
that are not block starts, so the `(i)'CLK_0` shifts cannot wrap into a block (the same check as
`blake3/compressor.pil`).

**R2. Feed-forward.** `H' = H + final state`, eight 32-bit additions per hash. In sliced form it
is one more addition per chunk with two terms. The true carry is 0 or 1, but checked against the
same `[0,6]` table the residual stays in `[-511, 1791]`, so Lemma 1 and Lemma 5 apply unchanged. Four write rows `68..71` put the output `(d,h), (c,g), (b,f),
(a,e)` in the order of the load rows. Row `68 + j` adds load row `j` (68 rows back) and final row
`64 + j` (4 rows back): the shifts are constant, and `sa/se/ca/ce` are reused. The cost is 72 clocks
instead of 68 and one new opening (shift 68). A chained compression can use its write rows as the
next block's load rows.

**R3. Initial state.** For a Merkle node, `H = IV` is a constant per sliced cell (digit `k` = bit
`i` of the IV word, the same in every slot). All 8 words × 32 cells must be pinned, not only some
limbs: blake2br's missing limb-1 copies (ZisK `68cd431af`) are the warning. For the ternary node or
the transcript, `H` comes from the band and falls under R4.

**R4. Per-lane binding (open: soundness AND cost).** A band cell carries one lane's value, while a
sliced cell carries one bit of all five lanes, and no linear map extracts slot `k`. Binding through
the lane-packed chunk works: `P(w)_h = Σ_k B_{k,h}·2^{12k}`, with witness bytes
`B_{k,h} ∈ [0,255]` and the band value `= Σ_h 256^h·B_{k,h}`. By Lemma 1 the bytes are forced
(non-negative digits below `2^12`). But that is 20 byte cells per word, and every message word,
every non-constant state word and every output word needs it. This cost is NOT in the
recursion-cell model ("the cost of loading proof data into 5-lane slots is not counted"), so the
self-fit there is a floor until it is designed and measured.

**R5. Canonical split.** A Goldilocks element entering as two words needs `(lo, hi)` to be the
canonical split: `v` and `v + p` both decompose for `v <= 2^32 - 2`, and the prover would pick which
bytes get hashed. Reuse `blake3SplitCanonicalGadget`. On the output side, packing two words into
one element reduces mod `p`; that map is not injective on `[p, 2^64)`, which costs a negligible
amount of collision resistance and is the same choice Blake3Node makes.

**R6. Slot k is hash k.** Every binding of R3-R5 must route slot `k` of every word to the same hash
instance and the same band row. Mixing slots across words is not caught by any lookup.

**What does not carry over from Blake3.** SHA2 has no message bus: the schedule is computed
in-trace, so there is no per-block bus key to keep unique across instances. The integrated tables
are identical in every instance, so a lookup served by another instance's table is still sound.
SHA2 has no byte-valued state words (no `blockLen`/`flags`), so there is no equivalent of
`blake3StateInitByteCell`; but the ternary node's free start has no flag left for domain
separation (see the recursion model's notes).
