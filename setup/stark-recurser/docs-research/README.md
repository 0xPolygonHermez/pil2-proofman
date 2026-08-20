# Recursion cell & GPU-memory decision map

[`recursion-cell-memory-model.html`](./recursion-cell-memory-model.html) is a
self-contained, interactive decision map for the recursive verifier. It needs
no server, no build, and no network — all CSS/JS is inlined in the one file.

## How to open it

It is a plain HTML file; just open it in any browser:

- **Double-click** `recursion-cell-memory-model.html` in your file manager, or
- from a terminal at the repo root:

  ```sh
  # Linux
  xdg-open setup/stark-recurser/docs-research/recursion-cell-memory-model.html
  # macOS
  open setup/stark-recurser/docs-research/recursion-cell-memory-model.html
  ```

- or drag the file into a browser tab / use **File → Open**.

If you are on a remote machine over SSH, copy it down first
(`scp <host>:.../setup/stark-recurser/docs-research/recursion-cell-memory-model.html .`) and open
the local copy — everything is embedded, so the copy works offline.

## What it shows

A heatmap with **N (rows = 2^N, 17–23) across the columns** and **blowup
(rate = 1/2^b, 2^1–2^3) down the rows**, at folding factor 3. The blowup axis
stops at 2^3 because the max constraint degree saturates at 8 there — a wider
blowup is the same arithmetization on a bigger domain. Each row header carries
the degree that blowup unlocks and the stage2/stageQ widths that come with it.
Each cell is a recursive-verifier configuration:

- **color** = the selected metric (prover memory, total cells, total hashes,
  minimum blake instances, or `n_queries`); cells over the 32 GB memory ceiling
  are greyed,
- **teal ring** = *feasible*: the verifier self-fits, a ZisK Main proof fits the
  recursion cell budget, and prover memory stays under 32 GB.

Live knobs: **blake instances packed**, hash family, grinding bits, and the
color metric. Click a cell for the breakdown. Cells-per-permutation is *derived*
from the lane width at the selected blowup rather than dialled in, so it is a
read-out in the Configuration panel, not a control.

The Configuration panel reports, for the selected cell:

- **geometry** — `fixed / stage1 / stage2 / stageQ` columns, the lane width, and
  the cells one permutation costs,
- **throughput** — how many hash permutations an air of 2^N rows with *k* lanes
  can hold (`(floor(2^N / clocks) - 1) * k`, mirroring `NUM_OPS` in `blake3.pil`),
  how many of those self-verification spends, and what is **left for payload**.
  At N=21, k=4, blowup 2^3 that reads 149,792 held, 38,106 spent, 111,686 spare.

The low-degree test is **FRI**.

## Self-fit is a fixed point, not a division

Stage widths are not free parameters: they come from packing `k` hash instances
side by side. So adding an instance widens the trace that the verifier must
itself hash, and the requirement grows with `k` too.

The panel therefore reports three states: **fits** (with instances to spare),
**needs k** (short by a finite amount), or **never fits** — the case where each
added instance costs more permutations than the `2^N` rows it contributes, so no
`k` ever catches up. The "cells and marginal cost" section shows that ratio
directly.

## Hash accounting

Every permutation the verifier performs is counted, mirroring
[`setup/pil2-stark/src/verifier_hashes.rs`](../../pil2-stark/src/verifier_hashes.rs)
rule for rule:

- **leaf hashes** — one *block compression* per opened oracle payload. Blake3 and
  blake2b are real hash functions, not rate-limited sponges: they absorb a whole
  64 / 128-byte block per call and carry the chaining value in the state, so a
  leaf of `w` Goldilocks elements costs `ceil(w / 8)` (blake3, plus one parent per
  extra 1024-byte chunk) or `ceil(w / 16)` (blake2b) — **not** `ceil(w / rate)`.
- **Merkle authentication paths** — one compression per level, stopping at
  `lastLevelVerification = 2` levels above the root,
- **per-tree root reductions** — folding that kept bottom level to the root, once
  per tree rather than once per query.
- **FRI** — the real `generate_stark_struct` schedule: fold the *extended domain*
  by 3 until it reaches `final_degree + 1 = 6`, snapping to 5 on the last step.
  Every step past the first commits a tree the queries open.

Paths dominate, typically 70–75% of the total.

The page reproduces `verifier_hashes.rs` **exactly** on all three airs of the
hashes example (21,576 / 27,156 / 29,208 leaf+merkle+fri), and that crate was in
turn validated against the native verifier temporarily instrumented to count its
own hashes. Only the Fiat-Shamir transcript and the grinding check are left out,
together about 1.5% of the total.

A leaf and a tree node use the same primitive but different shapes: the tree uses the full
compression block. The arity follows from that block — as many 4-element digests
as fit:

| | sponge rate | block | arity | clocks | fixed | stage1 | marginal cols/lane @ blowup 2^1 / 2^2 / 2^3 |
|---|---|---|---|---|---|---|---|
| Blake3  | 4  | 8 elems  | 2 (binary)     | 56 | 8 | 53k + 2 | 95 / 74 / 65 |
| Blake2b | 12 | 16 elems | 4 (quaternary) | 96 | 6 | 98k + 2 | 176 / 137 / 120 |

Those are *marginal* widths — what one more lane costs. The page's Configuration
panel instead reports the *amortized* width `(stage1 + stage2) / k`, which also
carries the shared columns, so it reads a little higher: 97 / 76 / 67 at k=4 and
103 / 79 / 70 at k=1. The cell budget uses the amortized figure on both sides of
the comparison, so the self-fit verdict is unaffected by the choice.

## How lanes and the blowup impact the widths

Two knobs move the committed width, and they act on different stages.

### Lanes

`Blake3(N, LANES)` puts `k = LANES` independent G evaluations on every row, so a
56-row cycle completes `k` permutations instead of one. Both committed stages are
**affine** in `k`, not proportional — each has a shared part paid once:

- **stage1 = 53k + 2.** Every witness column in the G function carries a lane
  index (`va[l][2]`, `vb[l][4]`, … `vb_pp_t[l]`) and sums to 53 per lane. The two
  that do not are `mul_table` and `mul_range`, the multiplicities of the XOR
  table and the range checker — one table serves all lanes. **stage1 does not
  depend on the degree at all.**
- **stage2** is the bus argument, and it is where the degree lands (below).
  Each lane issues **28 lookups**: 8 range checks on `va`/`vc`/`x`/`y`, 16
  XOR-table checks, 4 message-permutation proves/assumes. Two more are shared
  (the table and range-checker `lookup_proves`), so an air with `k` lanes runs
  `28k + 2` bus terms.
- **fixed = 8, always.** `CLK_0`, `BLOCK_ID`, `A`, `B`, `ROTATION`, `C_ROT[2]`,
  `__L1__` — the round schedule and the XOR table, all lane-independent.

### Blowup, through the max constraint degree

A blowup of `2^b` lets the quotient be split into `2^b` chunks, so the PIL can be
compiled at `set_max_constraint_degree(2^b + 1)`. The three cases that matter:

| blowup | 2^1 | 2^2 | 2^3 |
|---|---|---|---|
| max constraint degree | 3 | 5 | 8 |

(8 rather than 9 at `2^3` because that is what `Compressor.pil` pins.) Compile
below the blowup and the headroom is wasted; compile above it and the setup
inserts im pols to pull the degree back down — a blowup-1 setup fed a maxDeg-5
pilout degrades to 3 and pays 11 extra im pols, +33 columns.

The degree feeds two stages:

- **stage2.** `std_sum` folds as many bus terms as fit into the direct fraction
  (`maxDeg − 2` of them), then clusters the rest `maxDeg − 1` at a time behind
  one `im` — the im constraint is `im * prod(f_i + gamma)`, degree `1 + cluster`.
  Each `im`, and `gsum` itself, is one extension element = 3 Goldilocks columns:

  ```
  im     = ceil((28k + 2 − (maxDeg − 2)) / (maxDeg − 1))
  stage2 = 3 * (im + 1)
  ```

  So the per-lane stage2 cost is `3 * 28 / (maxDeg − 1)`: **42** at degree 3,
  **21** at degree 5, **12** at degree 8.
- **stageQ = 3 * (maxDeg − 1)** — the quotient chunks: 6, 12, 21.

### Measured grid

`proofman-setup stats` on `blake3.pil`, LANES 1–4 × blowup 1–3 (all twelve):

| | LANES 1 | 2 | 3 | 4 | | |
|---|---|---|---|---|---|---|
| **blowup 2^1, maxDeg 3** | 55 / 48 | 108 / 90 | 161 / 132 | 214 / 174 | stageQ 6 | fixed 8 |
| **blowup 2^2, maxDeg 5** | 55 / 24 | 108 / 45 | 161 / 66 | 214 / 87 | stageQ 12 | fixed 8 |
| **blowup 2^3, maxDeg 8** | 55 / 15 | 108 / 27 | 161 / 39 | 214 / 51 | stageQ 21 | fixed 8 |

(cells are `stage1 / stage2`.) Total committed width at LANES 4 goes
**394 → 313 → 286** columns as the blowup rises — the degree headroom is paid
back almost entirely out of stage2, and stageQ takes a small cut of it back.
The model in the page reproduces all twelve exactly, plus maxDeg 4, 9 and 16 and
LANES 8 as out-of-grid checks.

Packing lanes is mildly *sub*-linear: at blowup 2^1 the marginal lane costs 95
committed columns while the first costs 109. `blake2b.pil` has no `LANES`
parameter, so its row above is the same decomposition applied to its measured
LANES-1 geometry (cm1 100, cm2 84, const 6, 52 lane-local lookups + 2 shared) —
an extrapolation, not a measurement.

To reproduce one grid point:

```sh
cat > /tmp/l.pil <<'PIL'
require "blake3.pil"
set_max_constraint_degree(5);
airgroup Hashes { Blake3(2**20, 3); }
PIL
echo '{ "Blake3": { "blowupFactor": 2 } }' > /tmp/ss.json
cargo run --release --bin proofman-setup -- compile-pil --pil /tmp/l.pil \
    -I ./pil2-components/lib/std/pil -o /tmp/l.pilout
cargo run --release --bin proofman-setup -- stats -a /tmp/l.pilout -s /tmp/ss.json -o /tmp/l.txt
```

## Where the numbers come from

The page carries the model itself, in JS:

- leaf hashes per query — trace/Q stages plus FRI folding layers,
- Merkle-path hashes per query — one path per committed tree,
- `n_queries` from [`security::pcs::Fri`](../../setup/pil2-stark/src/types/security/pcs/fri.rs),
  the same calculator that sizes real proving keys,
- prover memory following `pil::info::get_prover_memory`.

The hash rules it applies are the ones
[`verifier_hashes.rs`](../../setup/pil2-stark/src/verifier_hashes.rs) measured against an
instrumented native verifier — leaf compressions, path levels stopping at
`lastLevelVerification`, and one root reduction per tree. That module is the reference to check
this page against; `proofman-setup stats` prints its per-air totals.

The in-browser query count is the closed form `pcs::Fri` itself uses:
`pp = 1 - sqrt(rate) - 1/300` (JBR at alpha = 0) and
`t = ceil((128 - grinding) / -log2(1 - pp))`. It has no N dependence — `pcs`
pays the dimension-dependent batching error out of its own grinding budget
rather than with extra queries.

## Modelling assumptions and limits

This is a research/estimation tool. It assumes the PIL is compiled at the degree
its blowup unlocks — `maxDeg = min(2^b + 1, 8)` — so every column of the map is a
*differently arithmetized* air, not just a differently sized domain. The
verifier's own `stageQ` is derived the same way (`3 * (maxDeg - 1)`) rather than
pinned. What it deliberately does not model:

- **custom gates**, and the **Fiat-Shamir transcript / grinding** permutations —
  `verifier_hashes.rs` counts both; here they are ~1.5% and would need the full
  challenge and air-value maps to reproduce (and its transcript term is the standalone
  sequence, so an aggregated verifier's is ~1-3% lower),
- **asymmetric recursion** — the two aggregated proofs are assumed to share the
  verifier's own N, blowup and stage widths; `n_proofs` is a plain multiplier.

Two hard bounds:

- **`n + blowup <= 31`** — `pcs::Fri` indexes the evaluation domain with a `u32`.
- The FRI schedule is `generate_stark_struct`'s own: the **extended domain** folds
  by 3 down to `final_degree = 5`. (An earlier version of this page folded the
  polynomial dimension to `2^6` instead, which counted 4 FRI trees where the real
  proof commits 6. With the blowup axis capped at 2^3 the domain rule has no
  degenerate cases, so there is no reason to approximate it.)
