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
  xdg-open docs/cells-model/recursion-cell-memory-model.html
  # macOS
  open docs/cells-model/recursion-cell-memory-model.html
  ```

- or drag the file into a browser tab / use **File → Open**.

If you are on a remote machine over SSH, copy it down first
(`scp <host>:.../docs/cells-model/recursion-cell-memory-model.html .`) and open
the local copy — everything is embedded, so the copy works offline.

## What it shows

A heatmap over **N (rows = 2^N)** × **blowup (rate = 1/2^b)**, folding factor 3.
Each cell is a recursive-verifier configuration:

- **color** = estimated prover GPU memory (cells over the 32 GB ceiling are
  greyed out),
- **teal ring** = *feasible*: the verifier self-fits, a ZisK Main proof fits the
  recursion cell budget, and prover memory stays under 32 GB.

Live knobs: stage1 / stage2 columns, cells-per-permutation, the linear hash
(**Blake3** rate 4 vs **Blake2** rate 12), and which metric drives the color.
Click any cell for the full breakdown.

- **Protocol** toggle (**FRI** vs **STIR**, ePrint 2024/390). STIR keeps the same
  stage-0 (trace/Q) commitment hashing as FRI, but folds with per-round query
  counts `t_i` that shrink as the rate improves: folding cells are
  `Σ_round t_i · ceil(k·3 / rate)` and stage-0 is opened `t_0` times (round-0
  reps) rather than the flat FRI `n_queries`. STIR folds by the same factor
  `k = 2^fold` down to degree `2^6` as FRI (apples-to-apples). Cells with no
  128-bit STIR schedule are greyed as "no schedule". The **grinding-bits slider**
  is a live knob feeding both protocols (default 20).

## Where the numbers come from

The page reimplements, in JS, the analytical model in
[`setup/pil2-stark/src/types/cells.rs`](../../setup/pil2-stark/src/types/cells.rs):

- linear-hash cells per query (trace/Q stages + FRI folding layers),
- `n_queries` from the FRI security calc (`security.rs`, JBR + 20 grinding bits),
- prover memory ported from `pil::info::get_prover_memory`.

The in-browser `n_queries` reproduces the Rust `security.rs` output exactly across
the swept range. To regenerate the underlying sweep from Rust:

```
cargo test -p pil2-stark-setup --lib types::cells::tests::cells_sweep_csv -- --ignored --nocapture
```

The STIR columns (`stir_t0, stir_total_queries, stir_total_cells,
stir_schedule_found`) in that CSV are the ground truth the in-browser STIR port is
checked against; the JS reproduces them exactly across the swept range (56/56 rows
verified). Grinding is configurable via the slider (default 20 bits for both protocols,
protocol security 108), matching the FRI path.

This is a research/estimation tool — it models linear-hash cells only and
deliberately ignores Merkle-path hashing, custom gates, and transcript costs.
