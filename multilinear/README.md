# Multilinear

A multilinear STARK: constraints expressed in PIL are proven with a
**sumcheck-based PIOP** (zerocheck) whose evaluation claims are discharged by
**Basefold** — a multilinear polynomial commitment scheme built from a sumcheck
run in lockstep with a FRI folding cascade.

The protocol background is in [`docs/multilinear-pcs.md`](../docs/multilinear-pcs.md)
(Basefold/WHIR) and [`docs/multilinear-STARK.md`](../docs/multilinear-STARK.md)
(the PIOP); the integration map is in
[`docs/multilinear-implementation.md`](../docs/multilinear-implementation.md).
This README is a guided tour of the code, bottom-up: read it next to the
sources and you should be able to follow everything from a field element to a
verified proof.

## 0. The three conventions everything relies on

Fix these in your head first; every module assumes them.

**Variables are LSB-first.** A trace column with $2^n$ rows is a function
$w : \{0,1\}^n \to \mathbb{F}$; row index $i$ has bits $(i_0, i_1, \dots, i_{n-1})$
and **variable $X_1$ is the least-significant bit** $i_0$. Consequence: binding
$X_1$ pairs adjacent entries $(t[2i],\, t[2i+1])$ — cache-friendly, and it is
the *same* pairing the FRI fold uses, so sumcheck round $t$ and codeword fold
$t$ bind the same variable with the same challenge.

**Commitments live in the monomial basis (Möbius transform).** The univariate
attached to a multilinear $\tilde{w}$ is

$$\hat{g}(x) = \tilde{w}\big(x, x^2, x^4, \dots, x^{2^{n-1}}\big).$$

Expanding it, the coefficients of $\hat{g}$ are the *monomial coefficients* of
$\tilde{w}$. In this basis the plain FRI fold $\hat{g}_e + \lambda\cdot\hat{g}_o$
is exactly "bind $X_1 = \lambda$":

| representation | bind $X_1 = \lambda$ |
|---|---|
| values (hypercube table) | $v'[t] = v[2t] + \lambda\cdot(v[2t+1] - v[2t])$ — `fold_mle` |
| monomial coefficients | $c'[t] = c[2t] + \lambda\cdot c[2t+1]$ — `fold_coeffs` = what one FRI fold does |

The test `codeword_fold_is_coefficient_fold` (in `encoding.rs`) pins this
identity — if you change any ordering convention and that test still passes,
the protocol is still sound.

**Merkle leaves are pair-packed.** A fold-consistency check at position $j$ of
a domain of size $N$ needs the values at $j$ **and** $j + N/2$ (they map to $x$
and $-x$). So every committed vector stores both halves in one leaf: leaf $j$ =
values at $j$ and $j + N/2$. One query ⇒ one Merkle opening per tree.

## 1. Building blocks

### `hypercube.rs` — multilinears as tables
`fold_mle` / `mle_eval` (value form), `values_to_coeffs` / `coeffs_to_values`
(Möbius and inverse), `fold_coeffs` / `monomial_eval` (coefficient form), and
`dot_base_ext` — the weighted sum $\sum_{b} w(b)\cdot K(b)$, which is how every
claim in the protocol is computed.

### `eq.rs` — the kernels
- `eq_evals(r)`: the tensor-product table $t[i] = \mathrm{eq}(\mathrm{bits}(i), \vec{r})$
  in $O(2^n)$, where
  $\mathrm{eq}(\vec{x},\vec{y}) = \prod_j \big(x_j y_j + (1-x_j)(1-y_j)\big)$.
- `rotate_table`: the prover's *rotation kernel* — the shifted column
  $w^{\to s}(b) = w(b+s \bmod 2^n)$ satisfies

$$\tilde{w}^{\to s}(\vec{\lambda}) = \sum_{y\in\{0,1\}^n} w(y)\cdot \mathrm{eq}(y-s,\ \vec{\lambda}),$$

  and the kernel $\mathrm{eq}(\cdot - s, \vec{\lambda})$ is just the
  $\mathrm{eq}(\cdot, \vec{\lambda})$ table cyclically rotated.
- `rot_kernel_eval`: the verifier's side of the same kernel — the MLE of the
  rotated-eq table evaluated at an arbitrary point $\vec{z}$,

$$\mathrm{rot}_s(\vec{z}, \vec{\lambda}) = \sum_{x\in\{0,1\}^n} \mathrm{eq}(x, \vec{\lambda})\cdot \mathrm{eq}(x+s \bmod 2^n,\ \vec{z}),$$

  computed in $O(n)$ by a 2-state dynamic program over the carry chain of
  $x + s$ (works for any offset, positive or negative; $s = 0$ degenerates to
  $\mathrm{eq}(\vec{\lambda}, \vec{z})$).
- `boolean_point`: hypercube corners, used by boundary constraints.

### `sumcheck.rs` — the reduction engine
The `SumcheckOracle` trait (`round_evals` → evaluations of the round polynomial
at $0, 1, \dots, d$; `bind` → fix the current variable). `ProductOracle` is the
degree-2 instance used by the Basefold opening ($\Phi\cdot W$). Verifier side:
`verify_sumcheck_round` checks $g_t(0) + g_t(1) = \text{claim}$ and returns
$g_t(r)$ via Lagrange interpolation on the integer nodes (`interpolate_at`).

### `encoding.rs` — Reed–Solomon encoding
`encode_column` = Möbius transform + coset LDE (`fields::coset_lde`, the
forward NTT added in `fields/src/ntt.rs`) onto the coset $g\cdot H$ with
$|H| = 2^{n + \log(\text{blowup})}$. `domain_point(n0_bits, level, j)` names
every point of every folding level:

$$x_{\ell,j} = g^{2^{\ell}} \cdot \omega_{\ell}^{\,j}, \qquad \omega_{\ell} = \text{generator of the order-}2^{n_0-\ell}\text{ subgroup}.$$

`eval_ext_poly_at_base` is the Horner evaluation used against the in-clear
final polynomial.

### `merkle.rs` — prover-side Merkle tree
Same layout as `fields::merkle` (Poseidon2, arity 4, 4-cell digests, zero
padding), but keeps all levels so it can produce sibling paths. Paths verify
with the *existing* `fields::verify_mt` — the test
`root_matches_partial_merkle_tree` pins compatibility with the rest of the
repo's Merkle stack.

### `transcript.rs` — Fiat–Shamir
Thin wrapper over `fields::Transcript` (Poseidon2 sponge): `absorb*`,
`challenge()` $\to \mathbb{E}$, `query_indices` (via `get_permutations`).

## 2. Basefold (`basefold.rs`) — commit and open

**Commit** (`commit_matrix`): RS-encode each column, pair-pack all columns of a
stage into one tree (leaf $j$ = all columns at $j$, then all at $j + N/2$).
One commitment for the fixed columns, one per witness stage.

**The opening statement.** Everything the STARK needs to open is a weighted
sum $v_{j,i} = \sum_b w_j(b)\cdot K_i(b)$ with a verifier-evaluable kernel
$K_i$. All claims are batched into **one** statement with two challenges
$\delta, \gamma \in \mathbb{E}$:

$$\Phi = \sum_j \delta^j\, w_j \quad (\text{batched column — this is what gets folded}),$$

$$W = \sum_i \gamma^i\, K_i \quad (\text{batched kernel}), \qquad \sigma = \sum_{j,i} \delta^j \gamma^i\, v_{j,i},$$

$$\text{prove:}\quad \sum_{b\in\{0,1\}^n} \Phi(b)\cdot W(b) = \sigma.$$

The cross terms $\Phi\cdot W = \sum_{j,i}\delta^j\gamma^i\, w_j K_i$ are why
the prover sends the **full claims matrix** $v_{j,i}$ for every
(column, kernel) pair, not only the entries the constraint check uses.

**Prove** (`prove_opening`): $n$ rounds of the degree-2 product sumcheck; the
round-$t$ challenge $\lambda'_t$ *also* folds the codeword one step
(`fold_codeword` — the Basefold identity: fold = bind $X_{t+1}$):

$$\Phi_{t+1}(x^2) = \frac{\Phi_t(x) + \Phi_t(-x)}{2} + \lambda'_t\cdot\frac{\Phi_t(x) - \Phi_t(-x)}{2x}.$$

Folded oracles $\Phi_1, \dots, \Phi_{L-1}$ are committed pair-packed; after
$L = n - \log|\text{final poly}|$ folds the remaining polynomial is sent
**in clear** (`final_poly`). Because of the Möbius convention, `final_poly` is
simultaneously
- the monomial-coefficient table of the partially-bound multilinear — used in
  the final check $s_n = \tilde{\Phi}(\vec{\lambda}')\cdot \tilde{W}(\vec{\lambda}')$
  via `monomial_eval`, and
- the coefficient vector of the remaining univariate — evaluated at domain
  points during queries.

**Query phase**: indices are pair positions in $[0, N_0/2)$. The verifier
recomputes $\Phi_0$ at the queried pair from the stage-tree leaves and the
$\delta$-powers, then walks the cascade: check the current value against the
matching half of the next oracle's opened pair, fold (`fold_pair`), repeat,
and finally compare against `final_poly` evaluated at the landing point.

`verify_opening` replays the transcript, runs the round checks, the final
algebraic check (the caller supplies $\tilde{W}(\vec{\lambda}')$ as a closure —
kernels are always verifier-evaluable), and all query checks.

## 3. The constraint layer (`ir.rs`, `evaluator.rs`)

`AirIr` is the compiled form of one AIR's PIL constraints, produced at setup
time from the pilout expression DAG
(`setup/pil2-stark/src/output/mlinfo.rs` → `<AIR>.mlinfo.bin`): a flat
instruction list (`add/sub/mul/neg` over witness/const/public/challenge/number
operands, each with a `row_offset`), constraint roots tagged
`EveryRow`/`FirstRow`/`LastRow`, per-constraint degrees, the set of row
offsets, column/publics layout, and `MlParams`. Shared subexpressions are
shared temps (instruction $i$ writes temp $i$).

`evaluator.rs` interprets that list generically over a `LeafSource` — the same
IR is executed by three different consumers:

| consumer | leaf source |
|---|---|
| prover, zerocheck round | folded column tables at the current point (`TablePoint`) |
| verifier, final check | the claimed openings matrix (`ClaimsAtPoint` / `ClaimsAtCorner`) |
| debug pre-check | raw trace rows (`check_constraints_on_trace`, mirrors the C++ verify-constraints) |

`eval_constraint_cone` evaluates only one constraint's dependency cone — used
for boundary constraints, whose leaf source is only defined at offset 0.

## 4. The PIOP (`zerocheck.rs`)

All `EveryRow` constraints $C_1, \dots, C_m$ are proven by **one** sumcheck of

$$G(\vec{X}) = \mathrm{eq}(\vec{r}, \vec{X}) \cdot \sum_{t=1}^{m} \alpha^{t-1}\, C_t(\vec{X}),$$

with $\vec{r} \in \mathbb{E}^n$, $\alpha \in \mathbb{E}$ sampled after the
commitments; $\sum_{\vec{x}\in\{0,1\}^n} G(\vec{x}) = 0$ iff every constraint
vanishes on every row (w.h.p. over $\vec{r}, \alpha$). Round-polynomial degree
= $\max_t \deg C_t + 1$ (columns are multilinear, $\mathrm{eq}$ adds one).
`ZerocheckOracle` keeps one extension table per (column, offset) leaf —
a shifted column starts as a rotated copy and all tables fold together each
round; per point it interpolates each leaf at $X = 0, \dots, d$ and runs the IR.

The sumcheck ends at a random point $\vec{\lambda}$ with the claim

$$s_n = \mathrm{eq}(\vec{r}, \vec{\lambda}) \cdot \sum_t \alpha^{t-1}\, C_t\big(\dots\big),$$

which the verifier can recompute itself *given the column openings at
$\vec{\lambda}$* — including shifted ones. That defines the claim set,
canonically ordered by `build_kernels`:

- one **rotation kernel** $\mathrm{rot}_s(\cdot, \vec{\lambda})$ per opening
  offset $s$ (with $\mathrm{rot}_0 = \mathrm{eq}(\cdot, \vec{\lambda})$),
- one **point kernel** $\mathrm{eq}(\cdot, \text{corner})$ per boundary corner
  referenced.

`FirstRow`/`LastRow` constraints never enter the sumcheck: their corner claims
are plain trace reads ($\tilde{w}$ at a Boolean point *is* the trace value),
and the verifier evaluates the constraint on them directly — zero extra rounds.

## 5. The whole STARK (`prover.rs`, `verifier.rs`)

`prove_air(ir, witness, consts, publics) → MlProof` — the transcript schedule,
which `verify_air` replays step by step (unicode in place of LaTeX since this
is a wire-format diagram):

```
absorb (airgroup_id, air_id, n_bits), publics
absorb const root, stage-1 root
absorb stage challenges (derived globally, see below)
absorb stage-2.. roots, air values, airgroup values
sample r ∈ E^n, α                          ┐
n zerocheck rounds: absorb g_t, sample λ_t │ zerocheck
absorb claims matrix v_{j,i}               ┘
sample δ, γ                                ┐
n opening rounds: absorb round poly,       │
  sample λ'_t, absorb fold root            │ Basefold opening
  (t < L−1) / final_poly (t = L−1)         │
absorb-derive query indices, check folds   ┘
```

**Global challenges (multi-stage / std).** With a shared bus (std
lookups/permutations across instances) every instance must use the *same*
stage challenges, so they are derived from **all** instances' stage-1 roots
(`derive_global_challenges[_for]`), ordered by global instance id. A single
proof carries its challenges; the **proof-set verifier** re-derives them from
the set's roots and enforces equality — a missing or tampered proof changes
the derivation and the whole set is rejected. Extension-valued stage-2
columns are committed as 3 base columns and reassembled by the IR evaluator;
air/airgroup values are prover messages bound before the zerocheck randomness,
and the airgroup values additionally enter the cross-instance *global
constraints* (bus balance), checked by `verify-multilinear` over the
aggregated set.

`MlProof` contains exactly the prover messages: stage/const roots, zerocheck
round polynomials, the claims matrix, and the `OpeningProof` (round polys,
fold roots, `final_poly`, per-query openings). Bincode-serialized;
`save`/`load`.

The verifier's checks, in order: proof shape ↔ IR; transcript replay;
$n$ zerocheck round checks; **zerocheck final check** (claim ↔ claims matrix);
**boundary checks** (corner claims); then the batched opening: $n$ round
checks, **final algebraic check**
$s_n = \tilde{\Phi}(\vec{\lambda}')\cdot \tilde{W}(\vec{\lambda}')$ (with
$\tilde{W}$ assembled from `rot_kernel_eval` / `eq_eval` — this is what makes
the claims matrix *binding*), and the query phase anchoring everything in the
Merkle roots.

Soundness intuition in one line: the zerocheck forces the claims to be
consistent with "all constraints vanish", and the opening forces the claims to
be consistent with the committed columns; the two share nothing but the
claims matrix, so a cheating prover must break one of them.

## 6. Tests as a map

Each protocol fact is pinned by a test — they double as usage examples:

| fact | test |
|---|---|
| Möbius/monomial ↔ value duality | `hypercube::mobius_roundtrip_and_monomial_eval` |
| fold = partial evaluation on codewords | `encoding::codeword_fold_is_coefficient_fold` |
| rotation kernel = shifted column (prover) | `eq::rotated_eq_table_evaluates_shifted_column` |
| carry-DP = rotated-eq MLE (verifier) | `eq::rot_kernel_eval_matches_table_mle` |
| Merkle compatibility with `fields` | `merkle::root_matches_partial_merkle_tree` |
| sumcheck completeness/soundness | `sumcheck::product_sumcheck_roundtrip`, `tampered_round_poly_rejected` |
| Basefold open/verify + negatives | `basefold::opening_roundtrip`, `wrong_claim_rejected`, `corrupted_codeword_rejected` |
| zerocheck ↔ true MLEs | `zerocheck::zerocheck_roundtrip_on_fib` |
| full STARK + negatives | `verifier::prove_verify_roundtrip`, `corrupted_trace_rejected`, `wrong_publics_rejected`, `tampered_proof_rejected` |
| against a real proving key | `tests/setup_artifact.rs` (skips if not generated) |

Run with `cargo test -p proofman-multilinear`. For the end-to-end CLI flow see
[`examples/fibonacci-multilinear`](../examples/fibonacci-multilinear/README.md).

## 7. Parameters and current limitations

`MlParams` (stored per-AIR in the mlinfo): `log_blowup` (default 2 → rate
$\rho = 1/4$), `n_queries` (default 50, **conjecture-level** ≈100 bits — no
formal analysis yet), `log_final_poly_len` (default 4), `grinding_bits` (must
be 0, not implemented).

Current scope (milestone 2): multi-stage AIRs with std arguments (lookups,
permutations, range checks), challenges, air/airgroup values — validated on
`pil2-components/test/simple`. Still out of scope: proof values, custom
commits (stage-0 `rom`), `everyFrame` boundaries, grinding; boundary
constraints must not reference shifted columns; CPU-only; no aggregation.
Im-pols are committed as-is (they are pure prover economics here — no rate
constraint forces them, see the discussion in
`docs/multilinear-implementation.md`); inlining them is a milestone-3
measurement. The prover is deliberately naive (scalar, unparallelized,
all-extension tables) — see the performance notes in
`docs/multilinear-implementation.md` for the optimization roadmap.
