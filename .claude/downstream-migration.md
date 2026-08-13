# Migrating a downstream project to pil2-proofman 1.1.0-alpha

**Audience:** Claude (or a human) working in a repo that *consumes* pil2-proofman — e.g. zisk — after
pil2-proofman merged the crates.io-preparation branch. This document lives in the pil2-proofman repo;
read it, then apply the changes in the consumer repo.

Every crate is now prefixed `proofman-` or `pil2-` so the names can be claimed on crates.io. Six
crates were renamed, one exported macro hardcodes a renamed path, generated code must be regenerated,
one implicit feature disappeared, and the Node tooling moved. None of it is subtle once you know to
look — but a plain find-and-replace of the crate names will leave you with three of the five breakages
still in place.

Work through the sections in order. Section 3 is the one people miss.

---

## 1. Crate renames

Six renames. Nothing was deleted, and no crate changed its public API as part of the rename — module
paths *inside* each crate are unchanged, so only the leading segment moves.

| Old crate | New crate | Old Rust path | New Rust path |
| --- | --- | --- | --- |
| `curves` | `proofman-curves` | `curves::` | `proofman_curves::` |
| `fields` | `proofman-fields` | `fields::` | `proofman_fields::` |
| `pil-std-lib` | `pil2-std-lib` | `pil_std_lib::` | `pil2_std_lib::` |
| `pilout` | `pil2-pilout` | `pilout::` | `pil2_pilout::` |
| `stark-recurser` | `pil2-stark-recurser` | `stark_recurser::` | `pil2_stark_recurser::` |
| `witness` | `proofman-witness` | `witness::` | `proofman_witness::` |

If you track pil2-proofman's `pre-develop-1.1.0-alpha` rather than the tagged release, one more applies:

| Old crate | New crate | Old Rust path | New Rust path |
| --- | --- | --- | --- |
| `exps-codegen` | `proofman-exps-codegen` | `exps_codegen::` | `proofman_exps_codegen::` |

**Unchanged** — do not touch these: `proofman`, `proofman-common`, `proofman-macros`, `proofman-hints`,
`proofman-util`, `proofman-verifier`, `proofman-soundness`, `proofman-starks-lib-c`, `proofman-cli`.

**New, internal:** `proofman-starks-src` vendors the pil2-stark C/C++/CUDA sources so
`proofman-starks-lib-c` can be published. It is a build-time dependency of `proofman-starks-lib-c` and
you should never depend on it directly.

### Applying it

In `Cargo.toml`, rename the dependency keys. If you use git dependencies, the key *is* the crate name
unless you set `package`, so the key has to change:

```toml
# before
witness      = { git = "https://github.com/0xPolygonHermez/pil2-proofman.git", branch = "..." }
fields       = { git = "https://github.com/0xPolygonHermez/pil2-proofman.git", branch = "..." }
pil-std-lib  = { git = "https://github.com/0xPolygonHermez/pil2-proofman.git", branch = "..." }

# after
proofman-witness = { git = "https://github.com/0xPolygonHermez/pil2-proofman.git", branch = "..." }
proofman-fields  = { git = "https://github.com/0xPolygonHermez/pil2-proofman.git", branch = "..." }
pil2-std-lib     = { git = "https://github.com/0xPolygonHermez/pil2-proofman.git", branch = "..." }
```

Do **not** paper over the rename with `package = "..."` aliases (`fields = { package = "proofman-fields" }`).
It compiles, but it leaves your source using a name that no longer exists upstream, and it will collide
with section 3 below — the `define_wc_with_std!` macro expands to a literal `pil2_std_lib::` path and does
not care what you aliased the crate to.

In Rust sources, rewrite the leading path segment. From the consumer repo root:

```bash
grep -rlE '\b(curves|fields|pilout|witness|pil_std_lib|stark_recurser)::' --include='*.rs' . \
  | xargs sed -i -E \
      -e 's/\bpil_std_lib::/pil2_std_lib::/g' \
      -e 's/\bstark_recurser::/pil2_stark_recurser::/g' \
      -e 's/\bpilout::/pil2_pilout::/g' \
      -e 's/\bwitness::/proofman_witness::/g' \
      -e 's/\bfields::/proofman_fields::/g' \
      -e 's/\bcurves::/proofman_curves::/g'
```

Review the diff before committing. `fields::` and `curves::` are generic enough to appear as your own
module names — check every hunk in files that define a local `mod fields` or `mod curves`.

---

## 2. Version

The workspace went `1.0.0-beta` → `1.1.0-alpha`, and every internal dependency now carries an explicit
`version` alongside its `path` (required for publishing). If you pin pil2-proofman by tag or rev, move
the pin. If you pin by branch, `cargo update` picks it up.

---

## 3. The breakages a find-and-replace will not catch

These are the ones that cost time. Each is independent of the renames.

### 3.1 `define_wc_with_std!` hardcodes `pil2_std_lib`

`witness/src/witness_component.rs` exports `define_wc_with_std!`, whose body expands to a literal
`use pil2_std_lib::Std;`. The macro resolves that path in *your* crate, so `pil2-std-lib` must be a
direct dependency of every crate that invokes the macro, under exactly that name. A `package = "..."`
alias will not satisfy it.

If you get `unresolved import pil2_std_lib` from a line you did not write, this is why.

### 3.2 `witness_library!` no longer needs a crate named `witness`

Same file, opposite direction — a fix, not a break. The macro used to expand to
`Box<dyn witness::WitnessLibrary<...>>`, which forced consumers to keep the dependency key literally
named `witness`. It now expands to `$crate::WitnessLibrary`, so the key is free. If you carried a
`witness = { package = "..." }` alias purely to satisfy this macro, delete it.

### 3.3 Generated `pil_helpers` must be regenerated

`proofman-cli pil-helpers` emits code containing `use proofman_common as common;` and
`use proofman_fields::PrimeField64;`. Files generated before the rename import `fields::PrimeField64`
and will not compile.

Regenerate rather than hand-editing — the templates changed, not just the crate names:

```bash
cargo run --bin proofman-cli -- pil-helpers --pilout <your.pilout> --path <crate>/src -o
```

(`-o` overwrites in place; it has no long form.)

Do this for every crate in your workspace that has a generated `pil_helpers` module. Missing one
produces a single confusing `unresolved import fields` in an autogenerated file whose header tells you
not to edit it.

### 3.4 The implicit `proofman-starks-lib-c` feature on `fields` is gone

`fields` used to declare `proofman-starks-lib-c = { ..., optional = true }`, which gave you an implicit
feature of the same name. `proofman-fields` now declares it non-optional under
`[target.'cfg(target_arch = "x86_64")'.dependencies]` — linked automatically on x86_64, with the
pure-Rust fallback elsewhere (Apple Silicon).

Any `features = ["proofman-starks-lib-c"]` on the old `fields` dependency is now a hard error
(`package does not have feature`). Delete it; you get the dependency automatically on x86_64.

### 3.5 Node tooling moved to `setup/pil2-stark/`

`package.json` (pil2-compiler, snarkjs, circomlib) moved from the pil2-proofman repo root to
`setup/pil2-stark/`. If your CI or scripts run `npm install` against a pil2-proofman checkout, add the
working directory:

```yaml
- name: Install Node dependencies
  run: npm install
  working-directory: <path-to-pil2-proofman>/setup/pil2-stark
```

Strictly this is belt-and-braces — `pil2-stark-setup` bootstraps its own Node deps when they are
missing, falling back to `~/.cache/pil2-proofman/node-deps`. But an explicit install keeps CI
deterministic and avoids paying for it inside a build script.

---

## 4. Things that did *not* change

Ruling these out saves you from chasing them:

- **Binary names.** `proofman-cli` and `proofman-setup` are unchanged.
- **Feature names.** `mpi` (default-on for `proofman` and `proofman-common`), `cpu-only` on
  `proofman-starks-lib-c`, `verify` on `proofman-fields` — all unchanged, same semantics.
- **MPI is still a default feature**, so a system MPI install (`libopenmpi-dev`) is still required.
- **Public APIs.** The rename moved crate names only. No types, traits, or functions were renamed,
  moved between crates, or changed signature as part of this work.

---

## 5. Verifying

In the consumer repo, in order — each step catches a different class of failure:

```bash
cargo metadata --no-deps >/dev/null   # manifests resolve; catches stale dep keys (§1) and features (§3.4)
cargo check --workspace --all-targets # catches missed `use` paths (§1) and macro expansion (§3.1)
cargo clippy --workspace -- -D warnings
cargo test --workspace                # build scripts run compile-pil; catches stale pil_helpers (§3.3)
```

If `cargo check` passes but `cargo test` fails inside a `build.rs` with a pil2com error, that is §3.5
(Node tooling), not a Rust problem.

---

## 6. Not yet on crates.io

As of this document, nothing is published — pil2-proofman is still consumed by git dependency. The
renames exist so the names *can* be claimed. Once published, `proofman-cli` and `proofman-soundness`
will remain unpublished (`publish = false`); everything else in the list above becomes available as a
normal registry dependency. Nothing in this migration changes when that happens.
