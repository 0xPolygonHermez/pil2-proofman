# Debug Configuration JSON Format

This document describes the structure and usage of the `debug.json` configuration file used
by the PIL2 Proofman debugger.

## Overview

This document describes the *input* schema. For the *output* schema — the structured
`DebugReport` produced by a run — see [DEBUG_REPORT.md](DEBUG_REPORT.md).

`debug.json` controls several independent debug subsystems plus their output destination:

| Section              | Purpose                                                        | Activation    |
|----------------------|----------------------------------------------------------------|---------------|
| `instances`          | Which instances to process / per-instance debug config         | Section present (defaults: `mode = "all"`, empty list) |
| `constraints`        | Per-air algebraic constraint verification                      | `enabled` (default `true`) |
| `global_constraints` | Global constraint verification                                 | `enabled` (default `true`) |
| `bus`                | Bus (std-lookup) verification — "do assumes match proves?"     | Section present |
| `output`             | Debug-output destination (stdout or file)                      | Section present |

**Activation is explicit.** Omit a section to disable its subsystem; the `constraints`
and `global_constraints` passes are additionally gated by their `enabled` field. There
are no hidden side effects: presence of one section never silently changes the behavior
of another.

## Root structure

```json
{
  "instances": { ... },
  "constraints": { ... },
  "global_constraints": { ... },
  "bus": { ... },
  "output": { ... }
}
```

All top-level fields are optional. An empty `debug.json` (`{}`) does nothing.

---

## `instances` — instance filtering and per-instance debug config

```json
"instances": {
  "mode": "all",
  "skip_tables": false,
  "list": [
    {
      "airgroup_id": 0,
      "airs": [
        {
          "air_id": 1,
          "store_row_info": false,
          "instances": [
            {
              "instance_id": 0,
              "constraints": [5, 10, 15],
              "store_row_info": true
            }
          ]
        }
      ]
    }
  ]
}
```

### Fields

- **`mode`** _(optional, default `"all"`)_ — either:
  - `"all"`: process every instance; `list` is used only to attach per-instance debug config.
  - `"only_listed"`: process *only* instances enumerated in `list`; skip everything else.
- **`skip_tables`** _(optional, default `false`)_ — only consulted in `"only_listed"` mode.
  By default, table (lookup) instances are **always** processed regardless of `list`, because
  skipping a lookup table would cause spurious bus mismatches across every air that consumes
  it. Set to `true` to subject tables to the same filter as everything else (e.g. to skip
  large tables you don't care about, accepting that bus debug for lookup-heavy opids will be
  meaningless without their tables). No effect in `"all"` mode.
- **`list`** _(optional)_ — hierarchical config keyed by airgroup → air → instance.

### Airgroup entry

```json
{
  "airgroup_id": <number>,    // OR
  "airgroup":    "<name>",    // exactly one of these two
  "airs":        [...]
}
```

- Exactly one of `airgroup_id` or `airgroup` must be specified.

### Air entry

```json
{
  "air_id":         <number>,     // OR
  "air":            "<name>",     // exactly one of these two
  "store_row_info": <boolean>,    // optional, default false
  "instances":      [...]
}
```

- Exactly one of `air_id` or `air` must be specified.
- `store_row_info` at the air level applies to every instance of this air. Bus-debug only.

### Instance entry

```json
{
  "instance_id":    <number>,         // optional, default 0
  "constraints":    [<indices>],      // constraint IDs to verify on this instance
  "store_row_info": <boolean>         // bus-debug detail flag
}
```

- `constraints` is consumed by the constraint-verification pass when the `constraints`
  section is also enabled.
- `store_row_info` at instance level overrides the air-level value if both are set.

### Row-info precedence

`store_row_info` can be set at three levels with OR-merge semantics:
- Root default: `bus.store_row_info`
- Air level: `instances.list[].airs[].store_row_info`
- Instance level: `instances.list[].airs[].instances[].store_row_info`

If *any* applicable level is `true`, row info is stored.

---

## `constraints` — per-air algebraic constraint verification

```json
"constraints": {
  "enabled":   true,
  "max_print": 10
}
```

### Fields

- **`enabled`** _(default `true`)_ — gates the per-air constraint verification pass.
- **`max_print`** _(optional, default 10)_ — maximum number of mismatched constraints
  to print per failing instance. Also inherited by the global pass (see below).

---

## `global_constraints` — global constraint verification

```json
"global_constraints": {
  "enabled":               true,
  "global_constraint_ids": [0, 1, 5]
}
```

### Fields

- **`enabled`** _(default `true`)_ — gates the global constraint verification pass.
  Omit the section to skip global verification entirely.
- **`global_constraint_ids`** _(optional)_ — array of global-constraint indices to
  verify. Empty/omitted means **all** global constraints.

`max_print` for the global pass is inherited from the `constraints` section,
falling back to the default (10) when `constraints` is absent.

---

## `bus` — bus / std-lookup verification

```json
"bus": {
  "opids":          [1, 2, 3, 4],
  "fast_mode":      true,
  "values_filter":  [["1302180"], ["0", "1", "0"]],
  "store_row_info": false,
  "max_print":      10,
  "group_by":       [ ... ]
}
```

### Fields

- **`opids`** _(optional)_ — bus IDs to debug. Empty/omitted means all opids.
- **`fast_mode`** _(optional, default `true`)_ — storage strategy:
  - `true`: per-opid balance counts only. Minimal overhead. Independent of `opids`.
  - `false`: full per-value detail (HashMap keyed by bus-value hash). Higher overhead but
    richer output.

  `fast_mode` is **independent** of `opids`. Setting `opids` no longer silently disables
  fast mode.
- **`values_filter`** _(optional)_ — list of complete bus values to track exclusively.
  Each inner array is one bus value tuple with its field components as strings (decimal or
  `0x`-prefixed hex). Has **no side effect** on `constraints`.
- **`store_row_info`** _(optional, default `false`)_ — root-level default for per-row
  detail tracking. Overridden by air/instance entries in the `instances` hierarchy.
- **`max_print`** _(optional, default 10)_ — maximum number of mismatched bus values to
  print per failing opid (regular mode only).
- **`group_by`** _(optional)_ — per-opid bucketing rules. See [§ Per-opid bucketing](#per-opid-bucketing).

Presence of this section enables bus debugging. Omit to skip.

---

## Per-opid bucketing

`bus.group_by` subdivides per-opid storage by extracting a value from the bus value tuple
and classifying it. Useful for narrowing down *which* values cause an imbalance, especially
in fast mode where you'd otherwise only learn that an opid is off.

```json
"group_by": [
  { "opid": 1, "column": 0, "by": "value" },
  {
    "opid": 2,
    "column": 0,
    "by": "range",
    "ranges": [
      { "max": "0x10000" },
      { "min": "0x10000", "max": "0x100000000" },
      { "min": "0x100000000" }
    ]
  },
  {
    "opid": 3,
    "column": 0,
    "by": "prefix",
    "prefixes": [
      { "value": "0xFF", "bits": 8 }
    ]
  }
]
```

Each rule:
- **`opid`** — the bus ID this rule applies to.
- **`column`** — index into the bus value tuple. One column per rule.
- **`by`** — classifier discriminator: `"value"`, `"range"`, `"prefix"`, or `"step"`.

Opids not listed in `group_by` keep the un-bucketed behavior (single balance/HashMap per opid).

### Filter mode (opt-in, per classifier)

Every classifier supports a **filter** opt-in that drops unmatched rows entirely instead
of tracking them. Useful workflow: run fast mode first to spot the interesting bucket,
then switch to slow mode with the same classifier in *filter* mode to drill down only on
that bucket without paying the cost of tracking everything else.

| Classifier | How filter activates | Effect |
|---|---|---|
| `value` | Add `values: [...]` field | Only listed column values are tracked; others dropped |
| `range` | Add `filter: true` | Gap-free coverage no longer required; values in gaps dropped |
| `prefix` | Add `filter: true` | Implicit "no match" catch-all dropped instead of bucketed |
| `step` | Add `filter: true` | Implicit "out of range" bucket dropped instead of bucketed |

Examples:

```json
// value filter: track only these specific column values
{ "opid": 102, "column": 2, "by": "value", "values": ["0x42", "0xff"] }

// range filter: only [0x100, 0x200) — gaps are allowed because filter=true
{ "opid": 102, "column": 2, "by": "range",
  "ranges": [{ "min": "0x100", "max": "0x200" }],
  "filter": true }

// prefix filter: only values whose top 8 bits are 0xFF — others dropped
{ "opid": 102, "column": 2, "by": "prefix",
  "prefixes": [{ "value": "0xFF", "bits": 8 }],
  "filter": true }

// step filter: only [0x1000_0000, 0x2000_0000) in 1MB buckets — OOR dropped
{ "opid": 102, "column": 2, "by": "step",
  "start": "0x10000000", "stop": "0x20000000", "step": "0x100000",
  "filter": true }
```

### `"value"` — raw column value as bucket

Bucket key is the raw `column[col_idx]` value (canonicalized to u64).
Cardinality equals the number of distinct values in that column. Unbounded.

Optional `values: ["0x42", ...]` engages filter mode: only listed column values are
tracked; other rows are dropped entirely.

### `"range"` — bucket by value range

```json
{
  "opid": 2,
  "column": 0,
  "by": "range",
  "ranges": [
    { "max": "0x10000" },
    { "min": "0x10000", "max": "0x100000000" },
    { "min": "0x100000000" }
  ]
}
```

Each range is `[min, max)`:
- Omit `min` to mean `-∞` (only allowed on the first range).
- Omit `max` to mean `+∞` (only allowed on the last range).
- Ranges must be **sorted, non-overlapping, and gap-free** — they collectively must cover
  all u64 values. The parser rejects gappy or overlapping range lists.

### `"prefix"` — bucket by top-bits prefix

```json
{
  "opid": 3,
  "column": 0,
  "by": "prefix",
  "prefixes": [
    { "value": "0xFF", "bits": 8 },
    { "value": "0xAB", "bits": 8 }
  ]
}
```

Each prefix means: *the top `bits` of the column value equal `value`*. For example,
`{value: "0xFF", bits: 8}` matches values whose top 8 bits are `0xFF`.

`value` and `bits` are independent: `bits` selects *which* top bits to compare, and the
column is always treated as 64 bits wide, so `value`'s own hex width is irrelevant. Leading
zeros in `value` are significant. For example, `{value: "0x1", bits: 8}` matches values whose
top **8** bits equal `0x01` (i.e. `col >> 56 == 1`) — *not* the same as `{value: "0x1", bits: 1}`,
which matches values whose single top bit is `1` (the entire upper half of the u64 range):

```json
// top 8 bits are 0x01 → matches [0x0100_0000_0000_0000, 0x01FF_FFFF_FFFF_FFFF]
{ "value": "0x1", "bits": 8 }

// top 1 bit is 1 → matches [0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]
{ "value": "0x1", "bits": 1 }
```

- `bits` must be in `1..=64`.
- `value` must fit in `bits` bits.
- First match wins.
- Values matching no prefix land in an implicit "no match" bucket.

### `"step"` — uniform-step buckets

```json
{
  "opid": 5,
  "column": 0,
  "by": "step",
  "start": "0x0",
  "stop":  "0x100000000",
  "step":  "0x100000"
}
```

Divides `[start, stop)` into uniform buckets of width `step`. Useful when you want a
regular grid (e.g. "one bucket per 1 MB") without listing every range explicitly.

- Bucket index for `col` in `[start, stop)`: `(col - start) / step`.
- Values outside `[start, stop)` land in an implicit "out of range" bucket.
- If `(stop - start)` is not a multiple of `step`, the last regular bucket is partial
  (`[start + (N-1)*step, stop)`).
- `step` must be > 0; `start` must be strictly less than `stop`.

### Bucketing in regular vs. fast mode

- **Fast mode**: per-bucket balance counts. Output reports which buckets are unbalanced.
- **Regular mode**: per-bucket per-value detail. Output groups mismatched values under
  each bucket descriptor (`◆ Bucket: col[0] in [0x10000, 0x100000000)`).

---

## `output` — output destination

```json
"output": {
  "to_file":   true,
  "file_path": "tmp/debug.log"
}
```

### Fields

- **`to_file`** _(optional, default `true`)_ — write debug output to a file. Set to
  `false` to redirect to stdout. The default is `true` because debug output is typically
  voluminous and would flood the terminal.
- **`file_path`** _(optional, default `"tmp/debug.log"`)_ — path of the output file. Parent
  directories are created on demand.

---

## Numeric string parsing

Wherever the schema accepts numeric strings (`values_filter`, range `min`/`max`, prefix
`value`), values may be:

- Decimal: `"123"`
- Hex: `"0xFF"` or `"0xfe"`

---

## Examples

### Minimal: verify all constraints, no bus debugging

```json
{ "constraints": {} }
```

### Bus debug only, fast mode

```json
{ "bus": {} }
```

This enables bus debugging on all opids in fast mode (default `fast_mode: true`),
without running constraint verification.

### Verify specific global constraints and debug specific opids

```json
{
  "constraints":        { "enabled": true, "max_print": 20 },
  "global_constraints": { "enabled": true, "global_constraint_ids": [0, 1, 5] },
  "bus":                { "opids": [3, 7], "fast_mode": false, "max_print": 50 }
}
```

### Process only one instance, log to a custom file

```json
{
  "instances": {
    "mode": "only_listed",
    "list": [
      {
        "airgroup": "Main",
        "airs": [
          {
            "air": "Binary",
            "instances": [{ "instance_id": 0, "constraints": [5, 10] }]
          }
        ]
      }
    ]
  },
  "constraints": {},
  "output": { "to_file": true, "file_path": "/tmp/my-debug.log" }
}
```

### Bucket opid 1 by column 0 value, opid 2 by ranges

```json
{
  "bus": {
    "fast_mode": true,
    "group_by": [
      { "opid": 1, "column": 0, "by": "value" },
      {
        "opid": 2,
        "column": 0,
        "by": "range",
        "ranges": [
          { "max": "0x10000" },
          { "min": "0x10000" }
        ]
      }
    ]
  }
}
```

When opid 1 or opid 2 mismatches, the output identifies the specific value (opid 1) or
range bucket (opid 2) responsible.

---

## Behavior notes

1. **Activation is explicit.** Omit a section to disable its subsystem; `constraints` and
   `global_constraints` are additionally gated by their `enabled` field (default `true`).
   This replaces the old implicit "presence of `std_mode` flips global mode to Debug"
   behavior.

2. **No hidden coupling between sections.** `bus.values_filter` no longer disables global
   constraint verification. `bus.opids` no longer silently overrides `bus.fast_mode`.

3. **Mutual exclusivity.** You cannot specify both `airgroup` and `airgroup_id`, or both
   `air` and `air_id`, in the same object.

4. **`mode: "only_listed"` skips everything not in `list`.** Useful for:
   - Isolating problematic instances during debugging.
   - Reducing proof generation time when testing specific components.

5. **Row-info hierarchy.** `bus.store_row_info` (root) → air-level → instance-level, OR-merged.

6. **Output destination.**
   - When `output.to_file` is `false` or omitted: debug output goes to stdout.
   - When `output.to_file` is `true`: output is written to `output.file_path` (default
     `tmp/debug.log`). Parent directories are created automatically.

7. **Performance considerations.**
   - **Fast mode** (`bus.fast_mode: true`): minimal overhead, only tracks per-opid (or per-bucket) balance.
   - **Regular mode** (`bus.fast_mode: false`): tracks per-value metadata.
   - **Row info enabled** (`store_row_info: true`): highest overhead, stores exact row
     locations of each mismatch. Only enable when needed.

8. **Migration from the old schema.** The new schema is a clean break. Old-format files
   using `std_mode`, `skip_prover_instances`, `n_print_constraints`, `air_ids`,
   `instance_ids`, `debug_values`, `hint_ids`, or `rows` no longer parse.

---

## Complete reference

Every supported field in one `debug.json`, with all classifier variants shown side-by-side.
You wouldn't normally combine all of these — this is a *schema sweep* for API review, not
a recommended config.

```json
{
  "instances": {
    "mode": "only_listed",
    "skip_tables": false,
    "list": [
      {
        "airgroup_id": 0,
        "airs": [
          {
            "air_id": 1,
            "store_row_info": false,
            "instances": [
              {
                "instance_id": 0,
                "constraints": [5, 10, 15],
                "store_row_info": true
              },
              { "instance_id": 2 }
            ]
          },
          { "air": "Module" }
        ]
      },
      {
        "airgroup": "Main",
        "airs": [
          {
            "air": "Binary",
            "instances": [{ "instance_id": 0, "constraints": [0, 1, 2] }]
          }
        ]
      }
    ]
  },

  "constraints": {
    "enabled": true,
    "max_print": 20
  },

  "global_constraints": {
    "enabled": true,
    "global_constraint_ids": [0, 1, 2]
  },

  "bus": {
    "opids": [1, 2, 3, 4, 5],
    "fast_mode": true,
    "store_row_info": false,
    "max_print": 10,
    "values_filter": [
      ["1302180"],
      ["0", "1", "0"],
      ["0xdeadbeef"]
    ],
    "group_by": [
      { "opid": 1, "column": 0, "by": "value" },
      { "opid": 102, "column": 2, "by": "value", "values": ["0x42", "0xff"] },
      {
        "opid": 2,
        "column": 0,
        "by": "range",
        "ranges": [
          { "max": "0x10000" },
          { "min": "0x10000", "max": "0x100000000" },
          { "min": "0x100000000" }
        ]
      },
      {
        "opid": 22,
        "column": 0,
        "by": "range",
        "ranges": [{ "min": "0x100", "max": "0x200" }],
        "filter": true
      },
      {
        "opid": 3,
        "column": 0,
        "by": "prefix",
        "prefixes": [
          { "value": "0xFF", "bits": 8 },
          { "value": "0xAB", "bits": 8 }
        ]
      },
      {
        "opid": 33,
        "column": 0,
        "by": "prefix",
        "prefixes": [{ "value": "0xFF", "bits": 8 }],
        "filter": true
      },
      {
        "opid": 5,
        "column": 0,
        "by": "step",
        "start": "0x0",
        "stop":  "0x100000000",
        "step":  "0x100000"
      },
      {
        "opid": 55,
        "column": 0,
        "by": "step",
        "start": "0x10000000",
        "stop":  "0x20000000",
        "step":  "0x100000",
        "filter": true
      }
    ]
  },

  "output": {
    "to_file": true,
    "file_path": "tmp/debug.log"
  }
}
```

### Field index (quick lookup)

| Path | Type | Default | Notes |
|---|---|---|---|
| `instances.mode` | `"all"` \| `"only_listed"` | `"all"` | `"only_listed"` skips instances not in `list` |
| `instances.skip_tables` | bool | `false` | `"only_listed"` only; subject table instances to the filter too |
| `instances.list[].airgroup_id` | u64 | — | mutually exclusive with `airgroup` |
| `instances.list[].airgroup` | string | — | mutually exclusive with `airgroup_id` |
| `instances.list[].airs[].air_id` | u64 | — | mutually exclusive with `air` |
| `instances.list[].airs[].air` | string | — | mutually exclusive with `air_id` |
| `instances.list[].airs[].store_row_info` | bool | `false` | air-level bus-debug detail |
| `instances.list[].airs[].instances[].instance_id` | u64 | `0` | |
| `instances.list[].airs[].instances[].constraints` | [u64] | `[]` | constraint indices to verify on this instance |
| `instances.list[].airs[].instances[].store_row_info` | bool | `false` | instance-level bus-debug detail |
| `constraints.enabled` | bool | `true` | enable per-air constraint verification |
| `constraints.max_print` | usize | `10` | max mismatched constraints printed per failure |
| `global_constraints.enabled` | bool | `true` | enable global constraint verification |
| `global_constraints.global_constraint_ids` | [usize] | `[]` (= all) | global constraint indices to verify |
| `bus.opids` | [u64] | `[]` (= all) | which opids to track |
| `bus.fast_mode` | bool | `true` | counts-only vs full per-value detail |
| `bus.store_row_info` | bool | `false` | root default for row-info storage |
| `bus.max_print` | usize | `10` | max mismatched bus values per opid (regular mode) |
| `bus.values_filter` | [[string]] | `[]` | exact bus values to track (decimal or `0x…`) |
| `bus.group_by[].opid` | u64 | — | which opid this rule applies to (must be unique) |
| `bus.group_by[].column` | u64 | — | column index in the bus value tuple |
| `bus.group_by[].by` | `"value"` \| `"range"` \| `"prefix"` \| `"step"` | — | classifier discriminator |
| `bus.group_by[].values` | [string] | — | `"value"` only; presence engages filter mode |
| `bus.group_by[].ranges[]` | `{min?, max?}` | — | `"range"` only; must cover (-∞,+∞) unless `filter: true` |
| `bus.group_by[].prefixes[]` | `{value, bits}` | — | `"prefix"` only; `bits` in 1..=64 |
| `bus.group_by[].start` / `.stop` / `.step` | string | — | `"step"` only; `step > 0`, `start < stop` |
| `bus.group_by[].filter` | bool | `false` | `"range"`/`"prefix"`/`"step"` only; drops unmatched rows |
| `output.to_file` | bool | `true` | write to file instead of stdout; set `false` for stdout |
| `output.file_path` | string | `"tmp/debug.log"` | output path; parent dirs auto-created |
