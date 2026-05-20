# Debug Report JSON Format

This document describes the JSON shape of the `DebugReport` value returned by
`ProofMan::get_debug_info(...)` (the `debug-info` CLI subcommand). The same data
is teed into `debug.log` (or stdout, per the `output` section of `debug.json`),
but the JSON form is intended for programmatic consumers — e.g. a frontend that
renders mismatches in whatever layout it likes.

## Overview

The report covers bus / std-lookup verification only. It is populated by the std
lib's bus checker as it consumes accumulated debug data, alongside the textual
report written to `debug.log`.

| Field          | Type                  | Description                                                  |
|----------------|-----------------------|--------------------------------------------------------------|
| `all_ok`       | bool                  | `true` if no mismatched bus values were found during the run |
| `bus_sections` | array of `BusSection` | One entry per opid that produced any mismatch                |

When `all_ok` is `true`, `bus_sections` is empty.

## Root structure

```json
{
  "all_ok": false,
  "bus_sections": [ ... ]
}
```

---

## `BusSection` — one opid

```json
{
  "opid": 7,
  "mismatched": true,
  "num_overassumed": 3,
  "num_overproven": 1,
  "buckets": [ ... ]
}
```

### Fields

- **`opid`** — bus operation ID this section reports on.
- **`mismatched`** — `true` when this opid has at least one mismatched value
  (always `true` in practice, since clean opids are omitted from `bus_sections`).
- **`num_overassumed`** — total number of values across all buckets where
  `num_assumes > num_proves` (assumed but not proven).
- **`num_overproven`** — total number of values across all buckets where
  `num_proves > num_assumes` (proven but not assumed).
- **`buckets`** — one entry per bucket within the opid. Opids without a
  per-opid bucketing rule have a single implicit bucket with `bucket_key = 0`.

---

## `BusBucket` — one bucket within an opid

```json
{
  "bucket_key": 0,
  "bucket_label": "col[2] in [0x100, 0x200)",
  "overassumed": [ ... ],
  "overproven":  [ ... ]
}
```

### Fields

- **`bucket_key`** — bucket discriminator, as set by the per-opid bucketing rule
  in `debug.json`'s `bus.group_by` section.
- **`bucket_label`** — human-readable bucket description rendered from that rule
  (e.g. `"col[2] in [0x100, 0x200)"`). `null` when the opid has no bucketing
  rule.
- **`overassumed`** — values appearing more as `assume` than `prove`.
- **`overproven`** — values appearing more as `prove` than `assume`.

Both lists may be empty (one side can be balanced while the other is not).

---

## `BusValueMismatch` — one mismatched bus value

```json
{
  "vals": [42, 0, 0],
  "hash": 14087324812907813462,
  "num_assumes": 3,
  "num_proves": 2,
  "global_origin": { ... },
  "local_origins": [ ... ]
}
```

### Fields

- **`vals`** — bus value tuple as canonical `u64` field elements. Extended-field
  components are flattened in order alongside base-field components, so a tuple
  of one cubic extension renders as a 3-element array. Frontends that need a
  specific tuple schema (e.g. `(payload..., multiplicity)`) should derive it
  from the opid's known width.
- **`hash`** — the value's debug-data hash key (stable per `vals`).
- **`num_assumes`** — total times this value appeared as `assume` across all
  origins.
- **`num_proves`** — total times this value appeared as `prove` across all
  origins.
- **`global_origin`** — aggregated global-level origin info when present, else
  `null`. See [`BusValueGlobalOrigin`](#busvalueglobalorigin).
- **`local_origins`** — per-AIR-instance occurrences. See
  [`BusValueLocalOrigin`](#busvaluelocalorigin). May be empty if
  `store_row_info` was not enabled for the contributing instances.

---

## `BusValueGlobalOrigin`

```json
{
  "airgroup_id": 0,
  "airgroup_name": "Main",
  "piop_name": "RangeCheck",
  "expression_names": ["value"],
  "is_prod": false
}
```

### Fields

- **`airgroup_id`** / **`airgroup_name`** — airgroup where the global origin was
  registered.
- **`piop_name`** — friendly name of the bus operation (PIOP).
- **`expression_names`** — names of the bus expressions, in tuple order.
- **`is_prod`** — `true` for product-style PIOPs (`gprod`), `false` for
  sum-style (`gsum`).

---

## `BusValueLocalOrigin`

```json
{
  "airgroup_id": 0,
  "airgroup_name": "Main",
  "air_id": 1,
  "air_name": "Range8",
  "instance_id": 0,
  "hint_id": 4,
  "piop_name": "RangeCheck",
  "expression_names": ["value"],
  "is_prod": false,
  "rows": [12, 47, 130, 412]
}
```

### Fields

- **`airgroup_id`** / **`airgroup_name`** — airgroup of the AIR instance that
  contributed.
- **`air_id`** / **`air_name`** — AIR within that airgroup.
- **`instance_id`** — process-local AIR instance ID.
- **`hint_id`** — index of the debug-data hint inside this AIR (matches the
  ordering used by the std lib).
- **`piop_name`** / **`expression_names`** — same semantics as on
  `BusValueGlobalOrigin`, but resolved against the AIR's local hints.
- **`is_prod`** — `true` for `gprod`, `false` for `gsum`.
- **`rows`** — row indices (sorted ascending) where this value appeared at this
  origin. **Not truncated** — the report keeps every row so the consumer can
  decide how to summarize. The text output in `debug.log` truncates to
  `bus.max_print` for readability; the structured form does not.

---

## Examples

### All bus values match

```json
{
  "all_ok": true,
  "bus_sections": []
}
```

### A mismatched opid

```json
{
  "all_ok": false,
  "bus_sections": [
    {
      "opid": 7,
      "mismatched": true,
      "num_overassumed": 1,
      "num_overproven": 0,
      "buckets": [
        {
          "bucket_key": 0,
          "bucket_label": null,
          "overassumed": [
            {
              "vals": [42, 0, 0],
              "hash": 14087324812907813462,
              "num_assumes": 3,
              "num_proves": 2,
              "global_origin": null,
              "local_origins": [
                {
                  "airgroup_id": 0,
                  "airgroup_name": "Main",
                  "air_id": 1,
                  "air_name": "Range8",
                  "instance_id": 0,
                  "hint_id": 4,
                  "piop_name": "RangeCheck",
                  "expression_names": ["value"],
                  "is_prod": false,
                  "rows": [12, 47, 130]
                }
              ]
            }
          ],
          "overproven": []
        }
      ]
    }
  ]
}
```

---

## Field index (quick lookup)

| Path | Type | Notes |
|---|---|---|
| `all_ok` | bool | `true` only when `bus_sections` is empty |
| `bus_sections[]` | array | one entry per mismatched opid |
| `bus_sections[].opid` | u64 | bus operation ID |
| `bus_sections[].mismatched` | bool | always `true` for entries in `bus_sections` |
| `bus_sections[].num_overassumed` | usize | sum across buckets |
| `bus_sections[].num_overproven` | usize | sum across buckets |
| `bus_sections[].buckets[]` | array | one entry per non-empty bucket |
| `bus_sections[].buckets[].bucket_key` | u64 | discriminator from `bus.group_by` |
| `bus_sections[].buckets[].bucket_label` | string \| null | rendered label, `null` if no rule |
| `bus_sections[].buckets[].overassumed[]` / `overproven[]` | array | one entry per mismatched value |
| `…[].vals` | [u64] | flattened field elements |
| `…[].hash` | u64 | debug-data hash key |
| `…[].num_assumes` / `num_proves` | u64 | totals across origins |
| `…[].global_origin` | object \| null | one aggregated global record per value |
| `…[].local_origins[]` | array | per-AIR-instance occurrences |
| `…[].local_origins[].rows` | [usize] | sorted, not truncated |
