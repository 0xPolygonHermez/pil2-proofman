# Debug Configuration JSON Format

This document describes the structure and usage of the `debug.json` configuration file used by the PIL2 Proofman debugger.

## Overview

The debug.json file allows fine-grained control over which constraints, instances, and bus operations to debug during proof generation. It supports both instance-specific debugging and standard mode debugging for bus operations.

## Root Structure

```json
{
  "instances": [...],
  "global_constraints": [...],
  "std_mode": {...},
  "n_print_constraints": <number>,
  "store_row_info": <boolean>,
  "skip_prover_instances": <boolean>
}
```

### Fields

- **`instances`** _(optional)_: Array of airgroup configurations to debug. If omitted or empty, no instance-specific debugging is performed.
- **`global_constraints`** _(optional)_: Array of constraint indices to debug at the global level. Defaults to empty array.
- **`std_mode`** _(optional)_: Configuration for standard mode bus debugging. See [Standard Mode](#standard-mode) section.
- **`n_print_constraints`** _(optional)_: Maximum number of constraints to print when errors occur. Defaults to system default (10).
- **`store_row_info`** _(optional)_: Global flag to enable row information storage for detailed debugging. Defaults to `false`.
- **`skip_prover_instances`** _(optional)_: When `true`, enables instance filtering mode where only instances listed in the `instances` array will be processed. When `false` or omitted, all instances are processed regardless of the `instances` configuration. Defaults to `false`.

## Standard Mode

The `std_mode` section controls bus operation debugging (checking that bus "assumes" match "proves"):

```json
"std_mode": {
  "opids": [1, 2, 3],
  "n_vals": 10,
  "print_to_file": true,
  "fast_mode": false
}
```

### Fields

- **`opids`** _(optional)_: Array of specific operation IDs (bus IDs) to debug. If empty or omitted, all operations are checked. When specified with non-empty array, `fast_mode` is automatically disabled.
- **`n_vals`** _(optional)_: Maximum number of mismatched values to print per operation. Defaults to 10.
- **`print_to_file`** _(optional)_: If `true`, writes debug output to `tmp/debug.log` instead of stdout. Defaults to `false`.
- **`fast_mode`** _(optional)_: Enable fast mode which only tracks counts without storing detailed location information. Defaults to `true`. Automatically set to `false` when `opids` is specified with non-empty array.
- **`debug_values`** _(optional)_: Array of arrays of string values representing complete bus values to track. Each inner array represents one complete bus value (which may consist of multiple field components). When specified, only these exact values will be tracked and detailed row information will be automatically enabled for them. See [Debug Values Format](#debug-values-format) for details.

## Debug Values Format

The `debug_values` field allows you to specify exact bus values to monitor. Each bus value is represented as an array of strings, where each string is a field component. When debug values are specified, row information storage is automatically enabled for those values.

### Format

```json
"std_mode": {
  "debug_values": [
    ["123"],           // Single field value
    ["1", "2", "3"],   // Three field components (e.g., extended field)
    ["0xff", "0x100"]  // Two field components in hex
  ]
}
```

### Value Parsing Rules

1. **Decimal notation**: `"123"` → parsed as base-10
2. **Hexadecimal notation**: `"0xff"` or `"0xFF"` → parsed as base-16
3. **Each inner array** represents one complete bus value to match
4. **All components** in an inner array are flattened and hashed together

### How It Works

1. Each inner array of strings is parsed into field values
2. A hash is computed for each complete bus value
3. During execution, only bus values matching these hashes are tracked
4. Row information is automatically stored for matching values

### Examples

**Single simple value:**
```json
"debug_values": [
  ["1302180"]
]
```

**Multiple simple values:**
```json
"debug_values": [
  ["123"],
  ["456"],
  ["0xdeadbeef"]
]
```

**Extended field values:**
```json
"debug_values": [
  ["1", "2", "3"],
  ["0", "1", "0"]
]
```

**Mixed:**
```json
"debug_values": [
  ["100"],
  ["1", "2", "3"],
  ["0xff"]
]
```

## Instance Configuration

The `instances` array defines which specific airgroups, airs, and instances to debug. **Important**: When `skip_prover_instances` is set to `true` and `instances` is specified with a non-empty array, only the listed instances will be processed during proof generation - all other instances will be skipped. This allows you to focus on debugging specific parts of the proof.

```json
"instances": [
  {
    "airgroup_id": 0,
    "air_ids": [
      {
        "air_id": 1,
        "instance_ids": [
          {
            "instance_id": 0,
            "constraints": [5, 10, 15],
            "hint_ids": [2, 4],
            "rows": [100, 200, 300],
            "store_row_info": true
          }
        ],
        "store_row_info": false
      }
    ]
  }
]
```

### Airgroup Object

```json
{
  "airgroup_id": <number>,
  "airgroup": "<name>",
  "air_ids": [...]
}
```

- **`airgroup_id`** _(conditional)_: Numeric identifier for the airgroup. Either this or `airgroup` must be specified, but not both.
- **`airgroup`** _(conditional)_: String name of the airgroup. Either this or `airgroup_id` must be specified, but not both.
- **`air_ids`** _(optional)_: Array of air configurations within this airgroup.

### Air Object

```json
{
  "air_id": <number>,
  "air": "<name>",
  "instance_ids": [...],
  "store_row_info": <boolean>
}
```

- **`air_id`** _(conditional)_: Numeric identifier for the air. Either this or `air` must be specified, but not both.
- **`air`** _(conditional)_: String name of the air. Either this or `air_id` must be specified, but not both.
- **`instance_ids`** _(optional)_: Array of instance configurations to debug.
- **`store_row_info`** _(optional)_: Enable row information storage for all instances in this air. Defaults to `false`.

### Instance Object

```json
{
  "instance_id": <number>,
  "constraints": [<indices>],
  "hint_ids": [<ids>],
  "rows": [<indices>],
  "store_row_info": <boolean>
}
```

- **`instance_id`** _(optional)_: Identifier for this specific instance. Defaults to 0.
- **`constraints`** _(optional)_: Array of constraint indices to debug. Empty array means no constraint-specific debugging.
- **`hint_ids`** _(optional)_: Array of hint IDs to debug. Empty array means no hint-specific debugging.
- **`rows`** _(optional)_: Array of specific row indices to debug. Empty array means no row-specific debugging.
- **`store_row_info`** _(optional)_: Enable row information storage for this instance. Defaults to `false`. **Note**: Storing row info has performance impact; only enable when you need to see exact row locations of mismatches.

## Examples

### Example 1: Fast Mode - Check All Bus Operations

Simple configuration to verify all bus operations match (assumes vs proves):

```json
{
  "std_mode": {
    "fast_mode": true
  }
}
```

This is the fastest mode - only counts are tracked, no detailed location information is stored.

### Example 2: Debug Specific Operations with Details

Debug only specific operation IDs with detailed output including row locations:

```json
{
  "std_mode": {
    "opids": [5, 12, 23],
    "n_vals": 20,
    "print_to_file": true
  },
  "store_row_info": true
}
```

**Note**: When `opids` is non-empty, `fast_mode` is automatically disabled.

### Example 2b: Debug Specific Values

Track only specific bus values across all operations:

```json
{
  "std_mode": {
    "debug_values": [
      ["1302180"],
      ["0", "1", "0"],
      ["0xdeadbeef"]
    ],
    "n_vals": 50,
    "print_to_file": true
  }
}
```

**Note**: Row information is automatically stored for values matching `debug_values`.

### Example 3: Instance-Specific Debugging

Debug specific constraints and hints in particular instances:

```json
{
  "skip_prover_instances": true,
  "instances": [
    {
      "airgroup": "Main",
      "air_ids": [
        {
          "air": "Binary",
          "instance_ids": [
            {
              "instance_id": 0,
              "constraints": [0, 1, 2],
              "hint_ids": [5, 10]
            }
          ]
        }
      ]
    }
  ]
}
```

**Note**: Setting `skip_prover_instances: true` means only the Binary air instance 0 will be processed.

### Example 4: Combined Configuration

Comprehensive debugging with both instance and standard mode:

```json
{
  "skip_prover_instances": true,
  "instances": [
    {
      "airgroup_id": 0,
      "air_ids": [
        {
          "air_id": 1,
          "instance_ids": [
            {
              "instance_id": 0,
              "constraints": [5, 10],
              "rows": [100, 200, 300],
              "store_row_info": true
            }
          ]
        }
      ]
    }
  ],
  "global_constraints": [0, 1, 2],
  "std_mode": {
    "opids": [1, 2, 3],
    "n_vals": 15,
    "print_to_file": true,
    "fast_mode": false
  },
  "n_print_constraints": 20,
  "store_row_info": true
}
```

### Example 5: Empty Configuration (Default Behavior)

Minimal configuration for standard mode with no specific filtering:

```json
{
  "std_mode": {
    "fast_mode": true
  }
}
```

Or simply:
```json
{}
```

## Behavior Notes

1. **Mutual Exclusivity**: You cannot specify both `airgroup` and `airgroup_id`, or both `air` and `air_id` in the same object.

2. **Instance Filtering**: When `skip_prover_instances` is set to `true` and `instances` is specified with a non-empty array, **only the listed instances will be processed** during proof generation. All other instances are skipped entirely. This is useful for:
   - Isolating problematic instances during debugging
   - Reducing proof generation time when testing specific components
   - Focusing on particular airgroups or airs

3. **Fast Mode Auto-Disable**: When `opids` is specified with a non-empty array, `fast_mode` is automatically set to `false` regardless of the configuration value. This is because detailed information is needed when debugging specific operations.

4. **Debug Values Auto-Store**: When `debug_values` is specified with a non-empty array, row information storage is automatically enabled for matching values, even if `store_row_info` is `false` globally. This allows precise tracking of specific values without the overhead of storing information for all values.

5. **Empty Arrays**: Empty arrays in `instances`, `global_constraints`, `constraints`, `hint_ids`, or `rows` effectively disable that aspect of debugging.

5. **Row Information Hierarchy**: The `store_row_info` flag can be set at three levels with the following precedence (most specific wins):
   - Root level: applies globally
   - Air level: applies to all instances in that air
   - Instance level: applies to specific instance only

6. **Output Destination**: 
   - When `print_to_file` is `false` (default): Debug output goes to stdout
   - When `print_to_file` is `true`: Debug output is written to `tmp/debug.log`
   - The `tmp` directory is created automatically if it doesn't exist

7. **Performance Considerations**:
   - **Fast mode** (`fast_mode: true`): Minimal overhead, only tracks counts
   - **Regular mode** (`fast_mode: false`): Tracks additional metadata but not row locations
   - **Row info enabled** (`store_row_info: true`): Highest overhead, stores exact row locations for each mismatch

8. **Parallel Processing**: The implementation uses parallel processing with multiple maps (currently 2) to reduce lock contention during debug data collection, improving performance on multi-core systems.