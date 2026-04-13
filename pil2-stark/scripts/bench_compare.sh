#!/usr/bin/env bash
# bench_compare.sh <baseline.txt> <fresh.txt>
#
# Compares Google Benchmark output files. Exits 1 if any benchmark regresses
# more than THRESHOLD (default 2%) vs the baseline.
#
# Usage:
#   ./benchscpu > /tmp/bench.txt 2>&1
#   ./scripts/bench_compare.sh src/goldilocks/benchs/baseline/$(hostname).txt /tmp/bench.txt

set -euo pipefail

THRESHOLD=${BENCH_THRESHOLD:-2}   # percent; override with env var

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <baseline.txt> <fresh.txt>" >&2
    exit 2
fi

BASELINE="$1"
FRESH="$2"

if [[ ! -f "$BASELINE" ]]; then
    echo "ERROR: baseline file not found: $BASELINE" >&2
    exit 2
fi
if [[ ! -f "$FRESH" ]]; then
    echo "ERROR: fresh file not found: $FRESH" >&2
    exit 2
fi

# Parse a benchmark file into tab-separated "name<TAB>time_ns" lines.
# Skips headers, separators, warnings, and date lines.
# Time units (ns/us/ms/s) are all converted to nanoseconds.
parse_file() {
    local file=$1
    awk '
    /^-+$/     { next }   # separator
    /^Benchmark/ { next } # header
    /^[[:space:]]*$/ { next }
    /^[0-9]/   { next }   # date/timestamp
    /^Running/ { next }
    /^Run on/  { next }
    /^CPU/     { next }
    /L[0-9]/   { next }   # cache info (may be indented)
    /^Load/    { next }
    /^\*\*\*/  { next }   # warnings
    NF >= 3 {
        name = $1
        val  = $2 + 0
        unit = $3
        if      (unit == "us") val *= 1e3
        else if (unit == "ms") val *= 1e6
        else if (unit == "s")  val *= 1e9
        # else ns — no conversion needed
        print name "\t" val
    }
    ' "$file"
}

# Build lookup tables
declare -A baseline_ns
declare -A fresh_ns

while IFS=$'\t' read -r name ns; do
    baseline_ns["$name"]="$ns"
done < <(parse_file "$BASELINE")

while IFS=$'\t' read -r name ns; do
    fresh_ns["$name"]="$ns"
done < <(parse_file "$FRESH")

if [[ ${#baseline_ns[@]} -eq 0 ]]; then
    echo "ERROR: no benchmarks parsed from $BASELINE" >&2
    exit 2
fi

# Compare
regressions=0
missing=0

printf "\n%-60s %14s %14s %9s\n" "Benchmark" "Baseline(ns)" "Fresh(ns)" "Change"
printf '%s\n' "$(printf '─%.0s' {1..100})"

for name in $(printf '%s\n' "${!baseline_ns[@]}" | sort); do
    base="${baseline_ns[$name]}"
    if [[ -z "${fresh_ns[$name]+x}" ]]; then
        printf "%-60s %14.0f %14s %9s  MISSING\n" "$name" "$base" "N/A" "?"
        missing=$((missing + 1))
        continue
    fi
    fresh="${fresh_ns[$name]}"

    pct_and_status=$(awk -v b="$base" -v f="$fresh" -v t="$THRESHOLD" 'BEGIN {
        if (b <= 0) { printf "0.00 OK\n"; exit }
        pct = (f - b) / b * 100
        printf "%.2f %s\n", pct, (pct > t ? "REGRESS" : "OK")
    }')
    pct=$(awk '{print $1}' <<< "$pct_and_status")
    status=$(awk '{print $2}' <<< "$pct_and_status")

    flag=""
    if [[ "$status" == "REGRESS" ]]; then
        flag="  <-- REGRESSION"
        regressions=$((regressions + 1))
    fi
    printf "%-60s %14.0f %14.0f %+8.1f%%%s\n" "$name" "$base" "$fresh" "$pct" "$flag"
done

printf '%s\n\n' "$(printf '─%.0s' {1..100})"

if (( missing > 0 )); then
    echo "WARNING: $missing benchmark(s) missing from fresh run" >&2
fi

if (( regressions > 0 )); then
    echo "FAIL: $regressions benchmark(s) regressed by more than ${THRESHOLD}%" >&2
    exit 1
fi

echo "PASS: no regressions above ${THRESHOLD}%"
