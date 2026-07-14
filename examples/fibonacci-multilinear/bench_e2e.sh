#!/usr/bin/env bash
#
# End-to-end multilinear-vs-univariate benchmark for the fibonacci-multilinear
# example, swept over trace shapes. Both trace size and the instance mix are
# fixed by the two N's in build.pil:
#
#   Fibonacci(N: 2^fib)   -> 1 Fibonacci instance of 2^fib rows
#   Module(N: 2^mod)      -> 2^(fib-mod) Module instances of 2^mod rows
#
# Usage (from repo root):
#   examples/fibonacci-multilinear/bench_e2e.sh [fib:mod ...]
#   examples/fibonacci-multilinear/bench_e2e.sh 10:8 12:8 14:8 16:10

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-14/lib}"

EX=examples/fibonacci-multilinear
PIL=$EX/pil/build.pil
PILOUT=$EX/pil/build.pilout
PK=$EX/build/provingKey
ROM=$EX/build/rom.bin
INPUTS=$EX/src/inputs.json
WLIB=target/release/libfibonacci_multilinear.so
LOGDIR=$(mktemp -d)

# Shapes as `fib:mod` pairs; default sweep if none given.
if [ "$#" -eq 0 ]; then SHAPES=(10:8 12:8 14:8 16:10); else SHAPES=("$@"); fi

echo "building tools + witness lib deps (release)…"
cargo build --release --bin proofman-setup --bin proofman-cli >/dev/null
SETUP=target/release/proofman-setup
CLI=target/release/proofman-cli

cp "$PIL" "$PIL.bak"
trap 'mv "$PIL.bak" "$PIL" 2>/dev/null || true; echo "logs in $LOGDIR"' EXIT

# Parse "<<< NAME (NNNms|N.Ns)" from a log into integer milliseconds (or NA).
extract() {
  local v
  v=$(grep -oE "<<< $2 \([0-9.]+m?s\)" "$1" | tail -1 | grep -oE '[0-9.]+m?s' || true)
  case "$v" in
    *ms) echo "${v%ms}" ;;
    *s)  awk "BEGIN{printf \"%.0f\", ${v%s}*1000}" ;;
    *)   echo "NA" ;;
  esac
}

# Sum a multilinear per-phase token (commit|zerocheck|claims|opening) across every
# `ml prove_air[...]` debug line (one per instance) → integer milliseconds. Needs
# `prove-multilinear -v` (the split is logged at Debug).
extract_ml_phase() {
  awk -v key="$2" '
    { s = $0
      while (match(s, key"=[0-9.]+m?s")) {
        tok = substr(s, RSTART, RLENGTH); sub(key"=", "", tok)
        if (tok ~ /ms$/) { sub(/ms$/, "", tok); sum += tok }
        else             { sub(/s$/,  "", tok); sum += tok * 1000 }
        seen = 1; s = substr(s, RSTART + RLENGTH)
      } }
    END { if (seen) printf "%.0f", sum; else print "NA" }' "$1"
}

printf '\n%-16s %6s %6s %6s | %-27s | %-27s | %s\n' \
  "shape" "n_fib" "n_mod" "#inst" "univariate (ms)" "multilinear (ms)" "inner speedup"
printf '%s\n' "-------------------------------------------------------------------------------------------------------------------"

for shape in "${SHAPES[@]}"; do
  fib=${shape%%:*}; mod=${shape##*:}
  ninst=$(( 1 + (1 << (fib - mod)) ))
  log=$LOGDIR/$shape

  # 1. Patch build.pil, 2. compile, 3. setup, 4. pil-helpers, 5. build wlib, 6. rom.
  sed -i -E "s/Fibonacci\(N: 2\*\*[0-9]+\)/Fibonacci(N: 2**$fib)/; s/Module\(N: 2\*\*[0-9]+\)/Module(N: 2**$mod)/" "$PIL"
  {
    $SETUP compile-pil --pil "$PIL" -I pil2-components/lib/std/pil -o "$PILOUT"
    $SETUP setup -a "$PILOUT" -b "$EX/build"
    $CLI pil-helpers --pilout "$PILOUT" --path "$EX/src" -o
    cargo build --release -p fibonacci-multilinear
    $CLI gen-custom-commits-fixed --witness-lib "$WLIB" --proving-key "$PK" --custom-commits "rom=$ROM"
  } >"$log.setup" 2>&1 || { echo "[$shape] setup failed — see $log.setup"; tail -5 "$log.setup"; continue; }

  # 7. Univariate (no aggregation) and 8. multilinear.
  $CLI prove -w "$WLIB" -k "$PK" -i "$INPUTS" --custom-commits "rom=$ROM" -o "$EX/build/proofs_uni" \
    >"$log.uni" 2>&1 || { echo "[$shape] univariate prove failed — see $log.uni"; tail -5 "$log.uni"; }
  $CLI prove-multilinear -v -w "$WLIB" -k "$PK" -i "$INPUTS" --custom-commits "rom=$ROM" -o "$EX/build/proofs_ml" \
    >"$log.ml" 2>&1 || { echo "[$shape] multilinear prove failed — see $log.ml"; tail -5 "$log.ml"; }

  uc=$(extract "$log.uni" CALCULATING_CONTRIBUTIONS); ui=$(extract "$log.uni" GENERATING_INNER_PROOFS)
  mc=$(extract "$log.ml"  CALCULATING_CONTRIBUTIONS); mi=$(extract "$log.ml"  GENERATING_INNER_PROOFS)
  spd=$(awk "BEGIN{ if(\"$ui\"!=\"NA\" && \"$mi\"!=\"NA\" && $mi>0) printf \"%.2fx\", $ui/$mi; else print \"-\" }")

  printf '%-16s %6s %6s %6s | contrib %-6s inner %-6s | contrib %-6s inner %-6s | %s\n' \
    "$shape" "$((1<<fib))" "$((1<<mod))" "$ninst" "$uc" "$ui" "$mc" "$mi" "$spd"

done
