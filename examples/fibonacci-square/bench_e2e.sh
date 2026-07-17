#!/usr/bin/env bash
#
# End-to-end multilinear-vs-univariate benchmark for the fibonacci-square
# example, swept over trace shapes. Both trace size and the instance mix are
# fixed by the two N's in build.pil:
#
#   FibonacciSquare(N: 2^fib) -> 1 FibonacciSquare instance of 2^fib rows
#   Module(N: 2^mod)          -> 2^(fib-mod) Module instances of 2^mod rows
#
# Each shape is proven twice, from two different setups:
#   univariate:  build.pil      (plain LogUp: committed gsum/im columns)
#   multilinear: build_ml.pil   (LogUp-GKR: no stage-2 bus columns)
#
# Each prove runs REPS times and the fastest repetition is reported.
#
# Usage (from repo root):
#   examples/fibonacci-square/bench_e2e.sh [fib:mod ...]
#   examples/fibonacci-square/bench_e2e.sh 10:8 12:8 14:8 16:10
#   REPS=5 examples/fibonacci-square/bench_e2e.sh 14:8

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-14/lib}"

EX=examples/fibonacci-square
PIL=$EX/pil/build.pil
PIL_ML=$EX/pil/build_ml.pil
PILOUT=$EX/pil/build.pilout
PK=$EX/build/provingKey
ROM=$EX/build/rom.bin
INPUTS=$EX/src/inputs.json
WLIB=target/release/libfibonacci_square.so
LOGDIR=$(mktemp -d)

# Shapes as `fib:mod` pairs; default sweep if none given.
if [ "$#" -eq 0 ]; then SHAPES=(10:8 14:12 18:16 10:6 14:10 18:14 10:4 14:8 18:12); else SHAPES=("$@"); fi
REPS=${REPS:-3}

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

# Run a prove command REPS times and echo "contrib inner total" (ms) of the
# fastest repetition by total. $1 = log prefix; the rest is the command.
bench_prove() {
  local prefix=$1; shift
  local best_tot="" best_c=NA best_i=NA r c i tot
  for r in $(seq 1 "$REPS"); do
    local log="$prefix.$r"
    "$@" >"$log" 2>&1 || { echo "  [rep $r] prove failed — see $log" >&2; tail -3 "$log" >&2; continue; }
    c=$(extract "$log" CALCULATING_CONTRIBUTIONS); i=$(extract "$log" GENERATING_INNER_PROOFS)
    if [ "$c" = NA ] || [ "$i" = NA ]; then continue; fi
    tot=$((c + i))
    if [ -z "$best_tot" ] || [ "$tot" -lt "$best_tot" ]; then best_tot=$tot; best_c=$c; best_i=$i; fi
  done
  echo "$best_c $best_i ${best_tot:-NA}"
}

# Proven area (columns × rows, summed over all instances) from a stats
# SUMMARY block: $1 = stats stdout log, $2 = per-AIR columns field,
# $3 = fib, $4 = mod, $5 = optional extra columns field added on top.
area_from_stats() {
  awk -v field="$2" -v fib="$3" -v mod="$4" -v extra="${5:-}" -F'|' '
    /^-+ SUMMARY -+$/ { insum = 1; next }
    insum && /^-+$/   { insum = 0 }
    insum && NF > 2 {
      name = $2; gsub(/ /, "", name)
      bits = -1; cols = 0; xcols = 0
      for (i = 3; i <= NF; i++) {
        if ($i ~ /(nBits|Bits): */)              { split($i, a, ":"); bits = a[2] + 0 }
        if ($i ~ " " field ": *")                { split($i, a, ":"); cols = a[2] + 0 }
        if (extra != "" && $i ~ " " extra ": *") { split($i, a, ":"); xcols = a[2] + 0 }
      }
      if (bits < 0 || cols == 0) next
      inst = (name == "Module") ? 2 ^ (fib - mod) : 1
      area += (cols + xcols) * 2 ^ bits * inst
    }
    END {
      if (area >= 1e9)      printf "%.2fG", area / 1e9
      else if (area >= 1e6) printf "%.1fM", area / 1e6
      else if (area >= 1e3) printf "%.0fK", area / 1e3
      else if (area > 0)    printf "%d", area
      else                  print "NA"
    }' "$1"
}

# Compile one pilout + regenerate the setup, helpers, witness lib and rom.
# $1 = pil entry file.
regen() {
  $SETUP compile-pil --pil "$1" -I pil2-components/lib/std/pil -o "$PILOUT"
  $SETUP setup -a "$PILOUT" -b "$EX/build"
  $CLI pil-helpers --pilout "$PILOUT" --path "$EX/src" -o
  cargo build --release -p fibonacci-square
  $CLI gen-custom-commits-fixed --witness-lib "$WLIB" --proving-key "$PK" --custom-commits "rom=$ROM"
}

printf '\n=== [CPU] Fibonacci Square AIR — Univariate vs Multilinear ===\n'
printf '\n%-8s %6s %6s %6s | %8s %8s | %-39s | %-39s | %-8s %s\n' \
  "shape" "n_fib" "n_mod" "#inst" "area(u)" "area(ml)" "univariate (ms)" "multilinear (ms)" "spd(in)" "spd(tot)"
printf '%s\n' "-------------------------------------------------------------------------------------------------------------------------------------------------------"

for shape in "${SHAPES[@]}"; do
  fib=${shape%%:*}; mod=${shape##*:}
  ninst=$(( 1 + (1 << (fib - mod)) ))
  log=$LOGDIR/$shape

  # Patch the trace shapes; build_ml.pil requires build.pil, so both legs
  # inherit the patched N's.
  sed -i -E "s/FibonacciSquare\(N: 2\*\*[0-9]+\)/FibonacciSquare(N: 2**$fib)/; s/Module\(N: 2\*\*[0-9]+\)/Module(N: 2**$mod)/" "$PIL"

  # Leg 1: Univariate setup.
  regen "$PIL" >"$log.setup-uni" 2>&1 \
    || { echo "[$shape] univariate setup failed — see $log.setup-uni"; tail -5 "$log.setup-uni"; continue; }
  $SETUP stats -a "$PILOUT" -o "$log.stats-uni.txt" >"$log.stats-uni" 2>&1 || true
  au=$(area_from_stats "$log.stats-uni" Total "$fib" "$mod")
  read -r uc ui ut <<<"$(bench_prove "$log.uni" \
    $CLI prove -w "$WLIB" -k "$PK" -i "$INPUTS" --custom-commits "rom=$ROM" -o "$EX/build/proofs_uni")"

  # Leg 2: Multilinear setup.
  regen "$PIL_ML" >"$log.setup-ml" 2>&1 \
    || { echo "[$shape] Multilinear setup failed — see $log.setup-ml"; tail -5 "$log.setup-ml"; continue; }
  $SETUP stats --multilinear -a "$PILOUT" -o "$log.stats-ml.txt" >"$log.stats-ml" 2>&1 || true
  am=$(area_from_stats "$log.stats-ml" Committed "$fib" "$mod" Fixed)
  read -r mc mi mt <<<"$(bench_prove "$log.ml" \
    $CLI prove-multilinear -v -w "$WLIB" -k "$PK" -i "$INPUTS" --custom-commits "rom=$ROM" -o "$EX/build/proofs_ml")"

  spd_i=$(awk "BEGIN{ if(\"$ui\"!=\"NA\" && \"$mi\"!=\"NA\" && $mi>0) printf \"%.2fx\", $ui/$mi; else print \"-\" }")
  spd_t=$(awk "BEGIN{ if(\"$ut\"!=\"NA\" && \"$mt\"!=\"NA\" && $mt>0) printf \"%.2fx\", $ut/$mt; else print \"-\" }")

  printf '%-8s %6s %6s %6s | %8s %8s | contrib %-6s inner %-6s tot %-7s | contrib %-6s inner %-6s tot %-7s | %-8s %s\n' \
    "$shape" "$((1<<fib))" "$((1<<mod))" "$ninst" "$au" "$am" "$uc" "$ui" "$ut" "$mc" "$mi" "$mt" "$spd_i" "$spd_t"

done
