#!/usr/bin/env bash
# The GPU check of the prover (pilfflonk/docs/performance.md#gpu): proves one provingKey/ and
# witness directory on the CPU and on the GPU, with the same --insecure-blinding-seed, checks that
# proof.json and publics.json are the same byte for byte, and prints the time of each phase on each,
# from the prover's PILFFLONK_* timers (-vv, pilfflonk/docs/performance.md#method). It needs neither
# Node.js nor the compiler: the key and the witness come from elsewhere (bench.sh with BENCH_KEEP=1,
# or setup-pilfflonk and pilfflonk_bench_inputs), and the proofs are verified there, with
# `pilfflonk verify`.
#
#   pilfflonk/bench/gpu_check.sh <provingKey> <witness dir> [<out dir>]
#
# From the repository root, after building the CLI with CUDA (no feature cpu-only; nvcc on the PATH
# or at /usr/local/cuda/bin):
#
#   cargo build --release -p proofman-cli
#
# Environment:
#   PILFFLONK_CLI      the proofman-cli to run (default target/release/proofman-cli): one built with
#                      CUDA, which proves on the CPU without --gpu and on the GPU with it
#   GPU_CHECK_SEED     the blinding seed, 64 hexadecimal digits (default a fixed one)
#   GPU_CHECK_REPEATS  proofs on each device (default 1); the times are their medians, and every
#                      proof must be the first CPU proof
#   OMP_NUM_THREADS    the prover's CPU threads (default: every CPU)
#
# It writes to <out dir> (default target/tmp/pilfflonk-gpu-check): each proof in {cpu,gpu}.<n>/, its
# log in {cpu,gpu}.<n>.log, and times.tsv, a line per proof (device, run, wall seconds and the
# phases). It exits 0 if every proof is the same, and 1 if a proof differs or a run fails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# phases_of and PHASES: bench.sh's parser of the timers.
# shellcheck source=pilfflonk/bench/bench.sh
source "$ROOT/pilfflonk/bench/bench.sh"

CLI="${PILFFLONK_CLI:-$ROOT/target/release/proofman-cli}"
SEED="${GPU_CHECK_SEED:-6d34330067707500636865636b0000000000000000000000000000000000002a}"
REPEATS="${GPU_CHECK_REPEATS:-1}"
# bench.sh's phases, the copy of the SRS's powers to the GPU (PILFFLONK_GPU_SRS) after its load, and
# the start of a proof (pilfflonk/docs/performance.md#the-start-of-a-proof): the Rust reading of the
# key's files (PILFFLONK_KEY_FILES), what is left of CUDA's initialisation (PILFFLONK_GPU_INIT), the
# witness's trace read while the key loads (PILFFLONK_WITNESS_READ), and the proof's files.
CHECK_PHASES="key_files gpu_init ${PHASES/load_srs/load_srs gpu_srs} witness_read write_proof"

fail() {
    echo "gpu_check.sh: $*" >&2
    exit 1
}

[ $# -ge 2 ] && [ $# -le 3 ] || fail "usage: gpu_check.sh <provingKey> <witness dir> [<out dir>]"
KEY="$1"
WITNESS="$2"
OUT="${3:-$ROOT/target/tmp/pilfflonk-gpu-check}"
[ -d "$KEY" ] || fail "no provingKey/ at $KEY"
[ -d "$WITNESS" ] || fail "no witness directory at $WITNESS"
[ -x "$CLI" ] || fail "no proofman-cli at $CLI: cargo build --release -p proofman-cli (with CUDA)"
case "$REPEATS" in '' | *[!0-9]* | 0) fail "GPU_CHECK_REPEATS must be a positive number" ;; esac
mkdir -p "$OUT"

echo "== $(date -u +%FT%TZ) on $(hostname), $(nproc --all) CPUs, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader | sed 's/^/== GPU: /'
fi
echo "== key $KEY, witness $WITNESS, seed $SEED, $REPEATS run(s) on each device"

TIMES="$OUT/times.tsv"
printf 'device\trun\twall_s\t%s\n' "$(tr ' ' '\t' <<<"$CHECK_PHASES")" >"$TIMES"

# Proves on `device` (cpu or gpu), run `n`: the proof to $OUT/<device>.<n>, the log beside it, and
# its line in times.tsv.
prove_on() {
    local device="$1" n="$2" flags=()
    [ "$device" = gpu ] && flags=(--gpu)
    local dir="$OUT/$device.$n" log="$OUT/$device.$n.log" start end status=0
    rm -rf "$dir"
    start="$(date +%s.%N)"
    "$CLI" pilfflonk prove -k "$KEY" --witness "$WITNESS" -o "$dir" --insecure-blinding-seed "$SEED" \
        ${flags[@]+"${flags[@]}"} -vv >"$log" 2>&1 || status=$?
    end="$(date +%s.%N)"
    [ "$status" = 0 ] || fail "pilfflonk prove ${flags[*]} failed ($status): $log"
    local wall
    wall="$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.2f", b - a }')"
    printf '%s\t%s\t%s\t%s\n' "$device" "$n" "$wall" "$(PHASES="$CHECK_PHASES" phases_of "$log")" >>"$TIMES"
    echo "   $device #$n: $wall s ($log)"
}

for n in $(seq "$REPEATS"); do
    prove_on cpu "$n"
    prove_on gpu "$n"
done

# Every proof against the first CPU one.
status=0
for device in cpu gpu; do
    for n in $(seq "$REPEATS"); do
        for file in proof.json publics.json; do
            if ! cmp -s "$OUT/cpu.1/$file" "$OUT/$device.$n/$file"; then
                echo "DIFFERENT: $device.$n/$file is not cpu.1/$file" >&2
                status=1
            fi
        done
    done
done

# The median of each phase on each device, and how many times faster the GPU is.
awk -F'\t' '
    NR == 1 { for (i = 3; i <= NF; i++) name[i] = $i; ncols = NF; next }
    { for (i = 3; i <= NF; i++) if ($i != "-") { k = $1 SUBSEP i; v[k, ++count[k]] = $i } }
    function median(d, i,    k, n, j, l, t, a) {
        k = d SUBSEP i; n = count[k]
        if (n == 0) return "-"
        for (j = 1; j <= n; j++) a[j] = v[k, j] + 0
        for (j = 2; j <= n; j++) for (l = j; l > 1 && a[l - 1] > a[l]; l--) { t = a[l]; a[l] = a[l - 1]; a[l - 1] = t }
        return n % 2 ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
    }
    END {
        printf "%-20s %10s %10s %9s\n", "phase (s)", "cpu", "gpu", "cpu/gpu"
        for (i = 3; i <= ncols; i++) {
            c = median("cpu", i); g = median("gpu", i)
            if (c == "-" && g == "-") continue
            ratio = (c != "-" && g != "-" && g > 0) ? sprintf("%.2f", c / g) : "-"
            printf "%-20s %10s %10s %9s\n", (name[i] == "wall_s" ? "wall" : name[i]), c, g, ratio
        }
    }' "$TIMES"

if [ "$status" = 0 ]; then
    echo "SAME: every proof.json and publics.json is cpu.1's, byte for byte ($OUT)"
fi
exit "$status"
