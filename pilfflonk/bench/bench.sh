#!/usr/bin/env bash
# The pilfflonk benchmark (pilfflonk/docs/performance.md#cpu): the time and the peak memory of the
# setup, of the prover, phase by phase, and of the JS verifier, for a benchmark program at
# N = 2^bits. Every proof it times must verify, or the run stops.
#
#   pilfflonk/bench/bench.sh ptau <powers>          writes $BENCH_PTAU: a test ptau (its τ is
#                                                   public: never for a real proof)
#   pilfflonk/bench/bench.sh run <program> <bits>…  measures each size of pilfflonk/bench/<program>.pil
#                                                   (fibonacci, all_sum, all_prod)
#   pilfflonk/bench/bench.sh summary                the median and the spread of every point
#
# From the repository root, after building the release binaries it runs:
#
#   cargo build --release --features proofman-starks-lib-c/cpu-only \
#       --bin proofman-cli --bin proofman-setup --example pilfflonk_bench_inputs
#
# Environment:
#   PIL2C_EXEC      pil2com (`run`), a compiler that has `--field`
#                   (pilfflonk/docs/README.md#compile-pil)
#   BENCH_DIR       where everything goes (default target/tmp/pilfflonk-bench); the results are
#                   $BENCH_DIR/{compile,setup,prove}.tsv
#   BENCH_PTAU      the ptau (default $BENCH_DIR/bench.ptau): at least as many powers as the
#                   largest layout degree of the keys (the setup refuses fewer)
#   BENCH_PACKING   the layouts, "packed" (the default of setup-pilfflonk), "nopacking" or both
#                   (default "packed")
#   BENCH_REPEATS   runs of each setup and each proof (default 2)
#   BENCH_KEEP      1 keeps each size's keys, witness and proofs (default: deleted once the
#                   size is measured, as they are large). The pilouts stay in $BENCH_DIR/pilouts,
#                   and later runs of the same program and size take them from there
#                   instead of compiling again
#   BENCH_NODE_HEAP_MB  pil2com's heap limit (node --max-old-space-size), in MB (default 250000:
#                   `all` at 2^22 takes 20 GB, and the machine is shared)
#   OMP_NUM_THREADS the prover's and the setup's threads, recorded with each run (default: every CPU)
#
# Each run records the 1- and 5-minute load averages when it starts. The prover runs with -vv, which
# prints the C++ timers (TimerStart/TimerStopAndLog, PILFFLONK_*), and a sampler reads its VmRSS
# every 0.2 s: the peak of each phase. The pilout carries the fixed columns (fixed-to-file does not
# work over BN254, pilfflonk/docs/README.md#compile-pil), so pil2com's time and memory grow with N
# too: `run` records them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${BENCH_DIR:-$ROOT/target/tmp/pilfflonk-bench}"
BENCH_PTAU="${BENCH_PTAU:-$BENCH_DIR/bench.ptau}"
BENCH_PACKING="${BENCH_PACKING:-packed}"
BENCH_REPEATS="${BENCH_REPEATS:-2}"
BENCH_KEEP="${BENCH_KEEP:-0}"
BENCH_NODE_HEAP_MB="${BENCH_NODE_HEAP_MB:-250000}"
THREADS="${OMP_NUM_THREADS:-$(nproc)}"

CLI="$ROOT/target/release/proofman-cli"
SETUP="$ROOT/target/release/proofman-setup"
INPUTS="$ROOT/target/release/examples/pilfflonk_bench_inputs"

die() {
    echo "bench.sh: $*" >&2
    exit 1
}

# The seconds and the peak RSS (kB) of a `/usr/bin/time -v` report.
wall_of() {
    awk -F': ' '/Elapsed \(wall clock\)/ {
        n = split($2, p, ":"); s = 0
        for (i = 1; i <= n; i++) s = s * 60 + p[i]
        printf "%.2f", s
    }' "$1"
}
rss_of() {
    awk -F': ' '/Maximum resident set size/ { print $2 }' "$1"
}
loads() {
    awk '{ print $1 "\t" $2 }' /proc/loadavg
}

# Runs a command under `/usr/bin/time -v`, its report to $1 and its output to $2; returns its status.
timed() {
    local report="$1" log="$2"
    shift 2
    local status=0
    /usr/bin/time -v -o "$report" "$@" >"$log" 2>&1 || status=$?
    return "$status"
}

# The VmRSS of the prover (the child of `/usr/bin/time`, pid $1) every 0.2 s, with the last
# top-level PILFFLONK_* marker of its log ($2) at that moment and the last marker of any timer:
# `<kB>\t<phase>\t<marker>` lines. The phases are those of the prover's steps
# (pilfflonk/docs/protocol.md#proof-sequence): the key's loading, each stage, Q, the evaluations and
# the opening.
TOP_MARKERS='(-->|<--) PILFFLONK_(LOAD_SRS|LOAD_AIRS|FIXED_COMMITMENTS|STAGE_[0-9]+|Q|EVALUATIONS|OPEN) '
sample_rss() {
    local timer="$1" log="$2" pid=""
    for _ in $(seq 100); do
        pid="$(pgrep -P "$timer" | head -1 || true)"
        [ -n "$pid" ] && break
        sleep 0.05
    done
    [ -n "$pid" ] || return 0
    while [ -e "/proc/$pid" ]; do
        local rss phase marker
        rss="$(awk '/^VmRSS/ { print $2 }' "/proc/$pid/status" 2>/dev/null || true)"
        phase="$(grep -oE "$TOP_MARKERS" "$log" 2>/dev/null | tail -1 || true)"
        marker="$(grep -oE '(-->|<--) PILFFLONK_[A-Z0-9_]+' "$log" 2>/dev/null | tail -1 || true)"
        [ -n "$rss" ] && printf '%s\t%s\t%s\n' "$rss" "${phase:-start}" "${marker:-start}"
        sleep 0.2
    done
}

# The C++ timers of a prover log, phase by phase (seconds), in the order of PHASES; `witness` is
# the time from the prover's "Reading the witness" to its "Committing stage 1" (the instance's
# C++ part, `instance`, included), from the timestamps of those two lines.
PHASES="load_srs load_airs fixed_intt fixed_commitments witness instance s1_hints s1_im s1_intt s1_msm s1_total s2_hints s2_im \
s2_intt s2_msm s2_total q_extend q_domain q_evaluate q_interpolate q_commit q_total evaluations shplonk_w \
shplonk_commit_w shplonk_wp shplonk_commit_wp open"
phases_of() {
    awk -v phases="$PHASES" '
        function seconds(line) {
            match(line, /T[0-9][0-9]:[0-9][0-9]:[0-9][0-9][.0-9]*/)
            split(substr(line, RSTART + 1, RLENGTH - 1), hms, ":")
            return hms[1] * 3600 + hms[2] * 60 + hms[3]
        }
        /INFO: ··· Reading the witness/ { read = seconds($0) }
        /INFO: ··· Committing stage 1$/ { if (read != "") t["witness"] = seconds($0) - read }
        /--> PILFFLONK_STAGE_[0-9]+ starting/ { match($0, /STAGE_[0-9]+/); sec = "s" substr($0, RSTART + 6, RLENGTH - 6) }
        /--> PILFFLONK_Q starting/ { sec = "q" }
        /<-- PILFFLONK_[A-Z0-9_]+ done: / {
            match($0, /PILFFLONK_[A-Z0-9_]+/); name = substr($0, RSTART + 10, RLENGTH - 10); secs = $(NF - 1)
            # Q is evaluated part by part: Q_EXTEND_<part> and so on add up.
            if (name ~ /^Q_[A-Z]+_[0-9]+$/) sub(/_[0-9]+$/, "", name)
            if (name ~ /^INTT_/) t[sec "_intt"] += secs
            else if (name ~ /^COMMIT_/) t[sec "_msm"] += secs
            else if (name ~ /^HINT_COLUMNS_/) t[sec "_hints"] += secs
            else if (name ~ /^IM_POLS_/) t[sec "_im"] += secs
            else if (name ~ /^STAGE_/) t[sec "_total"] += secs
            else if (name == "Q") t["q_total"] += secs
            else t[tolower(name)] += secs
        }
        END {
            n = split(phases, p, " ")
            for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? "\t" : ""), (p[i] in t ? sprintf("%.3f", t[p[i]]) : "-")
            printf "\n"
        }' "$1"
}

# The peak RSS (kB) of each phase of a sampler's output, in the order of RSS_PHASES: the key's
# loading (and what comes before it), the witness (read after the key), each stage (and the
# transcript after it), each step of Q, the evaluations (and the squeeze before them) and the
# opening.
RSS_PHASES="rss_load rss_witness rss_s1 rss_s2 rss_q_extend rss_q_evaluate rss_q_interpolate rss_q_commit \
rss_evaluations rss_open"
rss_phases_of() {
    awk -F'\t' -v phases="$RSS_PHASES" '
        {
            phase = $2; marker = $3; g = ""
            if (phase == "<-- PILFFLONK_FIXED_COMMITMENTS ") g = "rss_witness"
            else if (phase == "start" || phase ~ /LOAD_SRS|LOAD_AIRS|FIXED_COMMITMENTS/) g = "rss_load"
            else if (phase ~ /STAGE_[0-9]+/) { match(phase, /STAGE_[0-9]+/); g = "rss_s" substr(phase, RSTART + 6, RLENGTH - 6) }
            else if (phase == "--> PILFFLONK_Q ") {
                if (marker ~ /Q_DOMAIN|Q_EVALUATE/) g = "rss_q_evaluate"
                else if (marker ~ /Q_INTERPOLATE/) g = "rss_q_interpolate"
                else if (marker ~ /Q_COMMIT/) g = "rss_q_commit"
                else g = "rss_q_extend"
            }
            else if (phase == "<-- PILFFLONK_Q " || phase ~ /EVALUATIONS/) g = "rss_evaluations"
            else if (phase ~ /OPEN/) g = "rss_open"
            if (g != "" && $1 + 0 > r[g] + 0) r[g] = $1
        }
        END {
            n = split(phases, p, " ")
            for (i = 1; i <= n; i++) printf "%s%s", (i > 1 ? "\t" : ""), (p[i] in r ? r[p[i]] : "-")
            printf "\n"
        }' "$1"
}

# What the key says (tab-separated): nBits, nBitsExt (of Q's bound and of the columns with the most
# blinding, as proofman_pilfflonk::degrees; pilfflonk/docs/protocol.md#degrees), qDeg, the number of
# f, the SRS powers (the largest f degree) and the sizes of the .const and the SRS in bytes.
key_info() {
    local pk="$1" info srs const
    info="$(find "$pk" -name '*.pilfflonkinfo.json' | head -1)"
    srs="$(find "$pk" -name 'pilfflonk.srs.bin' | head -1)"
    const="$(find "$pk" -name '*.const' | head -1)"
    node -e '
        const info = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
        const n = 2 ** info.nBits;
        const committed = info.layout.filter((f) => f.stage >= 1 && f.stage <= info.nStages);
        const maxO = Math.max(...committed.map((f) => f.offsets.length));
        const q = info.qDeg * n + (info.qDeg + 1) * maxO + 1;
        const ext = Math.ceil(Math.log2(Math.max(q, n + maxO + 1)));
        const srs = Math.max(...info.layout.map((f) => f.degree));
        console.log([info.nBits, ext, info.qDeg, info.layout.length, srs].join("\t"));
    ' "$info" | tr -d '\n'
    printf '\t%s\t%s\n' "$(stat -c %s "$const")" "$(stat -c %s "$srs")"
}

# The size of a proof's bytes (pilfflonk/docs/formats.md#proof): 64 per G1 point and 32 per scalar
# of its JSON view.
proof_bytes() {
    node -e '
        const p = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
        console.log(64 * Object.keys(p.polynomials).length + 32 * Object.keys(p.evaluations).length);
    ' "$1"
}

header() {
    local file="$1" columns="$2"
    [ -s "$file" ] || printf '%s\n' "$columns" >"$file"
}

run_size() {
    local program="$1" bits="$2" generator
    case "$program" in
    fibonacci) generator=fibonacci ;;
    all_sum | all_prod) generator=all ;;
    *) die "no benchmark program $program: fibonacci, all_sum or all_prod" ;;
    esac
    local dir="$BENCH_DIR/$program/$bits"
    rm -rf "$dir"
    mkdir -p "$dir"
    echo "== $program, N = 2^$bits, $THREADS threads"

    # The pilout, with N from the compiler: pil2com compiles over BN254 (--field) with the define.
    # Its name is its file's stem, the program's; a later run of the size takes it as it is.
    local pilout="$BENCH_DIR/pilouts/$bits/$program.pilout" load status
    if [ ! -f "$pilout" ]; then
        # Written apart, and moved into place once whole.
        rm -rf "$pilout.tmp"
        mkdir -p "$pilout.tmp"
        load="$(loads)"
        status=0
        timed "$dir/compile.time" "$dir/compile.log" node --max-old-space-size="$BENCH_NODE_HEAP_MB" "$PIL2C_EXEC" \
            "$ROOT/pilfflonk/bench/$program.pil" -I "$ROOT/pil2-components/lib/std/pil" --field bn254 -D "BENCH_BITS=$bits" \
            -o "$pilout.tmp/$program.pilout" || status=$?
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$program" "$bits" "$load" "$status" "$(wall_of "$dir/compile.time")" \
            "$(rss_of "$dir/compile.time")" "$(stat -c %s "$pilout.tmp/$program.pilout" 2>/dev/null || echo -)" \
            >>"$BENCH_DIR/compile.tsv"
        [ "$status" = 0 ] || die "pil2com failed ($status): $dir/compile.log"
        mv "$pilout.tmp/$program.pilout" "$pilout"
        rmdir "$pilout.tmp"
    fi

    local packing rep
    for packing in $BENCH_PACKING; do
        local flags=()
        case "$packing" in
        packed) ;;
        nopacking) flags=(--no-packing) ;;
        *) die "no packing $packing: packed or nopacking" ;;
        esac
        local build="$dir/build_$packing" pk="$dir/build_$packing/provingKey"
        for rep in $(seq "$BENCH_REPEATS"); do
            rm -rf "$build"
            load="$(loads)"
            status=0
            timed "$dir/setup_$packing.$rep.time" "$dir/setup_$packing.$rep.log" "$SETUP" setup-pilfflonk \
                -a "$pilout" -b "$build" --powers-of-tau "$BENCH_PTAU" "${flags[@]}" || status=$?
            [ "$status" = 0 ] || die "setup-pilfflonk failed ($status): $dir/setup_$packing.$rep.log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$program" "$bits" "$packing" "$THREADS" "$rep" "$load" \
                "$status" "$(wall_of "$dir/setup_$packing.$rep.time")" "$(rss_of "$dir/setup_$packing.$rep.time")" \
                "$(key_info "$pk")" >>"$BENCH_DIR/setup.tsv"
        done

        # One witness for both layouts: they have the same stage-1 columns.
        if [ ! -d "$dir/witness" ]; then
            timed "$dir/witness.time" "$dir/witness.log" "$INPUTS" witness "$generator" "$pk" "$dir/witness" ||
                die "the witness failed: $dir/witness.log"
        fi

        local vkey
        vkey="$(find "$pk" -name pilfflonk.vkey.json | head -1)"
        for rep in $(seq "$BENCH_REPEATS"); do
            local out="$dir/proof_$packing.$rep" log="$dir/prove_$packing.$rep.log"
            local report="$dir/prove_$packing.$rep.time" samples="$dir/prove_$packing.$rep.rss"
            rm -rf "$out"
            load="$(loads)"
            status=0
            /usr/bin/time -v -o "$report" "$CLI" pilfflonk prove -k "$pk" --witness "$dir/witness" -o "$out" -vv \
                >"$log" 2>&1 &
            local timer=$!
            sample_rss "$timer" "$log" >"$samples" &
            local sampler=$!
            wait "$timer" || status=$?
            wait "$sampler" || true
            [ "$status" = 0 ] || die "pilfflonk prove failed ($status): $log"

            local vstatus=0
            timed "$dir/verify_$packing.$rep.time" "$dir/verify_$packing.$rep.log" "$CLI" pilfflonk verify \
                "$vkey" "$out/publics.json" "$out/proof.json" || vstatus=$?
            [ "$vstatus" = 0 ] || die "the proof does not verify ($vstatus): $dir/verify_$packing.$rep.log"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$program" "$bits" "$packing" \
                "$THREADS" "$rep" "$load" "$status" "$(wall_of "$report")" "$(rss_of "$report")" "$vstatus" \
                "$(wall_of "$dir/verify_$packing.$rep.time")" "$(rss_of "$dir/verify_$packing.$rep.time")" \
                "$(proof_bytes "$out/proof.json")" "$(phases_of "$log")" "$(rss_phases_of "$samples")" \
                >>"$BENCH_DIR/prove.tsv"
            echo "   $packing #$rep: prove $(wall_of "$report") s, $(rss_of "$report") kB; verify $(wall_of "$dir/verify_$packing.$rep.time") s"
        done
        [ "$BENCH_KEEP" = 1 ] || rm -rf "$build"
    done
    [ "$BENCH_KEEP" = 1 ] || rm -rf "$dir/witness" "$dir"/proof_*
}

main() {
    [ $# -ge 1 ] || die "usage: bench.sh ptau <powers> | run <program> <bits>... | summary"
    mkdir -p "$BENCH_DIR"
    header "$BENCH_DIR/compile.tsv" "$(printf 'program\tbits\tload1\tload5\texit\twall_s\trss_kb\tpilout_bytes')"
    header "$BENCH_DIR/setup.tsv" "$(printf 'program\tbits\tpacking\tthreads\trep\tload1\tload5\texit\twall_s\trss_kb\tnBits\tnBitsExt\tqDeg\tnF\tsrs_powers\tconst_bytes\tsrs_bytes')"
    header "$BENCH_DIR/prove.tsv" "$(printf 'program\tbits\tpacking\tthreads\trep\tload1\tload5\texit\twall_s\trss_kb\tverify_exit\tverify_s\tverify_rss_kb\tproof_bytes\t%s\t%s' \
        "$(tr ' ' '\t' <<<"$PHASES")" "$(tr ' ' '\t' <<<"$RSS_PHASES")")"
    case "$1" in
    ptau)
        [ $# = 2 ] || die "usage: bench.sh ptau <powers>"
        [ -x "$INPUTS" ] || die "build $INPUTS first (see the header of this script)"
        timed "$BENCH_PTAU.time" "$BENCH_PTAU.log" "$INPUTS" ptau "$2" "$BENCH_PTAU" || die "the ptau failed: $BENCH_PTAU.log"
        echo "wrote $BENCH_PTAU ($2 powers) in $(wall_of "$BENCH_PTAU.time") s, $(rss_of "$BENCH_PTAU.time") kB"
        ;;
    run)
        [ $# -ge 3 ] || die "usage: bench.sh run <program> <bits>..."
        [ -n "${PIL2C_EXEC:-}" ] || die "PIL2C_EXEC must name pil2com"
        [ -f "$BENCH_PTAU" ] || die "no ptau at $BENCH_PTAU: bench.sh ptau <powers>"
        for binary in "$CLI" "$SETUP" "$INPUTS"; do
            [ -x "$binary" ] || die "build $binary first (see the header of this script)"
        done
        local program="$2"
        shift 2
        for bits in "$@"; do
            run_size "$program" "$bits"
        done
        ;;
    summary)
        node "$ROOT/pilfflonk/bench/summary.mjs" "$BENCH_DIR"
        ;;
    *) die "usage: bench.sh ptau <powers> | run <program> <bits>... | summary" ;;
    esac
}

# Run when executed; sourced (gpu_check.sh), only its helpers are defined.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
