# Shared helpers of the golden scripts: generate.sh and check.sh here, and those
# of plonk2pil/. Sourced, not executed.

GOLDEN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$GOLDEN_DIR/../.." && pwd)
# The STARK golden's; plonk2pil/lib.sh sets its own.
GOLDEN_MANIFEST=$GOLDEN_DIR/manifest.sha256
GOLDEN_DEFAULT_OUT=$REPO_ROOT/target/golden-setup

golden_log() { echo "golden: $*" >&2; }
golden_die() {
    echo "golden: error: $*" >&2
    exit 1
}

# Make <out> an empty directory holding only <marker>. A directory that already
# holds <marker> was made here and is wiped; any other non-empty one is refused
# rather than wiped.
golden_fresh_out() {
    local out=$1 marker=$2
    if [[ -e $out ]]; then
        [[ -d $out ]] || golden_die "$out exists and is not a directory"
        if [[ -f $out/$marker ]]; then
            rm -rf -- "$out"
        elif [[ -n $(ls -A "$out") ]]; then
            golden_die "$out is not empty and was not created by generate.sh; refusing to wipe it"
        fi
    fi
    mkdir -p "$out"
    touch "$out/$marker"
}

# Run a command with its output in a log; on failure show the log's tail and stop.
golden_run_logged() {
    local log=$1
    shift
    if ! "$@" >"$log" 2>&1; then
        tail -n 50 "$log" >&2
        golden_die "command failed (full log: $log): $*"
    fi
}

# Print the manifest of every file under <tree>: `<sha256>  <path>`, paths
# relative to <tree>, byte-sorted. This is the only place the manifest format is
# defined.
golden_manifest() {
    local tree=$1
    [[ -d $tree ]] || golden_die "$tree does not exist; is its parent a generate.sh output dir?"

    local sha=(sha256sum)
    command -v sha256sum >/dev/null || sha=(shasum -a 256)

    local files=() f
    while IFS= read -r f; do
        files+=("$f")
    done < <(cd "$tree" && find . -type f | sed 's|^\./||' | LC_ALL=C sort)
    ((${#files[@]} > 0)) || golden_die "$tree holds no files"

    (cd "$tree" && "${sha[@]}" "${files[@]}")
}

# Compare the files under <tree>, of the generate.sh output dir <out>, with
# <manifest>. Names each entry that is MISSING, DIFFERS or UNEXPECTED, and exits
# 1 if there is one, 2 if the manifest is malformed.
golden_compare() {
    local manifest=$1 out=$2 tree=$3 actual
    [[ -s $manifest ]] || golden_die "$manifest is missing or empty"
    actual=$(golden_manifest "$tree")

    # A malformed or duplicated manifest line is an error, not something to skip.
    awk -v manifest="$manifest" '
        function bad(msg) { printf "golden: error: %s:%d: %s\n", manifest, FNR, msg; malformed = 1 }
        function parse(line) {
            hash = substr(line, 1, 64)
            path = substr(line, 67)
            return length(hash) == 64 && hash !~ /[^0-9a-f]/ && substr(line, 65, 2) == "  " && path != ""
        }
        FNR == NR {
            if (!parse($0)) { bad("not a `<sha256>  <path>` line"); next }
            if (path in want) { bad("duplicate entry " path); next }
            want[path] = hash; order[++n] = path
            next
        }
        parse($0) { have[path] = hash }
        END {
            if (malformed) exit 2
            for (i = 1; i <= n; i++) {
                p = order[i]
                if (!(p in have)) { print "golden: MISSING     " p; missing++ }
                else if (have[p] != want[p]) { print "golden: DIFFERS     " p; differs++ }
            }
            for (p in have) if (!(p in want)) extra[++m] = p
            # Sorted for a stable report (no asort in POSIX awk; the list is short).
            for (i = 2; i <= m; i++) for (j = i; j > 1 && extra[j - 1] > extra[j]; j--) {
                t = extra[j]; extra[j] = extra[j - 1]; extra[j - 1] = t
            }
            for (i = 1; i <= m; i++) print "golden: UNEXPECTED  " extra[i]
            if (missing + differs + m > 0) {
                printf "golden: FAIL: %d missing, %d differ, %d unexpected (%d entries in the manifest)\n", missing, differs, m, n
                exit 1
            }
            printf "golden: OK: all %d entries match\n", n
        }
    ' "$manifest" <(printf '%s\n' "$actual") >&2 || {
        local status=$?
        ((status == 1)) && golden_log "outputs are in $out; if the change is intended, replace $manifest with $out/manifest.sha256 from a fresh generate.sh run"
        exit "$status"
    }
}
