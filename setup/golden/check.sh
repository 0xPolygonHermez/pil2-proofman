#!/usr/bin/env bash
# Check the STARK setup outputs of the 10 CI programs against
# setup/golden/manifest.sha256. Exits non-zero, naming each file, if an entry
# is missing, a hash differs or a file is not in the manifest. See README.md.
#
# Usage: setup/golden/check.sh [BUILD_DIR]
#
#   no argument  Regenerate with generate.sh (into $GOLDEN_OUT, else
#                target/golden-setup), then compare.
#   BUILD_DIR    Compare an existing generate.sh output dir, without regenerating.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

(($# <= 1)) || golden_die "usage: $0 [BUILD_DIR]"
if (($# == 1)); then
    out=$1
else
    out=${GOLDEN_OUT:-$GOLDEN_DEFAULT_OUT}
    "$GOLDEN_DIR/generate.sh" "$out"
fi

[[ -s $GOLDEN_MANIFEST ]] || golden_die "$GOLDEN_MANIFEST is missing or empty"
actual=$(golden_manifest "$out")

# A malformed or duplicated manifest line is an error, not something to skip.
awk -v manifest="$GOLDEN_MANIFEST" '
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
' "$GOLDEN_MANIFEST" <(printf '%s\n' "$actual") >&2 || {
    status=$?
    ((status == 1)) && golden_log "outputs are in $out; if the change is intended, replace $GOLDEN_MANIFEST with $out/manifest.sha256 from a fresh generate.sh run"
    exit "$status"
}
