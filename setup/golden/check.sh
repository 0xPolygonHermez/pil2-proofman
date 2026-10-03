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

golden_compare "$GOLDEN_MANIFEST" "$out" "$out/programs"
