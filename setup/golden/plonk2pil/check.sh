#!/usr/bin/env bash
# Check plonk2pil's outputs for the test-recursive circuits against
# setup/golden/plonk2pil/manifest.sha256. Exits non-zero, naming each file, if an
# entry is missing, a hash differs or a file is not in the manifest. See
# README.md.
#
# Usage: setup/golden/plonk2pil/check.sh [BUILD_DIR]
#
#   no argument  Regenerate with generate.sh (into $PLONK2PIL_GOLDEN_OUT, else
#                target/golden-plonk2pil), then compare.
#   BUILD_DIR    Compare an existing generate.sh output dir, without regenerating.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

(($# <= 1)) || golden_die "usage: $0 [BUILD_DIR]"
if (($# == 1)); then
    out=$1
else
    out=${PLONK2PIL_GOLDEN_OUT:-$PLONK2PIL_DEFAULT_OUT}
    "$PLONK2PIL_DIR/generate.sh" "$out"
fi

golden_compare "$PLONK2PIL_MANIFEST" "$out" "$out/golden"
