#!/bin/sh
# Regenerates the C++ prover's SHPLONK fixtures and runs the JS tests on them
# (pilfflonk/docs/README.md#tests):
#
#     pilfflonk/js/test/fixtures.sh [dir]
#
# `make -C pil2-stark pilfflonk_test` builds pilfflonkTest if needed and runs the whole C++ suite,
# which writes <dir>/shplonk_<case>.json; then `node --test` runs every JS test with
# PILFFLONK_SHPLONK_FIXTURES=<dir>. Without <dir>, the fixtures go to a temporary directory that is
# removed on exit. OMP_NUM_THREADS passes through: on a loaded machine the suite's FFTs run faster
# with fewer threads (e.g. OMP_NUM_THREADS=16).
#
# The JS dependencies (package.json: ffjavascript and @noble/hashes, as snarkjs 0.7.6 has them) are
# installed by `npm install` in pilfflonk/js, which this script runs when node_modules/ is missing;
# node_modules/ and package-lock.json are git-ignored.
set -eu

js=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$js/../.." && pwd)

if [ $# -ge 1 ]; then
    mkdir -p "$1"
    dir=$(cd "$1" && pwd)
else
    dir=$(mktemp -d "${TMPDIR:-/tmp}/pilfflonk-shplonk-fixtures.XXXXXX")
    trap 'rm -rf "$dir"' EXIT
fi

PILFFLONK_SHPLONK_FIXTURES="$dir" make -C "$repo/pil2-stark" pilfflonk_test

if [ ! -d "$js/node_modules" ]; then
    (cd "$js" && npm install)
fi

cd "$js"
PILFFLONK_SHPLONK_FIXTURES="$dir" node --test test/*.test.js
