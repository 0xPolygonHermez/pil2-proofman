#!/bin/sh
# Sets up the Fibonacci fixture, proves it and takes the oracle's Q(ξ), and runs the JS tests on
# them (pilfflonk/docs/README.md#tests):
#
#     PIL2C_EXEC=<pil2-compiler>/src/pil.js pilfflonk/js/test/setup-fixtures.sh [dir]
#
# 1. `proofman-setup compile-pil` compiles pilfflonk/tests/fixtures/fibonacci over BN128 with the
#    compiler PIL2C_EXEC names, which must have `--field` (the pinned one silently compiles over
#    Goldilocks), and `proofman-setup setup-pilfflonk` writes <dir>/build/provingKey, grouped as it
#    does by default, with the official ptau of power 12 (pilfflonk/tests/download-ptau.sh) in
#    $PILFFLONK_PTAU_DIR;
# 2. `cargo test --test js_fixtures fibonacci_proof` writes <dir>/proof.json and <dir>/publics.json,
#    a proof of the prover;
# 3. `cargo test --test js_fixtures q_at_xi` writes <dir>/q_at_xi.json from the Rust oracle;
# 4. `node --test` runs every JS test with PILFFLONK_JS_FIXTURES=<dir>.
#
# Without <dir>, everything goes to a temporary directory that is removed on exit; it takes about
# 200 KB. The JS dependencies are installed by `npm install` in pilfflonk/js when node_modules/ is
# missing (see fixtures.sh). PILFFLONK_SHPLONK_FIXTURES, if set, passes through to node, so that
# the C++ SHPLONK fixtures' tests (fixtures.sh) run too.
set -eu

: "${PIL2C_EXEC:?must name a pil2com that has --field, e.g. ../pil2-compiler/src/pil.js}"
export PIL2C_EXEC
ptau="${PILFFLONK_PTAU_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pilfflonk/ptau}/powersOfTau28_hez_final_12.ptau"
[ -f "$ptau" ] || {
    echo "setup-fixtures.sh: no $ptau: run pilfflonk/tests/download-ptau.sh 12" >&2
    exit 1
}

js=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$js/../.." && pwd)

if [ $# -ge 1 ]; then
    mkdir -p "$1"
    dir=$(cd "$1" && pwd)
else
    dir=$(mktemp -d "${TMPDIR:-/tmp}/pilfflonk-js-fixtures.XXXXXX")
    trap 'rm -rf "$dir"' EXIT
fi
export PILFFLONK_JS_FIXTURES="$dir"

cd "$repo"
js_fixtures() {
    cargo test -q -p pilfflonk-setup --features proofman-starks-lib-c/cpu-only --test js_fixtures -- --ignored --exact "$1"
}
setup() {
    cargo run -q --features proofman-starks-lib-c/cpu-only --bin proofman-setup -- "$@"
}

setup compile-pil -p pilfflonk/tests/fixtures/fibonacci/fibonacci.pil -I ./pil2-components/lib/std/pil \
    --field bn128 -o "$dir/fibonacci.pilout"
setup setup-pilfflonk -a "$dir/fibonacci.pilout" -b "$dir/build" --powers-of-tau "$ptau"
js_fixtures fibonacci_proof
js_fixtures q_at_xi

if [ ! -d "$js/node_modules" ]; then
    (cd "$js" && npm install)
fi

cd "$js"
node --test test/*.test.js
