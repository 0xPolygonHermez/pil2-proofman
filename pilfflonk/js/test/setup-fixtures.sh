#!/bin/sh
# Sets up the Fibonacci fixture (M16) and the oracle's Q(ξ) (M14), and runs the JS tests on them
# (M19):
#
#     PIL2C_EXEC=<pil2-compiler>/src/pil.js pilfflonk/js/test/setup-fixtures.sh [dir]
#
# 1. `cargo test --test js_fixtures tau_one_ptau` writes <dir>/tau_one.ptau, τ = 1 (plan N13);
# 2. `proofman-setup compile-pil` compiles pilfflonk/tests/fixtures/fibonacci over BN254 with the
#    compiler PIL2C_EXEC names, which must honour `prime` (the pinned one silently compiles over
#    Goldilocks), and `proofman-setup setup-pilfflonk` writes <dir>/build/provingKey, grouped as it
#    does by default (plan M22);
# 3. `cargo test --test js_fixtures q_at_xi` writes <dir>/q_at_xi.json from the Rust oracle;
# 4. `node --test` runs every JS test with PILFFLONK_JS_FIXTURES=<dir>.
#
# Without <dir>, everything goes to a temporary directory that is removed on exit; it takes about
# 100 KB. The JS dependencies are installed by `npm install` in pilfflonk/js when node_modules/ is
# missing (see fixtures.sh). PILFFLONK_SHPLONK_FIXTURES, if set, passes through to node, so that
# the C++ SHPLONK fixtures' tests (fixtures.sh) run too.
set -eu

: "${PIL2C_EXEC:?must name a pil2com that honours prime, e.g. ../pil2-compiler/src/pil.js}"
export PIL2C_EXEC

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

js_fixtures tau_one_ptau
setup compile-pil -p pilfflonk/tests/fixtures/fibonacci/fibonacci.pil -I ./pil2-components/lib/std/pil \
    -P pilfflonk/tests/fixtures/fibonacci/bn254.json -o "$dir/fibonacci.pilout"
setup setup-pilfflonk -a "$dir/fibonacci.pilout" -b "$dir/build" --powers-of-tau "$dir/tau_one.ptau"
js_fixtures q_at_xi

if [ ! -d "$js/node_modules" ]; then
    (cd "$js" && npm install)
fi

cd "$js"
node --test test/*.test.js
