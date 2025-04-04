#!/bin/bash

set -e

echo "Launching all tests..."

test_pipeline() {
    NAME=$1             # Test name (e.g. simple)
    BASE=$2             # Base directory (e.g. ./pil2-components/test/simple)
    SO_NAME=$3          # Name of the .so file (e.g. libsimple.so)
    SETUP_ONLY=$4       # Whether to run only until the setup phase

    BUILD="$BASE/build"
    PIL_FILE="$BASE/$NAME.pil"
    SRC="$BASE/rs/src"
    PROVING_KEY="$BUILD/provingKey"
    PILOUT_FILE="$BUILD/$NAME.pilout"
    LIB="./target/debug/lib${SO_NAME}.so"
    LOG="$BUILD/$NAME.log"

    echo "  [$NAME] Starting..."

    # Start clean
    if [ "$SETUP_ONLY" != "true" ]; then
        rm -rf "$BUILD" && mkdir "$BUILD"
    fi

    {
        node --max-old-space-size=65536 ../pil2-compiler/src/pil.js "$PIL_FILE" \
            --include ./pil2-components/lib/std/pil \
            --output "$PILOUT_FILE"

        node --max-old-space-size=65536 ../pil2-proofman-js/src/main_setup.js \
            --airout "$PILOUT_FILE" \
            --builddir "$BUILD"

        if [ "$SETUP_ONLY" != "true" ]; then
            cargo run --bin proofman-cli check-setup \
                --proving-key "$PROVING_KEY" \

            cargo run --bin proofman-cli pil-helpers \
                --pilout "$PILOUT_FILE" \
                --path "$SRC" -o

            cargo build

            cargo run --bin proofman-cli verify-constraints \
                --witness-lib "$LIB" \
                --proving-key "$PROVING_KEY"

            cargo run --bin proofman-cli prove \
                --witness-lib "$LIB" \
                --proving-key "$PROVING_KEY" \
                --output-dir "$BUILD/proofs"
        fi

    } >"$LOG" 2>&1 && echo "  [$NAME] ✅" || echo "  [$NAME] ❌ (see $LOG)"
}

# Run tests
test_pipeline "simple" "./pil2-components/test/simple" "simple"
test_pipeline "connection" "./pil2-components/test/std/connection" "connection"
test_pipeline "diff_buses" "./pil2-components/test/std/diff_buses" "diff_buses"
test_pipeline "direct_update" "./pil2-components/test/std/direct_update" "direct_update"
test_pipeline "lookup" "./pil2-components/test/std/lookup" "lookup"
test_pipeline "permutation" "./pil2-components/test/std/permutation" "permutation"
test_pipeline "build" "./pil2-components/test/std/range_check" "range_check"

test_pipeline "array_size" "./pil2-components/test/std/special" "array_size" "true"
test_pipeline "direct_optimizations" "./pil2-components/test/std/special" "direct_optimizations" "true"
test_pipeline "expr_optimizations" "./pil2-components/test/std/special" "expr_optimizations" "true"
test_pipeline "intermediate_sums" "./pil2-components/test/std/special" "intermediate_sums" "true"
test_pipeline "table" "./pil2-components/test/std/special" "table" "true"

echo "✅ All tests completed."