#!/usr/bin/env bash
# Compile the circuits of examples/test-recursive with the pinned circom, run
# plonk2pil on them in the configurations the test-recursive CI job sets up, and
# in blake3's, then write the sha256 manifest of every input and output. See
# README.md in this directory.
#
# Usage: setup/golden/plonk2pil/generate.sh [OUT_DIR]
#
#   OUT_DIR  Where everything goes (default: $PLONK2PIL_GOLDEN_OUT, else
#            target/golden-plonk2pil). It is wiped first; a non-empty directory
#            that generate.sh did not create is refused rather than wiped.
#
# Layout of OUT_DIR:
#   golden/              the hashed tree:
#     r1cs/<input>.r1cs  each input, in canonical form
#     <config>/          plonk2pil's outputs (setup/stark-recurser/examples/plonk2pil_golden.rs)
#   build/<input>/       circom's output, with the r1cs as circom wrote it
#   logs/                circom and driver logs
#   manifest.sha256      manifest of golden/, same format as ./manifest.sha256
#
# Environment:
#   PLONK2PIL_GOLDEN_BIN  a prebuilt plonk2pil_golden driver to use instead of
#                         building one.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

(($# <= 1)) || golden_die "usage: $0 [OUT_DIR]"
out=${1:-${PLONK2PIL_GOLDEN_OUT:-$PLONK2PIL_DEFAULT_OUT}}

# One line per input: name | source, from the repo root. A .circom source is
# compiled with the flags and libraries gen_recursive_test_setup passes
# (setup/pil2-stark/src/proving_key/recursive_test.rs); these are the fixtures
# `setup-recursive-test -c examples/test-recursive/test.circom --hash <H>`
# resolves to. A .r1cs source, absolute or from the repo root, is an r1cs built
# elsewhere and is used as it is: the way in for a circuit no fixture compiles
# to, such as recursivef's, which needs a vadcop_final verifier.
inputs=(
    "poseidon1|examples/test-recursive/poseidon1/test.circom"
    "poseidon2|examples/test-recursive/poseidon2/test.circom"
    "blake3|examples/test-recursive/blake3/test.circom"
)
circom_libs=(
    setup/stark-recurser/stark2circom/circom_verifier/helper_circuits
    setup/stark-recurser/stark2circom/circom_verifier/circuits.gl
)

# One line per configuration: name | input | setup type | driver flags, which
# give the options as gen_recursive_test_setup builds them for that setup type
# and family. Its airgroup is always Compressor, and its maxDeg is
# max_constraint_degree_for_blowup(recursive_blowup(template, hash)): blowup 2
# for a poseidon compressor and 3 for its aggregation (built as Recursive2),
# 1 and 2 for blake3, hence 5 and 8 (the cap), 3 and 5. The CI job runs the
# poseidon ones; blake3 has a fixture beside them, and its own code path.
configs=(
    "poseidon1-compressor|poseidon1|compressor|--hash Poseidon1 --airgroup Compressor --max-degree 5"
    "poseidon1-aggregation|poseidon1|aggregation|--hash Poseidon1 --airgroup Compressor --max-degree 8"
    "poseidon2-compressor|poseidon2|compressor|--hash Poseidon2 --airgroup Compressor --max-degree 5"
    "poseidon2-aggregation|poseidon2|aggregation|--hash Poseidon2 --airgroup Compressor --max-degree 8"
    "blake3-compressor|blake3|compressor|--hash blake3 --airgroup Compressor --max-degree 3"
    "blake3-aggregation|blake3|aggregation|--hash blake3 --airgroup Compressor --max-degree 5"
)

# --- Output directory ---------------------------------------------------------

golden_fresh_out "$out" .golden-plonk2pil
out=$(cd "$out" && pwd)
mkdir -p "$out/golden/r1cs" "$out/build" "$out/logs"

# --- Tools --------------------------------------------------------------------

# The committed circom, as resolve_circom_exec picks it, never one from PATH: a
# different compiler can compile a different constraint system.
case $(uname -s) in
Darwin) circom=$REPO_ROOT/setup/circom/circom_mac ;;
*) circom=$REPO_ROOT/setup/circom/circom ;;
esac
[[ -x $circom ]] || golden_die "$circom not found"
golden_log "circom: $circom ($("$circom" --version))"

if [[ -n ${PLONK2PIL_GOLDEN_BIN:-} ]]; then
    [[ -f $PLONK2PIL_GOLDEN_BIN && -x $PLONK2PIL_GOLDEN_BIN ]] ||
        golden_die "PLONK2PIL_GOLDEN_BIN=$PLONK2PIL_GOLDEN_BIN is not an executable file"
    driver=$(cd "$(dirname "$PLONK2PIL_GOLDEN_BIN")" && pwd)/$(basename "$PLONK2PIL_GOLDEN_BIN")
else
    # Release: unoptimized, plonk2pil takes about seven times longer on these
    # circuits (some 90 s for the six configurations against 12 s). CPU-only, as
    # the STARK golden's build and for the same reason: the C++ library comes
    # along through proofman-common, and this keeps CUDA out of it.
    golden_log "building the plonk2pil_golden driver"
    (cd "$REPO_ROOT" && cargo build --release -p pil2-stark-recurser --example plonk2pil_golden \
        --features proofman-common/cpu-only)
    target_dir=$(cd "$REPO_ROOT" && cargo metadata --format-version 1 --no-deps |
        grep -o '"target_directory":"[^"\\]*"' | cut -d'"' -f4)
    [[ -n $target_dir ]] || golden_die "could not read the target directory off cargo metadata"
    driver=$target_dir/release/examples/plonk2pil_golden
fi
[[ -x $driver ]] || golden_die "plonk2pil_golden not found at $driver"
golden_log "driver: $driver"

# --- Inputs -------------------------------------------------------------------

# Compile one input into build/<name>/ and put its canonical r1cs in the tree.
# circom writes the same system in a different order from run to run, so it is
# the canonical form that is hashed, and that plonk2pil is run on.
prepare_input() {
    local name=$1 src=$2 r1cs
    case $src in
    *.circom)
        local build=$out/build/$name lib_args=() lib
        mkdir -p "$build"
        for lib in "${circom_libs[@]}"; do
            lib_args+=(-l "$lib")
        done
        golden_log "$name: circom $src"
        golden_run_logged "$out/logs/$name.circom.log" "$circom" --O2 --r1cs --prime goldilocks --c --verbose \
            "${lib_args[@]}" "$src" -o "$build"
        r1cs=$build/$(basename "$src" .circom).r1cs
        ;;
    *.r1cs) r1cs=$src ;;
    *) golden_die "$name: $src is neither a .circom nor a .r1cs file" ;;
    esac
    [[ -f $r1cs ]] || golden_die "$name: $r1cs not found"
    golden_run_logged "$out/logs/$name.canonical.log" "$driver" canonical-r1cs "$r1cs" "$out/golden/r1cs/$name.r1cs"
}

# In parallel: the blake3 circuit alone takes about two minutes to compile.
cd "$REPO_ROOT"
pids=()
names=()
for entry in "${inputs[@]}"; do
    IFS='|' read -r name src <<<"$entry"
    prepare_input "$name" "$src" &
    pids+=($!)
    names+=("$name")
done
failed=()
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed+=("${names[$i]}")
done
((${#failed[@]} == 0)) || golden_die "inputs failed: ${failed[*]} (logs in $out/logs)"

# --- Configurations -----------------------------------------------------------

for entry in "${configs[@]}"; do
    IFS='|' read -r name input setup_type flags <<<"$entry"
    r1cs=$out/golden/r1cs/$input.r1cs
    [[ -f $r1cs ]] || golden_die "$name: no input named $input"
    read -r -a extra <<<"$flags"
    golden_log "$name: plonk2pil --setup-type $setup_type $flags"
    golden_run_logged "$out/logs/$name.plonk2pil.log" "$driver" run "$r1cs" "$out/golden/$name" \
        --setup-type "$setup_type" ${extra[@]+"${extra[@]}"}
done

golden_manifest "$out/golden" >"$out/manifest.sha256"
golden_log "wrote $out/manifest.sha256 ($(wc -l <"$out/manifest.sha256" | tr -d ' ') entries)"
