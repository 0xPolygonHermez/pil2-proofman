#!/usr/bin/env bash
# Compile the 10 CI programs and run the non-recursive STARK setup on each,
# exactly as .github/workflows/ci.yaml does, then write the sha256 manifest of
# every input and output. See README.md in this directory.
#
# Usage: setup/golden/generate.sh [OUT_DIR]
#
#   OUT_DIR  Where everything goes (default: $GOLDEN_OUT, else target/golden-setup).
#            It is wiped first; a non-empty directory that generate.sh did not
#            create is refused rather than wiped.
#
# Layout of OUT_DIR:
#   programs/<name>/   the hashed tree: the .pilout, fixed/ (if any) and provingKey/
#   logs/<name>.*.log  compile and setup logs
#   manifest.sha256    manifest of programs/, same format as setup/golden/manifest.sha256
#
# Environment:
#   PROOFMAN_SETUP  a prebuilt proofman-setup to use instead of building one.
#   SETUP_JOBS      passed through to `proofman-setup setup` (parallel AIRs).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

(($# <= 1)) || golden_die "usage: $0 [OUT_DIR]"
out=${1:-${GOLDEN_OUT:-$GOLDEN_DEFAULT_OUT}}

# One line per program: name | entry .pil | .pilout file name | fixed columns to
# files (yes/no) | extra setup flags. The .pil paths and -I are passed from the
# repo root exactly as ci.yaml passes them (test-fibo, test-std). The one
# deviation: ci.yaml runs fibonacci-square's setup with -r; the recursive setup
# needs circom and is slow, so it is out of the golden's scope.
programs=(
    "fibonacci-square|examples/fibonacci-square/pil/build.pil|build.pilout|yes|--hash Poseidon1"
    "lookup|pil2-components/test/lookup/lookup.pil|build.pilout|no|"
    "permutation|pil2-components/test/permutation/permutation.pil|build.pilout|no|"
    "connection|pil2-components/test/connection/connection.pil|build.pilout|no|"
    "diff_buses|pil2-components/test/diff_buses/diff_buses.pil|build.pilout|no|"
    "direct_update|pil2-components/test/direct_update/direct_update.pil|direct_update.pilout|no|"
    "range_check|pil2-components/test/range_check/build.pil|build.pilout|no|"
    "one_instance|pil2-components/test/one_instance/one_instance.pil|one_instance.pilout|no|"
    "virtual_tables|pil2-components/test/virtual_tables/virtual_tables.pil|build.pilout|no|"
    "simple|pil2-components/test/simple/simple.pil|build.pilout|no|"
)
std_include=./pil2-components/lib/std/pil

# Files the setup writes for every air, whatever its fixed columns.
air_files=(starkinfo.json expressionsinfo.json verifierinfo.json bin verifier.bin)
global_files=(pilout.globalInfo.json pilout.globalConstraints.json pilout.globalConstraints.bin)

# --- Output directory ---------------------------------------------------------

golden_fresh_out "$out" .golden-setup
out=$(cd "$out" && pwd)
mkdir -p "$out/programs" "$out/logs"

# --- Tools --------------------------------------------------------------------

command -v node >/dev/null || golden_die "node not found; pil2-compiler needs Node.js"

# The pinned compiler (setup/pil2-stark/package.json), never a PIL2C_EXEC from
# the environment: a different compiler changes every .pilout.
pil2com=$REPO_ROOT/setup/pil2-stark/node_modules/.bin/pil2com
[[ -x $pil2com ]] || golden_die "$pil2com not found; run \`npm install\` in setup/pil2-stark"
export PIL2C_EXEC=$pil2com
compiler=$(node -p 'require(process.argv[1]).packages["node_modules/pil2-compiler"].resolved' \
    "$REPO_ROOT/setup/pil2-stark/node_modules/.package-lock.json" 2>/dev/null || echo unknown)
golden_log "pil2-compiler: $compiler (node $(node --version))"

if [[ -n ${PROOFMAN_SETUP:-} ]]; then
    [[ -f $PROOFMAN_SETUP && -x $PROOFMAN_SETUP ]] || golden_die "PROOFMAN_SETUP=$PROOFMAN_SETUP is not an executable file"
    # Absolute, since the programs run from the repo root.
    setup_bin=$(cd "$(dirname "$PROOFMAN_SETUP")" && pwd)/$(basename "$PROOFMAN_SETUP")
else
    # CPU-only: the setup computes everything on the CPU, and this keeps the
    # build free of CUDA and of the GPU submodules on machines that have nvcc.
    # The CI runners have no CUDA, so there it is the same build as `cargo run`.
    golden_log "building proofman-setup"
    (cd "$REPO_ROOT" && cargo build -p pil2-stark-setup --bin proofman-setup --features proofman-starks-lib-c/cpu-only)
    target_dir=$(cd "$REPO_ROOT" && cargo metadata --format-version 1 --no-deps |
        node -p 'JSON.parse(require("fs").readFileSync(0, "utf8")).target_directory')
    setup_bin=$target_dir/debug/proofman-setup
fi
[[ -x $setup_bin ]] || golden_die "proofman-setup not found at $setup_bin"
golden_log "proofman-setup: $setup_bin (SETUP_JOBS=${SETUP_JOBS:-<default>})"

# --- Programs -----------------------------------------------------------------

cd "$REPO_ROOT"
for entry in "${programs[@]}"; do
    IFS='|' read -r name pil pilout fixed_to_file setup_flags <<<"$entry"
    dir=$out/programs/$name
    mkdir -p "$dir"

    compile_args=(compile-pil --pil "./$pil" -I "$std_include" -o "$dir/$pilout")
    setup_args=(setup -a "$dir/$pilout" -b "$dir")
    if [[ $fixed_to_file == yes ]]; then
        compile_args+=(-u "$dir/fixed" --fixed-to-file)
        setup_args+=(-u "$dir/fixed")
    fi
    read -r -a extra <<<"$setup_flags"
    setup_args+=(${extra[@]+"${extra[@]}"})

    golden_log "$name: compile-pil"
    golden_run_logged "$out/logs/$name.compile.log" "$setup_bin" "${compile_args[@]}"
    golden_log "$name: setup ${setup_args[*]:1}"
    golden_run_logged "$out/logs/$name.setup.log" "$setup_bin" "${setup_args[@]}"

    # A setup that exits 0 but skips something must not produce a golden.
    for f in "${global_files[@]}"; do
        [[ -f $dir/provingKey/$f ]] || golden_die "$name: setup did not write provingKey/$f"
    done
    air_dirs=("$dir"/provingKey/*/*/airs/*/air)
    [[ -d ${air_dirs[0]} ]] || golden_die "$name: setup wrote no air"
    for air_dir in "${air_dirs[@]}"; do
        air=$(basename "$(dirname "$air_dir")")
        for f in "${air_files[@]}"; do
            [[ -f $air_dir/$air.$f ]] || golden_die "$name: setup did not write $air.$f"
        done
    done
    # Setup warnings (e.g. a missing .fixed) are shown, never swallowed.
    grep -a 'WARN' "$out/logs/$name.setup.log" | sed "s|^|golden: $name: |" >&2 || true
done

golden_manifest "$out/programs" >"$out/manifest.sha256"
golden_log "wrote $out/manifest.sha256 ($(wc -l <"$out/manifest.sha256" | tr -d ' ') entries)"
