# Shared helpers for generate.sh and check.sh. Sourced, not executed.

GOLDEN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$GOLDEN_DIR/../.." && pwd)
GOLDEN_MANIFEST=$GOLDEN_DIR/manifest.sha256
GOLDEN_DEFAULT_OUT=$REPO_ROOT/target/golden-setup

golden_log() { echo "golden: $*" >&2; }
golden_die() {
    echo "golden: error: $*" >&2
    exit 1
}

# Print the manifest of every file under <out>/programs: `<sha256>  <path>`,
# paths relative to that directory, byte-sorted. This is the only place the
# manifest format is defined.
golden_manifest() {
    local tree=$1/programs
    [[ -d $tree ]] || golden_die "$tree does not exist; is $1 a generate.sh output dir?"

    local sha=(sha256sum)
    command -v sha256sum >/dev/null || sha=(shasum -a 256)

    local files=() f
    while IFS= read -r f; do
        files+=("$f")
    done < <(cd "$tree" && find . -type f | sed 's|^\./||' | LC_ALL=C sort)
    ((${#files[@]} > 0)) || golden_die "$tree holds no files"

    (cd "$tree" && "${sha[@]}" "${files[@]}")
}
