# Shared by plonk2pil/generate.sh and plonk2pil/check.sh, on top of ../lib.sh.
# Sourced, not executed.

source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"
PLONK2PIL_DIR=$GOLDEN_DIR/plonk2pil
PLONK2PIL_MANIFEST=$PLONK2PIL_DIR/manifest.sha256
PLONK2PIL_DEFAULT_OUT=$REPO_ROOT/target/golden-plonk2pil
