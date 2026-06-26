#!/bin/bash
# Cubin-in-setup: generate each AIR's expression kernel(s) for a provingKey and compile EACH into its
# own self-contained shared library next to that AIR's .bin: <provingKey>/<air-dir>/<basename>.exps.so.
#
# Usage: build_exps.sh <provingKey-dir> [archspec] [keep-dir]
#
#   archspec — CUDA architectures to build for; same convention as the pil2-stark Makefile:
#     auto | (omitted)  auto-detect the host GPU; fall back to 'major' if detection fails
#     major             all major archs since Ampere (80 86 89 90 100 120)
#     89,120 | sm_120   one or more archs, comma-separated; each may be a bare number or sm_NN
#                       (the two are interchangeable: '89,120' == 'sm_89,sm_120', '120' == 'sm_120')
#     Builds SASS (code=sm_NN) for every listed arch + PTX (code=compute_NN) for the newest arch of each
#     Blackwell lineage (datacenter sm_100-11x vs the rest) for forward compatibility — same as the Makefile.
#   keep-dir — if given, the generated source/objects are written here and kept (to inspect the
#              generated kernels); default is a temp dir, removed on exit. (To pass keep-dir, give an
#              explicit archspec, e.g. 'auto'.)
set -euo pipefail
PK="${1:?usage: build_exps.sh <provingKey-dir> [archspec] [keep-dir]}"
ARCHSPEC="${2:-auto}"
KEEP_DIR="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STARK="$(cd "$HERE/../../../.." && pwd)"
SRC="$STARK/src"

# ---- resolve the CUDA arch list (same convention as the Makefile) ----
MAJOR_ARCHS="80 86 89 90 100 120"
spec_lc="$(printf '%s' "$ARCHSPEC" | tr 'A-Z' 'a-z')"
if [ "$spec_lc" = "major" ]; then
  ARCH_LIST="$MAJOR_ARCHS"
elif [ "$spec_lc" = "auto" ]; then
  det="$(__nvcc_device_query 2>/dev/null | head -1 | tr -dc '0-9' || true)"
  if [ -n "$det" ]; then ARCH_LIST="$det"; echo "[build_exps] auto-detected host GPU arch: sm_$det";
  else ARCH_LIST="$MAJOR_ARCHS"; echo "[build_exps] arch auto-detect failed -> building 'major': $MAJOR_ARCHS"; fi
else
  ARCH_LIST="$(printf '%s' "$ARCHSPEC" | tr ',' ' ' | sed 's/sm_//g')"
fi
for a in $ARCH_LIST; do case "$a" in ''|*[!0-9]*)
  echo "[build_exps] bad arch '$a' (expected 'auto', 'major', or integers like 89 / '89,120' / sm_120)"; exit 1;; esac; done
ARCH_LIST="$(printf '%s\n' $ARCH_LIST | sort -n -u)"
# gencode: SASS for every arch + PTX for the newest arch of each Blackwell lineage (sm_100-11x
# datacenter vs the rest) — the lineages aren't cross-compatible, so each carries its own forward-PTX.
GENCODE=""
for a in $ARCH_LIST; do GENCODE="$GENCODE -gencode arch=compute_${a},code=sm_${a}"; done
ptx_dc="$(printf '%s\n' $ARCH_LIST | grep -E '^(10|11)' | tail -1 || true)"
ptx_rest="$(printf '%s\n' $ARCH_LIST | grep -vE '^(10|11)' | tail -1 || true)"
for p in $ptx_dc $ptx_rest; do GENCODE="$GENCODE -gencode arch=compute_${p},code=compute_${p}"; done

# ---- work dir: temp (removed on exit) by default, or a keep-dir to retain the generated code ----
if [ -n "$KEEP_DIR" ]; then WORK="$KEEP_DIR"; mkdir -p "$WORK"; echo "[build_exps] keeping generated code in $WORK";
else WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT; fi

# Comprehensive include set (mirrors the real pil2-stark gen-kernel compile).
INCS="-I$WORK \
 -I$STARK/external/sppark/ff -I$STARK/external/sppark -I$STARK/external/sppark/util \
 -I$STARK/external/sppark/ec -I$STARK/external/blst/src \
 -I$SRC -I$SRC/utils -I$SRC/goldilocks -I$SRC/goldilocks/utils -I$SRC/goldilocks/src \
 -I$SRC/binfile -I$SRC/XKCP -I$SRC/bctree -I$SRC/config -I$SRC/api \
 -I$SRC/bn128 -I$SRC/bn128/src -I$SRC/bn128/src/ffiasm -I$SRC/bn128/src/poseidon \
 -I$SRC/bn128/src/msm -I$SRC/bn128/src/ntt -I$SRC/bn128/src/poseidon2 -I$SRC/bn128/src/curve \
 -I$SRC/starkpil -I$SRC/starkpil/expressions -I$SRC/starkpil/transcript -I$SRC/starkpil/merkleTree -I$SRC/starkpil/fri \
 -I$SRC/rapidsnark -I$SRC/rapidsnark/polynomial -I$SRC/fflonk_setup \
 -I/usr/include -I/usr/local/include -I/usr/lib/x86_64-linux-gnu/openmpi/include"
DEFS="-D__USE_CUDA__ -DGL64_PARTIALLY_REDUCED -D__AVX2__ -D__USE_ASSEMBLY__ -D__ADX__ \
 -DFEATURE_BN254 -DUSE_CUDA_GRAPH -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX -D__USE_MPI_RMA__ \
 --diag-suppress 114"
CCFLAGS="-Xcompiler -fPIC -Xcompiler -mavx2 -std=c++17 -O3 $GENCODE $DEFS $INCS"
# nvcc command for the auto-tuner: compiles one kernel to an object so cuobjdump can read its spill.
# Full gencode => (a) the no-spill check covers every target arch (cuobjdump reports STACK per-arch)
# and (b) the resulting object is exactly what the final .so links — no recompile.
NVCC_CMD="nvcc -c $CCFLAGS"

echo "[build_exps] generating kernels for $PK (archs: $ARCH_LIST; auto-tuning chunk for zero spill)"
python3 -u "$HERE/gen_exps.py" "$PK" "$WORK" --nvcc-cmd "$NVCC_CMD" 2>&1 | sed 's/^/[gen] /'

NJ="$(nproc)"
LOG="$WORK/gen.log"
NK="$(wc -l < "$LOG" 2>/dev/null || echo 0)"
echo "[build_exps] linking $NK per-AIR .exps.so (archs: $ARCH_LIST) — parallel -j$NJ ..."

while IFS=$'\t' read -r reldir base sym slots; do
  [ -n "${sym:-}" ] || continue
  dest="$PK/$reldir/${base}.exps.so"
  objs=""; for o in "$WORK/gen_$sym.o" "$WORK"/gen_${sym}_c*.o; do [ -f "$o" ] && objs="$objs $o"; done
  if [ -n "$objs" ]; then printf '%s\0' "nvcc -shared $GENCODE $objs -lcudart -o $dest"
  else
    cus=""; for c in "$WORK/gen_$sym.cu" "$WORK"/gen_${sym}_c*.cu; do [ -f "$c" ] && cus="$cus $c"; done
    printf '%s\0' "nvcc -shared $CCFLAGS $cus -lcudart -o $dest"
  fi
done < "$LOG" | xargs -0 -P "$NJ" -I{} bash -c '{}' 2>"$WORK/cc.err" || {
  echo "[build_exps] LINK/COMPILE ERRORS:"; cat "$WORK/cc.err"; exit 1; }

echo "[build_exps] done -> $NK kernels, one <base>.exps.so per AIR under $PK"
