#!/bin/bash
# Detect the host GPU compute capability and write CUDA_ARCH = sm_XX to CudaArch.mk.
# Used by the Makefile when neither CUDA_GENCODE_FLAGS, CUDA_ARCHS nor CUDA_ARCH
# are set externally.
#
# On failure exits 1 — the Makefile then falls back to building for all major
# architectures (CUDA_MAJOR_ARCHS). nvidia-smi is the only probe, 

set -eu

OUT_FILE="CudaArch.mk"

# Drop any previous result up front: the file must only ever hold the outcome
# of the CURRENT run, never a stale arch cached from another machine.
rm -f "$OUT_FILE"

CAP=""
if command -v nvidia-smi >/dev/null 2>&1; then
    SMI_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n1 \
        | tr -d ' .')
    case "$SMI_CAP" in
        ''|*[!0-9]*) ;; # empty or non-numeric (e.g. driver up but no GPU) — ignore
        *) CAP=$SMI_CAP ;;
    esac
fi
if [ -z "$CAP" ]; then
    echo "[configure] nvidia-smi probe failed — no GPU arch detected, falling back to major archs." >&2
    exit 1
fi

# Cap to the highest arch the installed nvcc supports.
NVCC_ARCHS=""
if command -v nvcc >/dev/null 2>&1; then
    if nvcc --list-gpu-code >/dev/null 2>&1; then
        # Use the more reliable --list-gpu-code option
        NVCC_ARCHS=$(nvcc --list-gpu-code | grep -oE "sm_[0-9]+" | sed 's/sm_//g' | sort -n -u)
    else
        # Fallback to parsing help text
        NVCC_ARCHS=$(nvcc --help | grep -oE "sm_[0-9]+" | sed 's/sm_//g' | sort -n -u)
    fi
fi

SELECTED_CAP=0
for arch in $NVCC_ARCHS; do
    if [ "$arch" -le "$CAP" ]; then
        SELECTED_CAP=$arch
    fi
done
if [ "$SELECTED_CAP" -eq 0 ]; then
    echo "[configure] No compatible CUDA architecture found for capability $CAP!" >&2
    exit 1
fi
if [ "$SELECTED_CAP" -lt "$CAP" ]; then
    echo "Warning: CUDA capability $CAP detected, capping to highest supported sm_$SELECTED_CAP."
fi
echo "CUDA_ARCH = sm_$SELECTED_CAP" > "$OUT_FILE"
echo "[configure] Host GPU compute capability $CAP → CUDA_ARCH = sm_$SELECTED_CAP"
