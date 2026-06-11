#!/bin/bash

# sudo apt-get -y install libgtest-dev libomp-dev libgmp-dev libbenchmark-dev

touch CudaArch.mk

cd utils
make
cd ..

# Probe the GPU's compute capability: deviceQuery first, nvidia-smi as a
# fallback 
CAP=""
if [ -e utils/deviceQuery ]; then
    CAP=`./utils/deviceQuery | grep "CUDA Capability" | head -n 1 | tr -d ' ' | cut -d ':' -f 2 | tr -d '.'`
else
    echo "Error building CUDA deviceQuery!"
fi
if [ -z "$CAP" ] && command -v nvidia-smi >/dev/null 2>&1; then
    SMI_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')
    case "$SMI_CAP" in
        ''|*[!0-9]*) ;; # empty or non-numeric (e.g. driver up but no GPU) — ignore
        *)
            echo "deviceQuery unavailable, using nvidia-smi: compute capability $SMI_CAP"
            CAP=$SMI_CAP
            ;;
    esac
fi
if [ -z "$CAP" ]; then
    echo "Unable to get CUDA capability on this system!"
    exit 1
fi
# Try to use nvcc --list-gpu-code if available, otherwise fallback to parsing help
if command -v nvcc >/dev/null 2>&1; then
    # nvcc exists — now check capabilities
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
    echo "No compatible CUDA architecture found for capability $CAP!"
    exit 1
fi
if [ "$SELECTED_CAP" -lt "$CAP" ]; then
    echo "Warning: CUDA capability $CAP detected, capping to highest supported sm_$SELECTED_CAP."
fi
echo "CUDA_ARCH = sm_$SELECTED_CAP" > CudaArch.mk
