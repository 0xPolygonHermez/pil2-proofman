#!/bin/bash

# sudo apt-get -y install libgtest-dev libomp-dev libgmp-dev libbenchmark-dev

touch CudaArch.mk

cd utils
make
cd ..
if ! [ -e utils/deviceQuery ]; then
    echo "Error buidling CUDA deviceQuery!"
    exit 1
fi

CAP=`./utils/deviceQuery | grep "CUDA Capability" | head -n 1 | tr -d ' ' | cut -d ':' -f 2 | tr -d '.'`
if [ -z "$CAP" ]; then
    echo "Unable to get CUDA capability on this system!"
    exit 1
fi
# Find all supported sm_XX archs from nvcc, sort, and pick the highest <= CAP
NVCC_ARCHS=$(nvcc --help | grep -oE "sm_[0-9]+" | sort -u | sed 's/sm_//g' | sort -n)
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
