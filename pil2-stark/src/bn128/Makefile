# BN128 GPU Makefile - CUDA only (Linux)

BUILD_DIR_GPU := ./build_gpu

# CUDA configuration
NVCC := /usr/local/cuda/bin/nvcc
include CudaArch.mk

CUDA_SRC_DIRS := ./src/ffigpu ./tests
CUDA_INC_FLAGS := $(addprefix -I,$(CUDA_SRC_DIRS))
NVCCFLAGS := -D__USE_CUDA__ $(CUDA_INC_FLAGS) -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -I./../../external/sppark/ff

# ----------------------
# Targets
# ----------------------
.PHONY: all testsgpu clean help

all: testsgpu

testsgpu: $(BUILD_DIR_GPU)/tests/tests.cu.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lgtest -lgmp

$(BUILD_DIR_GPU)/tests/%.cu.o: tests/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I./tests -c $< --output-file $@

$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	g++ -D__USE_CUDA__ -mavx2 -std=c++17 -O3 -fopenmp $(CUDA_INC_FLAGS) -c $< -o $@

# ----------------------
# Clean
# ----------------------
clean:
	rm -rf ./build_gpu testsgpu

help:
	@echo "make [target]"
	@echo "  all (default) - Build GPU tests"
	@echo "  clean         - Remove build artifacts"

MKDIR_P ?= mkdir -p
