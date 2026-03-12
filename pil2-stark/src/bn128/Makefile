# BN128 GPU Makefile (Linux only)

# ----------------------
# Directory Configuration
# ----------------------
BUILD_DIR     := ./build
BUILD_DIR_GPU := ./build_gpu
SRC_DIR       := ./src
MKDIR_P       ?= mkdir -p

# ----------------------
# Toolchain
# ----------------------
NVCC := /usr/local/cuda/bin/nvcc
CXX  := g++
AS   := nasm

include CudaArch.mk

# ----------------------
# Compiler Flags
# ----------------------
CUDA_SRC_DIRS  := $(SRC_DIR)/ffigpu $(SRC_DIR)/msm $(SRC_DIR)/ntt ./tests $(SRC_DIR)/poseidon $(SRC_DIR)/poseidon2 $(SRC_DIR)/curve $(SRC_DIR)/../../utils ./../goldilocks/src
CUDA_INC_FLAGS := $(addprefix -I,$(CUDA_SRC_DIRS))
BLST_INC       := -I./../../external/blst/src
BLST_LIB       := ./../../external/blst/libblst.a
SPPARK_INC     := -I./../../external/sppark -I./../../external/sppark/ff -I./../../external/sppark/util -I./../../external/sppark/ec
SPPARK_UTIL    := ./../../external/sppark/util/all_gpus.cpp
# Use -iquote for ffiasm so nested includes (fq.hpp, fr.hpp, etc.) resolve correctly
FFIASM_QUOTE   := -Xcompiler -iquote$(SRC_DIR)/ffiasm -Xcompiler -iquote$(SRC_DIR)
NVCCFLAGS      := -D__USE_CUDA__ -DFEATURE_BN254 $(SPPARK_INC) $(BLST_INC) $(CUDA_INC_FLAGS) $(FFIASM_QUOTE) \
                  -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -Xcompiler -D__ADX__ \
                  -D__USE_ASSEMBLY__ -arch=$(CUDA_ARCH) --diag-suppress 114

CXXFLAGS := -std=c++17 -Wall -O3 -fPIC -D__USE_ASSEMBLY__ -fopenmp
CPPFLAGS := -I$(SRC_DIR) -I$(SRC_DIR)/ffiasm -I$(SRC_DIR)/poseidon -I$(SRC_DIR)/poseidon2 -I./../utils -I./../goldilocks/src
ASFLAGS  := -f elf64 -g -F dwarf

# Fix linker warning about executable stack
LDFLAGS_COMMON := -z noexecstack -fopenmp

# Optional BLST support
ifdef BLST
  CXXFLAGS += -D__BLST__
  CPPFLAGS += -I$(BLST_PATH)/bindings
  LDFLAGS_COMMON += -L$(BLST_PATH) -lblst
endif

# Benchmark flags (optional)
BENCHMARK_FLAGS := $(shell pkg-config --cflags benchmark 2>/dev/null)
BENCHMARK_LIBS  := $(shell pkg-config --libs benchmark 2>/dev/null)

# ----------------------
# Source Files
# ----------------------
FFIASM_SRCS := $(SRC_DIR)/ffiasm/fr.cpp $(SRC_DIR)/ffiasm/fq.cpp $(SRC_DIR)/ffiasm/naf.cpp $(SRC_DIR)/ffiasm/misc.cpp $(SRC_DIR)/ffiasm/splitparstr.cpp $(SRC_DIR)/ffiasm/alt_bn128.cpp
FFIASM_ASMS := $(SRC_DIR)/ffiasm/fr.asm $(SRC_DIR)/ffiasm/fq.asm
FFIASM_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.cpp.o,$(FFIASM_SRCS)) \
               $(patsubst $(SRC_DIR)/%.asm,$(BUILD_DIR)/%.asm.o,$(FFIASM_ASMS))

# CPU test objects
TESTS_CPU_OBJS    := $(BUILD_DIR)/tests/tests.o
POSEIDON_CPU_OBJS := $(BUILD_DIR)/src/poseidon/poseidon_bn128.o
POSEIDON2_CPU_OBJS := $(BUILD_DIR)/src/poseidon2/poseidon2_bn128.o

# GPU test objects
POSEIDON_GPU_OBJS  := $(BUILD_DIR_GPU)/src/poseidon/poseidon_bn128.cu.o
POSEIDON2_GPU_OBJS := $(BUILD_DIR_GPU)/src/poseidon2/poseidon2_bn128.cu.o
MSM_GPU_OBJS       := $(BUILD_DIR_GPU)/src/msm/msm_bn128.cu.o
NTT_GPU_OBJS       := $(BUILD_DIR_GPU)/src/ntt/ntt_bn128.cu.o
SPPARK_GPU_OBJS    := $(BUILD_DIR_GPU)/sppark/all_gpus.o

# Benchmark CPU
BENCH_BUILD_DIR := $(BUILD_DIR)/benchs
BENCH_TARGET    := benchscpu
BENCH_SRCS      := ./benchs/bench.cpp $(FFIASM_SRCS)
BENCH_OBJS      := $(BENCH_BUILD_DIR)/bench.cpp.o \
                   $(patsubst $(SRC_DIR)/ffiasm/%.cpp,$(BENCH_BUILD_DIR)/%.cpp.o,$(FFIASM_SRCS)) \
                   $(patsubst $(SRC_DIR)/ffiasm/%.asm,$(BENCH_BUILD_DIR)/%.asm.o,$(FFIASM_ASMS))

# Benchmark GPU
BENCH_GPU_BUILD_DIR := $(BUILD_DIR_GPU)/benchs
BENCH_GPU_TARGET    := benchsgpu

# ----------------------
# Targets
# ----------------------
.PHONY: all testsgpu testscpu benchscpu benchsgpu clean help

all: testsgpu

# GPU tests
testsgpu: $(BUILD_DIR_GPU)/tests/tests.cu.o $(POSEIDON_GPU_OBJS) $(POSEIDON2_GPU_OBJS) $(MSM_GPU_OBJS) $(NTT_GPU_OBJS) $(SPPARK_GPU_OBJS) $(FFIASM_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(BLST_LIB) -lgtest -lgmp

# CPU tests
testscpu: $(TESTS_CPU_OBJS) $(POSEIDON_CPU_OBJS) $(POSEIDON2_CPU_OBJS) $(FFIASM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lgtest -lgtest_main -lgmp -lpthread $(LDFLAGS_COMMON)

# Benchmarks CPU
benchscpu: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_OBJS) $(POSEIDON_CPU_OBJS) $(POSEIDON2_CPU_OBJS)
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lgmp $(LDFLAGS_COMMON) $(BENCHMARK_LIBS)

# Benchmarks GPU
benchsgpu: $(BENCH_GPU_BUILD_DIR)/bench.cu.o $(POSEIDON_GPU_OBJS) $(POSEIDON2_GPU_OBJS) $(MSM_GPU_OBJS) $(NTT_GPU_OBJS) $(SPPARK_GPU_OBJS) $(FFIASM_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(BLST_LIB) -lgmp $(BENCHMARK_LIBS)

# ----------------------
# Pattern Rules - GPU
# ----------------------
$(BUILD_DIR_GPU)/tests/%.cu.o: tests/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I./tests -c $< -o $@

$(BUILD_DIR_GPU)/src/poseidon/%.cu.o: $(SRC_DIR)/poseidon/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/src/poseidon2/%.cu.o: $(SRC_DIR)/poseidon2/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/src/msm/%.cu.o: $(SRC_DIR)/msm/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/src/ntt/%.cu.o: $(SRC_DIR)/ntt/%.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I./../../external/sppark/ntt -c $< -o $@

# sppark utility (compile as CUDA to get proper GPU support)
$(BUILD_DIR_GPU)/sppark/all_gpus.o: $(SPPARK_UTIL)
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

# GPU benchmarks
$(BENCH_GPU_BUILD_DIR)/bench.cu.o: ./benchs/bench.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I./benchs -c $< -o $@

# ----------------------
# Pattern Rules - CPU
# ----------------------
$(BUILD_DIR)/tests/%.o: ./tests/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/src/poseidon/%.o: $(SRC_DIR)/poseidon/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/src/poseidon2/%.o: $(SRC_DIR)/poseidon2/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# FFIASM (C++ and ASM)
$(BUILD_DIR)/ffiasm/%.cpp.o: $(SRC_DIR)/ffiasm/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/ffiasm/%.asm.o: $(SRC_DIR)/ffiasm/%.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# ----------------------
# Pattern Rules - Benchmarks
# ----------------------
$(BENCH_BUILD_DIR)/bench.cpp.o: ./benchs/bench.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BENCHMARK_FLAGS) -c $< -o $@

$(BENCH_BUILD_DIR)/%.cpp.o: $(SRC_DIR)/ffiasm/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BENCHMARK_FLAGS) -c $< -o $@

$(BENCH_BUILD_DIR)/%.asm.o: $(SRC_DIR)/ffiasm/%.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# ----------------------
# Utilities
# ----------------------
clean:
	rm -rf $(BUILD_DIR) $(BUILD_DIR_GPU)
	rm -f benchscpu benchsgpu testsgpu testscpu

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all (default) - Build GPU tests"
	@echo "  testsgpu      - Build GPU unit tests (output: ./testsgpu)"
	@echo "  testscpu      - Build CPU unit tests (output: ./testscpu)"
	@echo "  benchscpu     - Build CPU benchmarks (output: ./benchscpu)"
	@echo "  benchsgpu     - Build GPU benchmarks (output: ./benchsgpu)"
	@echo "  clean         - Remove build artifacts"
