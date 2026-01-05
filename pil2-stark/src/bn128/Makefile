# BN128 GPU Makefile (Linux only)

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build_gpu

NVCC := /usr/local/cuda/bin/nvcc
include CudaArch.mk

CUDA_SRC_DIRS := ./src/ffigpu ./tests
CUDA_INC_FLAGS := $(addprefix -I,$(CUDA_SRC_DIRS))
NVCCFLAGS := -D__USE_CUDA__ $(CUDA_INC_FLAGS) -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -I./../../external/sppark/ff

CXX := g++
AS := nasm
ASFLAGS := -f elf64

BLST_PATH :=
CXXFLAGS := -std=c++17 -Wall -O3 -fPIC -D__USE_ASSEMBLY__
CPPFLAGS := -I./src/ffiasm
LDFLAGS_BENCH := -lgmp -z noexecstack

ifdef BLST
CXXFLAGS += -D__BLST__
CPPFLAGS += -I$(BLST_PATH)/bindings
LDFLAGS_BENCH += -L$(BLST_PATH) -lblst
endif

BENCHMARK_FLAGS := $(shell pkg-config --cflags benchmark 2>/dev/null)
BENCHMARK_LIBS := $(shell pkg-config --libs benchmark 2>/dev/null)

# Bench build dir and target
BENCH_BUILD_DIR := $(BUILD_DIR)/benchs
BENCH_TARGET := benchscpu
BENCH_SRCS := ./benchs/bench.cpp ./src/ffiasm/fr.cpp ./src/ffiasm/fq.cpp
BENCH_ASMS := ./src/ffiasm/fr.asm ./src/ffiasm/fq.asm
BENCH_OBJS := $(addprefix $(BENCH_BUILD_DIR)/,$(notdir $(BENCH_SRCS:.cpp=.cpp.o) $(BENCH_ASMS:.asm=.asm.o)))

# ----------------------
# Targets
# ----------------------
.PHONY: all testsgpu clean help benchscpu

all: testsgpu

benchscpu: $(BENCH_TARGET)

$(BENCH_TARGET): $(BENCH_OBJS)
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS_BENCH) $(BENCHMARK_LIBS)

$(BENCH_BUILD_DIR)/%.cpp.o: ./benchs/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BENCHMARK_FLAGS) -c $< -o $@

$(BENCH_BUILD_DIR)/%.cpp.o: ./src/ffiasm/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(BENCHMARK_FLAGS) -c $< -o $@

$(BENCH_BUILD_DIR)/%.asm.o: ./src/ffiasm/%.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@


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
	rm -rf $(BUILD_DIR) $(BUILD_DIR_GPU)
	rm -f benchscpu testsgpu

help:
	@echo "make [target]"
	@echo "  all (default) - Build GPU tests"
	@echo "  benchscpu     - Build CPU benchmarks (output: ./benchscpu)"
	@echo "  clean         - Remove build artifacts"

MKDIR_P ?= mkdir -p
