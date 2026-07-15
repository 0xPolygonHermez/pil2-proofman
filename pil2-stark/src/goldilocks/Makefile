# Directories
BUILD_DIR := ./build
SRC_DIR := ./src
TEST_DIR := ./tests

# Platform detection for compiler and library paths
UNAME_S := $(shell uname -s)

# ----------------------
# Platform-specific configuration 
# ----------------------

ifeq ($(UNAME_S),Darwin)
    HOMEBREW_PREFIX := $(shell brew --prefix)
    CXX      = clang++
    SDKPATH  = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    CXXFLAGS = -std=c++17 -Wall -pthread -O3 -I./src -I$(HOMEBREW_PREFIX)/include -I$(HOMEBREW_PREFIX)/opt/libomp/include -isysroot $(SDKPATH) -Xpreprocessor -fopenmp
    CPPFLAGS = -isysroot $(SDKPATH)
    LDFLAGS  = -L$(HOMEBREW_PREFIX)/lib -L$(HOMEBREW_PREFIX)/opt/libomp/lib -L$(SDKPATH)/usr/lib
    LDLIBS   = -lomp -lbenchmark -lbenchmark_main -lgtest -lgtest_main -lgmpxx -lgmp -lpthread
endif

ifeq ($(UNAME_S),Linux)
    CXX      = g++
    CXXFLAGS = -std=c++17 -Wall -pthread -O3 -I./src -I/usr/include -I/usr/local/include -fopenmp -mavx2 -I/../utils
    CPPFLAGS =
    LDFLAGS  = -L/usr/lib -L/usr/local/lib
    LDLIBS_COMMON = -fopenmp -lgmpxx -lgmp -lpthread -lstdc++
    LDLIBS_TEST   = $(LDLIBS_COMMON) -lgtest -lgtest_main
    LDLIBS_BENCH  = $(LDLIBS_COMMON) -lbenchmark -lbenchmark_main

	  CXXFLAGS += -D__AVX2__
  	CXXFLAGS += -D__USE_ASSEMBLY__

    # Standalone tests/benches always exercise the Poseidon2 path. Force-undef
    # STARK_POSEIDON1 in case the environment inherited it from the outer build, like this both paths are reached
    CXXFLAGS += -U STARK_POSEIDON1
    # Auto-enable AVX512 if the host CPU supports it
    AVX512_SUPPORTED := $(shell grep -m1 avx512f /proc/cpuinfo)
    ifneq ($(AVX512_SUPPORTED),)
      CXXFLAGS += -mavx512f -D__AVX512__
    endif

    # CUDA/GPU build options (Linux only)
    NVCC := /usr/local/cuda/bin/nvcc
    include CudaArch.mk
    BUILD_DIR_GPU := ./build_gpu
    CUDA_SRC_DIRS := ./src ../utils ./utils
    CUDA_INC_FLAGS := $(addprefix -I,$(CUDA_SRC_DIRS))
    NVCCFLAGS := --expt-relaxed-constexpr -D__USE_CUDA__ -D__GOLDILOCKS_ENV__ $(CUDA_INC_FLAGS) -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -I./../../external/sppark/ff -U STARK_POSEIDON1
endif

# Debug build flags
ifeq ($(dbg),1)
  CXXFLAGS += -g
else
  CXXFLAGS += -O3
endif

# Source files
SRCS := $(shell find $(SRC_DIR) -name *.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
DEPS := $(OBJS:.o=.d)

# Include directories
INC_DIRS := $(shell find $(SRC_DIR) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

# Preprocessor flags
CPPFLAGS := $(INC_FLAGS) -MMD -MP

# ----------------------
# Test and benchmark source collection (multi-file support)
# ----------------------
TEST_SRCS  := $(wildcard $(TEST_DIR)/*_cpu.cpp)
TEST_OBJS  := $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/tests/%.o, $(TEST_SRCS))
BENCH_SRCS := $(wildcard benchs/*_cpu.cpp)
BENCH_OBJS := $(patsubst benchs/%.cpp, $(BUILD_DIR)/benchs/%.o, $(BENCH_SRCS))

ifeq ($(UNAME_S),Linux)
GPU_SRC_OBJS := $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o \
                $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o \
                $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o \
                $(BUILD_DIR_GPU)/src/sppark_lde.cu.o \
                $(BUILD_DIR_GPU)/external/sppark/util/all_gpus.o \
                $(BUILD_DIR_GPU)/src/poseidon2_goldilocks.cu.o \
                $(BUILD_DIR_GPU)/src/poseidon2_goldilocks.cpp.o \
                $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o \
                $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o \
                $(BUILD_DIR_GPU)/src/goldilocks_tooling.cu.o

GPU_TEST_SRCS := $(wildcard $(TEST_DIR)/*_gpu.cu)
GPU_TEST_OBJS := $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR_GPU)/tests/%.cu.o, $(GPU_TEST_SRCS))
GPU_BENCH_SRCS := $(wildcard benchs/*_gpu.cu)
GPU_BENCH_OBJS := $(patsubst benchs/%.cu, $(BUILD_DIR_GPU)/benchs/%.cu.o, $(GPU_BENCH_SRCS))
endif

# ----------------------
# Default target
# ----------------------
.PHONY: all testscpu benchscpu testsgpu benchsgpu clean help

ifeq ($(UNAME_S),Linux)
all: testscpu benchscpu testsgpu benchsgpu
else
all: testscpu benchscpu
endif

testscpu: $(OBJS) $(TEST_OBJS)
	$(CXX) $(OBJS) $(TEST_OBJS) $(LDFLAGS) $(LDLIBS_TEST) -o $@

benchscpu: $(BENCH_OBJS) $(OBJS)
	$(CXX) $(BENCH_OBJS) $(OBJS) $(LDFLAGS) $(LDLIBS_BENCH) -o $@

testsgpu: $(GPU_TEST_OBJS) $(GPU_SRC_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lgtest -lgtest_main -lgmp

benchsgpu: $(GPU_BENCH_OBJS) $(GPU_SRC_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lgtest -lgmp -lbenchmark

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/benchs/%.o: benchs/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

SPPARK_DIR := ./../../external/sppark
SPPARK_INC := -I$(SPPARK_DIR) -I$(SPPARK_DIR)/ff -I$(SPPARK_DIR)/util -I$(SPPARK_DIR)/ec
# sppark's fr_t==gl64_t typedef is gated on FEATURE_GOLDILOCKS; the gl64_t backend needs the same
# reduction/asm macros the top-level build passes (see NVCCFLAGS / NVCCFLAGS_GL there).
SPPARK_DEFS := -DFEATURE_GOLDILOCKS -DGOLDILOCKS_ZISK -DGL64_PARTIALLY_REDUCED -D__ADX__ -D__USE_ASSEMBLY__
$(BUILD_DIR_GPU)/src/sppark_lde.cu.o: src/sppark_lde.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(SPPARK_INC) $(SPPARK_DEFS) -c $< --output-file $@

$(BUILD_DIR_GPU)/external/sppark/util/all_gpus.o: $(SPPARK_DIR)/util/all_gpus.cpp
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(SPPARK_INC) $(SPPARK_DEFS) -x cu -c $< --output-file $@

$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< --output-file $@

$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ -D__GOLDILOCKS_ENV__ -mavx2 $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -c $< -o $@

# ----------------------
# Clean target
# ----------------------
clean:
	rm -rf ./build testscpu benchscpu
ifeq ($(UNAME_S),Linux)
	rm -rf ./build_gpu testsgpu benchsgpu
endif

# ----------------------
# Help target
# ----------------------
help:
	@echo "make [target]"
	@echo "  all (default) - Build all CPU (and GPU on Linux) binaries"
	@echo "  testscpu      - Build the CPU test binary (scalar + AVX2 + AVX512 auto-detected)"
	@echo "  benchscpu     - Build the CPU benchmark binary (Google Benchmark)"
ifeq ($(UNAME_S),Linux)
	@echo "  testsgpu      - Build the GPU test binary (CUDA, Linux only)"
	@echo "  benchsgpu     - Build the GPU benchmark binary (CUDA, Linux only)"
endif
	@echo "  clean         - Remove build artifacts (CPU and, on Linux, GPU)"

-include $(DEPS)

MKDIR_P ?= mkdir -p
