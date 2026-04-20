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
    CXXFLAGS = -std=c++17 -Wall -pthread -O3 -I./src -I../bn128/src/ffiasm -I$(HOMEBREW_PREFIX)/include -I$(HOMEBREW_PREFIX)/opt/libomp/include -isysroot $(SDKPATH) -Xpreprocessor -fopenmp
    CPPFLAGS = -isysroot $(SDKPATH)
    LDFLAGS  = -L$(HOMEBREW_PREFIX)/lib -L$(HOMEBREW_PREFIX)/opt/libomp/lib -L$(SDKPATH)/usr/lib
    LDLIBS_COMMON = -lomp -lgmpxx -lgmp -lpthread
    LDLIBS_TEST   = $(LDLIBS_COMMON) -lgtest -lgtest_main
    LDLIBS_BENCH  = $(LDLIBS_COMMON) -lbenchmark -lbenchmark_main
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
    NVCCFLAGS := -D__USE_CUDA__ -D__GOLDILOCKS_ENV__ $(CUDA_INC_FLAGS) -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -I./../../external/sppark/ff
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
                $(BUILD_DIR_GPU)/src/poseidon2_goldilocks.cu.o \
                $(BUILD_DIR_GPU)/src/poseidon2_goldilocks.cpp.o \
                $(BUILD_DIR_GPU)/src/goldilocks_tooling.cu.o

GPU_TEST_SRCS := $(wildcard $(TEST_DIR)/*_gpu.cu)
GPU_TEST_OBJS := $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR_GPU)/tests/%.cu.o, $(GPU_TEST_SRCS))
GPU_BENCH_SRCS := $(wildcard benchs/*_gpu.cu)
GPU_BENCH_OBJS := $(patsubst benchs/%.cu, $(BUILD_DIR_GPU)/benchs/%.cu.o, $(GPU_BENCH_SRCS))
endif

ifeq ($(UNAME_S),Darwin)
# Metal Obj-C++ bridge sources (compiled with ARC). Only enumerated on
# Darwin — the .mm files are ifdef'd on PIL2_HAS_METAL internally but the
# `.mm` extension is only meaningful to clang on Apple platforms.
#
# Metal-flavoured test objects go in $(BUILD_DIR)/tests-metal/ so they get a
# dedicated pattern rule (with -DPIL2_HAS_METAL=1) without competing against
# the generic $(BUILD_DIR)/tests/%.o rule used by testscpu. Using separate
# directories avoids pattern-rule stem tiebreaker surprises.
METAL_SRCS      := $(wildcard $(SRC_DIR)/metal/*.mm)
METAL_OBJS      := $(patsubst $(SRC_DIR)/%.mm, $(BUILD_DIR)/%.mm.o, $(METAL_SRCS))
METAL_TEST_SRCS := $(wildcard $(TEST_DIR)/test_metal_*.cpp)
METAL_TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/tests-metal/%.o, $(METAL_TEST_SRCS))
endif

# ----------------------
# Default target
# ----------------------
.PHONY: all testscpu benchscpu testsgpu benchsgpu testsmetal clean help

ifeq ($(UNAME_S),Linux)
all: testscpu benchscpu testsgpu benchsgpu
else
all: testscpu benchscpu
endif

testscpu: $(OBJS) $(TEST_OBJS) $(EXTRA_TEST_OBJS)
	$(CXX) $(OBJS) $(TEST_OBJS) $(EXTRA_TEST_OBJS) $(LDFLAGS) $(LDLIBS_TEST) -o $@

benchscpu: $(BENCH_OBJS) $(OBJS)
	$(CXX) $(BENCH_OBJS) $(OBJS) $(LDFLAGS) $(LDLIBS_BENCH) -o $@

testsgpu: $(GPU_TEST_OBJS) $(GPU_SRC_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lgtest -lgtest_main -lgmp

benchsgpu: $(GPU_BENCH_OBJS) $(GPU_SRC_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lgtest -lgmp -lbenchmark

ifeq ($(UNAME_S),Darwin)
# testsmetal: Apple-only smoke harness for the Metal backend. Reuses the
# scalar CPU objects ($(OBJS)) so we exercise the Metal bridge alongside the
# host code, and adds the Obj-C++ bridge objects plus Metal-only test
# translation units. -framework Metal/Foundation are linked here rather than
# into $(LDFLAGS) to keep testscpu free of Metal coupling.
testsmetal: $(OBJS) $(METAL_OBJS) $(METAL_TEST_OBJS) $(EXTRA_TEST_OBJS)
	$(CXX) $(OBJS) $(METAL_OBJS) $(METAL_TEST_OBJS) $(EXTRA_TEST_OBJS) \
	       $(LDFLAGS) $(LDLIBS_TEST) \
	       -framework Metal -framework Foundation \
	       -o $@
endif

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/benchs/%.o: benchs/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Metal-only test translation units live in a separate build directory so
# their recipe (adds -DPIL2_HAS_METAL=1) doesn't compete with the generic
# $(BUILD_DIR)/tests/%.o rule used by testscpu.
$(BUILD_DIR)/tests-metal/%.o: $(TEST_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -DPIL2_HAS_METAL=1 $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Obj-C++ Metal bridge. -fobjc-arc enables ARC memory management for the
# MTLDevice / MTLCommandQueue pointers; -x objective-c++ tells clang the
# file mixes C++ and Obj-C even though the project uses clang's default
# language detection for .mm elsewhere (being explicit avoids a surprise if
# someone adds a stricter CXXFLAGS in the future).
$(BUILD_DIR)/metal/%.mm.o: $(SRC_DIR)/metal/%.mm
	$(MKDIR_P) $(dir $@)
	$(CXX) -DPIL2_HAS_METAL=1 -fobjc-arc -x objective-c++ $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

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
ifeq ($(UNAME_S),Darwin)
	rm -f testsmetal
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
