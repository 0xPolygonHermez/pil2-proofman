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
    CXXFLAGS = -std=c++17 -Wall -pthread -O3 -I./src -I/usr/include -I/usr/local/include -fopenmp
    CPPFLAGS =
    LDFLAGS  = -L/usr/lib -L/usr/local/lib
    LDLIBS   = -lomp -lbenchmark -lbenchmark_main -lgtest -lgtest_main -lgmpxx -lgmp -lpthread -lstdc++

	CXXFLAGS += -D__AVX2__
  	CXXFLAGS += -D__USE_ASSEMBLY__
    # Auto-enable AVX512 if supported, or allow override
    ifndef USE_AVX512
      AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)
      USE_AVX512 := $(if $(AVX512_SUPPORTED),1,0)
    endif
    ifeq ($(USE_AVX512),1)
      CXXFLAGS += -mavx512f
    endif

    # CUDA/GPU build options (Linux only)
    NVCC := /usr/local/cuda/bin/nvcc
    include CudaArch.mk
    BUILD_DIR_GPU := ./build_gpu
    CUDA_SRC_DIRS := ./src ./utils
    CUDA_INC_FLAGS := $(addprefix -I,$(CUDA_SRC_DIRS))

    # CUDA object rules
    $(BUILD_DIR_GPU)/%.cu.o: %.cu
    	$(MKDIR_P) $(dir $@)
    	$(NVCC) -D__USE_CUDA__ $(CUDA_INC_FLAGS) -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -dc $< --output-file $@

    $(BUILD_DIR_GPU)/%.cpp.o: %.cpp
    	$(MKDIR_P) $(dir $@)
    	$(CXX) -D__USE_CUDA__ -mavx2 $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -c $< -o $@

    # GPU test and bench targets
    testsgpu: $(BUILD_DIR_GPU)/tests/tests.cu.o $(BUILD_DIR)/src/goldilocks_base_field.o $(BUILD_DIR)/src/goldilocks_cubic_extension.o $(BUILD_DIR)/utils/timer_gl.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR)/src/poseidon_goldilocks.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o
    	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

    benchgpu: $(BUILD_DIR_GPU)/benchs/bench.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.o $(BUILD_DIR)/src/goldilocks_cubic_extension.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
    	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp -lbenchmark

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
# Default target
# ----------------------
.PHONY: all testscpu benchscpu testsgpu benchgpu clean help

ifeq ($(UNAME_S),Linux)
all: testscpu benchscpu testsgpu benchgpu
else
all: testscpu benchscpu
endif

testscpu: $(OBJS) $(BUILD_DIR)/tests/tests.o
	$(CXX) $(OBJS) $(BUILD_DIR)/tests/tests.o $(LDFLAGS) $(LDLIBS) -o $@

benchscpu: $(BUILD_DIR)/benchs/bench.o $(OBJS)
	$(CXX) $(BUILD_DIR)/benchs/bench.o $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/benchs/%.o: benchs/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# ----------------------
# Clean target
# ----------------------
.PHONY: clean
clean:
	rm -rf ./build testscpu benchscpu
ifeq ($(UNAME_S),Linux)
	rm -rf ./build_gpu testsgpu benchgpu
endif

# ----------------------
# Help target
# ----------------------
.PHONY: help
help:
	@echo "make [target] [USE_AVX512=1|0]"
	@echo "  clean         - Remove build artifacts (CPU and, on Linux, GPU)"
	@echo "  distclean     - Remove all build artifacts and backup files"
	@echo "  all (default) - Build all CPU (and GPU on Linux) binaries"
	@echo "  testscpu      - Build the CPU test binary (unit and integration tests)"
	@echo "  benchscpu     - Build the CPU benchmark binary (Google Benchmark)"
ifeq ($(UNAME_S),Linux) 
	@echo "  testsgpu      - Build the GPU test binary (CUDA, Linux only)"
	@echo "  benchgpu      - Build the GPU benchmark binary (CUDA, Linux only)"
endif
	@echo "  USE_AVX512   - Force AVX512 on/off (default: auto-detect on Linux)"

-include $(DEPS)

MKDIR_P ?= mkdir -p