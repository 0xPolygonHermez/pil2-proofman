# Platform detection for compiler and library paths
UNAME_S := $(shell uname -s)
ifneq ($(UNAME_S),Darwin)
    INFO := $(shell cd src/goldilocks && ./configure.sh && cd ../.. && sleep 2)
    include src/goldilocks/CudaArch.mk
endif

NVCC := /usr/local/cuda/bin/nvcc

TARGET_BCT := bctree
TARGET_STARKS_LIB := libstarks.a
TARGET_SETUP := fflonkSetup
TARGET_STARKS_LIB_GPU := libstarksgpu.a

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build-gpu
LIB_DIR := ./lib
LIB_DIR_GPU := ./lib-gpu
SRC_DIRS := ./src
SETUP_DIRS := ./src/rapidsnark
SETUP_DPNDS_DIR := src/ffiasm
WITNESS_LIB := ./witness_lib


CXX := g++
AS := nasm
NVCCFLAGS := -std=c++17 -Xcompiler -fPIC -Xcompiler -mavx2 -O3 -arch=$(CUDA_ARCH) -DGL64_PARTIALLY_REDUCED

# -G -lineinfo -Xptxas=-v



# Define include directories first
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

ifeq ($(UNAME_S),Darwin)
    HOMEBREW_PREFIX := $(shell brew --prefix)
    CXX      = clang++
    SDKPATH  = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    CPPFLAGS = -isysroot $(SDKPATH) -I$(HOMEBREW_PREFIX)/include -I$(HOMEBREW_PREFIX)/opt/libomp/include \
        -I$(HOMEBREW_PREFIX)/opt/libsodium/include -D__EXCLUDE_BN128__ -MMD -MP $(INC_FLAGS)
    CXXFLAGS = -std=c++17 -Wall -pthread -Wno-unused-label -Wno-vla-cxx-extension -fPIC \
        -Xpreprocessor -fopenmp
    LDFLAGS  = -rdynamic -L$(HOMEBREW_PREFIX)/lib -L$(HOMEBREW_PREFIX)/opt/libomp/lib \
        -L$(HOMEBREW_PREFIX)/opt/libsodium/lib -L$(SDKPATH)/usr/lib \
        -lomp -lgmpxx -lgmp -lpthread -lsodium
    CFLAGS   =
    GPU_TARGETS :=
    NVCC :=
else ifeq ($(UNAME_S),Linux)
    CXX      = g++
    NVCC := /usr/local/cuda/bin/nvcc
    include src/goldilocks/CudaArch.mk
    CPPFLAGS = -I/usr/include -I/usr/local/include -MMD -MP $(INC_FLAGS)
    CXXFLAGS = -std=c++17 -Wall -pthread -Wno-unused-label -fPIC -fopenmp
    LDFLAGS  = -rdynamic -fopenmp -lsodium -lpthread -lgmp -lstdc++
    LDFLAGS_GPU := -lcudart -lsodium -lpthread -lgmp -lstdc++
    CFLAGS   = -fopenmp
    # AVX512/AVX2 detection
    AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)
    ifneq ($(AVX512_SUPPORTED),)
        CXXFLAGS += -mavx512f -D__AVX512__ -D__USE_ASSEMBLY__
        $(info AVX-512 is supported by the CPU)
    else
        AVX2_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx2' -m 1)
        ifneq ($(AVX2_SUPPORTED),)
            CXXFLAGS += -mavx2 -D__AVX2__ -D__USE_ASSEMBLY__
            $(info AVX2 is supported by the CPU)
        endif
    endif
    GPU_TARGETS := $(BUILD_DIR_GPU)/$(TARGET_STARKS_LIB_GPU)
else
    $(error Unsupported platform: $(UNAME_S))
endif

# Debug vs Release build flags
ifeq ($(dbg),1)
    CXXFLAGS += -g -D__DEBUG__ -O0
else
    CXXFLAGS += -O3
endif

# Define source files based on platform
ifeq ($(UNAME_S),Darwin)
    # macOS - exclude poseidon_opt, rapidsnark, and ffiasm
    SRCS_STARKS_LIB := $(shell find ./src/api/starks_api.* ./src/goldilocks/src ./src/config ./src/starkpil ./src/XKCP ./src/utils -name *.cpp -or -name *.c -or -name  -or -name *.cc)
else
    # Linux - include all directories
    SRCS_STARKS_LIB := $(shell find ./src/api/starks_api.* ./src/goldilocks/src ./src/config ./src/starkpil ./src/XKCP ./src/poseidon_opt ./src/rapidsnark/*  ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
endif

SRCS_STARKS_LIB_GPU := $(shell find ./src/api/starks_api.* ./src/goldilocks/src ./src/goldilocks/utils ./src/config ./src/starkpil ./src/XKCP ./src/poseidon_opt ./src/rapidsnark/*  ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu)
OBJS_STARKS_LIB := $(SRCS_STARKS_LIB:%=$(BUILD_DIR)/%.o)
OBJS_STARKS_LIB_GPU := $(SRCS_STARKS_LIB_GPU:%=$(BUILD_DIR_GPU)/%.o)
DEPS_STARKS_LIB := $(OBJS_STARKS_LIB:.o=.d)
DEPS_STARKS_LIB_GPU := $(OBJS_STARKS_LIB_GPU:.o=.d)

# Define BCT source files based on platform
ifeq ($(UNAME_S),Darwin)
    # macOS - exclude merkleTreeBN128.cpp, poseidon_opt, and ffiasm but include logger
    SRCS_BCT := $(shell find ./src/bctree/build_const_tree.cpp ./src/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/rapidsnark/logger.* ./src/utils/* -name *.cpp -or -name *.c -or -name *.cc)
else
    # Linux - include all files
    SRCS_BCT := $(shell find ./src/bctree/build_const_tree.cpp ./src/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/rapidsnark/logger.*  ./src/poseidon_opt/poseidon_opt.cpp ./src/ffiasm ./src/utils/* -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
endif
OBJS_BCT := $(SRCS_BCT:%=$(BUILD_DIR)/%.o)
DEPS_BCT := $(OBJS_BCT:.o=.d)

SRCS_SETUP := $(shell find $(SETUP_DIRS) ! -path "./src/sm/*" ! -path "./src/main_sm/*" -name *.cpp)
SRCS_SETUP += $(shell find src/XKCP -name *.cpp)
SRCS_SETUP += $(shell find src/fflonk_setup -name fflonk_setup.cpp)
SRCS_SETUP += $(addprefix $(SETUP_DPNDS_DIR)/, alt_bn128.cpp fr.cpp fq.cpp fnec.cpp fec.cpp misc.cpp naf.cpp splitparstr.cpp)
SRCS_SETUP += $(shell find $(SETUP_DPNDS_DIR) -name *.asm)
OBJS_SETUP := $(patsubst %,$(BUILD_DIR)/%.o,$(SRCS_SETUP))
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
OBJS_SETUP := $(filter-out $(BUILD_DIR)/src/main_test.cpp.o, $(OBJS_SETUP)) # Exclude main.cpp from test build
DEPS_SETUP := $(OBJS_SETUP:.o=.d)

all: $(BUILD_DIR)/$(TARGET_STARKS_LIB) $(GPU_TARGETS)

starks_lib: $(LIB_DIR)/$(TARGET_STARKS_LIB)

starks_lib_gpu: LDFLAGS := $(LDFLAGS_GPU)
starks_lib_gpu: $(LIB_DIR_GPU)/$(TARGET_STARKS_LIB_GPU)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

fflonk_setup: $(BUILD_DIR)/$(TARGET_SETUP)

$(LIB_DIR)/$(TARGET_STARKS_LIB): $(OBJS_STARKS_LIB)
	mkdir -p $(LIB_DIR)
	mkdir -p $(LIB_DIR)/include
	$(AR) rcs $@ $^
	cp src/api/starks_api.hpp $(LIB_DIR)/include/starks_lib.h

$(LIB_DIR_GPU)/$(TARGET_STARKS_LIB_GPU): $(OBJS_STARKS_LIB_GPU)
	$(MKDIR_P) $(LIB_DIR_GPU)
	$(MKDIR_P) $(LIB_DIR_GPU)/include
	$(AR) rcs $@ $^
	cp src/api/starks_api.hpp $(LIB_DIR_GPU)/include/starks_lib_gpu.h

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) -o $@ $(LDFLAGS) $(CFLAGS) $(CPPFLAGS)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

$(BUILD_DIR_GPU)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__  $(NVCCFLAGS) $(CPPFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR) $(BUILD_DIR_GPU)
	$(RM) -r $(LIB_DIR) $(LIB_DIR_GPU)

-include $(DEPS_SETUP)
-include $(DEPS_BCT)
-include $(DEPS_STARKS_LIB)
-include $(DEPS_STARKS_LIB_GPU)

MKDIR_P ?= mkdir -p
