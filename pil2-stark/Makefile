INFO := $(shell cd src/goldilocks && ./configure.sh && cd ../.. && sleep 2)
include src/goldilocks/CudaArch.mk
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
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -fPIC
NVCCFLAGS := -std=c++17 -Xcompiler -fPIC -Xcompiler -mavx2 -O3 -arch=$(CUDA_ARCH) -DGL64_PARTIALLY_REDUCED

# -G -lineinfo -Xptxas=-v

LDFLAGS_GPU := -lcudart -lsodium -lpthread -lgmp -lstdc++
LDFLAGS := -fopenmp -lsodium -lpthread -lgmp -lstdc++
CFLAGS := -fopenmp
ASFLAGS := -felf64

# Check if AVX-512 is supported
AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)
ifneq ($(AVX512_SUPPORTED),)
    CXXFLAGS += -mavx512f -D__AVX512__
    $(info AVX-512 is supported by the CPU)
    USE_ASSEMBLY := 1
else
    $(info AVX-512 is not supported by the CPU)
    
    # Check if AVX2 is supported
    AVX2_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx2' -m 1)
    ifneq ($(AVX2_SUPPORTED),)
        CXXFLAGS += -mavx2 -D__AVX2__
        $(info AVX2 is supported by the CPU)
        USE_ASSEMBLY := 1
    else
        $(info AVX2 is not supported by the CPU)
        USE_ASSEMBLY := 0  # Neither AVX-512 nor AVX2 is supported
    endif
endif

# Add USE_ASSEMBLY to CXXFLAGS
CXXFLAGS += -DUSE_ASSEMBLY=$(USE_ASSEMBLY)

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g -D__DEBUG__
else
      CXXFLAGS += -O3
endif

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) $(INC_FLAGS_EXT) -MMD -MP

INC_DIRS := $(shell find $(SRC_DIRS) -type d) $(sort $(dir))
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

SRCS_STARKS_LIB := $(shell find ./src/api/starks_api.* ./src/goldilocks/src ./src/config ./src/starkpil ./src/XKCP ./src/poseidon_opt ./src/rapidsnark/*  ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
SRCS_STARKS_LIB_GPU := $(shell find ./src/api/starks_api.* ./src/goldilocks/src ./src/goldilocks/utils ./src/config ./src/starkpil ./src/XKCP ./src/poseidon_opt ./src/rapidsnark/*  ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc -or -name *.cu)
OBJS_STARKS_LIB := $(SRCS_STARKS_LIB:%=$(BUILD_DIR)/%.o)
OBJS_STARKS_LIB_GPU := $(SRCS_STARKS_LIB_GPU:%=$(BUILD_DIR_GPU)/%.o)
DEPS_STARKS_LIB := $(OBJS_STARKS_LIB:.o=.d)
DEPS_STARKS_LIB_GPU := $(OBJS_STARKS_LIB_GPU:.o=.d)

SRCS_BCT := $(shell find ./src/bctree/build_const_tree.cpp ./src/bctree/main.cpp ./src/goldilocks/src ./src/starkpil/merkleTree/merkleTreeBN128.cpp ./src/starkpil/merkleTree/merkleTreeGL.cpp ./src/rapidsnark/logger.*  ./src/poseidon_opt/poseidon_opt.cpp ./src/ffiasm ./src/utils/* -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
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

all: $(BUILD_DIR)/$(TARGET_STARKS_LIB) $(BUILD_DIR_GPU)/$(TARGET_STARKS_LIB_GPU)

starks_lib: CXXFLAGS_EXT := -D__ZKEVM_LIB__ -fPIC#we decided to use the same flags for both libraries
starks_lib: $(LIB_DIR)/$(TARGET_STARKS_LIB)

starks_lib_gpu: CXXFLAGS_EXT := -D__ZKEVM_LIB__ -fPIC -fopenmp#we decided to use the same flags for both libraries
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
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

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
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# cuda source
$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__  $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(CXXFLAGS_EXT)" -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR) $(BUILD_DIR_GPU)
	$(RM) -r $(LIB_DIR) $(LIB_DIR_GPU)

-include $(DEPS_SETUP)
-include $(DEPS_BCT)
-include $(DEPS_STARKS_LIB)
-include $(DEPS_STARKS_LIB_GPU)

MKDIR_P ?= mkdir -p
