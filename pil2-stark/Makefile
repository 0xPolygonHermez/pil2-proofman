TARGET_BCT := bctree
TARGET_STARKS_LIB := libstarks.a
TARGET_SETUP := fflonkSetup
TARGET_SETUP_PLONK := plonkSetup
TARGET_PROVE_PLONK := plonkProve

BUILD_DIR := ./build
LIB_DIR := ./lib
SRC_DIRS := ./src
SETUP_DIRS := ./src/rapidsnark
SETUP_DPNDS_DIR := src/ffiasm
WITNESS_LIB := ./witness_lib


CXX := g++
AS := nasm
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic #-Wfatal-errors
LDFLAGS := -lsodium -lpthread -lgmp -lstdc++ -fopenmp 
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


SRCS_STARKS_LIB := $(shell find ./src/api/starks_api.* ./src/goldilocks/src ./src/config ./src/starkpil ./src/XKCP ./src/poseidon_opt ./src/rapidsnark/* ./src/rapidsnark/* ./src/ffiasm ./src/utils -name *.cpp -or -name *.c -or -name *.asm -or -name *.cc)
OBJS_STARKS_LIB := $(SRCS_STARKS_LIB:%=$(BUILD_DIR)/%.o)
DEPS_STARKS_LIB := $(OBJS_STARKS_LIB:.o=.d)

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

SRCS_SETUP_PLONK := $(shell find $(SETUP_DIRS) ! -path "./src/sm/*" ! -path "./src/main_sm/*" -name *.cpp)
SRCS_SETUP_PLONK += $(shell find src/XKCP -name *.cpp)
SRCS_SETUP_PLONK += $(shell find src/plonk_setup -name plonk_setup.cpp)
SRCS_SETUP_PLONK += $(addprefix $(SETUP_DPNDS_DIR)/, alt_bn128.cpp fr.cpp fq.cpp fnec.cpp fec.cpp misc.cpp naf.cpp splitparstr.cpp)
SRCS_SETUP_PLONK += $(shell find $(SETUP_DPNDS_DIR) -name *.asm)
OBJS_SETUP_PLONK := $(patsubst %,$(BUILD_DIR)/%.o,$(SRCS_SETUP_PLONK))
OBJS_SETUP_PLONK := $(filter-out $(BUILD_DIR)/src/main.cpp.o, $(OBJS_SETUP_PLONK)) # Exclude main.cpp from test build
OBJS_SETUP_PLONK := $(filter-out $(BUILD_DIR)/src/main_test.cpp.o, $(OBJS_SETUP_PLONK)) # Exclude main.cpp from test build
DEPS_SETUP_PLONK := $(OBJS_SETUP_PLONK:.o=.d)

SRCS_PROVE_PLONK := $(shell find $(SETUP_DIRS) ! -path "./src/sm/*" ! -path "./src/main_sm/*" -name *.cpp)
SRCS_PROVE_PLONK += $(shell find src/XKCP -name *.cpp)
SRCS_PROVE_PLONK += $(shell find src/plonk_setup -name main_prover.cpp)
SRCS_PROVE_PLONK += $(addprefix $(SETUP_DPNDS_DIR)/, alt_bn128.cpp fr.cpp fq.cpp fnec.cpp fec.cpp misc.cpp naf.cpp splitparstr.cpp)
SRCS_PROVE_PLONK += $(shell find $(SETUP_DPNDS_DIR) -name *.asm)
OBJS_PROVE_PLONK := $(patsubst %,$(BUILD_DIR)/%.o,$(SRCS_PROVE_PLONK))
OBJS_PROVE_PLONK := $(filter-out $(BUILD_DIR)/src/main.cpp.o, $(OBJS_PROVE_PLONK)) # Exclude main.cpp from test build
OBJS_PROVE_PLONK := $(filter-out $(BUILD_DIR)/src/main_test.cpp.o, $(OBJS_PROVE_PLONK)) # Exclude main.cpp from test build
DEPS_SETUP_PLONK := $(OBJS_PROVE_PLONK:.o=.d)

all: $(LIB_DIR)/$(TARGET_STARKS_LIB)

starks_lib: CXXFLAGS_EXT := -D__ZKEVM_LIB__ -fPIC#we decided to use the same flags for both libraries
starks_lib: $(LIB_DIR)/$(TARGET_STARKS_LIB)

bctree: $(BUILD_DIR)/$(TARGET_BCT)

fflonk_setup: $(BUILD_DIR)/$(TARGET_SETUP)

plonk_setup: $(BUILD_DIR)/$(TARGET_SETUP_PLONK)

plonk_prove: $(BUILD_DIR)/$(TARGET_PROVE_PLONK)

$(LIB_DIR)/$(TARGET_STARKS_LIB): $(OBJS_STARKS_LIB)
	mkdir -p $(LIB_DIR)
	mkdir -p $(LIB_DIR)/include
	$(AR) rcs $@ $^
	cp src/api/starks_api.hpp $(LIB_DIR)/include/starks_lib.h

$(BUILD_DIR)/$(TARGET_BCT): $(OBJS_BCT)
	$(CXX) $(OBJS_BCT) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_SETUP): $(OBJS_SETUP)
	$(CXX) $(OBJS_SETUP) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_SETUP_PLONK): $(OBJS_SETUP_PLONK)
	$(CXX) $(OBJS_SETUP_PLONK) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

$(BUILD_DIR)/$(TARGET_PROVE_PLONK): $(OBJS_PROVE_PLONK)
	$(CXX) $(OBJS_PROVE_PLONK) $(CXXFLAGS) $(CXXFLAGS_EXT) -o $@ $(LDFLAGS) $(LDFLAGS_EXT) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT)

# assembly
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS_SETUP)
-include $(DEPS_SETUP_PLONK)
-include $(DEPS_PROVE_PLONK)
-include $(DEPS_BCT)

MKDIR_P ?= mkdir -p
