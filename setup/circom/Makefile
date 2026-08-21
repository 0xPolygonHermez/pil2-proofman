# Detect the operating system
UNAME_S := $(shell uname -s)

WITNESS_DIR ?= ./tmp
ifeq ($(UNAME_S),Darwin)
WITNESS_FILE ?= libwitness.dylib
else
WITNESS_FILE ?= libwitness.so
endif

BUILD_DIR := ./build

OPT_LEVEL := -O1

GOLDILOCKS_SRC ?= ../../pil2-stark/src/goldilocks/src

INC_FLAGS := -I. -I$(GOLDILOCKS_SRC)

ifeq ($(UNAME_S),Darwin)
    HOMEBREW_PREFIX := $(shell brew --prefix)
    CXX      = clang++
    SDKPATH  = /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    CPPFLAGS = -isysroot $(SDKPATH) -I$(HOMEBREW_PREFIX)/include -I$(HOMEBREW_PREFIX)/opt/libomp/include \
        -I$(HOMEBREW_PREFIX)/opt/libsodium/include -I$(HOMEBREW_PREFIX)/opt/gmp/include -MMD -MP $(INC_FLAGS)
    CXXFLAGS = -std=c++17 -Wall -pthread -Wno-unused-label -Wno-vla-cxx-extension -fPIC $(OPT_LEVEL)\
        -Xpreprocessor -fopenmp
    CFLAGS = -rdynamic -L$(HOMEBREW_PREFIX)/lib -L$(HOMEBREW_PREFIX)/opt/libomp/lib \
         -L$(HOMEBREW_PREFIX)/opt/gmp/lib -L$(SDKPATH)/usr/lib -lpthread -lgmpxx -lgmp -lomp
else ifeq ($(UNAME_S),Linux)
    CXX := g++
    CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2 $(OPT_LEVEL) -fPIC #-Wfatal-errors
    CFLAGS := -fopenmp -Wno-unused-variable
    CPPFLAGS := $(INC_FLAGS) -MMD -MP
    # AVX2 detection
    AVX2_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx2' -m 1)
    ifneq ($(AVX2_SUPPORTED),)
        CXXFLAGS += -mavx2 -D__AVX2__ -D__USE_ASSEMBLY__
        $(info AVX2 is supported by the CPU)
    endif
endif

witness: $(WITNESS_DIR)/$(WITNESS_FILE)

clean:
	$(RM) -r $(BUILD_DIR)

SRCS_WITNESS_LIB := calcwit.cpp main.cpp verifier.cpp poseidon2_goldilocks.cpp poseidon_goldilocks.cpp blake3_goldilocks.cpp goldilocks_base_field.cpp cmul.cpp
OBJS_WITNESS_LIB := $(SRCS_WITNESS_LIB:%=$(BUILD_DIR)/%.o)
DEPS_WITNESS_LIB := $(OBJS_WITNESS_LIB:.o=.d)

$(WITNESS_DIR)/$(WITNESS_FILE): $(OBJS_WITNESS_LIB)
	$(MKDIR_P) $(WITNESS_DIR)
	$(CXX) -shared -o $@ $^ $(CFLAGS) -lgmp -lgmpxx


# C++ source compilation rule
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

MKDIR_P ?= mkdir -p

.PHONY: clean