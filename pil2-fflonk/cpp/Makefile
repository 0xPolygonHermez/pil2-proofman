# pil2-fflonk C++ Makefile
#
# pil2-fflonk is a BN254 SNARK. Its include closure is deliberately limited to
# pil2-stark's snark-side primitives (rapidsnark) and BN254 field/curve
# arithmetic (bn128/ffiasm) -- no Goldilocks, no starks_api. Keep it that way:
# pil2-stark's own utils/timer.hpp reaches both, which is why cpp/timer.hpp
# exists.

BUILD_DIR := ./build
STARK_SRC := ../../pil2-stark/src
MKDIR_P   ?= mkdir -p

CXX := g++
AS  := nasm

CXXFLAGS := -std=c++17 -Wall -O3 -fPIC -D__USE_ASSEMBLY__ -fopenmp
CPPFLAGS := -I. -I$(STARK_SRC)/rapidsnark -I$(STARK_SRC)/bn128/src/ffiasm \
            -I$(STARK_SRC)/XKCP -MMD -MP
ASFLAGS  := -f elf64 -g -F dwarf

LDFLAGS_COMMON := -z noexecstack -fopenmp

# ----------------------
# Sources
# ----------------------
SHPLONK_SRCS := shplonk.cpp
SHPLONK_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SHPLONK_SRCS))

# Snark-side support pulled from pil2-stark: the logger fflonk/plonk already
# use, and Keccak for the Fiat-Shamir transcript.
SUPPORT_OBJS := $(BUILD_DIR)/support/logger.o \
                $(BUILD_DIR)/support/keccak_wrapper.o \
                $(BUILD_DIR)/xkcp/Keccak-more-compact.o

FFIASM_DIR  := $(STARK_SRC)/bn128/src/ffiasm
FFIASM_SRCS := fr.cpp fq.cpp naf.cpp misc.cpp splitparstr.cpp alt_bn128.cpp
FFIASM_ASMS := fr.asm fq.asm
FFIASM_OBJS := $(patsubst %.cpp,$(BUILD_DIR)/ffiasm/%.cpp.o,$(FFIASM_SRCS)) \
               $(patsubst %.asm,$(BUILD_DIR)/ffiasm/%.asm.o,$(FFIASM_ASMS))

TESTS_SRCS := $(wildcard tests/*.cpp)
TESTS_OBJS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests/%.o,$(TESTS_SRCS))

DEPS := $(SHPLONK_OBJS:.o=.d) $(SUPPORT_OBJS:.o=.d) $(TESTS_OBJS:.o=.d) \
        $(patsubst %.cpp,$(BUILD_DIR)/ffiasm/%.cpp.d,$(FFIASM_SRCS))

# ----------------------
# Targets
# ----------------------
.PHONY: all tests clean help

all: tests

tests: testscpu

testscpu: $(TESTS_OBJS) $(SHPLONK_OBJS) $(SUPPORT_OBJS) $(FFIASM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lgtest -lgtest_main -lgmp -lpthread $(LDFLAGS_COMMON)

# ----------------------
# Pattern rules
# ----------------------
$(BUILD_DIR)/%.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/support/%.o: $(STARK_SRC)/rapidsnark/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/xkcp/%.o: $(STARK_SRC)/XKCP/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/ffiasm/%.cpp.o: $(FFIASM_DIR)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/ffiasm/%.asm.o: $(FFIASM_DIR)/%.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR) testscpu

-include $(DEPS)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  tests (default) - Build CPU unit tests (output: ./testscpu)"
	@echo "  clean           - Remove build artifacts"
