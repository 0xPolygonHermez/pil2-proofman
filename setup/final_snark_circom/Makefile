WITNESS_DIR ?= ./tmp
WITNESS_FILE ?= libwitness.so

BUILD_DIR := ./build

CXX := g++
CXXFLAGS := -std=c++17 -Wall -pthread -flarge-source-files -Wno-unused-label -rdynamic -mavx2 -O2 #-Wfatal-errors
CXXFLAGS_EXT := -fPIC
CFLAGS := -fopenmp -Wno-unused-variable

CPPFLAGS := -I ./src/setup/final_snark_circom -MMD -MP

AS := nasm
ASFLAGS := -f elf64 -F dwarf


witness: $(WITNESS_DIR)/$(WITNESS_FILE)

clean:
	$(RM) -r $(BUILD_DIR)

SRCS_WITNESS_LIB := calcwit.cpp fr.cpp fr.asm main.cpp verifier.cpp
OBJS_WITNESS_LIB := $(SRCS_WITNESS_LIB:%=$(BUILD_DIR)/%.o)
DEPS_WITNESS_LIB := $(OBJS_WITNESS_LIB:.o=.d)

$(WITNESS_DIR)/$(WITNESS_FILE): $(OBJS_WITNESS_LIB)
	$(MKDIR_P) $(WITNESS_DIR)
	$(CXX) $(CXXFLAGS_EXT) -shared -o $@ $^ -lgmp -lgmpxx

# Assembly compilation rule
$(BUILD_DIR)/%.asm.o: %.asm
	$(MKDIR_P) $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

# C++ source compilation rule
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_EXT) -c $< -o $@

MKDIR_P ?= mkdir -p

.PHONY: clean