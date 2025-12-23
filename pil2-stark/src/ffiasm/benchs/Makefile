CXX := g++
AS := nasm
ASFLAGS := -f elf64

BLST_PATH := #/path/to/blst
CXXFLAGS := -std=c++17 -Wall -O3 -fPIC -D__USE_ASSEMBLY__
CPPFLAGS := -I.. -I../..
LDFLAGS := -lgmp -z noexecstack

# Optional BLST support
ifdef BLST
CXXFLAGS += -D__BLST__
CPPFLAGS += -I$(BLST_PATH)/bindings
LDFLAGS += -L$(BLST_PATH) -lblst
endif

BENCHMARK_FLAGS := $(shell pkg-config --cflags benchmark)
BENCHMARK_LIBS := $(shell pkg-config --libs benchmark)

BUILD_DIR := build
TARGET := $(BUILD_DIR)/bench

SRCS := bench.cpp ../fr.cpp ../fq.cpp
ASMS := ../fr.asm ../fq.asm
OBJS := $(addprefix $(BUILD_DIR)/,$(notdir $(SRCS:.cpp=.cpp.o) $(ASMS:.asm=.asm.o)))

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(BENCHMARK_LIBS)

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(BENCHMARK_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: ../%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(BENCHMARK_FLAGS) -c $< -o $@

$(BUILD_DIR)/%.asm.o: %.asm
	@mkdir -p $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

$(BUILD_DIR)/%.asm.o: ../%.asm
	@mkdir -p $(dir $@)
	$(AS) $(ASFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
