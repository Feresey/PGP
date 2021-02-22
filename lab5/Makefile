SRCDIR   = $(CURDIR)/src
BUILDDIR = $(CURDIR)/build
BINDIR   = $(CURDIR)/bin
SEND_DIR = $(CURDIR)/send

$(CONVERT_TOOL):
	cd $(dir $@) && go build -v

TARGET = lab5
TARGET_BIN = $(BINDIR)/$(TARGET)
SEND_TARGET_DIR = $(SEND_DIR)/$(TARGET)

CXX = nvcc

CFLAGS  = -I$(SRCDIR)
LDFLAGS = -lm

GCCFLAGS = -Wall -Wextra -g3 \
			-Wno-sign-compare -Wno-long-long \
			-Wconversion -Wsign-conversion \
			-Wunreachable-code -Wno-unused-parameter

NVCCFLAGS += --std=c++11 -Werror cross-execution-space-call
NVCCFLAGS += -arch=sm_35 -Wno-deprecated-gpu-targets
NVCCFLAGS += -Wno-deprecated-declarations
NVCCFLAGS += -DBENCHMARK
NVCCFLAGS += -G
NVCCFLAGS += --compiler-options "$(GCCFLAGS)"


CU_SOURCES  = main.cu
C_SOURCES   = helpers.c
# CPP_SOURCES = main.cpp solve.cpp helpers.cpp
HEADERS     = helpers.cuh

CPP_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CPP_SOURCES)))
C_OBJECTS   = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(C_SOURCES)))
CU_OBJECTS  = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CU_SOURCES)))

ifeq ($(CUDA),off)
CXX = g++
CFLAGS = $(GCCFLAGS)
OBJECTS = $(CPP_OBJECTS) $(C_OBJECTS)
else
CFLAGS = $(NVCCFLAGS)
OBJECTS = $(CU_OBJECTS) $(C_OBJECTS)
endif


.PHONY: all clean
all: $(TARGET)
clean:
	-rm $(OBJECTS) $(TARGET_BIN)
	-$(MAKE) -C test clean

$(BUILDDIR) $(BINDIR) $(SEND_DIR) $(SEND_TARGET_DIR):
	mkdir -p $@

.PHONY: test
test: $(TARGET_BIN) | $(CONVERT_TOOL)
	$(MAKE) -j1 -C test TEST_TOOL="$(PREFIX) $^"

.PHONY: send
send: | $(SEND_DIR) $(SEND_TARGET_DIR)
	cp $(addprefix $(SRCDIR)/,$(C_SOURCES) $(CU_SOURCES) $(HEADERS)) $(SEND_DIR)/$(TARGET)
	cp $(SEND_DIR)/Makefile $(SEND_TARGET_DIR)
	cd $(SEND_DIR) && tar cvf $(TARGET).tar $(TARGET)
	gpg -r 703FD5D038947C5B57CCE5A64DF7496E16FD3CBC -ab $(SEND_DIR)/$(TARGET).tar

.PHONY: $(TARGET)
$(TARGET): $(TARGET_BIN)

$(TARGET_BIN): $(OBJECTS) | $(BINDIR)
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@

define compile
	$(CXX) $(CFLAGS) -c $^ -o $@
endef

$(CU_OBJECTS): $(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(compile)

$(CPP_OBJECTS): $(BUILDDIR)/%.cpp.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(compile)

$(C_OBJECTS): $(BUILDDIR)/%.c.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(compile)


