SRCDIR   = $(CURDIR)/src
BUILDDIR = $(CURDIR)/build
BINDIR   = $(CURDIR)/bin
SEND_DIR = $(CURDIR)/send

$(CONVERT_TOOL):
	cd $(dir $@) && go build -v

TARGET = lab4
TARGET_BIN = $(BINDIR)/$(TARGET)
SEND_TARGET_DIR = $(SEND_DIR)/$(TARGET)

CXX = nvcc
CC  = nvcc

CFLAGS  = -I$(SRCDIR)
LDFLAGS = -lm

GCCFLAGS = -Wall -Wextra -g3 \
			-Wno-sign-compare -Wno-long-long \
			-Wunreachable-code \
			-Wunused-result -Wunused-but-set-variable

NVCCFLAGS += --std=c++14 -Werror cross-execution-space-call
NVCCFLAGS += -arch=sm_35 -Wno-deprecated-gpu-targets
NVCCFLAGS += -D_FORCE_INLINES -Xcompiler -fPIC
NVCCFLAGS += -Wno-deprecated-declarations

ifeq ($(BENCH),on)
GCCFLAGS   = -O3
# NVCCFLAGS += -DBENCHMARK
NVCCFLAGS += -O3
else
GCCFLAGS  += -g3
NVCCFLAGS += -G
endif
NVCCFLAGS += --compiler-options "$(GCCFLAGS)"

CU_SOURCES  = transponse.cu
# C_SOURCES   = helpers.c
CPP_SOURCES = main.cpp helpers.cpp
HEADERS     = helpers.cuh

CPP_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CPP_SOURCES)))
# C_OBJECTS   = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(C_SOURCES)))
CU_OBJECTS  = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CU_SOURCES)))

ifeq ($(CUDA),off)
CXX = g++
CC  = gcc
CFLAGS = $(GCCFLAGS)
CPP_SOURCES += transponse.cpp
OBJECTS = $(CPP_OBJECTS)
else
CFLAGS = $(NVCCFLAGS)
OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS)
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
	$(MAKE) -j1 -C test TEST_TOOL="$(PREFIX) $^" SKIP_DIFF=$(SKIP_DIFF)

.PHONY: send
send: | $(SEND_DIR) $(SEND_TARGET_DIR)
	cp $(addprefix $(SRCDIR)/,$(C_SOURCES) $(CPP_SOURCES) $(CU_SOURCES) $(HEADERS)) $(SEND_DIR)/$(TARGET)
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
