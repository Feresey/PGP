SRCDIR   = $(CURDIR)/src
BUILDDIR = $(CURDIR)/build
BINDIR   = $(CURDIR)/bin
SEND_DIR = $(CURDIR)/send

TARGET = lab3
TARGET_BIN = $(BINDIR)/$(TARGET)
SEND_TARGET_DIR = $(SEND_DIR)/$(TARGET)

CC = nvcc

CFLAGS += --std=c++14 -Werror cross-execution-space-call
CFLAGS += -arch=sm_35 -Wno-deprecated-gpu-targets
CFLAGS += -Wno-deprecated-declarations
CFLAGS += -G
CFLAGS += --compiler-options "-Wall -Werror -Wextra -g3 -Wconversion -Wsign-conversion -Wunreachable-code"
CFLAGS += -I$(SRCDIR)

LDFLAGS = -lm

CU_SOURCES = main.cu
C_SOURCES  = serialize.c
HEADERS    = serialize.h helpers.cuh

C_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(C_SOURCES)))
CU_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CU_SOURCES)))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)

.PHONY: all clean
all: $(TARGET)
clean:
	-rm $(OBJECTS) $(TARGET_BIN)
	-$(MAKE) -C test clean

$(BUILDDIR) $(BINDIR) $(SEND_DIR) $(SEND_TARGET_DIR):
	mkdir -p $@

.PHONY: test
test:
	$(MAKE) -j1 -C test test TEST_TOOL=$(TARGET)

.PHONY: send
send: | $(SEND_DIR) $(SEND_TARGET_DIR)
	cp $(addprefix $(SRCDIR)/,$(C_SOURCES) $(CU_SOURCES) $(HEADERS)) $(SEND_DIR)/$(TARGET)
	cp $(SEND_DIR)/Makefile $(SEND_TARGET_DIR)
	cd $(SEND_DIR) && tar cvf $(TARGET).tar $(TARGET)
	gpg -r 703FD5D038947C5B57CCE5A64DF7496E16FD3CBC -ab $(SEND_DIR)/$(TARGET).tar

.PHONY: $(TARGET)
$(TARGET): $(TARGET_BIN)

$(TARGET_BIN): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

define compile
	$(CC) $(CFLAGS) -c $^ -o $@
endef

$(CU_OBJECTS): $(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(compile)

$(C_OBJECTS): $(BUILDDIR)/%.c.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(compile)
