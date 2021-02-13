SRCDIR   = $(CURDIR)/src
BUILDDIR = $(CURDIR)/build
BINDIR   = $(CURDIR)/bin
SEND_DIR = $(CURDIR)/send/lab2

CC = nvcc

CFLAGS += --std=c++14 -Werror cross-execution-space-call
CFLAGS += -arch=sm_35 -Wno-deprecated-gpu-targets
CFLAGS += -Wno-deprecated-declarations
CFLAGS += -G
CFLAGS += --compiler-options "-Wall -Werror -Wextra -g3 -Wconversion -Wsign-conversion -Wunreachable-code"
CFLAGS += -I$(SRCDIR)

LDFLAGS = -lm

TARGET = $(BINDIR)/lab2

CU_SOURCES = main.cu
C_SOURCES  = serialize.c
HEADERS    = serialize.h

C_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(C_SOURCES)))
CU_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CU_SOURCES)))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)

.PHONY: all clean
all: $(TARGET)
clean:
	-rm $(OBJECTS) $(TARGET)
	-$(MAKE) -C test clean

$(BUILDDIR) $(BINDIR) $(SEND_DIR):
	mkdir -p $@

.PHONY: test
test:
	$(MAKE) -j1 -C test test TEST_TOOL=$(TARGET)

.PHONY: send
send: | $(SEND_DIR)
	cp $(addprefix $(SRCDIR)/,$(C_SOURCES) $(CU_SOURCES) $(HEADERS)) $(SEND_DIR)
	cp $(SEND_DIR)/../Makefile $(SEND_DIR)
	cd $(SEND_DIR)/.. && tar cvf lab2.tar lab2
	gpg -r 703FD5D038947C5B57CCE5A64DF7496E16FD3CBC -ab $(SEND_DIR)/../lab2.tar

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

define compile
	$(CC) $(CFLAGS) -c $^ -o $@
endef

$(CU_OBJECTS): $(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(compile)

$(C_OBJECTS): $(BUILDDIR)/%.c.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(compile)
