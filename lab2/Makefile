SRCDIR   = src
BUILDDIR = build
BINDIR   = bin

CC = nvcc

CFLAGS += --std=c++14 -Werror cross-execution-space-call
CFLAGS += -arch=sm_35 -Wno-deprecated-gpu-targets
CFLAGS += -Wno-deprecated-declarations
CFLAGS += -G
CFLAGS += --compiler-options "-Wall -Werror -Wextra -g3 -Wconversion -Wsign-conversion -Wunreachable-code"
CFLAGS += -I$(SRCDIR)

LDFLAGS = -lm

TARGET = $(BINDIR)/lab2

CU_SOURCES += main.cu
C_SOURCES += serialize.c

C_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(C_SOURCES)))
CU_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CU_SOURCES)))
OBJECTS = $(C_OBJECTS) $(CU_OBJECTS)

.PHONY: all clean
all: $(TARGET)
clean:
	-rm $(OBJECTS) $(TARGET)

$(BUILDDIR) $(BINDIR):
	mkdir -p $@

.PHONY: test
test:
	@echo $(OBJECTS)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

define compile
	$(CC) $(CFLAGS) -c $^ -o $@
endef

$(CU_OBJECTS): $(BUILDDIR)/%.cu.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(compile)

$(C_OBJECTS): $(BUILDDIR)/%.c.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(compile)
