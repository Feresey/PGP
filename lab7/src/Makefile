BUILDDIR ?= $(CURDIR)/build
BINDIR   ?= $(CURDIR)/bin

TARGET_BIN ?= $(BINDIR)/$(TARGET)

CXX ?= g++
CC  ?= gcc

.PHONY: all clean

all: $(TARGET_BIN)

clean:
	rm $(TARGET_OBJECTS)

CPP_SOURCES = main.cpp grid.cpp dim3.cpp solver.cpp
HEADERS     = helpers.hpp dim3.hpp grid.hpp solver.hpp

grid.cpp solver.cpp dim3.cpp: helpers.hpp

grid.hpp: dim3.hpp
solver.hpp: dim3.hpp

main.cpp: solver.hpp helpers.hpp
dim3.cpp: dim3.hpp
grid.cpp: grid.hpp
solver.cpp: solver.hpp grid.hpp

CPP_OBJECTS = $(addprefix $(BUILDDIR)/,$(addsuffix .o,$(CPP_SOURCES)))

TARGET_OBJECTS = $(CPP_OBJECTS)

LDFLAGS = -lm -lmpi

CFLAGS = -Wall -Wextra \
		-Wno-sign-compare -Wno-long-long \
		-Wconversion -Wsign-conversion \
		-Wunreachable-code

ifeq ($(BENCH),on)
CFLAGS = -O3
else
CFLAGS += -g3
endif

$(TARGET_BIN): CXX=mpic++
$(TARGET_BIN): $(TARGET_OBJECTS) | $(BINDIR)
	$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(CPP_OBJECTS): $(BUILDDIR)/%.cpp.o: %.cpp | $(BUILDDIR)
	$(CXX) $(CFLAGS) -c $^ -o $@
