BUILDDIR := build
BINDIR   ?= bin
DEPS_DIR := deps

TARGET_BIN ?= $(BINDIR)/target

CXX   ?= g++
MPICC ?= mpic++

.PHONY: all clean

all: $(TARGET_BIN)

clean:
	rm $(TARGET_OBJECTS)

$(DEPS_DIR) $(BUILDDIR) $(BINDIR):
	-mkdir -p "$@"


CPP_SOURCES = main.cpp grid.cpp dim3.cpp \
			$(addprefix solver/,solver.cpp common.cpp problem.cpp)
HEADERS     = $(shell find . -name '*.hpp')

SOURCES = $(shell find . -name '*.cpp') $(HEADERS)

DEPS_LIST = $(foreach dep,$(SOURCES),$(DEPS_DIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(dep))).mk)

$(DEPS_LIST): $(DEPS_DIR)/%.mk: $(subst __,/,$@) | $(DEPS_DIR)
	@echo $*
	$(file > $@,$(patsubst %.cpp,$(BUILDDIR)/%.cpp.o,$*): $(shell $(CXX) -E $(subst __,/,$*) | \
		grep -Eo '#\s[0-9]+\s".*$$' - | \
		sed -re 's|#\s[0-9]+\s"(.*?)".*$$|\1|gi' | \
		grep -v -e "^/usr" -e "^<" | \
		sort -u | \
		$(if $(patsubst %.hpp,,$*),,grep -v -e "$*" |) \
		xargs -n1 realpath --relative-to $(CURDIR)))

deps.mk: $(DEPS_LIST) | $(DEPS_DIR)
	@echo $(DEPS_LIST) | xargs -n1 printf "-include %s\n" > $@

-include deps.mk

CPP_OBJECTS = $(foreach obj,$(CPP_SOURCES),$(BUILDDIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(obj))).o)

TARGET_OBJECTS = $(CPP_OBJECTS)

LDFLAGS = -lm -lmpi

CFLAGS = -W -Wall -Wextra -Wpedantic \
		-Wno-long-long \
		-Wconversion -Wsign-conversion -Wno-sign-compare \
		-Winit-self \
		-Wctor-dtor-privacy -Wnon-virtual-dtor \
		-Wold-style-cast \
		-Wunreachable-code -fstack-protector \
		--std=c++14

ifeq ($(BENCH),on)
CFLAGS = -O3
else
CFLAGS += -g3
endif

$(TARGET_BIN): $(TARGET_OBJECTS) | $(BINDIR)
	@echo $@
	$(MPICC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(CPP_OBJECTS): $(BUILDDIR)/%: $(patsubst %.cpp.o,%.cpp,$(subst __,/,$@)) | $(BUILDDIR)
	@echo $@
	$(CXX) $(CFLAGS) -c $(patsubst $(BUILDDIR)/%.cpp.o,%.cpp,$(subst __,/,$@)) -o $@
