BUILDDIR := build
BINDIR   ?= bin
DEPS_DIR := $(BUILDDIR)/deps

TARGET_BIN ?= $(BINDIR)/target

CXX    ?= g++
MPICXX ?= mpic++

.PHONY: all clean

all: $(TARGET_BIN)

clean:
	rm $(TARGET_OBJECTS)

$(DEPS_DIR) $(BUILDDIR) $(BINDIR):
	-mkdir -p "$@"


CPP_SOURCES = main.cpp grid/grid.cpp dim3/dim3.cpp \
			solver.cpp common.cpp problem.cpp exchange.cpp sides.cpp
HEADERS     = helpers.hpp grid/grid.hpp dim3/dim3.hpp \
			problem.hpp solver.hpp exchange.hpp sides.hpp

SOURCES = $(CPP_SOURCES)

COPY_TARGETS = $(addprefix COPY_,$(CPP_SOURCES) $(HEADERS))

.PHONY: $(COPY_TARGETS)
$(COPY_TARGETS): COPY_%:
	mkdir -p $(DESTDIR)/$(dir $*)
	cp $* $(DESTDIR)/$(dir $*)

define stupidmake
all:
	$(MPICXX) $(IFLAGS) $(LDFLAGS) $(CPP_SOURCES) -o $(TARGET)
endef

copy: $(COPY_TARGETS)
	$(file > $(DESTDIR)/Makefile,$(stupidmake))

DEPS_LIST = $(foreach dep,$(SOURCES),$(DEPS_DIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(dep))).mk)

$(DEPS_LIST): $(DEPS_DIR)/%.mk: | $(DEPS_DIR)
	@echo $*
	$(file > $@,$(patsubst %.cpp,$(BUILDDIR)/%.cpp.o,$*): $(shell $(CXX) $(IFLAGS) -E $(subst __,/,$*) | \
		grep -Eo '#\s[0-9]+\s".*$$' - | \
		sed -re 's|#\s[0-9]+\s"(.*?)".*$$|\1|gi' | \
		grep -v -e "^/usr" -e "^<" | \
		sort -u | \
		$(if $(patsubst %.hpp,,$*),,grep -v -e "$*" |) \
		xargs -n1 realpath --relative-to $(CURDIR)))

.PHONY: deps.mk
deps: $(DEPS_LIST) | $(DEPS_DIR)
	@cat $(sort $(DEPS_LIST)) > deps.mk

-include deps.mk

CPP_OBJECTS = $(foreach obj,$(CPP_SOURCES),$(BUILDDIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(obj))).o)

TARGET_OBJECTS = $(CPP_OBJECTS)

show:
	@echo $(TARGET_OBJECTS)

LDFLAGS = -lm -lmpi

IFLAGS = -I. -fopenmp

CFLAGS = -W -Wall -Wextra -Wpedantic \
		-Wno-long-long \
		-Wconversion -Wsign-conversion -Wno-sign-compare \
		-Winit-self \
		-Wctor-dtor-privacy -Wnon-virtual-dtor \
		-Wold-style-cast \
		-Wunreachable-code -fstack-protector \
		--std=c++14 \
		-Wno-variadic-macros

ifeq ($(BENCH),on)
CFLAGS = -O3
else
CFLAGS += -g3
endif

$(TARGET_BIN): $(TARGET_OBJECTS) | $(BINDIR)
	@echo $@
	$(MPICXX) $(CFLAGS) $(IFLAGS) $(LDFLAGS) $^ -o $@

$(CPP_OBJECTS): $(BUILDDIR)/%: | $(BUILDDIR)
	@echo $@
	$(CXX) $(CFLAGS) $(IFLAGS) -c $(patsubst $(BUILDDIR)/%.cpp.o,%.cpp,$(subst __,/,$@)) -o $@
