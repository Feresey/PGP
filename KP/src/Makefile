BUILDDIR := build
BINDIR   ?= bin
DEPS_DIR := $(BUILDDIR)/deps

TARGET ?= target
TARGET_BIN ?= $(BINDIR)/$(TARGET)

CXX    ?= g++
NVCC   ?= nvcc
MPICXX ?= mpic++

.PHONY: all clean

all: $(TARGET_BIN)

clean:
	rm $(TARGET_OBJECTS)

$(DEPS_DIR) $(BUILDDIR) $(BINDIR):
	-mkdir -p "$@"

HEADERS = helpers.cuh helpers.hpp ssaa.hpp scene.hpp \
		render/render.cuh \
		dim3/dim3.hpp \
		mat/mat3.hpp \
		vec/vec3.hpp \

CU_SOURCES  = mat/mat.cu vec/vec.cu ssaa.cu render/render.cu helpers.cu
CPP_SOURCES = main.cpp scene.cpp render/render.cpp render/objects.cpp

SOURCES = $(CU_SOURCES)

COPY_TARGETS = $(addprefix COPY_,$(CU_SOURCES) $(CPP_SOURCES) $(HEADERS))

.PHONY: $(COPY_TARGETS)
$(COPY_TARGETS): COPY_%:
	mkdir -p $(DESTDIR)/$(dir $*)
	cp $* $(DESTDIR)/$(dir $*)

define stupidmake
CXX    ?= $(CXX)
NVCC   ?= $(NVCC)
MPICXX ?= $(MPICXX)

.PHONY: all
all: $(TARGET)

CU_SOURCES  = $(CU_SOURCES)
CPP_SOURCES = $(CPP_SOURCES)

CPP_OBJECTS = $$(foreach obj,$$(CPP_SOURCES),$$(obj).o)
CU_OBJECTS = $$(foreach obj,$$(CU_SOURCES),$$(obj).o)

TARGET_OBJECTS = $$(CPP_OBJECTS) $$(CU_OBJECTS)

LDFLAGS = -lm -lmpi
IFLAGS = -I.
NVCCFLAGS = -rdc=true
GCCFLAGS = --std=c++11

$(TARGET): $$(TARGET_OBJECTS)
	$$(NVCC) -ccbin=$$(MPICXX) $$(NVCCFLAGS) $$(IFLAGS) $$(LDFLAGS) $$^ -o $$@

$$(CPP_OBJECTS): %.o:
	$$(CXX) $$(GCCFLAGS) $$(IFLAGS) -c $$* -o $$@

$$(CU_OBJECTS): %.o:
	$$(NVCC) $$(NVCCFLAGS) $$(IFLAGS) -c $$* -o $$@
endef

copy: $(COPY_TARGETS)
	$(file > $(DESTDIR)/Makefile,$(stupidmake))

CPP_DEPS_LIST = $(foreach dep,$(CPP_SOURCES),$(DEPS_DIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(dep))).mk)
CU_DEPS_LIST = $(foreach dep,$(CU_SOURCES),$(DEPS_DIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(dep))).mk)

$(CPP_DEPS_LIST): COMPILE=$(CXX) $(IFLAGS)
$(CU_DEPS_LIST): COMPILE=$(NVCC) $(IFLAGS)

DEPS_LIST = $(CU_DEPS_LIST) $(CPP_DEPS_LIST)
$(DEPS_LIST): $(DEPS_DIR)/%.mk: | $(DEPS_DIR)
	@echo $*
	$(file > $@,$(patsubst %.cu,$(BUILDDIR)/%.cu.o,$(patsubst %.cpp,$(BUILDDIR)/%.cpp.o,$*)): \
		$(shell $(COMPILE) -E $(subst __,/,$*) | \
		grep -Eo '#\s[0-9]+\s".*$$' - | \
		sed -re 's|#\s[0-9]+\s"(.*?)".*$$|\1|gi' | \
		grep -v -e "^/usr" -e "^<" -e "cuda" | \
		sort -u | \
		$(if $(patsubst %.hpp,,$*),,grep -v -e "$*" |) \
		xargs -n1 realpath --relative-to $(CURDIR) | \
		sort -u \
		))

deps: $(DEPS_LIST) | $(DEPS_DIR)
	@cat $(sort $(DEPS_LIST)) > deps.mk

-include deps.mk

CPP_OBJECTS = $(foreach obj,$(CPP_SOURCES),$(BUILDDIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(obj))).o)
CU_OBJECTS = $(foreach obj,$(CU_SOURCES),$(BUILDDIR)/$(subst /,__,$(shell realpath --relative-to $(CURDIR) $(obj))).o)

TARGET_OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

LDFLAGS = -lm -lmpi

IFLAGS = -I.

CFLAGS ?= --std=c++14 \
		-W -Wall -Wextra \
		-fopenmp

GCCFLAGS = $(CFLAGS) \
		-Wconversion -Wsign-conversion -Wno-sign-compare \
		-Wno-long-long \
		-Winit-self \
		-Wctor-dtor-privacy -Wnon-virtual-dtor \
		-Wunreachable-code -fstack-protector

NVCCFLAGS ?= --std=c++14 -rdc=true\
		-Werror cross-execution-space-call \
		-arch=sm_35 -Wno-deprecated-gpu-targets \
		-D_FORCE_INLINES -Xcompiler -fPIC \
		-Wno-deprecated-declarations \
		--expt-relaxed-constexpr

ifeq ($(BENCH),on)
CFLAGS    = -O3 -DBENCHMARK
NVCCFLAGS += -O3
else
CFLAGS    += -g3
NVCCFLAGS += -G
endif
NVCCFLAGS += --compiler-options "$(CFLAGS)"


$(TARGET_BIN): $(TARGET_OBJECTS) | $(BINDIR)
	$(NVCC) -ccbin=$(MPICXX) $(NVCCFLAGS) $(IFLAGS) $(LDFLAGS) $^ -o $@

$(CPP_OBJECTS): $(BUILDDIR)/%: | $(BUILDDIR)
	$(CXX) $(GCCFLAGS) $(IFLAGS) -c $(patsubst $(BUILDDIR)/%.cpp.o,%.cpp,$(subst __,/,$@)) -o $@

$(CU_OBJECTS): $(BUILDDIR)/%: | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(IFLAGS) -c $(patsubst $(BUILDDIR)/%.cu.o,%.cu,$(subst __,/,$@)) -o $@
