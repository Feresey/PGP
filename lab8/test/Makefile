CONVERT_TOOL ?= $(CURDIR)/../../convert/convert
TEST_TOOL    ?= $(CURDIR)/../bin/lab8

TESTS = t1 t2 t3

DIFFS = $(addprefix test-,$(TESTS))

STDIN  = $(addsuffix /stdin,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))

.PHONY: all clean

all: $(DIFFS)

clean:
	-rm $(RESULT)

test: $(DIFFS)

.PHONY: $(DIFFS)
$(DIFFS): test-%: %/want %/res
	-diff -q $^

.PHONY: $(RESULT)
$(RESULT): %/res: %/stdin
	cd $* && mpirun -np $(shell cat $(dir $^)/params) $(TEST_TOOL) < stdin > res
