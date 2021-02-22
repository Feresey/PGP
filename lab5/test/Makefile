TEST_TOOL ?= ../bin/lab5

TESTS = t1 t2 t3 t4 t5

INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))
DIFFS = $(addprefix test-,$(TESTS))

.PHONY: all clean

all: $(DIFFS)

clean:
	-rm $(RESULT)

.PHONY: $(DIFFS)
$(DIFFS): test-%: %/want %/res
	-diff -q $^

.PHONY: $(RESULT) # run always
$(RESULT): %/res: %/in
	cd $* && cat in | $(TEST_TOOL) > res
