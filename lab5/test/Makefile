TEST_TOOL ?= ../bin/lab4

TESTS = t1

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
	-diff $^

.PHONY: $(RESULT) # run always
$(RESULT): %/res: %/in
	cat $^ | $(TEST_TOOL) > $@
