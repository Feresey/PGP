TEST_TOOL    ?= ../bin/lab4

TESTS = t1

INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))

.PHONY: all clean
.PHONY: $(RESULT)
all: $(RESULT)
	
clean:
	-rm $(RESULT)

DIFFS = $(addprefix test-,$(TESTS))
test: $(DIFFS)

$(DIFFS): test-%: %/want %/res
	-diff $^

$(RESULT): %/res: %/in
	cat $^ | $(TEST_TOOL) > $@
