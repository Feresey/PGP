CONVERT_TOOL ?= $(CURDIR)/../../convert/convert
TEST_TOOL    ?= $(CURDIR)/../bin/lab5

TESTS = t1 t2 t3 t4 t5 t6 t7

DIFFS = $(addprefix test-,$(TESTS))

INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))

.PHONY: all clean

all: $(DIFFS)

clean:
	-rm $(RESULT)

EQUAL_TOOL = $(CURDIR)/equal/equal
$(EQUAL_TOOL): $(dir $(EQUAL_TOOL))
	go build -v -o $@ $^

.PHONY: $(DIFFS)
$(DIFFS): test-%: %/want %/res | $(EQUAL_TOOL)
	$(EQUAL_TOOL) --want $(word 1,$^) --result $(word 2,$^)

.PHONY: $(RESULT) # run always
$(RESULT): %/res: %/in
	cd $* && $(TEST_TOOL) < in > res
