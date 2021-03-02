CONVERT_TOOL ?= $(CURDIR)/../../convert/convert
TEST_TOOL    ?= $(CURDIR)/../bin/lab4

TESTDIRS = t1 t2 t3 t4 t5
TESTS = $(addprefix $(CURDIR)/,$(TESTDIRS))

INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))

.PHONY: all clean

all: $(TESTS)

clean:
	-rm $(RESULT)

EQUAL_TOOL = $(CURDIR)/equal/equal
$(EQUAL_TOOL): $(dir $(EQUAL_TOOL))/main.go
	go build -v -o $@ $^

.PHONY: $(TESTDIRS)
$(TESTDIRS): $(CURDIR)/$@
$(TESTDIRS): ;

.PHONY: $(TESTS)
$(TESTS): %: %/want %/res | $(EQUAL_TOOL)
	-$(EQUAL_TOOL) --want $(word 1,$^) --result $(word 2,$^)

.PHONY: $(RESULT) # run always
$(RESULT): %/res: %/in
	$(TEST_TOOL) < $^ > $@
