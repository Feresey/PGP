CONVERT_TOOL ?= ../../convert/convert
TEST_TOOL    ?= ../bin/lab5

TESTS = t1 t2 t3 t4 t5 t6

DIFFS = $(addprefix test-,$(TESTS))

INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))

HEX_INPUT  = $(addsuffix .hex,$(INPUT))
HEX_WANT   = $(addsuffix .hex,$(WANT))
HEX_RESULT = $(addsuffix .hex,$(RESULT))

BINARY_INPUT  = $(addsuffix .bin,$(INPUT))
BINARY_RESULT = $(addsuffix .bin,$(RESULT))

HEXS   = $(HEX_INPUT) $(HEX_RESULT)

.PHONY: all clean

all: $(DIFFS)

clean:
	-rm $(BINARY_INPUT) $(BINARY_RESULT) $(HEX_RESULT)

.PHONY: $(DIFFS)
$(DIFFS): test-%: %/want.hex %/res.hex
	-diff -q $^

$(HEX_RESULT): %.hex: %.bin
	@$(CONVERT_TOOL) hex-array --first-length=false -i $^ -o $@ --mode=decode

$(BINARY_INPUT): %.bin: %.hex
	@$(CONVERT_TOOL) hex-array --first-length=true -i $^ -o $@ --mode=encode

.PHONY: $(BINARY_RESULT) # run always
$(BINARY_RESULT): %/res.bin: %/in.bin
	cat $^ | $(TEST_TOOL) > $@
