CONVERT_TOOL ?= ../../conv/convert
TEST_TOOL    ?= ../bin/lab3

TESTS = t1 t2 t3

INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))
STDIN  = $(addsuffix /stdin,$(TESTS))

HEX_INPUT  = $(addsuffix .hex,$(INPUT))
HEX_RESULT = $(addsuffix .hex,$(RESULT))

BINARY_INPUT  = $(addsuffix .bin,$(INPUT))
BINARY_RESULT = $(addsuffix .bin,$(RESULT))

BINS   = $(BINARY_INPUT)
HEXS   = $(HEX_INPUT) $(HEX_RESULT)

.PHONY: $(BINARY_RESULT)

.PHONY: all clean
all: $(BINS)
	
clean:
	-rm $(BINS) $(HEX_RESULT)

DIFFS = $(addprefix test-,$(RESULT))
test: $(DIFFS)

$(DIFFS): test-%/res: %/in.hex %/res.hex
	-diff $^

$(HEX_RESULT): %.hex: %.bin
	@$(CONVERT_TOOL) hex -i $^ -o $@ --mode=decode

$(BINS): %.bin: %.hex
	@$(CONVERT_TOOL) hex -i $^ -o $@ --mode=encode

# $(IMAGES): %.png: %.bin
# 	@$(CONVERT_TOOL) image -i $^ -o $@ --mode=decode

$(BINARY_RESULT): %/res.bin: %/in.bin
	echo $^ $@ $(shell cat $*/stdin) | $(TEST_TOOL)
