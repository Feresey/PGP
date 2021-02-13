CONVERT_TOOL ?= ../conv/convert
TEST_TOOL    ?= ../lab2

TESTS = t1 t2

TESTS_INPUT  = $(addsuffix /in,$(TESTS))
TESTS_WANT   = $(addsuffix /want,$(TESTS))
TESTS_RESULT = $(addsuffix /res,$(TESTS))

TESTS_HEX_INPUT  = $(addsuffix .hex,$(TESTS_INPUT))
TESTS_HEX_WANT   = $(addsuffix .hex,$(TESTS_WANT))
TESTS_HEX_RESULT = $(addsuffix .hex,$(TESTS_RESULT))

TESTS_BINARY_INPUT  = $(addsuffix .bin,$(TESTS_INPUT))
TESTS_BINARY_WANT   = $(addsuffix .bin,$(TESTS_WANT))
TESTS_BINARY_RESULT = $(addsuffix .bin,$(TESTS_RESULT))

TESTS_IMAGE_INPUT  = $(addsuffix .png,$(TESTS_INPUT))
TESTS_IMAGE_WANT   = $(addsuffix .png,$(TESTS_WANT))
TESTS_IMAGE_RESULT = $(addsuffix .png,$(TESTS_RESULT))

BINS   = $(TESTS_BINARY_INPUT) $(TESTS_BINARY_WANT)
HEXS   = $(TESTS_HEX_INPUT) $(TESTS_HEX_WANT) $(TESTS_HEX_RESULT)
IMAGES = $(TESTS_IMAGE_INPUT) $(TESTS_IMAGE_WANT) $(TESTS_IMAGE_RESULT)

.PHONY: all clean

all: $(BINS)

clean:
	-rm $(BINS) $(IMAGES)

test:
	@echo $(TESTS_BINARY_RESULT)

%.hex: 

$(BINS): %.bin: %.hex
	$(CONVERT_TOOL) hex -i $^ -o $@ --mode=encode

$(TESTS_BINARY_RESULT): %/res.bin: %/in.bin
	echo $@ $^ | $(TEST_TOOL)
