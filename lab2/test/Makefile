CONVERT_TOOL ?= ../../convert/convert
TEST_TOOL    ?= ../bin/lab2

TESTS = t1 t2 t3 t4 t5

INPUT  = $(addsuffix /in,$(TESTS))
WANT   = $(addsuffix /want,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))

HEX_INPUT  = $(addsuffix .hex,$(INPUT))
HEX_WANT   = $(addsuffix .hex,$(WANT))
HEX_RESULT = $(addsuffix .hex,$(RESULT))

BINARY_INPUT  = $(addsuffix .bin,$(INPUT))
BINARY_WANT   = $(addsuffix .bin,$(WANT))
BINARY_RESULT = $(addsuffix .bin,$(RESULT))

IMAGE_INPUT  = $(addsuffix .png,$(INPUT))
IMAGE_WANT   = $(addsuffix .png,$(WANT))
IMAGE_RESULT = $(addsuffix .png,$(RESULT))

BINS   = $(BINARY_INPUT) $(BINARY_WANT)
HEXS   = $(HEX_INPUT) $(HEX_WANT) $(HEX_RESULT)
IMAGES = $(IMAGE_INPUT) $(IMAGE_WANT) $(IMAGE_RESULT)

.PHONY: all test images clean
all: test
	
clean:
	-rm $(BINS) $(IMAGES) $(HEX_RESULT)

run: $(IMAGES)

DIFFS = $(addprefix test-,$(RESULT))
test: $(DIFFS)

images: $(IMAGES)

.PHONY: $(DIFFS)
$(DIFFS): test-%/res: %/want.hex %/res.hex
	-diff -q $^

$(HEX_RESULT): %.hex: %.bin
	@$(CONVERT_TOOL) hex -i $^ -o $@ --mode=decode

$(BINS): %.bin: %.hex
	@$(CONVERT_TOOL) hex -i $^ -o $@ --mode=encode

$(IMAGES): %.png: %.bin
	@$(CONVERT_TOOL) image -i $^ -o $@ --mode=decode

.PHONY: $(BINARY_RESULT)
$(BINARY_RESULT): %/res.bin: %/in.bin
	echo $^ $@ | $(TEST_TOOL)
