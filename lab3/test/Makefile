CONVERT_TOOL ?= ../../convert/convert
TEST_TOOL    ?= ../bin/lab3

TESTS = t1 t2 t3 t4

STDIN  = $(addsuffix /stdin,$(TESTS))
INPUT  = $(addsuffix /in,$(TESTS))
RESULT = $(addsuffix /res,$(TESTS))

HEX_INPUT  = $(addsuffix .hex,$(INPUT))
HEX_RESULT = $(addsuffix .hex,$(RESULT))

BINARY_INPUT  = $(addsuffix .bin,$(INPUT))
BINARY_RESULT = $(addsuffix .bin,$(RESULT))

IMAGE_INPUT  = $(addsuffix .png,$(INPUT))
IMAGE_RESULT = $(addsuffix .png,$(RESULT))

BINS   = $(BINARY_INPUT) $(BINARY_WANT)
HEXS   = $(HEX_INPUT) $(HEX_RESULT)
IMAGES = $(IMAGE_INPUT) $(IMAGE_RESULT)

.PHONY: all test images clean
all: test
	
clean:
	-rm $(BINS) $(IMAGES) $(HEX_RESULT)

run: $(IMAGES)

DIFFS = $(addprefix test-,$(RESULT))
test: $(DIFFS)

images: $(IMAGES)

.PHONY: $(DIFFS)
$(DIFFS): test-%/res: %/in.hex %/res.hex
	-diff -q $^

$(HEX_RESULT): %.hex: %.bin
	@$(CONVERT_TOOL) hex -i $^ -o $@ --mode=decode

$(BINS): %.bin: %.hex
	@$(CONVERT_TOOL) hex -i $^ -o $@ --mode=encode

$(IMAGES): %.png: %.bin
	$(CONVERT_TOOL) image -i $^ -o $@ --mode=decode

.PHONY: $(BINARY_RESULT)
$(BINARY_RESULT): %/res.bin: %/in.bin
	echo $^ $@ $(shell cat $*/stdin) | $(TEST_TOOL)
