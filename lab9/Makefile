SRCDIR   = $(CURDIR)/src
BUILDDIR = $(CURDIR)/build
BINDIR   = $(CURDIR)/bin
SEND_DIR = $(CURDIR)/send

$(CONVERT_TOOL):
	cd $(dir $@) && go build -v

TARGET = lab9
TARGET_BIN = $(BINDIR)/$(TARGET)
SEND_TARGET_DIR = $(SEND_DIR)/$(TARGET)

.PHONY: all clean
all: $(TARGET)
clean:
	-rm -rf $(BINDIR)/*
	$(MAKE) -C src clean

$(BUILDDIR) $(BINDIR) $(SEND_DIR) $(SEND_TARGET_DIR):
	mkdir -p $@

.PHONY: test
test: $(TARGET_BIN) | $(CONVERT_TOOL)
	$(MAKE) -j1 -C test TEST_TOOL="$(PREFIX) $^"

.PHONY: send
send: | $(SEND_DIR) $(SEND_TARGET_DIR)
	mkdir -p $(SEND_TARGET_DIR)
	$(MAKE) -C $(SRCDIR) copy DESTDIR=$(SEND_TARGET_DIR) TARGET=$(TARGET)
	cd $(SEND_DIR) && tar cvf $(TARGET).tar $(TARGET)
	gpg -r 703FD5D038947C5B57CCE5A64DF7496E16FD3CBC -ab $(SEND_DIR)/$(TARGET).tar

.PHONY: $(TARGET)
$(TARGET): $(TARGET_BIN)

.PHONY: $(TARGET_BIN)
$(TARGET_BIN): | $(BUILDDIR) $(BINDIR)
	$(MAKE) -C src BINDIR=$(BINDIR) TARGET_BIN=$(TARGET_BIN)
