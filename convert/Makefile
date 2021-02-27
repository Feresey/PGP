TARGET = convert

.PHONY: all
all: $(TARGET)

$(TARGET):
	CGO_ENABLED=0 go build -v -o $(TARGET)

test:
	go test -v .