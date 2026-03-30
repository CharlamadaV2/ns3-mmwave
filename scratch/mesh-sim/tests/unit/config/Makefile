CXX      ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wpedantic
ROOT     := ../../..

SRCS := config-validator-test.cc \
        $(ROOT)/src/config/config-validator.cc \
        $(ROOT)/src/config/config-loader.cc \
        $(ROOT)/src/util/string-utils.cc \
        $(ROOT)/src/util/ini-parser.cc

OUT := config-validator-test

.PHONY: all test clean

all: $(OUT)

$(OUT): $(SRCS)
	$(CXX) $(CXXFLAGS) -I$(ROOT) -o $@ $^

test: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OUT)
