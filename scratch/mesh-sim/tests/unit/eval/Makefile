CXX      ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wpedantic
ROOT     := ../../..
COMMON   := ../../common

SRCS := eval-test.cc \
        $(ROOT)/src/eval/link-table.cc

OUT := eval-test

# The ns3/ stub directory lets #include "ns3/log.h" resolve without real ns-3.
NS3_STUB := ns3/log.h

.PHONY: all test clean

all: $(OUT)

$(OUT): $(NS3_STUB) $(SRCS)
	$(CXX) $(CXXFLAGS) -I. -I$(ROOT) -o $@ $(SRCS)

$(NS3_STUB): $(COMMON)/ns3-log-stub.h
	@mkdir -p ns3
	@ln -sf ../../../common/ns3-log-stub.h ns3/log.h

test: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OUT)
	rm -rf ns3
