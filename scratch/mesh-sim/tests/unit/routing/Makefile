CXX      ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wpedantic
ROOT     := ../../..
COMMON   := ../../common

SRCS := mesh-router-test.cc \
        $(ROOT)/src/routing/mesh-router.cc \
        $(ROOT)/src/eval/link-table.cc

OUT := mesh-router-test

# ns3/ stub directory lets #include "ns3/..." resolve without real ns-3.
NS3_STUBS := ns3/log.h ns3/random-variable-stream.h

.PHONY: all test clean

all: $(OUT)

$(OUT): $(NS3_STUBS) $(SRCS)
	$(CXX) $(CXXFLAGS) -I. -I$(ROOT) -o $@ $(SRCS)

ns3/log.h: $(COMMON)/ns3-log-stub.h
	@mkdir -p ns3
	@ln -sf ../../../common/ns3-log-stub.h ns3/log.h

ns3/random-variable-stream.h: $(COMMON)/ns3-random-stub.h
	@mkdir -p ns3
	@ln -sf ../../../common/ns3-random-stub.h ns3/random-variable-stream.h

test: $(OUT)
	@./$(OUT)

clean:
	rm -f $(OUT)
	rm -rf ns3
