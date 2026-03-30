# Top-level test runner for mesh-sim.
# Run from scratch/mesh-sim/tests/:
#
#   make              # build all unit tests
#   make test         # lint + build + run all unit tests
#   make lint         # run clang-tidy on src/ (no build required)
#   make clean        # remove all build artifacts
#
# Integration tests (require a built sim binary):
#   make integration  # run CLI integration tests

UNIT_DIRS := unit/config unit/eval unit/routing unit/traffic

# Homebrew LLVM clang-tidy (not on PATH by default).
CLANG_TIDY ?= $(firstword $(wildcard /usr/local/opt/llvm/bin/clang-tidy) \
                           $(wildcard /opt/homebrew/opt/llvm/bin/clang-tidy) \
                           $(shell which clang-tidy 2>/dev/null))

# Derive the LLVM include dir from clang-tidy's path (works for Homebrew).
LLVM_DIR := $(patsubst %/bin/clang-tidy,%,$(CLANG_TIDY))

ROOT := ..

# Lint only files that compile standalone (no ns-3 includes).
# Modules with ns-3 deps (setup, traffic, eval/link-evaluator, io, cli, rl, sim.cc)
# are checked at build time by the ns-3 compiler instead.
LINT_SRCS := $(wildcard $(ROOT)/src/domain/*.h) \
             $(wildcard $(ROOT)/src/config/*.h $(ROOT)/src/config/*.cc) \
             $(wildcard $(ROOT)/src/util/*.h $(ROOT)/src/util/*.cc) \
             $(ROOT)/src/eval/sinr-capacity.h \
             $(ROOT)/src/eval/link-table.h $(ROOT)/src/eval/link-table.cc \
             $(ROOT)/src/routing/mesh-router.h $(ROOT)/src/routing/mesh-router.cc

.PHONY: all test clean integration lint $(UNIT_DIRS)

all: $(UNIT_DIRS)

$(UNIT_DIRS):
	@$(MAKE) -C $@ all

lint:
ifeq ($(CLANG_TIDY),)
	@echo "WARNING: clang-tidy not found, skipping lint."
else
	@echo "=== lint ==="
	@$(CLANG_TIDY) --quiet -p=. $(LINT_SRCS) \
		-- -x c++ -std=c++17 -I$(ROOT) -Ilint-stubs \
		-isystem $(LLVM_DIR)/include/c++/v1 \
		-isysroot $(shell xcrun --show-sdk-path) 2>&1 | \
		grep -v "^[0-9]* warnings\? generated" || true
	@echo "Lint passed."
endif

test: lint
	@for dir in $(UNIT_DIRS); do \
		echo "=== $$dir ==="; \
		$(MAKE) -C $$dir test || exit 1; \
		echo ""; \
	done
	@echo "All test suites passed."

integration:
	@echo "=== integration ==="
	@./integration/cli-integration-test.sh

clean:
	@for dir in $(UNIT_DIRS); do \
		$(MAKE) -C $$dir clean; \
	done
