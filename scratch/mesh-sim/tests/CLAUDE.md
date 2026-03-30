# tests/

## Scope
Standalone unit and integration tests for mesh-sim.
Unit tests compile without ns-3 -- they only test pure-logic code (config, eval math, link table).

## Running tests

```bash
# From scratch/mesh-sim/tests/:
make test          # build + run all unit tests
make clean         # remove all build artifacts
make integration   # run CLI integration tests (requires built sim binary)
```

## Directory structure

```
tests/
  common/                  # shared test infrastructure
    ns3-log-stub.h         #   minimal ns3/log.h stub for standalone compilation
  unit/                    # unit tests (no ns-3 dependency)
    config/                #   config validation + seed parsing
    eval/                  #   SinrToCapacity + LinkTable
  integration/             # tests that run the sim binary
    cli-integration-test.sh
  Makefile                 # top-level runner: delegates to unit/*/Makefile
  CMakeLists.txt           # empty guard (prevents ns-3 from auto-targeting .cc files)
```

## Adding a new unit test suite

1. Create a directory under `unit/` (e.g. `unit/routing/`).
2. Add a `<name>-test.cc` using the `g_pass`/`g_fail`/`check()` pattern from existing tests.
3. Add a `Makefile` following the pattern in `unit/config/Makefile` or `unit/eval/Makefile`.
   - Set `ROOT := ../../..` to reach the mesh-sim root.
   - If the code under test includes `ns3/log.h`, use the stub setup from `unit/eval/Makefile`.
4. Add the new directory to `UNIT_DIRS` in the top-level `tests/Makefile`.
5. Add the binary name to `.gitignore`.

## Dependencies
- Unit tests depend only on `domain/`, `config/`, `eval/`, and `util/` -- never on ns-3.
- Integration tests depend on a built sim binary.
