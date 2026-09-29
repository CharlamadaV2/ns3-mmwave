# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# tests/

C++ unit tests that compile without ns-3, the CLI integration script, and
regression fixtures. Python tests live beside their packages
(`scripts/rl/tests`, `scripts/validation/tests`), not here.

## Commands

From `scratch/mesh-sim/tests/`. Never build the simulator; integration needs
an existing binary.

```bash
make test                       # lint, then build + run every suite in UNIT_DIRS
make                            # build all suites without running
make -C unit/config test        # one suite: config, eval, routing, traffic
make clean
MESH_SIM_BIN=<BIN> make integration
```

- `make test` stops at the first failing suite; run suites individually to see
  every failure.
- `make lint` (run first by `make test`) is macOS-oriented: it expects Homebrew
  LLVM paths and `xcrun`. On Linux its output isn't meaningful, and it never
  fails the run (`|| true`). It lints only the ns-3-free sources in `LINT_SRCS`.
- CI runs `make integration` and the Python suites, not `make test`.

## Layout

- `unit/<suite>/` -- one `<name>-test.cc` + `Makefile` compiling it directly
  against the `src/` files under test with `-I$(ROOT)`. Assertions use the
  `g_pass`/`g_fail`/`check()` pattern; `main()` calls each `test_*` function
  explicitly, so a new test must also be added there.
- `common/` -- shared ns-3 header stubs (`ns3-log-stub.h`,
  `ns3-random-stub.h`). Suite Makefiles symlink them into a local, gitignored
  `ns3/` dir so `#include "ns3/..."` resolves. `traffic/` has its own
  `ns3-traffic-stub.h`; `config/` needs no stubs.
- `integration/cli-integration-test.sh` -- Tests 1-9 against a real binary
  (`$MESH_SIM_BIN`, else the argument, else the single
  `build/scratch/mesh-sim/ns3*-sim-*` match). Tests 4-9 use `inputs/baselines`
  fixtures; Test 9 checks the legacy RL stream shape.
- `fixtures/regression/p0/` -- only `manifest.json` is tracked. The reference
  `*.json` snapshots are external (gitignored); never recapture them from
  changed code. Used by `scripts/validation/regression_check.py`.
- `CMakeLists.txt` files here are intentionally empty so ns-3's scratch build
  doesn't turn test `.cc` files into targets. Keep them when adding suites.

## Adding a unit suite

1. `unit/<name>/<name>-test.cc` and a `Makefile` modeled on a suite with the
   same stub needs (`ROOT := ../../..`).
2. An empty-guard `CMakeLists.txt` copied from an existing suite.
3. Add the directory to `UNIT_DIRS` in `tests/Makefile` and the binary name to
   `tests/.gitignore`.
4. Only test ns-3-free code (`domain`, `config`, `util`, and the pure parts of
   `eval`, `routing`, `traffic`). If code under test pulls in an ns-3 header,
   stub it in `common/` rather than linking ns-3.
