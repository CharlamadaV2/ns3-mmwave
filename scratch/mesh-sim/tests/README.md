@page tests Tests

@brief C++ unit tests that build without ns-3, a CLI integration script, and regression fixtures.

The unit suites compile selected @c src/ files directly with stubbed ns-3
headers, so they run on any machine with a C++17 compiler. The integration
script drives an already-built simulator binary. Python tests live beside their
packages, see the [RL test map](@ref scripts_rl_tests) and the
[validation package](@ref scripts_validation).

## Setup

No setup is needed for the unit suites. For integration, build the simulator
first (this directory never builds it) and point at the binary:

```bash
export MESH_SIM_BIN=<path-to-built-sim-binary>
```

## Run

Run from `scratch/mesh-sim/tests/`.

```bash
make test                  # lint, then build and run every suite in UNIT_DIRS
make                       # build all suites without running
make -C unit/config test   # one suite: config, eval, routing, traffic
make integration           # 19 real-binary CLI checks, needs MESH_SIM_BIN
make clean                 # remove built binaries and stub directories
```

- `make test` stops at the first failing suite. Run suites one by one to see every failure.
- `make lint` is macOS-oriented (Homebrew LLVM paths and `xcrun`). On Linux its output is not meaningful, and it never fails the run.
- The integration script also accepts the binary as its first argument. If neither is given it uses the single match of `build/scratch/mesh-sim/ns3*-sim-*`.
- CI runs `make integration` and the Python suites, not `make test`.

## Module Layout

| File | Role |
| --- | --- |
| @c Makefile | Top-level runner: @c UNIT_DIRS, @c lint, @c test, @c integration, @c clean. |
| @c CMakeLists.txt | Intentionally empty so the ns-3 scratch build ignores test sources. |
| @c .gitignore | Ignores built test binaries and the generated @c ns3/ stub directories. |
| @c unit/config/config-validator-test.cc | Config validation, band, reward, baseline selectors, RL control resolution, layout override, seed parsing. |
| @c unit/eval/eval-test.cc | SinrToCapacity, SinrToMcsIndex, LinkTable. |
| @c unit/routing/mesh-router-test.cc | MeshRouter paths, algorithms, max hops, latency, congestion. |
| @c unit/traffic/traffic-matrix-test.cc | TrafficMatrix topologies, expiry, on/off phases. |
| @c unit/traffic/ns3-traffic-stub.h | Deterministic RNG stubs used only by the traffic suite. |
| @c unit/*/Makefile | Per-suite build, compiling the test against the @c src/ files under test. |
| @c unit/*/CMakeLists.txt | Empty guards, same purpose as the top-level one. |
| @c common/ns3-log-stub.h | No-op stub for @c ns3/log.h (eval, routing, traffic suites). |
| @c common/ns3-random-stub.h | Empty-type stub for @c ns3/random-variable-stream.h (routing suite). |
| @c integration/cli-integration-test.sh | Tests 1-19 against a real simulator binary. |
| @c fixtures/regression/p0/manifest.json | Case list for the regression suite in @ref scripts_validation. |

## Output

| Step | Where |
| --- | --- |
| Unit build | Test binary in each @c unit/&lt;suite&gt;/ (for example @c config-validator-test), plus a local @c ns3/ symlink directory. Both are gitignored. |
| Unit run | Console only. Failures print @c FAIL to stderr, then a pass/fail count. Exit code 0 means all passed. |
| Integration | Console only: one @c PASS or @c FAIL line per check and a final count. Temporary output is deleted on exit. |

## Conventions

- Assertions use the @c g_pass / @c g_fail / @c check() pattern. @c main() calls each @c test_* function by name, so a new test must also be added there.
- Only ns-3-free code is tested (@c domain, @c config, @c util, and the pure parts of @c eval, @c routing, @c traffic). If code under test includes an ns-3 header, stub it in @c common/ instead of linking ns-3.
- Suite Makefiles symlink stubs into a local @c ns3/ directory so `#include "ns3/..."` resolves.
- The @c CMakeLists.txt files are empty on purpose. Keep them when adding suites.
- Regression snapshots under @c fixtures/regression/p0/ other than @c manifest.json are external and untracked. Never recapture them from changed code.
- To add a suite: create `unit/<name>/<name>-test.cc` and a @c Makefile (copy one with the same stub needs, @c ROOT is `../../..`), copy an empty @c CMakeLists.txt, add the directory to @c UNIT_DIRS in the top-level @c Makefile, and add the binary name to @c .gitignore.

### Integration checks

All checks use the smoke scenarios in @c inputs/baselines (@c p0-smoke and @c p0-jammer-smoke). Test 17 also uses @c 16-random-walk-urban.

| Test | Checks |
| --- | --- |
| 1-3 | `--PrintHelp` works, missing `--run-config` and missing file exit nonzero. |
| 4-5 | Bad `--seeds` value and nonexistent positions override give named errors. |
| 6-7 | A valid run writes @c run.log and @c seed-1 outputs. `--band` overrides the scenario band. |
| 8 | Same jammer scenario under `sub-6` and `mmwave` gives matching link rows with different SINR. |
| 9 | Legacy `--rl-mode` stream with closed stdin: five `step` lines, no `init`. |
| 10-12 | Active baseline algorithm is refused without `--rl-mode`, `none` runs normally, explicit `--rl-mode` runs with a notice. |
| 13-19 | `--channel-query` worker: init line, one-layout reply, error handling, one-seed rule, out-of-bounds layout, large reply drain, SIGTERM cleanup. |

Tests 13-19 also need `python3`.

## Dependencies

- C++17 compiler (`c++` by default, override with @c CXX; flags with @c CXXFLAGS) and GNU make for the unit suites.
- Optional @c clang-tidy for `make lint`; without it lint prints a warning and is skipped.
- Integration: a built simulator binary, bash, and `python3` for Tests 13-19. Fixtures come from @c inputs/baselines.
