# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# mesh-sim

Lightweight time-stepped mmWave / sub-6 mesh simulator built on ns3-mmwave.
All nodes are identical peers (no eNB/UE distinction). Uses ns-3 propagation
models as library functions. Uses `Simulator::Stop()`+`Run()` as a controlled
time-stepper each tick but does not use the ns-3 event loop for control flow.
C++ for simulation; Python for field-data pipelines, validation, sweeps, the RL
environment/training, and plotting.
Entry point: `sim.cc`.

## Commands

Never run `./ns3 build` or `./ns3 run` -- the user handles builds. Python tools
that launch the simulator take `--sim-binary <BIN>` for an already-built binary.

```bash
# C++ unit tests (no ns-3 needed), from scratch/mesh-sim/tests/
make test                    # lint + build + run all suites in UNIT_DIRS
make -C unit/config test     # one suite (config, eval, routing, traffic)
make integration             # CLI integration tests; needs a built sim binary

# Python env, from scratch/mesh-sim/
python3 scripts/rl/bootstrap_venv.py          # creates .venv from requirements.txt
python3 scripts/rl/bootstrap_venv.py --check  # verify installed versions only

# Python tests, from scratch/mesh-sim/
.venv/bin/python -m pytest scripts/rl/tests -q
.venv/bin/python -m pytest scripts/validation/tests
.venv/bin/python -m pytest scripts/rl/tests/test_mesh_env.py::<test_name>
```

- `make lint` (run by `make test`) uses clang-tidy with `xcrun`, so it is
  macOS-oriented; it only lints the ns-3-free sources.
- Python scripts use package-relative imports: always run them as
  `python -m scripts.<pkg>.<module>` from `scratch/mesh-sim/`.
- `requirements-tuning.txt` (Optuna) is deliberately separate from
  `requirements.txt`, whose contents are recorded in every RL manifest.
- Simulator CLI: `--run-config=<run.ini>` (the file, not the dir), `--band`,
  `--seeds=1,2,...`, `--output-dir`, `--rl-mode`, `--debug-links`, `--channel-query`. See `README.md`.

## Commit guidelines

- Use conventional commit format: `type(scope): short summary` (e.g. `feat(mesh-sim):`, `fix(mesh-sim):`).
- Keep the summary under 72 characters; use the body for the *why*, not the *what*.
- Group changes into logical commits -- one concern per commit. Prefer many focused commits over one monolithic commit.
- Types: `feat` (new functionality), `fix` (bug fix), `refactor` (restructure, no behavior change), `docs` (docs only), `chore`/`build` (tooling/CI/build).
- Scope should reflect the module being changed (e.g. `mesh-sim`, `link-evaluator`, `mesh-router`).
- Use the human's configured Git author/committer identity. Never add
  `Co-Authored-By`, `Signed-off-by`, or AI-generated attribution to commits or
  PR text; verify the final message before publishing.

## Code guidelines

- **File scope awareness**: Understand what each directory owns before editing.
  Don't pile unrelated logic into one file. Most subdirectories have a CLAUDE.md
  or README.md describing their scope -- read it first.
- **Module boundaries**: Before substantially extending a script, check whether
  the new behavior belongs to an existing module or a cohesive extraction.
  File size is a review signal, not a quota; avoid both monoliths and tiny
  wrappers created only to lower a line count.
- Comments, Doxygen blocks, and README/doc pages follow the documenter agent's rules
  (`~/.claude/agents/documenter.md`); use that agent for documentation work.
- When adding new source files, update `CMakeLists.txt`'s source list.
- New C++ unit test suites: follow `tests/CLAUDE.md` (add to `UNIT_DIRS`).
- Before changing a user-visible behavior (`run.ini`/CLI option, RL
  action/observation/reward semantics, output files, scenario identity), check
  the "update together" table in `CONTRIBUTING.md`. Bump the relevant
  `manifest_version` when a saved JSON schema changes; change the RL `contract`
  name (`mesh_move_2d_v1`) only for incompatible protocol changes.

## Architecture

- `src/` is split by concern: cli, config, domain, eval, io, jammer, query, routing, rl, setup, traffic, util
- `domain/` types are pure POD with no ns-3 dependency -- everything else depends on them
- `eval/` wraps ns-3 propagation models to compute per-link SINR and capacity
- `jammer/` adds jammer received power to the SINR denominator; active only when `band == sub-6` and `jammers.json` defines jammers
- `traffic/` generates demand matrices (no packets -- flow-level abstraction)
- `routing/` routes flows over the mesh using the link table
- `setup/` is the only layer that creates ns-3 objects (nodes, mobility, buildings, propagation models)
- `io/` writes CSV and JSON output compatible with the mmwave-sim GUI
- `rl/` is the C++ side of a stdin/stdout JSON bridge to the Python Gymnasium env in `scripts/rl/`; the contract is in `src/rl/README.md`
- `config/` has no ns-3 headers -- can be compiled/tested independently
- All C++ code lives in namespace `mesh_sim`

### Run flow (`sim.cc`)

1. Parse CLI, load `run.ini` + `nodes.json` (+ `buildings.json`, `jammers.json`),
   apply CLI overrides, validate, resolve RL control.
2. Archive scenario inputs into the output dir and write `run.log`
   (its "Resolved config" block is the source of truth for a run).
3. Per seed, into `seed-<N>/`: `TopologyBuilder` -> `LinkEvaluator` ->
   per tick { advance ns-3 clock, evaluate all links into `LinkTable`,
   `TrafficMatrix::Tick`, `MeshRouter::Route`, write metrics/viz, RL step }.

Band resolves as `--band` > `[channel] band` in `run.ini` > default `mmwave`.

## Dependency layers

```
domain  <--  config, cli, eval, io, jammer, query, routing, rl, setup, traffic
jammer  <--  domain (sim-config.h includes jammer-spec.h), eval
util    <--  cli, config, io
cli     <--  io, query, sim.cc
config  <--  query, setup, sim.cc
setup   <--  query, sim.cc
eval    <--  io, query, routing, rl (link-table), sim.cc
traffic <--  routing, sim.cc
routing <--  io, rl, sim.cc
io      <--  sim.cc
rl      <--  sim.cc
query   <--  sim.cc
```

## Python side (`scripts/`)

- `arpo_data/` -- turns raw ARPO field data into per-day / per-scenario traces
- `validation/` -- builds scenarios from field data, batch-runs the sim,
  compares sim vs field distributions; also the baseline regression and smoke checks
- `sweep/` -- parameter sweeps driven by a `sweep.ini`
- `rl/` -- Gymnasium env, training (MaskablePPO), evaluation, experiment/ops tooling
- `baselines/` -- gateway-free `geometric` / `optimization` placement planners; score candidate
  layouts through the binary's `--channel-query` worker (`src/query/`); see its README
- `plotting/` -- post-sim plots
- `sim_support.py` -- shared simulator launcher helpers
- `stats.py` -- shared sample statistics (t critical values for 95% CIs)

## Input/output conventions

- Scenarios live under `inputs/` (`baselines/`, `calfex/`, `experiments/`, `sweeps/`),
  each with `run.ini` + `nodes.json` + optional `buildings.json` / `jammers.json`
- Default outputs go to `outputs/YYYY-MM/DD/HH-MM-SS/seed-N/`; `--output-dir` overrides
- Scenario inputs are archived into each output run for reproducibility
- Regression reference snapshots (`tests/fixtures/regression/p0/*.json` except
  `manifest.json`) are external and untracked; never recapture them from changed code

## Key differences from mmwave-sim

- No EPC, RRC, MAC, HARQ, RLC, PDCP -- direct propagation model calls only
- `Simulator::Stop()`+`Run()` advances the clock each tick; for-loop remains the master
- All nodes are peers (no base station / UE roles)
- Evaluates all N*(N-1)/2 links per tick (not just eNB-UE pairs)
- Traffic is flow-level demands, not packet-level
- Routing layer computes multi-hop paths over the mesh
