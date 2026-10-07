@mainpage Overview

@brief Lightweight time-stepped mmWave / sub-6 mesh simulator built on ns3-mmwave.

The sim loads a scenario (@c run.ini + @c nodes.json), steps through time evaluating per-link SINR / capacity / MCS
against an ns-3 propagation model, routes traffic, and writes metrics. It also
supports an optional reinforcement-learning (RL) bridge and an optional jammer /
interference model.

For changes to configuration, the RL protocol, or saved results, see the
[contributor checklist](@ref Contributing).

## Module Layout

| File | Role |
|------|------|
| `sim.cc` | Simulator entry point: parses the CLI, loads the scenario, runs the per-tick loop. |
| `CMakeLists.txt` | Source list for the ns-3 build. |
| [src](@ref src) | C++ modules (config, eval, routing, jammer, traffic, io, rl, ...). |
| [scripts](@ref scripts) | Python tools: field-data pipeline, validation, sweeps, RL, baselines, plotting. |
| [inputs](@ref Inputs) | Scenario definitions (`run.ini` + JSON files), sweeps, experiment matrices. |
| [tests](@ref tests) | C++ unit tests, CLI integration script, regression fixtures. |
| `docs/` | `Doxyfile` and `topics.dox` for this documentation site. |
| [CONTRIBUTING.md](@ref Contributing) | "Update together" checklist for user-visible changes. |
| `requirements.txt` | Pinned Python dependencies. |
| `requirements-tuning.txt` | Optuna pin, used only by the tuning smoke. |
| `TODO.md` | Open work items. |
| `data/`, `outputs/`, `third_party/` | Field data, generated results (git-ignored), vendored code. |

## Setup {#readme_setup}

### Build {#readme_build}

You build the simulator yourself; none of the tools here build it. All commands
run from the **ns3-mmwave repo root** (two levels above this directory).

```bash
./ns3 clean
./ns3 configure --build-profile=debug -- -DCMAKE_OSX_ARCHITECTURES=arm64
./ns3 build
```

If `./ns3 clean` does not clear the cache fully, remove it manually first:

```bash
rm -rf cmake-cache build
```

### Python environment {#readme_python_environment}

The simulator itself needs no Python. The RL, sweep, validation, and plotting
scripts do. One command from `scratch/mesh-sim/` creates `.venv` and installs
`requirements.txt`:

```bash
python3 scripts/rl/bootstrap_venv.py
.venv/bin/python -m pytest scripts/rl/tests -q
```

- On Windows the interpreter is `.venv\Scripts\python.exe`.
- `requirements.txt` pins the direct dependencies at the versions tested on the
  implementer's platform; it is not a universal lock file.
- `python3 scripts/rl/bootstrap_venv.py --check` verifies imports and exact
  installed versions without installing anything.
- Tuning needs Optuna from a separate file; see [scripts/rl](@ref scripts_rl).

## Run {#readme_run}

Simulator commands run from the **ns3-mmwave repo root**. Python commands run
from **scratch/mesh-sim**.

### Single scenario {#readme_run_single}

```bash
./ns3 run scratch/mesh-sim/sim -- \
  --run-config=scratch/mesh-sim/inputs/calfex/1602-1605/run.ini \
  --band=sub-6 \
  --seeds=1,2,5,6,8,10 \
  --output-dir=scratch/mesh-sim/outputs/calfex/1602-1605
```

Key flags (the full list is in the [CLI guide](@ref src_cli)):

| Flag | Meaning |
|------|---------|
| @c --run-config | Path to the scenario's @c run.ini (the file, not the dir). |
| @c --band | @c sub-6 or @c mmwave. Interference/jammers only apply on @c sub-6. Optional. |
| @c --seeds | Comma-separated seeds; each gets its own `seed-<N>/` output subdir. |
| @c --output-dir | Where results are written. |
| @c --rl-mode | Enable the RL bridge. |
| @c --seed / @c --run-id | Override the single seed or the run id from @c run.ini (@c --seeds takes precedence for multi-seed runs). |
| @c --positions-override | Optional JSON file overriding node start positions. |
| @c --debug-links | Verbose per-link (and per-jammer) log output. |
| @c --channel-query | Serve candidate-layout channel queries on stdin/stdout for the placement baselines ([contract](@ref src_query)); needs exactly one seed and writes no outputs. |

The band resolves as `--band`, then `[channel] band` in `run.ini`, then the
default `mmwave` (see the [run.ini reference](@ref src_config_run_ini_reference)).
After a run, check the "Resolved config" block in `run.log` (frequency,
duration, bandwidth, band) before trusting the results.

### Generating a scenario from field data {#readme_run_pipeline}

Run from **scratch/mesh-sim**. Each step needs the output of the one before it.

```bash
# 1. plot raw per-node data into per-day traces
python -m scripts.arpo_data.cli plot --nodes --day 2026-06-25

# 2. slice one scenario window out of the day (UTC HH.MM)
python -m scripts.arpo_data.cli split-scenario \
  -i data/arpo_extracted/_plots/per_day/2026-06-25 --start 12.27 --end 14.13

# 3. build run.ini + nodes.json for that window
python -m scripts.validation.build_config_files \
  -i data/arpo_extracted/_plots/per_day/2026-06-25/scenarios \
  --day 1227-1413 --csv-dir data/arpo_extracted/csv \
  --band sub-6 --mode node --name jeddoc --tx-gain 1 --rx-gain 1

# 4. fill node trajectories from the sliced GPS trace
python -m scripts.validation.build_waypoints node \
  -i inputs/calfex/1227-1413 \
  -f data/arpo_extracted/_plots/per_day/2026-06-25/scenarios/1227-1413/gps_all_nodes_trace.csv \
  --all-nodes --time-mode raw
```

### Validating against field data {#readme_run_validate}

```bash
python -m scripts.validation.validate_days \
  --day 1227-1413 \
  --field-root data/arpo_extracted/_plots/per_day/2026-06-25/scenarios \
  -t 6360
```

See [scripts/validation](@ref scripts_validation) for the other validation tools.

### Jammer / interference model (optional) {#readme_run_jammer}

Run a scenario with field-logged electronic-warfare (jamming) events. The full
workflow is in the [jammer model](@ref src_jammer); in brief:

```bash
python -m scripts.validation.make_jammers \
  --trials data/trials2.csv \
  --trace data/arpo_extracted/_plots/per_day/2026-06-24/scenarios/1602-1605/gps_all_nodes_trace.csv \
  --trial directional_stationary_mss_20_mss --beamwidth 60 \
  -o inputs/calfex/1602-1605/jammers.json \
  --run-ini inputs/calfex/1602-1605/run.ini

./ns3 run scratch/mesh-sim/sim -- \
  --run-config=scratch/mesh-sim/inputs/calfex/1602-1605/run.ini \
  --band=sub-6 --seeds=1,2,5,6,8,10 \
  --output-dir=scratch/mesh-sim/outputs/calfex/1602-1605-jammed
```

### Sweep {#readme_run_sweep}

```bash
python -m scripts.sweep.cli --config inputs/sweeps/example.ini
```

Sweep format and flags, including sweeping `band`, are in
[scripts/sweep](@ref scripts_sweep).

### Reinforcement learning {#readme_run_rl}

Smoke training run from **scratch/mesh-sim** (replace `<BIN>` with your built
simulator binary):

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/p0-smoke/run.ini \
  --output-dir outputs/rl-smoke-verification/rl \
  m-ppo --total-timesteps 16 --n-steps 16 --seed 1
```

Where to go next:

| Topic | Page |
|-------|------|
| `[rl]` keys, reward types, centralized control | [run.ini reference](@ref src_config_run_ini_reference) |
| Message, mode, mask, and action contract | [src/rl](@ref src_rl) |
| Observation, reward, and telemetry selection | [policy inputs](@ref src_rl_policy_inputs) |
| Train, inspect, evaluate, compare, experiment matrices | [scripts/rl](@ref scripts_rl) |
| Benchmark, tuning, SLURM, fetch | [scripts/rl/ops](@ref scripts_rl_ops) |

### Placement baselines {#readme_run_baselines}

The gateway-free `geometric` and `optimization` planners place nodes once, before
the run, scoring layouts through the simulator's `--channel-query` mode:

```bash
.venv/bin/python -m scripts.baselines.runner \
  --sim-binary <BIN> --run-config <INI> --seeds 1,2
```

Keys, objective, outputs, and limits are in the
[placement-baseline guide](@ref scripts_baselines).

## Verify {#readme_verify}

From `scratch/mesh-sim/tests/`:

```bash
make test                                   # standalone unit tests
MESH_SIM_BIN=<BIN> make integration         # 19 real-binary CLI checks (see tests/README.md)
```

`make test` stops at the first failing suite. To inspect every suite despite a
failure, run `make -C unit/config test`, `make -C unit/eval test`,
`make -C unit/routing test`, and `make -C unit/traffic test` separately.
The [RL test map](@ref scripts_rl_tests) lists each centralized-control
test, its input, and its expected output.

For a quick check without cloud reference data, run two tiny synthetic
simulations and check output contracts and same-seed repeatability. Run from
`scratch/mesh-sim/`:

```bash
python3 -m scripts.validation.smoke_check \
  --sim-binary <BIN> --out outputs/smoke-check/<new-name>
```

GitHub Actions runs Python contracts, builds mesh-sim, and runs CLI/synthetic
smoke checks on Linux and macOS for PRs into `arpo-main`. This is not a training
or cross-platform bit-identical-results claim.

The historical regression suite requires the approved cloud baseline bundle.
Ask the project team for it and unpack its reference JSON files under
`tests/fixtures/regression/p0/`; snapshots and gzip archives are not tracked.
The small tracked manifest retains the original hashes and scenario provenance:

```bash
python3 -m scripts.validation.regression_check verify-suite \
  --sim-binary <BIN> \
  --manifest tests/fixtures/regression/p0/manifest.json \
  --out outputs/baseline-regression/<name>
```

See [scripts/validation](@ref scripts_validation) for the suite's cases, skip
behavior, and exit codes.

## Output {#readme_where_output_lands}

| Invocation | Where |
|---|---|
| Direct run | `<output-dir>/run.log`, `<output-dir>/inputs/` (archived scenario files), `<output-dir>/seed-N/{positions,links,rx-power,mcs,flows,routes}.csv` + `summary.json` |
| Sweep point / validation scenario | Same layout, plus `console.log` (launcher-captured stdout/stderr; absent for direct runs) |
| RL training and evaluation | Manifests, models, and `episode-NNNN/` folders; see [scripts/rl](@ref scripts_rl_output) |

With no `[output] dir`, the simulator auto-generates
`outputs/YYYY-MM/DD/HH-MM-SS/`. For the direct-run CSV columns, see the
[I/O guide](@ref src_io_output).

## Conventions {#readme_conventions}

- Python tools run as modules from `scratch/mesh-sim/`
  (`python -m scripts.<pkg>.<module>`) because they use package-relative imports.
- Python tools that launch the simulator take `--sim-binary <BIN>`; they never
  build it.
- `run.log` "Resolved config" is the source of truth for what a run used.
- Scenario inputs are archived into every output folder for reproducibility.
- Generated `outputs/` and `docs/html/` are git-ignored.
- Code comments follow Doxygen style; Python uses `##` blocks and C++ uses
  `/** ... */`.

## Dependencies {#readme_dependencies}

- **Simulator:** ns3-mmwave, CMake, and a C++ compiler (not installed by the
  helper scripts).
- **Python tools:** Python 3 and the packages in `requirements.txt`; Optuna from
  `requirements-tuning.txt` only for tuning.
- **Data:** field data under `data/` for the scenario pipeline; the cloud baseline
  bundle for the regression suite.
- **Docs:** `doxygen` (and `graphviz` for diagrams).

## Accessing the documentation {#readme_docs}

This documentation is generated by **Doxygen** into a static HTML site under
@c scratch/mesh-sim/docs/html/. It is a set of local files; there is no server
to start.

### Generating the site {#readme_docs_build}

From @c scratch/mesh-sim/docs/ (where the @c Doxyfile lives; its input paths are
relative to that directory):

```bash
cd scratch/mesh-sim/docs
doxygen Doxyfile
```

This (re)builds @c docs/html/. Re-run it after changing source comments or these
README pages. If @c doxygen is not installed:

```bash
sudo apt install doxygen graphviz     # Linux (graphviz enables the diagrams)
brew install doxygen graphviz         # macOS
```

### Opening the site {#readme_docs_open}

Open the generated entry page in any browser:

```bash
# Linux
xdg-open scratch/mesh-sim/docs/html/index.html
# macOS
open scratch/mesh-sim/docs/html/index.html
# Windows
start scratch/mesh-sim/docs/html/index.html
```

Or paste the absolute path into the browser's address bar, for example
`/home/<USER>/ns3-mmwave/scratch/mesh-sim/docs/html/index.html`
(replace @c USER and the repo path with yours). This @c index.html is this
Overview page; use the navigation tree / search at the top to reach the module
pages and the per-file API docs.

## About {#readme_about}

This sim is being worked on by the University of Massachusetts's ACNL.
