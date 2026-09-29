@page scripts_sweep scripts/sweep

@brief Runs a grid of mesh-sim simulations from one sweep INI file and collects them in a single output folder.

A sweep takes a base scenario (`run.ini` + `nodes.json`), changes chosen
`run.ini` values, and runs the simulator once per combination of values
(a "point"), for a list of seeds. Points are the cartesian product of every
line in the `[sweep]` section. Optionally it plots each point.

## Module Layout

| File | Role |
|------|------|
| `cli.py` | Command-line entry point (`--config`, `--dry-run`, `--resume`, `--sim-binary`). |
| `config.py` | Parses and validates the sweep INI into `SweepConfig` / `SweepDimension`. |
| `ini_writer.py` | Writes each point's `run.ini` and copies `nodes.json` / buildings file. |
| `runner.py` | Builds the point matrix, launches the simulator, writes the manifest, triggers plots. |
| `__init__.py` | Empty package marker. |

## Setup

From `scratch/mesh-sim/`, with the simulator already built by you (this tool never builds it):

```bash
python3 scripts/rl/bootstrap_venv.py
```

## Run

Working directory: `scratch/mesh-sim/`.

Preview the points (starts no simulations, needs no binary):

```bash
.venv/bin/python -m scripts.sweep.cli --config inputs/sweeps/example.ini --dry-run
```

Run the sweep:

```bash
.venv/bin/python -m scripts.sweep.cli --config inputs/sweeps/example.ini --sim-binary <BIN>
```

| Flag | Meaning |
|------|---------|
| `--config` | Required. Path to the sweep INI. |
| `--dry-run` | Print the sweep matrix and exit. |
| `--resume` | Skip points whose `seed-<N>/summary.json` files all exist. |
| `--sim-binary` | Simulator binary. Default: first `ns3*-sim-*` under `<ns3>/build/scratch/mesh-sim/`. |

## Sweep INI

`inputs/sweeps/example.ini` runs 3 frequencies x 3 TX powers = 9 points, 3 seeds each:

```ini
[sweep.meta]
base_scenario = inputs/baselines/01-static-los-baseline
seeds = 1, 2, 3
auto_plot = all
plot_config = scripts/plotting/plot.example.ini
label = freq-power-sweep

[sweep.override]
scenario.duration_s = 2.0

[sweep]
channel.frequency_ghz = 28.0, 39.0, 60.0
channel.tx_power_dbm  = 20.0, 25.0, 30.0
```

### `[sweep.meta]`

| Key | Default | Meaning |
|-----|---------|---------|
| `base_scenario` | required | Scenario directory (must hold `run.ini`), relative to the mesh-sim root. |
| `seeds` | `1` | Seeds per point; comma list and `A-B` ranges. |
| `auto_plot` | `none` | `each` (after every successful point), `all` (after the sweep), or `none`. |
| `plot_config` | empty | Plot INI relative to the mesh-sim root; empty uses default plots. |
| `label` | `sweep` | Used in each point's `[scenario] name` (`<label>_point-NNN`) and the console summary. |

### `[sweep.override]` and `[sweep]`

- `[sweep.override]` (optional): `section.key = value` constants applied to every point.
- `[sweep]` (required): `section.key = v1, v2, ...`. Every key is a `run.ini` section and key.
- Sweep values win over overrides, and overrides win over the base `run.ini`.

## Output

| Step | Where |
|------|-------|
| Sweep folder | `outputs/YYYY-MM/DD/HH-MM-SS/` under the mesh-sim root (new every run). |
| Copy of the sweep INI | `<sweep folder>/sweep.ini` |
| Manifest (rewritten after each point) | `<sweep folder>/sweep_manifest.json` |
| Per-point inputs | `<sweep folder>/point-NNN/run.ini`, `nodes.json`, optional buildings file |
| Per-point simulator output | `<sweep folder>/point-NNN/seed-<N>/`, plus the sim's `run.log` |
| Simulator stdout/stderr | `<sweep folder>/point-NNN/console.log` |
| Plots (if enabled) | `<sweep folder>/point-NNN/plot.ini` and the plotting CLI's output |

## Conventions

- Run as `python -m scripts.sweep.cli` from `scratch/mesh-sim/`.
- Points are numbered from 1 in the order of the cartesian product (last `[sweep]` line varies fastest).
- A failed point is logged as `failed` in the manifest and the sweep continues.
- Comments are Doxygen `##` blocks above each function.

## Dependencies

- Python 3.10+ (uses `X | None` type hints); standard library only.
- `scripts/sim_support.py` (binary lookup, seed parsing) and `scripts/plotting/` (only for `auto_plot`).
- A built simulator binary, unless using `--dry-run`.
