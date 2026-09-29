# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/sweep

Runs the simulator once per point of a cartesian grid of `run.ini` values,
built from one sweep INI (example: `inputs/sweeps/example.ini`). The sweep INI
format, flags, and output layout are in `README.md`; keep it in sync when
they change.

## Commands

Run from `scratch/mesh-sim/` as a module (package-relative imports). Never
build the simulator; pass an existing binary.

```bash
.venv/bin/python -m scripts.sweep.cli --config inputs/sweeps/example.ini --dry-run
.venv/bin/python -m scripts.sweep.cli --config inputs/sweeps/example.ini --sim-binary <BIN>
```

The package has no test directory of its own. Its launcher behavior (stdin
closed, loader paths cleaned) is covered in the validation suite:

```bash
.venv/bin/python -m pytest scripts/validation/tests/test_regression_check.py::test_unattended_launchers_close_stdin_and_clean_loader_paths
```

## Architecture

`cli` -> `config.parse_sweep_config` -> `runner.run_sweep`, which per point
calls `ini_writer.write_point_ini` + `copy_scenario_files`, then launches
`<bin> --run-config=<point>/run.ini --seeds=...`.

- Precedence: `[sweep]` value > `[sweep.override]` > base `run.ini`. The
  writer then forces `[output] dir` = the point dir and
  `[scenario] name` = `<label>_point-NNN`.
- Swept values stay raw strings; nothing checks them against the simulator's
  config schema. The simulator validates the generated `run.ini` itself.
- The point `run.ini` is rewritten with `configparser`: base comments are
  dropped and keys are lowercased.
- Mesh-sim root discovery differs by module: `config` walks up from the
  sweep INI looking for `sim.cc` (`base_scenario` is relative to that root);
  `runner` uses `scripts.sim_support.find_mesh_root`. Binary lookup
  (`find_sim_binary`), the subprocess env (`simulator_env`), and seed parsing
  (`parse_seed_spec`) also come from `scripts/sim_support.py`. Launcher
  changes should go there, not here.
- `auto_plot` shells out to `python -m scripts.plotting.cli` with a generated
  per-point `plot.ini`. If that INI has no `[plots]` section, seven default
  plots are enabled. The plotting exit code is ignored.
- `sweep_manifest.json` is rewritten after every non-skipped point. A failed
  point is recorded as `failed` and the sweep continues.

## Conventions

- Doxygen `##` / `@fn` / `@brief` blocks above each function (the one-line
  docstrings are kept too).
- Config errors print to stderr and `sys.exit(1)`. A bad `seeds` value raises
  an uncaught `ValueError`.

## Known issues (left unfixed)

- `--resume` does nothing useful: every run creates a new timestamped
  `outputs/YYYY-MM/DD/HH-MM-SS/` dir, so no finished points are ever found.
- `copy_scenario_files` copies only the nodes and buildings files, never
  `jammers.json`.
- A `scenario.buildings_file` in `[sweep.override]` is ignored when
  choosing which file to copy. Only `nodes_file` overrides (and per-point
  `[sweep]` values of either) are honored.
