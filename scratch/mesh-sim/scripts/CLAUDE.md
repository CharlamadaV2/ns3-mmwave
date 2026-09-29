# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/

Python tooling around the simulator. Each subpackage has its own CLAUDE.md;
read it before editing there.

| Package | Role |
|---|---|
| `arpo_data/` | raw ARPO field CSVs -> per-day / per-scenario traces and plots |
| `validation/` | field traces -> scenarios, batch runs, sim-vs-field comparison, regression and smoke checks |
| `sweep/` | one sim run per point of a `run.ini` grid |
| `rl/` | Gymnasium env, training, evaluation, experiments, cluster ops |
| `plotting/` | post-sim figures from finished `seed-N/` folders |

Data flow: `arpo_data` -> `validation` (build scenarios, run, compare);
`sweep`, `validation`, and `rl` launch the simulator; `plotting` reads its
outputs.

## Commands

From `scratch/mesh-sim/`, always as modules (package-relative imports break
`python file.py`). Never build the simulator; pass `--sim-binary <BIN>`.

```bash
python3 scripts/rl/bootstrap_venv.py              # creates .venv
.venv/bin/python -m scripts.<pkg>.<module> --help
.venv/bin/python -m pytest scripts/rl/tests scripts/validation/tests -q   # what CI runs
```

Only `rl/` and `validation/` have test suites; `sweep` launcher behavior is
covered in `validation/tests`, and `plotting` by one test in
`rl/tests/test_policy_comparison.py`.

## Shared modules

Reuse these instead of re-implementing them in a subpackage:

- `sim_support.py` -- `find_mesh_root` (ancestor containing `sim.cc`),
  `find_sim_binary` (`<ns3>/build/scratch/mesh-sim/ns3*-sim-*`),
  `simulator_env` (prepends `<ns3>/build/lib` to `LD_LIBRARY_PATH` /
  `DYLD_LIBRARY_PATH`), `parse_seed_spec` (`1,3,5-7`), `strip_inline_comment`
  (INI values, matching the C++ loader), `tail_lines`.
- `stats.py` -- `sample_stats` and `t_critical_95` for 95% CIs. The t table
  rounds df down so intervals are never too narrow. Used by `plotting` and
  `rl/policy/compare.py`, so changing it changes both.

Any launcher that spawns the simulator should use `simulator_env` (the binary
needs ns-3's shared libraries) and close its stdin unless it is the RL bridge.

## Conventions

- `requirements.txt` is the recorded dependency set for RL manifests; add
  packages there only when needed repo-wide, and keep tuning-only packages in
  `requirements-tuning.txt`.
