# inputs/

Scenario definitions for mesh-sim.  Each baseline scenario lives in its own
directory under `baselines/` and contains the configuration files that drive
a simulation run.

## Directory layout

```
inputs/
  baselines/          <-- validation scenarios (numbered 01-16)
    01-static-los-baseline/
      run.ini          simulation parameters
      nodes.json       node positions and mobility
      buildings.json   (optional) building geometry
    ...
```

## How to run a scenario

```bash
./ns3 run "mesh-sim --scenario=scratch/mesh-sim/inputs/baselines/01-static-los-baseline"
```

Outputs are written to `scratch/mesh-sim/outputs/` in a timestamped directory.

## Design philosophy

The scenarios are ordered by complexity.  Each one changes **exactly one
knob** from the baseline (scenario 01), so you can always diff back to
understand what changed.  See `baselines/README.md` for the full scenario
table and a guide to interpreting the output graphs.
