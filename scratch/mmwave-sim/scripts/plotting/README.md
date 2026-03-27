# mmwave-sim plotting

Generates plots from mmwave-sim C++ simulation output.

## Prerequisites

```
pip install matplotlib numpy pandas
```

## Usage

```bash
# Copy the example config and edit data_dir to point at your output
cp scripts/plotting/plot.example.ini plot.ini
vim plot.ini

# Run
python scripts/plotting/cli.py --config plot.ini

# Override output directory
python scripts/plotting/cli.py --config plot.ini --output-dir /tmp/my-plots
```

## Output

Plots are saved to `<data_dir>/figures/` by default. Override with `--output-dir`.

## Expected output directory structure

The `data_dir` in your INI must contain one or more `seed-N/` subdirectories:

```
data_dir/
  seed-0/
    summary.json    # required - per-UE and network-level metrics
    links.csv       # optional - needed for sinr_timeseries plot
  seed-1/
    summary.json
    links.csv
  ...
```

## Available plots

| INI flag             | Description                                  | Seeds    |
|----------------------|----------------------------------------------|----------|
| `per_ue_sinr`        | Bar chart of mean SINR per UE                | single   |
| `per_ue_throughput`   | Bar chart of DL throughput per UE             | single   |
| `per_ue_delay`        | Bar chart of DL delay per UE                  | single   |
| `sinr_timeseries`    | SINR vs simulation time (from links.csv)     | any      |
| `sim_runtime`        | Wall-clock runtime per seed                  | multi    |
| `network_summary`    | Table of network-level metrics               | any      |

- **Single seed**: per-UE bar charts + network summary table.
- **Multiple seeds**: aggregated bars with 95% CI error bars + runtime comparison.

All flags default to `true` if the `[plots]` section is omitted.
