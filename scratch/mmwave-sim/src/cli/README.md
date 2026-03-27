# cli/

Command-line argument parsing and pre-simulation setup helpers. Keeps `sim.cc` focused on orchestration.

## Files

### cli-parser.h / cli-parser.cc

Parses command-line arguments and handles pre-run setup tasks.

**Key types:**
- `CliArgs` -- struct holding parsed CLI arguments (`run_config_path`, `positions_override_path`, `seeds_arg`, `seed_override`, `run_id_override`)

**Key functions:**
- `ParseCommandLine(argc, argv)` -- parses args via `ns3::CommandLine`. Exits if `--run-config` is missing.
- `ResolveSeeds(args, cfg)` -- determines which seeds to run. Priority: `--seeds` > `--seed` > `cfg.seed` > default 42.
- `ArchiveScenarioInputs(base_output_dir, run_config_path)` -- copies all files from the scenario input directory into `<output_dir>/inputs/` for reproducibility.

## Usage

Called at the top of `sim.cc`:
```cpp
auto args = mmwave_sim::ParseCommandLine(argc, argv);
// ... load config ...
auto seeds = mmwave_sim::ResolveSeeds(args, cfg);
mmwave_sim::ArchiveScenarioInputs(baseOutputDir, args.run_config_path);
```
