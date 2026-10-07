# query/

## Scope
The `--channel-query` worker: NDJSON requests on stdin, candidate layouts scored by forked
children through `TopologyBuilder` + `LinkEvaluator`, results on stdout. Contract
`mesh_channel_query_v1` and the isolation argument are in `README.md`.

## Files
- **channel-query.h/cc** -- `RunChannelQuery(cfg, args, in, out)`.

## Behavior to preserve
- The parent never creates an ns-3 object or random variable and stays single-threaded. Anything
  that touches ns-3 belongs in `EvaluateInChild`, which runs only after `fork()`.
- Child order mirrors `sim.cc` per seed: `cfg.seed = planning seed`, `SetSeed(planning seed)` /
  `SetRun(run_id)`, `Build()`, `Configure(cfg, propagation, condition, cfg.band, jammerMobs)`,
  `EvaluateAll(mobs, 0.0)`, then probes node-major. Do not evaluate anything before `EvaluateAll` or skip pairs; draw order is
  part of the realization.
- `jammer_seed` equals the planning seed because `sim.cc` assigns the resolved run seed to
  `cfg.seed` before configuring the evaluator; the field is reported separately so that a future
  simulator change stays visible to clients.
- Stdout carries only NDJSON: the child redirects its stdout to stderr and answers through its pipe.
- Drain the child pipe while it runs, reap every child, and terminate/kill/reap on deadline.
- Never serialise NaN/inf; report them as a layout error.
- Request-level errors keep the worker serving; only `shutdown`/EOF (exit 0) or a broken stdout
  (exit 1) end it.

## Changing the contract
Update together: this module, `scripts/baselines/planners/channel.py`,
`scripts/baselines/tests/fake_query.py`, `README.md` here, the query tests in
`tests/integration/cli-integration-test.sh`, and the `CONTRIBUTING.md` row. Rename the contract
for incompatible changes. There is no unit suite (needs ns-3); `ApplyLayout` is tested in
`tests/unit/config`.

## Dependencies
- Depends on: `domain/`, `cli/` (`ResolveQuerySeed`), `config/` (`ApplyLayout`,
  `ControlledStartPosition`), `setup/`, `eval/`, `third_party/json.hpp`, ns-3 core
  (`RngSeedManager`), POSIX (`fork`, `poll`, `waitpid`)
- Depended on by: `sim.cc`
