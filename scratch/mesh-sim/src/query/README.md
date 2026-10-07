@page src_query src/query

@brief `--channel-query`: scores candidate start layouts through the simulator's own channel path.

The placement baselines (`scripts/baselines/`) propose node layouts and need
each one scored with the run's resolved band, per-node gains, buildings,
jammers and propagation models. `--channel-query` turns the binary into a
long-lived worker that answers those requests on stdin/stdout. The only
Python client is `scripts/baselines/planners/channel.py` (`ChannelScorer`).

## Module Layout

| File | Role |
|------|------|
| @ref channel-query.h "channel-query.h" | `RunChannelQuery(cfg, args, in, out)`. |
| @ref channel-query.cc "channel-query.cc" | Request loop, validation, fork-per-layout child, pipe draining, reaping, JSON framing. |
| `CLAUDE.md` | Scope and invariants for contributors. |

## Run

The user builds the binary. A client launches it like this:

```bash
<BIN> --run-config=<staged run.ini> --channel-query --seed=<planning seed> \
      [--band=sub-6] [--rl-mode]
```

`sim.cc` loads and validates the config, applies `--run-id` / `--band` /
`--rl-mode`, resolves RL control, then calls `RunChannelQuery`. Nothing is
written to disk: no output directory, archive or `run.log`. Seeds must resolve
to exactly one value (`Error: --channel-query needs exactly one seed`, exit 1);
`--output-dir` is ignored with a note on stderr. The `[baseline]` guard is not
applied in this mode. Logs go to stderr; stdout carries only NDJSON.

## Wire contract mesh_channel_query_v1

One JSON object per line in each direction.

**Init** (first stdout line):

```json
{"type":"init","contract":"mesh_channel_query_v1","isolation":"fork_per_layout",
 "node_ids":[...],"node_types":[...],"mobility":[...],
 "start_positions":[[x,y,z],...],"controlled_indices":[...],"rl_enabled":false,
 "band":"sub-6","band_source":"cli","seed":1,"run_id":1,"jammer_seed":1,
 "sinr_threshold_db":-6.7,"jammer_path_enabled":true,"num_buildings":0,
 "channel":{"frequency_ghz":2.4,"tx_power_dbm":30.0,"bandwidth_mhz":400.0,
            "noise_figure_db":5.0,"amc_model":"shannon","channel_model":"3gpp",
            "scenario":"RMa","condition_model":"static_los",
            "tx_array_gain_dbi":12.0,"rx_array_gain_dbi":12.0},
 "limits":{"max_layouts":1024,"max_probes":10000,
           "max_request_line_bytes":16777216,"max_child_response_bytes":16777216,
           "child_deadline_s":60.0},
 "time_s":0.0}
```

- `seed` is the planning seed (the single resolved CLI/INI seed) given to
  `RngSeedManager::SetSeed`; `run_id` goes to `SetRun`.
- `jammer_seed` (used by `JammerModel` for `random` bursts) equals the planning
  seed because `sim.cc` assigns the resolved run seed to `cfg.seed` before
  configuring the evaluator; the field is reported separately so that a future
  simulator change stays visible to clients.
- `start_positions` use `ControlledStartPosition`: the first waypoint for
  waypoint mobility, otherwise `position`.
- `channel` values are the resolved channel defaults; per-node gain overrides
  still apply inside the evaluation.

**Requests:**

```json
{"type":"evaluate","request_id":7,"layouts":[[[x,y,z],...N],...],
 "probes":{"height_m":1.5,"rx_gain_dbi":12.0,"sinr_db":-6.7,"points":[[x,y],...]}}
{"type":"shutdown"}
```

`probes` is optional; when present all four fields are required. Each layout
holds one `[x, y, z]` per node in roster order. `shutdown` or EOF ends the
worker with exit code 0.

**Response:**

```json
{"type":"result","request_id":7,"layouts":[L,...],"wall_s":0.42}
```

with each `L` either

```json
{"links":[[i,j,sinr_db,capacity_mbps,is_los,connected],...],"coverage":[[k,...],...],"wall_s":0.01}
```

or `{"error":"..."}`. `links` has N(N-1)/2 rows in `EvaluateAll` order
(i < j, row-major). `connected` is `sinr_db >= sinr_threshold_db`.
`coverage[i]` lists the probe indices node i covers (`sinr_db >=
probes.sinr_db`), or `coverage` is `null` without probes. SINR and capacity are
the `LinkResult` values unchanged.

## Isolation and evaluation order

The worker process creates no ns-3 object or random variable and stays
single-threaded. Each layout is scored by a `fork()`ed child that copies the
config, applies the layout (`ApplyLayout`), sets `cfg.seed` to the planning
seed, calls `RngSeedManager::SetSeed(seed)` / `SetRun(run_id)`, builds the topology
(`TopologyBuilder`, with probes when requested), configures `LinkEvaluator`
like `sim.cc`, runs `EvaluateAll(mobs, 0.0)` first, then probe links
node-major (node 0..N-1, probe 0..G-1), writes one JSON object to its pipe and
`_exit`s. The event loop never runs before scoring and jammer power is taken
at `t = 0`.

Every child starts from the same global state as a fresh simulator process:
automatic RNG stream counter 0, empty node/building registries, empty
condition caches. ns-3 has no API to reset the stream counter, so forking from
a clean parent is what makes this hold; no per-variable `AssignStreams` is
needed. Probes are created without random variables (see `src/setup/`), so
mesh draws do not depend on whether probes were requested.

Parity, stated at two levels:

1. **Mechanics parity** (by construction): every candidate goes through
   `LinkEvaluator::Evaluate` with the run's resolved config.
2. **Identical t = 0 realization** with the first seed of an ordinary run on
   the same effective inputs, seed and mode flag: expected because
   construction order and evaluation order match and both start from counter
   0. It is claimed only once the real-binary parity checks pass, and never
   for the second or later seed of a multi-seed run (stream-counter drift).

Determinism is per layout: the same scenario files, flags, seed, run ID,
layout and probe grid always give the same result, independent of request
order or batch composition. It is not a stable per-pair random stream across
layouts: condition-dependent branches (shadowing, O2I, foliage) can change
how many draws earlier pairs consume, so two layouts that share a pair can see
different draws for it. Evaluating a subset of pairs would change the draw
indices, so every layout is a full `EvaluateAll` plus all probe links.

## Limits and errors

| Limit | Value |
|-------|-------|
| Request line | 16 MiB; reading stops buffering at the limit and discards the rest of the line. |
| Child response | 16 MiB per layout. |
| Layouts per request | 1024 |
| Probe points | 10000 |
| Child deadline | 60 s (the child also arms a 65 s `alarm` as a backstop). |

Request-level problems (malformed JSON, non-object, bad `request_id`, unknown
`type`, wrong shapes, limits exceeded, non-finite numbers, oversized line)
produce `{"type":"error","request_id":<int|null>,"message":"..."}` and the
worker keeps serving. Per-layout problems produce `{"error":"..."}` in that
layout's slot: `ApplyLayout` errors (changed z, outside random-walk bounds),
a non-finite SINR or capacity (never serialised as a number), a child that
times out, exceeds the response limit, exits non-zero or is killed by a
signal, or prints unparsable output.

## Process cleanup

- The parent flushes its streams before each `fork`; the child redirects its
  stdin to `/dev/null` and stdout to stderr, so it can only answer through its
  pipe.
- The parent closes the pipe's write end, drains the read end with `poll`
  while the child runs (large responses cannot deadlock), then `waitpid`s.
  On deadline, oversize or read failure it sends `SIGTERM`, waits 1 s, sends
  `SIGKILL` and reaps. Every child is reaped; no zombie remains.
- `SIGTERM` / `SIGINT` / `SIGHUP` to the parent kill the active child before
  the parent exits. Clients should still run the worker in its own process
  group and clean up the whole group on cancellation.
