@page scripts_rl_ops scripts/rl/ops

Operations plumbing *around* an existing `experiment_plan.json`: run one array
task, measure one task's cost, search the already-wired PPO knobs, submit the
plan to SLURM, and copy results to another machine.

## Start here

Create an experiment plan before benchmarking or submitting cluster tasks;
tuning reads a study spec directly, and fetch copies an existing run. For a
resource estimate, start at [Benchmark](@ref scripts_rl_ops_benchmark); for a small search, see
[Tuning smoke](@ref scripts_rl_ops_tuning_smoke); for job submission and recovery, see
[Cluster runs](@ref scripts_rl_ops_cluster_runs); for copying results home, see [Fetch](@ref scripts_rl_ops_fetch).
The [module map](@ref scripts_rl_ops_module_map) below tells contributors which file owns each
part. The cluster guide documents the current CLI, but live SLURM and real
`rsync` behavior still need validation on the target site.

## What this package owns

- Mapping a plan onto array tasks and reading each task's filesystem state.
- An in-job runner that executes one task, or the plan's compare step.
- A process-tree benchmark of one task and arithmetic scaling of that
  measurement onto another matrix.
- An Optuna smoke over the PPO knobs that already reach `MaskablePPO`.
- SLURM submission bookkeeping: task table, submit lock, receipts, state
  reconciliation, and the `plan / submit / status / resume / cancel / compare`
  front end.
- An rsync wrapper that copies selected result files to a destination machine
  and records what arrived.

## What this package does not own

- No new training capability and no new PPO wiring. `train.py`,
  `evaluate.py`, `compare.py`, `mask_ppo.py`, and `experiment.py` are imported
  or executed, never modified; only `n_steps`, `gamma`, and `ent_coef` are
  searchable because only they reach the constructor today.
- No ETA and no queue position. The only time-like scheduler value shown is
  SLURM's own estimated start for a queued element, printed as
  `scheduler-estimated start: … (may change)`.
- No cluster constants. Partition, account, QOS, constraint, modules, wall
  time, memory, CPU count, array limits, and venv path are required inputs in a
  config file; the code supplies none of them.
- No statistics. `fetch` computes nothing; `compare` transfers nothing.
- No study resume, no array chunking beyond the site limit, and no automatic
  deletion or renaming of a blocked step directory.

## Module map {#scripts_rl_ops_module_map}

```
tasks.py       plan -> array tasks, per-task filesystem state, exit-code tolerance,
               compare prerequisites, comparison outcome. Pure; no subprocess.
run_task.py    in-job entry point: one task, or the compare step. Scheduler-agnostic.
benchmark.py   measure one task's steps via ps, and derive a resource estimate.
tune.py        Optuna smoke driver.
slurm.py       cluster-config validation, sbatch/squeue/sacct/scancel argv builders,
               subprocess calls, output parsers, job-script rendering.
receipts.py    cluster/ layout, task table, submit lock, submission receipts.
reconcile.py   pure: filesystem state + receipts + scheduler snapshot -> task state.
cluster.py     CLI: argument parsing, orchestration, printing.
fetch.py       CLI: rsync selected results to a local destination.
```

Dependency direction, no cycles: `tasks` is imported by `run_task`,
`benchmark`, `reconcile`, `cluster`, and `fetch`. `reconcile` imports only
`tasks` and takes receipts and the scheduler snapshot as plain dicts.
`slurm` and `receipts` import neither each other nor `reconcile`. `cluster`
imports `tasks`, `slurm`, `receipts`, `reconcile`, and `run_task`. `tune` and
`fetch` import no scheduler module. Only `tasks`, `tune`, and `benchmark`
import `scripts.rl.experiment`, and only its public names (`load_matrix`,
`build_plan`, `load_plan`, `step_state`, `PLAN_NAME`).

## Task unit

One array task is one `(row, training seed)` pair: its `train` step followed by
its `evaluate` step, serially, in one array element. The task index is the
0-based position of the train step among the plan's train steps, and the task
id is the train step id without its `train/` prefix, for example
`local-delivery/train-seed-101`. `build_tasks` pairs each train step with the
evaluate step whose `needs` names it and raises `ValueError` when a pair or the
compare step is missing.

An evaluation needs only its own training run, and evaluation seeds cannot be
split across tasks (`scripts.rl.compare` requires one manifest per label with
sorted seeds), so one element per pair keeps each task self-contained.

### Why the compare job depends with `afterany`

The comparison is one separate job submitted with
`--dependency=afterany:<every active array job id>`. Correctness comes from a
filesystem prerequisite check inside the compare runner, not from scheduler
exit codes: `afterok` on an array with one failed element never clears, even
after that element is resubmitted as part of a later array. `afterany` only
orders the jobs; `compare_prerequisites` decides whether a comparison may be
written at all.

## Runner: `run_task.py`

```bash
.venv/bin/python -m scripts.rl.ops.run_task --output-root R --task-index I [--record PATH]
.venv/bin/python -m scripts.rl.ops.run_task --output-root R --compare [--allow-incomplete] [--record PATH]
```

Exactly one of `--task-index` or `--compare` is required;
`--allow-incomplete` applies to `--compare` only.

Task mode, in order:

1. Train `done` is skipped. `pending` is executed as a fresh
   `python -m <module> <args>` process from the mesh root. Any other state
   prints the step's own retry detail and exits 1 without touching the
   directory.
2. A train exit code that is not tolerated exits 1; the evaluation is not
   attempted.
3. Evaluate `done` exits 0. `partial` or `blocked` prints the detail and exits
   1. `pending` is executed; a tolerated exit gives 0, anything else 1.

Compare mode refuses by default when any evaluation is missing or partial: it
prints each offending step id, **writes nothing at all — no comparison output
and no `--record` file** — and exits 1. With `--allow-incomplete` it executes
the compare step and prints the step id, the raw exit code, and the state read
back from `comparison.json`.

### Exit-code mapping

`tolerated(kind, code)` accepts `0` for train, `0` and `2` for evaluate, and
`0` and `2` for compare, matching the tolerated codes of the corresponding CLIs
(a `2` means health counters or a not-held-out evaluation, not a failure).
A tolerated `2` therefore makes the runner exit 0, so `sacct` does not show a
healthy task as FAILED, while the raw code survives in the record.

### Record schema (`--record`)

```json
{"record_version": 1, "mode": "task|compare", "task_index": 0, "task_id": "row/train-seed-101",
 "host": "...", "started_at": "...", "ended_at": "...",
 "steps": [{"id": "...", "kind": "train", "action": "executed|skipped",
            "exit_code": 2, "tolerated": true}],
 "exit_code": 0}
```

`task_index` and `task_id` are `null` in compare mode. A skipped step records
`exit_code: null` and `tolerated: null`.

## Cluster runs {#scripts_rl_ops_cluster_runs}

For now, use one SLURM account per output root. Scheduler lookups use the
current account, so a second operator may not see an existing job and could
incorrectly abandon its receipt and submit a duplicate task. This is a known
limitation, not a substitute for the planned code safeguard.

### Commands

```bash
# on the cluster, after the human built the binary and prepared the venv
<venv>/bin/python -m scripts.rl.experiment plan --matrix M --output-root R --sim-binary /abs/BIN
<venv>/bin/python -m scripts.rl.ops.cluster plan   --output-root R --cluster-config C
<venv>/bin/python -m scripts.rl.ops.cluster submit --output-root R --cluster-config C [--tasks 0] [--no-compare] [--dry-run]
<venv>/bin/python -m scripts.rl.ops.cluster status --output-root R [--json]
<venv>/bin/python -m scripts.rl.ops.cluster resume --output-root R --cluster-config C [--inactive-job ID] [--abandon-intent NNNN:tasks] [--dry-run]
<venv>/bin/python -m scripts.rl.ops.cluster cancel --output-root R (--submission 0001 | --tasks 2,5) [--dry-run]
<venv>/bin/python -m scripts.rl.ops.cluster compare --output-root R [--allow-incomplete]
```

`--inactive-job` and `--abandon-intent` are repeatable; each occurrence names
one job id or one `NNNN:tasks` / `NNNN:compare` token. Exit 0 means the command
did what was asked; exit 1 is a refusal or an error, with the reason on stderr.
`status` exits 0 whenever it could render, including when tasks failed.

- **plan** validates, writes `cluster/tasks.json`, prints the task table, and
  prints the `sbatch` argv and rendered scripts with a clearly marked
  provisional receipt number `NNNN` and a `meshops-<random>-tasks` placeholder
  name. It submits nothing.
- **submit** submits tasks whose state is `unsubmitted`, optionally narrowed by
  `--tasks`; naming a task in any other state is refused and points at `resume`.
- **resume** submits every `unsubmitted`, `failed`, or `canceled` task whose
  remaining step directories are clean.
- **cancel** requires exactly one of `--submission` or `--tasks`.
- **compare** runs the compare step on the current host.

Validation shared by `plan`, `submit`, and `resume`, all refusals: the plan's
own root differs from `--output-root` (the plan was copied from another
filesystem — regenerate it here instead); `sim_binary` is not absolute, missing,
or not executable; a row's `run_config` is missing; the task count exceeds
`max_array_size`; the config is invalid; and, for a non-dry-run `submit` or
`resume`, `sbatch` is not on `PATH`.

`--dry-run` on `submit`/`resume` prints the state table and, when something
would be submitted, the provisional argv and scripts. It takes no lock, writes
no receipt, script, or `tasks.json`, and **does not query the scheduler** — its
state table is computed against an offline snapshot, so every receipt-covered
task reads as `unknown` there.

### Cluster config

`cluster-config.example.json` is the schema. Every key is required and the key
set is closed; there is no free-form `sbatch` option, because a value outside
the schema could override the array id, log path, dependency, or the
`--no-requeue` safety flag. A site-required option therefore needs a reviewed
addition to the schema.

```json
{
  "cluster_config_version": 1,
  "name": "<REQUIRED label for receipts>",
  "venv": "<REQUIRED absolute path to the prepared venv>",
  "setup_lines": [],
  "task":    {"partition": null, "account": null, "qos": null, "constraint": null,
              "time": "<REQUIRED HH:MM:SS>", "mem": "<REQUIRED e.g. 4G>",
              "cpus_per_task": "<REQUIRED integer>"},
  "compare": {"time": "<REQUIRED HH:MM:SS>", "mem": "<REQUIRED>", "cpus_per_task": 1},
  "max_array_size": "<REQUIRED integer: site MaxArraySize>",
  "max_concurrent_tasks": null
}
```

- `null` is accepted only for `partition`, `account`, `qos`, `constraint`, and
  `max_concurrent_tasks`; it means "omit the flag and take the site default".
- Any string starting with `<` is refused as a leftover placeholder, so the
  example file cannot be used unedited.
- `time` must match `[D-]H:MM:SS`, `mem` a digit string with an optional
  `K`/`M`/`G`/`T` suffix, and `cpus_per_task` / `max_array_size` /
  `max_concurrent_tasks` an integer ≥ 1.
- `setup_lines` is a possibly empty list of shell lines, for example
  `module load …`, emitted verbatim near the top of each job script.
- `compare` inherits `partition`, `account`, `qos`, and `constraint` from
  `task`; only its `time`, `mem`, and `cpus_per_task` are its own.
- `venv` must be absolute and contain `bin/python`; prepare it with
  `python3 scripts/rl/bootstrap_venv.py --venv <path>`.

The validated config and its SHA-256 are copied into every receipt, so
submissions may legitimately differ — for example more memory after an
out-of-memory failure. The config is not part of the immutable plan.

Job scripts run `set -euo pipefail`, the configured `setup_lines`, export
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` equal to `cpus_per_task` and an empty
`CUDA_VISIBLE_DEVICES`, `cd` to the mesh root recorded at submit time, run
`bootstrap_venv.py --venv <venv> --check` so a pin mismatch fails before any
step starts, and then `exec` the runner. Every interpolated path is quoted with
`shlex.quote`. Jobs are submitted with `--no-requeue`: a requeued task would
restart into a dirty directory and be refused anyway, so a silent scheduler
retry is worse than an explicit failure.

### Layout under the output root

```
<root>/experiment_plan.json                    existing, immutable, never written here
<root>/train/… <root>/eval/… <root>/comparison/ existing, guarded step directories
<root>/cluster/tasks.json                      deterministic task table
<root>/cluster/submit.lock                     present only during submit/resume
<root>/cluster/receipts/0001.json
<root>/cluster/scripts/0001-tasks.sh  0001-compare.sh
<root>/cluster/logs/0001/slurm-%A_%a.out  compare-%j.out
<root>/cluster/records/0001/task-0003.json  compare.json
```

`cluster/` sits outside every guarded step directory, so nothing written here
can block a train or evaluate run. `cluster/tasks.json` is
`{"tasks_version": 1, "plan_sha256", "tasks": [{index, id, train_id,
evaluate_id}]}`; an existing file describing a different plan is refused rather
than overwritten.

### Receipts and the submission protocol

One receipt per submission, `cluster/receipts/NNNN.json`:

```json
{"receipt_version": 1, "submission": "0001", "state": "submitting|submitted|submit_uncertain|abandoned",
 "created_at": "...", "host": "...", "user": "...",
 "job_name": "meshops-<32 hex chars>-tasks", "indices": [0, 1],
 "array_spec": "0-1", "plan_sha256": "...", "cluster_config": {…},
 "cluster_config_sha256": "...", "script": "...", "argv": [ … ],
 "job_id": null, "compare": null, "cancel_requests": [], "human_assertions": []}
```

`compare`, when present, holds `{"job_name", "job_id", "state", "created_at",
"script", "argv", "depends_on"}` with the same state values and its own random
job name.

`submit` and `resume` follow the same order:

1. Create `cluster/submit.lock` with `O_CREAT|O_EXCL`. An existing lock is a
   refusal that prints the holder's pid, host, and time; a stale lock is removed
   by hand after confirming no submit is running. The lock is released in a
   `finally` block.
2. Recover any no-ID intents by exact job name, take a scheduler snapshot,
   apply any `--inactive-job` / `--abandon-intent` assertions, reconcile, and
   compute the indices. Nothing to submit prints the state table and exits 0.
3. Allocate `NNNN` by exclusive creation of the receipt file, then write the
   **intent** — including a random 128-bit job name
   (`meshops-<token>-tasks`) — *before* `sbatch` runs, so an accepted job is
   never nameless.
4. Render the script, create the log and record directories, and run
   `sbatch --parsable …`. A non-zero exit, an interruption, or an unparseable
   response may still follow scheduler acceptance, so the receipt is left as a
   `submit_uncertain` no-ID intent with a stderr excerpt and the command exits
   1. **It is never retried automatically and never abandoned automatically.**
   A parsed job id finalizes the receipt as `submitted`.
5. Unless `--no-compare`, and only when every task is finished or covered by an
   active job, queue the compare job (below). When some tasks are uncovered —
   for example after a `--tasks 0` canary — the command says so and queues no
   compare job. If the array was submitted but compare submission is refused
   or its outcome is uncertain, a later `resume` with no tasks left to submit
   does **not** queue compare by itself. Check `status` and the receipts first.
   Once every evaluation is complete and no compare job can still write, use
   `cluster compare` on a host where the site permits it. There is currently no
   CLI command to submit a compare-only SLURM job; do not assume the comparison
   will appear automatically.

The `sbatch` argv is
`sbatch --parsable --no-requeue --job-name=J [--array=SPEC] --output=LOG_PATTERN
[--dependency=…] [--partition=] [--account=] [--qos=] [--constraint=] --time= --mem=
--cpus-per-task= SCRIPT`, with each `null` config value omitting its flag.
`SPEC` is the compressed index list (`0-7`, `1,3`) plus `%N` when
`max_concurrent_tasks` is set.

Recovery of a no-ID intent is by exact job name only: `squeue --name=…` and
`sacct --name=…` are filtered for exact equality, and their matches are merged.
An id is written back **only when exactly one job matches across the two
queries** — one exact match is a positive observation even if the other query
failed. Several matches are reported as too ambiguous and nothing is written.
Zero matches leaves the intent unresolved, because an empty or failed query is
*not* proof that `sbatch` failed; it keeps blocking resubmission.

Two human assertions can clear that, both recorded in the receipt with the
asserting account and time:

- `resume --inactive-job <job id>` asserts that a known-id job is no longer
  active. It is refused when no receipt holds that id or when the scheduler
  still shows the job active.
- `resume --abandon-intent NNNN:tasks` (or `NNNN:compare`) asserts that a no-ID
  submission never reached SLURM. It is refused when that element is not an
  unresolved no-ID intent, or when the job name still matches a job.

Neither assertion can bypass a positive active-job observation or a populated
step directory; the filesystem rules still apply afterwards. `status` never
writes anything.

### Reported states

Eight states, first match wins. `unsubmitted` is separate from `pending` so
"never submitted" and "queued" are not the same word.

| State | Rule |
|---|---|
| `pending` | a covering receipt's element is `PENDING`, `CONFIGURING`, or `REQUEUED` |
| `running` | a covering receipt's element is in another active state (`RUNNING`, `COMPLETING`, `SUSPENDED`, `RESIZING`, `SIGNALING`, `STAGE_OUT`) |
| `completed` | train `done` and evaluate `done` |
| `partial` | train `done` and evaluate `partial` |
| `unknown` | a receipt covers the task and the queue query failed; or an unresolved no-ID intent covers it; or a submitted element is absent from the queue and accounting holds no terminal record for it; or accounting says `COMPLETED` while the filesystem is incomplete; or no receipt covers the task and its train manifest says `running` |
| `canceled` | the latest covering receipt's element is `CANCELLED` in accounting; or accounting is unavailable, the queue query succeeded without the element, and the receipt records a cancellation covering it |
| `failed` | the latest covering receipt's element has an explicit terminal failure in accounting; or no receipt covers the task and a step directory is `blocked`; or every known job id of an otherwise-`unknown` task carries a recorded `--inactive-job` assertion (the row then also reports `asserted_inactive: true`) |
| `unsubmitted` | no receipt covers the task and both step directories are `pending` |

Each row carries a `detail` — the scheduler state and reason, the blocked
directory's `move or delete <dir> to retry` message, or why it is unknown — and
a queued row adds the scheduler's own estimated start when it is not `N/A`.
A parent-array accounting row is not evidence that each element finished:
elements are reconciled by exact `<array_job_id>_<index>` records, or left
`unknown`.

The compare row uses the same active/terminal logic over the latest compare job
plus the state inside `comparison.json`: `complete` maps to `completed`,
`incomplete` to `partial`, and absent with no compare job to `unsubmitted`.

### Resume rules

A task is resubmitted only when its state is `unsubmitted`, `failed`, or
`canceled`, **and** no element of it is active in any receipt, **and** every
step the runner would execute is `pending` on disk. Train `done` plus evaluate
`pending` qualifies, because the runner skips the finished train.

A task blocked by a dirty directory is listed as not submitted with
`move or delete <dir> to retry`; nothing is ever deleted or renamed by this
tool. `unknown` is never resubmitted without one of the recorded human
assertions above.

### Cancellation

`cancel --tasks 2,5` resolves the selection to exact
`<array_job_id>_<index>` element ids taken from the receipts and never passes a
parent array id, so sibling tasks are untouched. `cancel --submission 0001`
deliberately targets that receipt's whole active array plus its compare job.
Jobs absent from the receipts are never selected. `--dry-run` prints the
`scancel` argv and cancels nothing. A failed `scancel` is reported on stderr
with exit 1 and **no cancellation is recorded**; only a successful call appends
to `cancel_requests`. A failed queue query is a refusal with exit 1 — including
under `--dry-run` — because active elements cannot be resolved without it;
nothing is cancelled and nothing is recorded.

### Only one writer of `comparison.json`

If a compare submission is lost after the array is queued, the same safety
checks can prevent another automatic submission. Inspect the compare receipt
and scheduler state before using `cluster compare`; an unresolved no-ID intent
or a job that may still be active blocks it. The deferred compare-only
submission path is tracked in `TODO-RL-OPS-2`.

Before queuing a new compare job, an earlier compare job that is still active is
cancelled, the `scancel` call must have succeeded, and a fresh queue snapshot
must show no remaining blocker; otherwise the new compare job is refused rather
than risking two writers. `cluster compare` applies the same check.

A compare job blocks both of them while it is active, while it is an unresolved
no-ID intent, or — for any non-abandoned compare job carrying a job id — while
nothing has been observed that proves it can no longer write. Evidence of
termination is a non-active state in the queue snapshot, a non-active accounting
state, a recorded cancel request naming that job id (recorded only after a
verified `scancel`), or a recorded `resume --inactive-job <compare job id>`
assertion. A failed queue query blocks on its own once any compare element
exists. The consequence: where `sacct` accounting is unavailable, a compare job
that has simply left the queue keeps blocking until the human asserts it
inactive.

### `cluster compare` versus `fetch`

`cluster compare` computes statistics **where the plan lives**: it runs the
compare step in-process on the current host, and transfers nothing. With every
evaluation done it may simply run; otherwise it needs `--allow-incomplete`, and
even then refuses while any task is `pending`, `running`, or `unknown`. It
prints one line from the comparison outcome plus the raw exit code:
`complete` (0), `complete with health counters or seed overlap` (2), or
`incomplete` (1).

`fetch` copies files **to another machine** and computes nothing.

## Fetch {#scripts_rl_ops_fetch}

```bash
.venv/bin/python -m scripts.rl.ops.fetch --remote user@host:/abs/output-root \
  --dest outputs/fetched/<name> --select comparison,manifests [--update] [--dry-run]
```

`--remote` and `--dest` are required and have no defaults; a local path is
accepted as `--remote`. `--select` is optional — without it only the
always-included files are copied.

Always included: `experiment_plan.json`, `cluster/tasks.json`, and
`cluster/receipts/*.json`. Categories (an unknown name is refused):

| Category | Included |
|---|---|
| `comparison` | `comparison/comparison.json`, `comparison/episodes.csv` |
| `manifests` | `train/**/train_manifest.json`, `eval/**/eval_manifest.json`, `train/**/episode-*/rl_episode.json`, `eval/**/episode-*/rl_episode.json`, `cluster/records/**`, `benchmark/**` |
| `models` | `train/**/maskable_ppo_mesh.zip`, `train/**/best_model.zip` |
| `selection-logs` | `train/**/evaluations.npz` |
| `inputs` | `train/**/episode-*/inputs/**`, `eval/**/episode-*/inputs/**` |
| `telemetry` | `train/**/episode-*/steps.jsonl`, `eval/**/episode-*/steps.jsonl` |
| `episode-data` | `train/**/episode-*/**`, `eval/**/episode-*/**` (whole trees; potentially large, explicit opt-in) |
| `logs` | `cluster/logs/**` |

The argv is `rsync -a --prune-empty-dirs --ignore-existing --include=… --include='*/'
--exclude='*' REMOTE/ DEST/`. `--ignore-existing` is always present, so a
local file is never overwritten. A non-empty destination is refused unless
`--update`, which only adds files that are absent locally and keeps an earlier
manifest as `fetch_manifest.<n>.json`. The destination is also refused when it
resolves to the filesystem root, a home directory, the mesh-sim checkout root,
an existing non-directory, or a path containing a local `--remote` source; a
destination symlink is checked both as written and as resolved. `--dry-run`
prints the argv and transfers nothing. A non-zero `rsync` exit gives exit 1 and
no manifest; a missing `rsync` on `PATH` is a reported refusal.

`<dest>/fetch_manifest.json`:

```json
{"fetch_manifest_version": 1, "remote": "...", "selection": ["comparison"],
 "argv": [ … ], "fetched_at": "...",
 "files": [{"path": "...", "bytes": 0, "sha256": "..."}],
 "tasks": [{"index": 0, "id": "...", "state": "completed|partial|failed|missing|not_fetched"}],
 "comparison": "not_fetched|absent|complete|incomplete|unreadable",
 "snapshot_of_incomplete_run": true}
```

The distinction matters: `not_fetched` means the category was not selected, not
that the remote run lacked the file. Task states are read from the fetched
files only when `manifests` was selected — otherwise every task is
`not_fetched` and `snapshot_of_incomplete_run` is `null` rather than inferred
from omitted files. `comparison` is `not_fetched` unless `comparison` was
selected, and may be `unreadable` when the file arrived but could not be
parsed. With `manifests` selected, each step's `output_dir` is mapped from the
remote plan root onto the destination, because the fetched plan's absolute
paths do not exist locally.

Fetched trees live under the git-ignored `outputs/`; nothing fetched is
committed.

## Benchmark {#scripts_rl_ops_benchmark}

This is a measurement run, not a dry run: it executes one planned train step
and its evaluation, then records wall-clock time and sampled memory. Use a
small representative task first; the estimate does not measure the target
matrix or predict queue wait time.

```bash
.venv/bin/python -m scripts.rl.experiment plan \
  --matrix inputs/experiments/bypass-smoke-matrix.json \
  --output-root outputs/bench/bypass-smoke --sim-binary <BIN> --rows local-delivery

.venv/bin/python -m scripts.rl.ops.benchmark run \
  --output-root outputs/bench/bypass-smoke --task-index 0

.venv/bin/python -m scripts.rl.ops.benchmark estimate \
  --benchmark outputs/bench/bypass-smoke/benchmark/task-0000.json \
  --target-matrix <matrix.json> --safety-factor 2.0 \
  --output outputs/bench/bypass-smoke/benchmark/estimate.json
```

`run` needs both step directories `pending`, refuses an existing
`benchmark/task-NNNN.json`, launches each step in its own process group, applies
the same exit tolerance as the runner, and stops after a train that is not
tolerated. While a step runs, a sampler thread takes one sample immediately
after spawn and then every `--sample-interval-s` (default 0.5) seconds: it
parses `ps -A -o pid=,ppid=,rss=`, walks the parent chain from the step's pid,
and keeps the peak summed RSS, the peak process count, and the largest single
process RSS over the tree. Process-tree sampling is used because a training step
can have the Python process and up to two simulator subprocesses alive at once.

RSS means *resident set size*: memory currently resident in RAM, reported here
in KiB. `peak_tree_rss_kb` is the largest sampled sum across the task's Python
process and its descendants, not a precise peak of unique memory; shared pages
may be counted more than once.

**Sampling misses spikes shorter than the interval**; the record says so in
`sampling.note`, and the safety factor of the estimate is the human's lever for
that. A `ps` failure sets the three memory fields to `null` and records
`measurement_error`; timing is still measured and the benchmark file is still
written.

`benchmark/task-NNNN.json` carries `benchmark_version`, `measured_at`, `host`
(`system`, `machine`, `cpu_count`, `python_version`), `plan`
(`root`, `matrix_name`, `matrix_sha256`), `task_index`, `task_id`, `sampling`
(`method: ps-process-tree`, `interval_s`, `note`), `package_versions`, and one
entry per step with `id`, `kind`, `exit_code`, `tolerated`, `wall_seconds`,
`peak_tree_rss_kb`, `peak_process_count`, `peak_single_process_rss_kb`, and
`samples`. Train steps add:

- `requested_timesteps`, `n_steps`, and `effective_timesteps`
  (`ceil(requested / n_steps) * n_steps`, because rollouts run whole);
- `seconds_per_timestep` = wall ÷ effective, which **includes** the
  model-selection evaluations and the PPO updates;
- `scenario_identity`, `selection`, `eval_every_steps`, `eval_episodes`;
- `training_episodes` / `selection_episodes` and `episode_seconds_sum` from the
  episode manifests;
- `non_episode_seconds` = wall − episode sum. This is Python start-up, PPO
  updates, and the gaps between simulator launches together; **it is not a
  launch-overhead figure** and must not be labelled as one.

Evaluate steps add `episodes_expected`, `episodes_completed`, and
`seconds_per_episode`.

`estimate` runs nothing and is arithmetic only; `--safety-factor` and
`--output` are required. A target row is estimated only when its scenario
matches the benchmarked one — compared by the `*_sha256` fields of
`scenario_identity`, because `run.ini` paths differ between machines — and its
`n_steps` and evaluation cadence match the measured ones. Otherwise the row gets
`per_task_seconds: null` with `reason` `"different scenario"` or
`"different cadence"`. For a matching row:

```
per_task_seconds = seconds_per_timestep × effective target timesteps
                 + seconds_per_episode × policies × held-out seeds
```

A row whose observation or reward selection differs but whose scenario and
cadence match is still estimated, with `selection_differs: true` and a warning
that its Python and agent work was not measured. The output also records the
per-row safety-scaled seconds and `HH:MM:SS`, the task count, the summed serial
seconds, the memory estimate — `memory.peak_tree_rss_kb` is the largest measured
`peak_tree_rss_kb` over the benchmark's steps, and `with_safety_kb` is that value
times the safety factor; both are `null` only when no step's memory was measured
— and the assumptions as data:
`{"linear_in_timesteps": true, "measured_on": "<system> <machine>",
"is_cluster_estimate": false}`. A measurement taken on a laptop, for example on
macOS, is never a cluster estimate; there is no queue, ETA, or throughput
figure anywhere in the output.

## Tuning smoke {#scripts_rl_ops_tuning_smoke}

```bash
.venv/bin/python -m pip install -r requirements-tuning.txt

.venv/bin/python -m scripts.rl.ops.tune --study inputs/experiments/bypass-smoke-study.json \
  --output-root outputs/tune/bypass-smoke --sim-binary <BIN> --dry-run

.venv/bin/python -m scripts.rl.ops.tune --study inputs/experiments/bypass-smoke-study.json \
  --output-root outputs/tune/bypass-smoke --sim-binary <BIN>
```

Exit 0 means every trial produced an objective; exit 1 means a refusal, or that
at least one trial failed — records are still written in that case.

### What a trial is, and what it is not

- Distinct trials are **different configurations, not replicate seeds** of one
  comparison group. `scripts.rl.compare` builds an across-runs group from
  several training seeds of *one* configuration; the way to use a tuning result
  is to freeze one configuration by hand, then train it on several training
  seeds through a matrix and evaluate it on held-out seeds.
- The objective is one deterministic episode on one model-selection seed at a
  smoke budget. **It ranks nothing reliably and is no evidence of learning.**
- Rows with different reward definitions produce returns on different scales, so
  each needs its own study.
- The numbers in `inputs/experiments/bypass-smoke-study.json` exist to exercise
  the code path on the smoke fixture. They are not recommended ranges.

### Study spec

```json
{
  "study_version": 1,
  "name": "bypass-smoke-study",
  "description": "Plumbing smoke: three tiny trials; values are not recommendations.",
  "matrix": "inputs/experiments/bypass-smoke-matrix.json",
  "row": "local-delivery",
  "training_seed": 101,
  "sampler": {"type": "tpe", "seed": 7},
  "n_trials": 3,
  "search_space": {
    "n_steps":  {"type": "categorical", "choices": [32, 64]},
    "gamma":    {"type": "float", "low": 0.90, "high": 0.99},
    "ent_coef": {"type": "float", "low": 0.001, "high": 0.05, "log": true}
  }
}
```

These are different seeds: `training_seed` fixes the PPO and training-scenario
randomness for every trial in this study. `sampler.seed` fixes Optuna's sequence
of candidate hyperparameters; it is **not** passed to PPO or the simulator.
The matrix's `model_selection` seed scores each trial, and held-out seeds are
not used for tuning. Vary training seeds later to test whether the selected
configuration is robust, rather than treating Optuna trials as independent
training-seed replicates.

Every key is validated before anything is launched:

- Unknown keys at any level are refused, naming the valid ones. A relative
  `matrix` path is resolved against the mesh root.
- `search_space` must be non-empty and its keys a subset of `n_steps`, `gamma`,
  and `ent_coef` — the only knobs that reach `MaskablePPO` today.
  `total_timesteps` gets its own message: the budget is fixed per study and
  comes from the matrix. Any other name is reported as not wired, pointing at
  the open `TODO-RL-TUNE-1`.
- `n_steps` is `int` or `categorical` with distinct integer choices ≥ 2;
  `gamma` is a float range inside `(0, 1]`; `ent_coef` is a float range with
  `low ≥ 0`; `log: true` requires `low > 0`; every range needs `low < high`.
- `row` must exist in the matrix and `training_seed` must be one of the
  matrix's `seeds.training`, which `load_matrix` has already proven disjoint
  from the model-selection and held-out seeds.
- The matrix must set `seeds.model_selection` and `training.total_timesteps`,
  and satisfy `0 < training.eval_every_steps <= total_timesteps` — otherwise no
  model selection ever runs and every objective would be `null`.
- `sampler.type` must be `tpe` with an integer `seed`; `1 <= n_trials <= 50`.
- An `--output-root` already holding `study_manifest.json` or `trials/` is
  refused: studies are not resumed, so choose a new root.
- `--sim-binary` is always required, and must be an existing executable file
  unless `--dry-run`.

### Trials, objective, and sampler

For each trial the sampled values replace the same keys in a copy of the
matrix's `training` block, the copy's `seeds.training` becomes just the study's
training seed, and `build_plan` supplies the train step. Only that step's
module, args, and output directory are used; no plan file is written. The trial
command is therefore byte-for-byte the command a matrix with those values would
run, carrying `--eval-seed <model_selection>`; a command whose `--seed` or
`--eval-seed` value is a held-out seed is refused — the guard reads only the
value following each of those two flags, not every token of the command. Trials run sequentially as subprocesses.

The sampler is `TPESampler(seed=<spec seed>, n_startup_trials=10)`, and that
constant is recorded in the manifest. With ten or fewer trials every draw is an
objective-independent startup draw, so `--dry-run` asks `min(n_trials, 10)`
times without telling and prints exactly those parameter sets and commands; for
a larger `n_trials` it says the remaining trials depend on earlier objectives.
A dry run creates no output directory, record, training process, or simulator
process.

The objective is `train_manifest.json`'s `best_mean_reward`, maximized. A
non-zero training exit, an unreadable manifest, or a non-finite or `null` value
fails that trial with a recorded reason and the study continues.

### Outputs

```
<output-root>/study_manifest.json
<output-root>/trials/trial-0000/trial.json
<output-root>/trials/trial-0000/train/<row>/train-seed-<S>/   (train.py's own output)
```

`trial.json` holds `trial_version`, `number`, `params`, the resolved full
`training` block, `module`, `args`, `train_dir`, `state`
(`complete`/`failed`), `objective`, `failure`, `exit_code`, `started_at`, and
`ended_at`.

`study_manifest.json` is rewritten atomically after every trial with
`study_manifest_version`, `status` (`running`/`completed`/`failed`), the `spec`
(path, sha256, body), the `matrix` (path, sha256, name), `row`, `seed_roles`
(`training`, `model_selection`, `held_out_used: false`), the `objective`
description, `sampler`, `fixed_training`, `optuna_version`,
`package_versions`, `python_version`, `platform`, `sim_binary`, the per-trial
summaries, `best`, `started_at`, and `ended_at`. `best.training` is a full
resolved training block, ready to be copied by hand into a new matrix file —
nothing edits a matrix automatically.

### Optuna pin

Optuna is pinned in `requirements-tuning.txt` and deliberately kept out of
`requirements.txt`: the training manifests record the direct dependency set, so
adding a tuner there would change every manifest and force the cluster venv to
carry a package no job imports. `tune.py` imports Optuna lazily; a missing
install or a version other than the pin is a refusal naming both versions and
the install command. Nothing else in this package imports Optuna, and no
cluster job does.

## Walkthrough: the tracked smoke matrix

`inputs/experiments/bypass-smoke-matrix.json` has 4 rows × training seeds
`[101, 102]`, held-out seeds `301-303`, and 3 evaluation policies, giving 17
plan steps and **8 tasks**:

| Index | Task id |
|---|---|
| 0, 1 | `local-delivery/train-seed-101`, `…-102` |
| 2, 3 | `raw-delivery/…` |
| 4, 5 | `local-connectivity/…` |
| 6, 7 | `local-legacy/…` |
| — | compare, one job depending `afterany` on the array |

1. `submit --tasks 0` writes receipt `0001` with `--array=0` and queues no
   compare job, because seven tasks are uncovered. `status` shows task 0 move
   `pending` → `running` → `completed` while 1–7 stay `unsubmitted`.
2. `resume` writes receipt `0002` with `--array=1-7` plus a compare job
   depending `afterany` on that array.
3. Task 3 hits the wall-time limit: accounting reports `TIMEOUT`, its train
   manifest still says `running`, and its directory is `blocked`, so `status`
   reports `failed` with the retry message. The compare job starts, finds
   `evaluate/raw-delivery/train-seed-102` missing, writes nothing, and exits 1;
   the compare row shows `failed`.
4. The human moves the blocked directory aside and raises `task.time` in the
   config. `resume` writes receipt `0003` with `--array=3` and a new compare
   job; tasks 0–2 and 4–7 are `completed` and are not resubmitted.
5. Task 3 finishes and its evaluation exits 2 (health counters). The runner
   exits 0 and the record keeps `exit_code: 2, tolerated: true`. The comparison
   runs and exits 2, which also maps to 0; the compare row shows `completed`,
   and `cluster compare` would print
   `complete with health counters or seed overlap`.
6. On the laptop:
   `fetch --remote user@host:<root> --dest outputs/fetched/bypass-smoke --select comparison,manifests`.

## Validation status

The cluster commands are exercised **against a fake scheduler only**: `sbatch`,
`squeue`, `sacct`, and `scancel` shims in the test suite, never a real SLURM
installation. `fetch` is exercised against a fake `rsync` shim. `squeue` and
`sacct` output and state names vary by SLURM version and site configuration, so
the parsers remain unproven until a live run. Every query failure degrades to
`unknown`, which blocks resubmission rather than risking a duplicate job.

### Needs a live cluster

- Which scheduler and version, whether `sacct --array` and `--parsable` are
  accepted, and whether `<job id>_<index>` element ids behave as parsed here.
- Partition, account, QOS, constraint, site `MaxArraySize`, and any per-user
  concurrent-task cap, for the config file.
- Whether environment modules are needed, which Python backs the shared venv,
  and whether the pinned wheels install there.
- Per-task wall time and memory, taken from a benchmark repeated **on the
  cluster** rather than from a laptop measurement; whether preemption exists
  and whether the site overrides `--no-requeue`.
- Absolute paths for the venv, checkout, binary, and output root, whether they
  are visible to compute nodes, and whether any of them is purged on a
  schedule.
- Whether `sacct` accounting is enabled and how long its history is kept, and
  whether a short pure-Python `cluster compare` may run on a login node.
- The SSH/rsync access pattern for `fetch` (jump host, key, allowed transfer
  node).
- Whether the site exposes a meaningful start estimate at all.

Findings from a live run belong in the open `TODO-RL-OPS-1` entry.
