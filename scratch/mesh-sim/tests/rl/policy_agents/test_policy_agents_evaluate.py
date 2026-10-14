"""policy/evaluate.py loop, metrics, and policies, plus policy/preferences.py.

Contract sources: scripts/rl/policy/README.md ("Evaluation") and CLAUDE.md: one env
per policy, every policy over every seed; eval_manifest.json is written before the
first episode and after every episode, so a crash leaves `failed` / `partial`, never
none; unattempted seeds are `not_run`; hold never masked; random_valid reseeded per
episode; model deterministic and masked. The loop is driven by an in-process fake env
(no simulator) that writes rl_episode.json / steps.jsonl like the bridge does.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.rl.env.telemetry import TELEMETRY_FILE
from scripts.rl.policy import evaluate as ev
from scripts.rl.policy.preferences import PreferenceCapture

SLOTS = 2


# Fake environment --------------------------------------------------------------------

class _Selection:
    def describe(self):
        return {"observation_preset": "raw_links_v1", "telemetry": "steps"}


class FakeEnv:
    """Writes an episode dir per reset; `decisions` steps per episode, reward -0.5 each."""

    def __init__(self, root: Path, name: str, decisions=3, step_fail=(), reset_fail=(),
                 interrupt=(), wrong_return=(), log=None):
        self.root, self.name, self.decisions = Path(root), name, decisions
        self.step_fail, self.reset_fail = set(step_fail), set(reset_fail)
        self.interrupt, self.wrong_return = set(interrupt), set(wrong_return)
        self.log = log if log is not None else []
        self._cmd = ["sim"]
        self.contract = None
        self.selection = _Selection()
        self.observation_schema = {"sha256": "o" * 64}
        self.reward_schema = {"sha256": "r" * 64}
        self.closed = False
        self.resets = []
        self.actions = []
        self._episodes = 0

    # helpers
    def _dir(self) -> Path:
        return Path(self._cmd[1].split("=", 1)[1])

    def _write_manifest(self, status, **extra):
        payload = {"seed": self._seed, "status": status, "exit_code": 0,
                   "decisions": self._t, "cumulative_reward": self._total,
                   "reward_components_sum": {}}
        payload.update(extra)
        (self._dir() / "rl_episode.json").write_text(json.dumps(payload))

    def _append(self, record):
        with open(self._dir() / TELEMETRY_FILE, "a") as handle:
            handle.write(json.dumps(record) + "\n")

    def _step_record(self, decision):
        window = {"ticks": 5, "demand_mbps_sum": 10.0, "delivered_mbps_sum": 8.0,
                  "connected_pairs_sum": 15, "los_pairs_sum": 15,
                  "flow_ticks_with_demand": 4, "unroutable_flow_ticks": 1}
        nodes = [[float(decision), 0.0, 10.0], [0.0, 0.0, 10.0], [5.0, 5.0, 10.0]]
        return {"type": "step", "decision": decision,
                "facts": {"window": window, "nodes": nodes}}

    # env API used by run_episode / evaluate
    def reset(self, seed=None, options=None):
        self.log.append(("reset", self.name, seed))
        self.resets.append(seed)
        if seed in self.reset_fail:
            raise RuntimeError(f"reset failed for seed {seed}")
        self._episodes += 1
        episode = self.root / self.name / f"episode-{self._episodes:04d}"
        episode.mkdir(parents=True)
        self._cmd = ["sim", f"--output-dir={episode}"]
        self._seed, self._t, self._total = seed, 0, 0.0
        self.contract = {"num_links": 3, "node_ids": ["a", "b", "c"], "mask_dim": 5 * SLOTS}
        self._append({"type": "header", "contract": self.contract})
        self._append(self._step_record(0))
        self._write_manifest("running")
        return np.zeros(4, dtype=np.float32), {}

    def action_masks(self):
        return np.ones(5 * SLOTS, dtype=bool)

    def step(self, action):
        self.actions.append([int(a) for a in action])
        if self._seed in self.interrupt:
            raise KeyboardInterrupt
        self._t += 1
        self._total += -0.5
        self._append(self._step_record(self._t))
        if self._seed in self.step_fail and self._t == 2:
            self._write_manifest("failed", exit_code=3)
            raise RuntimeError("simulator exited")
        done = self._t >= self.decisions
        if done:
            total = self._total + (1.0 if self._seed in self.wrong_return else 0.0)
            self._write_manifest("completed", cumulative_reward=total)
        return np.zeros(4, dtype=np.float32), -0.5, done, False, {"revalidated_slots": []}

    def close(self):
        self.log.append(("close", self.name))
        self.closed = True


def _hold_spec(name="hold", initial=False):
    def build(env, seed):
        start = env.reset(seed=seed, options={"seed_source": "eval"}) if initial else None
        return ev.Prepared(ev.HoldPolicy(), initial=start)

    return ev.PolicySpec(name, build)


def _manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / ev.EVAL_MANIFEST_NAME).read_text())


# 1. Policies -------------------------------------------------------------------------

def test_hold_policy_shape_and_dtype():
    action = ev.HoldPolicy().act(None, np.ones(15, dtype=bool), {})
    assert action.dtype == np.int64 and action.tolist() == [4, 4, 4]
    assert ev.HoldPolicy().act(None, np.ones(0, dtype=bool), {}).tolist() == []


def test_random_valid_picks_hold_when_it_is_the_only_valid_action():
    mask = np.zeros(10, dtype=bool)
    mask[[4, 9]] = True
    policy = ev.RandomValidPolicy()
    policy.start_episode(3)
    for _ in range(20):
        assert policy.act(None, mask, {}).tolist() == [4, 4]


class _RecordingModel:
    def __init__(self, action):
        self.calls, self._action = [], action

    def predict(self, obs, **kwargs):
        self.calls.append(kwargs)
        return self._action, None


class _Capture:
    def __init__(self):
        self.attached = []

    def attach(self, model):
        self.attached.append(model)


def test_model_policy_is_deterministic_masked_and_flattened():
    model, capture = _RecordingModel(np.array([[1, 4]])), _Capture()
    policy = ev.ModelPolicy(model, capture)
    action = policy.act(np.zeros(4), [1, 1, 1, 1, 1, 0, 0, 0, 0, 1], {})
    assert action.dtype == np.int64 and action.tolist() == [1, 4]
    call = model.calls[0]
    assert call["deterministic"] is True
    assert call["action_masks"].dtype == bool
    assert call["action_masks"].tolist() == [True] * 5 + [False] * 4 + [True]
    assert capture.attached == [model]


# 2. episode_metrics ------------------------------------------------------------------

def _write_steps(path: Path, records, header=None):
    path.mkdir(parents=True, exist_ok=True)
    lines = [header or {"type": "header", "contract": {"node_ids": ["a", "b"]}}] + records
    (path / TELEMETRY_FILE).write_text("\n".join(json.dumps(r) for r in lines) + "\n")


def _step(decision, nodes=((0.0, 0.0), (0.0, 0.0)), **window):
    base = {"ticks": 2, "demand_mbps_sum": 10.0, "delivered_mbps_sum": 5.0,
            "connected_pairs_sum": 1, "los_pairs_sum": 0, "flow_ticks_with_demand": 2,
            "unroutable_flow_ticks": 1}
    base.update(window)
    return {"type": "step", "decision": decision,
            "facts": {"window": base, "nodes": [list(n) for n in nodes]}}


def test_metrics_are_window_sums_excluding_decision_zero(tmp_path):
    records = [_step(0, demand_mbps_sum=1000.0, delivered_mbps_sum=0.0),
               _step(1, nodes=((3.0, 4.0), (0.0, 0.0))),
               {"type": "note", "decision": 2},
               _step(2, nodes=((0.0, 4.0), (0.0, 1.0)), delivered_mbps_sum=10.0,
                     connected_pairs_sum=2, los_pairs_sum=2, unroutable_flow_ticks=0)]
    _write_steps(tmp_path, records)
    metrics = ev.episode_metrics(tmp_path, num_links=1)
    assert metrics["delivery_ratio"] == pytest.approx(15.0 / 20.0)
    assert metrics["connectivity"] == pytest.approx(3 / 4)
    assert metrics["los_fraction"] == pytest.approx(2 / 4)
    assert metrics["unroutable_fraction"] == pytest.approx(1 / 4)
    assert metrics["first_all_los_decision"] == 2
    # node a: 0,0 -> 3,4 (5 m) -> 0,4 (3 m); node b: 0,0 -> 0,0 -> 0,1 (1 m)
    assert metrics["per_node_travel_m"] == pytest.approx({"a": 8.0, "b": 1.0})
    assert metrics["travel_m_total"] == pytest.approx(9.0)
    assert metrics["per_node_displacement_m"] == pytest.approx({"a": 4.0, "b": 1.0})
    assert metrics["displacement_m_final"] == pytest.approx(5.0)


def test_degenerate_windows_leave_metrics_null_not_zero(tmp_path):
    _write_steps(tmp_path, [_step(1, demand_mbps_sum=0.0, delivered_mbps_sum=0.0,
                                  flow_ticks_with_demand=0, unroutable_flow_ticks=0)])
    metrics = ev.episode_metrics(tmp_path, num_links=0)
    assert metrics["delivery_ratio"] is None
    assert metrics["connectivity"] is None and metrics["los_fraction"] is None
    assert metrics["unroutable_fraction"] is None
    # Without a decision-0 record no travel origin exists.
    assert metrics["travel_m_total"] is None and "per_node_travel_m" not in metrics


def test_missing_or_header_only_telemetry_gives_all_null(tmp_path):
    names = ("delivery_ratio", "connectivity", "los_fraction", "unroutable_fraction",
             "first_all_los_decision", "travel_m_total", "displacement_m_final")
    assert ev.episode_metrics(tmp_path / "absent", 3) == {n: None for n in names}
    _write_steps(tmp_path / "header", [])
    assert ev.episode_metrics(tmp_path / "header", 3) == {n: None for n in names}
    _write_steps(tmp_path / "zero", [_step(0)])
    assert set(ev.episode_metrics(tmp_path / "zero", 3).values()) == {None}


# 3. summarize ------------------------------------------------------------------------

def _result(status="completed", ret=-1.0, decisions=2, **kwargs):
    defaults = dict(seed=1, episode_dir=None, status=status, exit_code=0,
                    decisions=decisions, total_return=ret, reward_components_sum={},
                    revalidated_slots_total=1, mask_violations=2, actions_sha256=None,
                    metrics={})
    defaults.update(kwargs)
    return ev.EpisodeResult(**defaults)


def test_summarize_counts_only_completed_episodes():
    summary = ev.summarize([_result(ret=-2.0, decisions=4), _result(ret=-1.0, decisions=0),
                            _result(status="failed", ret=-50.0)], expected=5)
    assert summary["expected_episodes"] == 5
    assert summary["completed_episodes"] == 2
    assert summary["mean_return"] == pytest.approx(-1.5)
    assert (summary["min_return"], summary["max_return"]) == (-2.0, -1.0)
    assert summary["mean_reward_per_decision"] == pytest.approx(-0.5)  # 0-decision skipped
    assert summary["mask_violations_total"] == 4
    assert summary["revalidated_slots_total"] == 2


# 4. evaluate() loop ------------------------------------------------------------------

def test_every_policy_runs_every_seed_in_its_own_env(tmp_path):
    envs, log = {}, []
    out = tmp_path / "eval"

    def make_env(name):
        assert _manifest(out)["status"] == "running"   # written before any episode
        envs[name] = FakeEnv(tmp_path / "episodes", name, log=log)
        return envs[name]

    manifest = ev.evaluate(make_env, [_hold_spec("hold"), _hold_spec("other")],
                           [5, 6, 7], out, {"label": "x"})
    assert list(envs) == ["hold", "other"]
    assert all(env.closed and env.resets == [5, 6, 7] for env in envs.values())
    # Policies run one after the other; each env is closed before the next is used.
    assert log.index(("close", "hold")) < log.index(("reset", "other", 5))
    assert manifest == _manifest(out)
    assert manifest["status"] == "completed"
    assert (manifest["episodes_expected"], manifest["episodes_completed"]) == (6, 6)
    assert manifest["eval_manifest_version"] == ev.EVAL_MANIFEST_VERSION == 2
    assert manifest["deterministic"] is True and manifest["seed_source"] == "eval"
    assert manifest["label"] == "x" and manifest["ended_at"]
    assert manifest["contract"]["num_links"] == 3
    assert manifest["selection"] == _Selection().describe()
    for block in manifest["policies"].values():
        assert [e["seed"] for e in block["episodes"]] == [5, 6, 7]
        assert all(e["status"] == "completed" and e["return"] == pytest.approx(-1.5)
                   for e in block["episodes"])
        assert block["episodes"][0]["actions_sha256"] == ev._actions_sha256([[4, 4]] * 3)
        assert block["episodes"][0]["metrics"]["delivery_ratio"] == pytest.approx(0.8)


def test_initial_reset_from_build_is_reused_only_for_the_first_seed(tmp_path):
    env = FakeEnv(tmp_path / "episodes", "hold")
    manifest = ev.evaluate(lambda name: env, [_hold_spec(initial=True)], [5, 6],
                           tmp_path / "eval", {})
    assert env.resets == [5, 6]          # build's reset for 5, the loop's reset for 6
    assert manifest["status"] == "completed"


def test_random_valid_is_reseeded_with_each_episode_seed(tmp_path):
    seen = []

    class _Spy(ev.RandomValidPolicy):
        def start_episode(self, seed):
            seen.append(seed)
            super().start_episode(seed)

    spec = ev.PolicySpec("random_valid", lambda env, seed: ev.Prepared(_Spy()))
    ev.evaluate(lambda name: FakeEnv(tmp_path / "episodes", name), [spec], [3, 1, 2],
                tmp_path / "eval", {})
    assert seen == [3, 1, 2]


def test_a_failed_seed_is_kept_and_the_run_is_partial(tmp_path):
    manifest = ev.evaluate(lambda name: FakeEnv(tmp_path / "episodes", name, step_fail={6}),
                           [_hold_spec()], [5, 6, 7], tmp_path / "eval", {})
    assert manifest["status"] == "partial"
    episodes = manifest["policies"]["hold"]["episodes"]
    assert [e["status"] for e in episodes] == ["completed", "failed", "completed"]
    failed = episodes[1]
    assert "simulator exited" in failed["error"]
    assert failed["exit_code"] == 3 and failed["decisions"] == 2
    assert set(failed["metrics"].values()) == {None}
    assert manifest["policies"]["hold"]["summary"]["completed_episodes"] == 2


def test_every_seed_failing_gives_a_failed_manifest_without_raising(tmp_path):
    manifest = ev.evaluate(
        lambda name: FakeEnv(tmp_path / "episodes", name, step_fail={5, 6}),
        [_hold_spec()], [5, 6], tmp_path / "eval", {})
    assert manifest["status"] == "failed"
    assert manifest["episodes_completed"] == 0


def test_a_return_mismatch_is_recorded_as_a_failed_episode(tmp_path):
    manifest = ev.evaluate(
        lambda name: FakeEnv(tmp_path / "episodes", name, wrong_return={5}),
        [_hold_spec()], [5], tmp_path / "eval", {})
    episode = manifest["policies"]["hold"]["episodes"][0]
    assert episode["status"] == "failed" and "does not match" in episode["error"]
    assert manifest["status"] == "failed"


def test_a_failed_reset_is_not_attributed_to_the_previous_episode(tmp_path):
    manifest = ev.evaluate(lambda name: FakeEnv(tmp_path / "episodes", name, reset_fail={6}),
                           [_hold_spec()], [5, 6], tmp_path / "eval", {})
    first, second = manifest["policies"]["hold"]["episodes"]
    assert second["status"] == "failed"
    assert second["episode_dir"] is None and second["decisions"] == 0
    assert second["return"] is None
    assert first["episode_dir"] != second["episode_dir"]


def test_a_crash_while_building_a_later_env_leaves_a_failed_manifest(tmp_path):
    out = tmp_path / "eval"

    def make_env(name):
        if name == "second":
            raise OSError("cannot launch")
        return FakeEnv(tmp_path / "episodes", name)

    with pytest.raises(OSError, match="cannot launch"):
        ev.evaluate(make_env, [_hold_spec("first"), _hold_spec("second")], [5, 6], out, {})
    manifest = _manifest(out)
    assert manifest["status"] == "failed"
    assert "OSError: cannot launch" in manifest["error"] and manifest["ended_at"]
    assert [e["status"] for e in manifest["policies"]["first"]["episodes"]] == [
        "completed", "completed"]
    second = manifest["policies"]["second"]
    assert [(e["seed"], e["status"]) for e in second["episodes"]] == [(5, "not_run"),
                                                                     (6, "not_run")]
    assert second["summary"]["expected_episodes"] == 2
    assert manifest["episodes_completed"] == 2


def test_an_interrupt_mid_episode_marks_the_rest_not_run(tmp_path):
    out = tmp_path / "eval"
    envs = []

    def make_env(name):
        envs.append(FakeEnv(tmp_path / "episodes", name, interrupt={6}))
        return envs[-1]

    with pytest.raises(KeyboardInterrupt):
        ev.evaluate(make_env, [_hold_spec("first"), _hold_spec("second")], [5, 6, 7],
                    out, {})
    manifest = _manifest(out)
    assert manifest["status"] == "failed" and "KeyboardInterrupt" in manifest["error"]
    assert [e["status"] for e in manifest["policies"]["first"]["episodes"]] == [
        "completed", "not_run", "not_run"]
    assert [e["status"] for e in manifest["policies"]["second"]["episodes"]] == [
        "not_run"] * 3
    assert len(envs) == 1 and envs[0].closed


def test_a_failing_build_runs_no_episode_and_closes_the_env(tmp_path):
    env = FakeEnv(tmp_path / "episodes", "hold")

    def build(env, seed):
        raise ValueError("bad plan")

    with pytest.raises(ValueError, match="bad plan"):
        ev.evaluate(lambda name: env, [ev.PolicySpec("hold", build)], [5, 6],
                    tmp_path / "eval", {})
    manifest = _manifest(tmp_path / "eval")
    assert env.resets == [] and env.closed
    assert [e["status"] for e in manifest["policies"]["hold"]["episodes"]] == [
        "not_run", "not_run"]


def test_the_manifest_on_disk_grows_by_one_record_per_episode(tmp_path):
    out = tmp_path / "eval"
    counts = []

    class _Counting(ev.HoldPolicy):
        def start_episode(self, seed):
            block = _manifest(out)["policies"].get("hold") or {"episodes": []}
            counts.append(len(block["episodes"]))

    spec = ev.PolicySpec("hold", lambda env, seed: ev.Prepared(_Counting()))
    ev.evaluate(lambda name: FakeEnv(tmp_path / "episodes", name), [spec], [1, 2, 3, 4],
                out, {})
    assert counts == [0, 1, 2, 3]
    assert len(_manifest(out)["policies"]["hold"]["episodes"]) == 4


def test_policy_metadata_is_stored_as_the_baseline_block(tmp_path):
    spec = ev.PolicySpec("geo", lambda env, seed: ev.Prepared(ev.HoldPolicy()),
                         metadata={"method": "geometric", "fingerprint": "f" * 64})
    plain = _hold_spec()
    manifest = ev.evaluate(lambda name: FakeEnv(tmp_path / "episodes", name), [spec, plain],
                           [1], tmp_path / "eval", {})
    assert manifest["policies"]["geo"]["baseline"]["fingerprint"] == "f" * 64
    assert "baseline" not in manifest["policies"]["hold"]


# 5. PreferenceCapture ----------------------------------------------------------------


def test_capture_copies_the_last_forward_and_detaches_cleanly():
    torch = pytest.importorskip("torch")

    class _Policy:
        action_net = torch.nn.Linear(3, 4)

    class _Model:
        policy = _Policy()

    capture = PreferenceCapture()
    capture.attach(_Model())
    assert capture.registered is True and capture.error is None
    with torch.no_grad():
        output = _Model.policy.action_net(torch.ones(1, 3))
        expected = output.numpy().reshape(-1).copy()
        output.add_(100.0)                      # mutate after the hook ran
    captured = capture.take()
    assert captured.dtype == np.float32 and captured.shape == (4,)
    assert np.allclose(captured, expected)
    assert capture.take() is None               # take() consumes the value

    capture.detach()
    capture.detach()                            # idempotent
    assert capture.registered is False
    with torch.no_grad():
        _Model.policy.action_net(torch.ones(1, 3))
    assert capture.take() is None
