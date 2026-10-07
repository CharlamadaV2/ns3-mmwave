"""Opt-in per-decision records: policy_decisions.jsonl and its manifest."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .telemetry import TELEMETRY_FILE, obs_sha256, reward_field

DECISIONS_FILE = "policy_decisions.jsonl"
DECISIONS_MANIFEST = "policy_decisions_manifest.json"
DECISION_RECORD_VERSION = 1
OBS_VECTOR_MODES = ("auto", "always")
PREFERENCE_MODES = ("auto", "off")
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
HOLD_INDEX = 4
MODEL_INPUT_CONVERSION = "torch.Tensor.float() == numpy.astype(float32)"
PREPROCESS_NOTE = "stable_baselines3.common.preprocessing.preprocess_obs: obs.float()"
CONTEXT_MODES = ("training", "evaluation")
CONTEXT_SOURCES = ("train", "train_eval", "evaluate")
PREFERENCE_REPRESENTATION = "raw_action_net_logits_float32"
PREFERENCE_CAPTURE = "action_net_forward_hook"
RESET_NOTE = "reset snapshot; no policy inference"
_SLOT_ACTIONS = 5
_MAX_ERROR_CHARS = 1000
_FLAGS = {
    "enabled": "--decision-records",
    "obs_vector": "--decision-records-obs-vector",
    "preferences": "--decision-records-preferences",
    "record_every": "--decision-records-every",
    "max_bytes_per_episode": "--decision-records-max-bytes",
}
_DEFAULTS = {
    "enabled": False,
    "obs_vector": "auto",
    "preferences": "auto",
    "record_every": 1,
    "max_bytes_per_episode": DEFAULT_MAX_BYTES,
}


@dataclass(frozen=True)
class DecisionRecordSettings:
    """Validated decision-record settings with the source of every key."""

    enabled: bool = False
    obs_vector: str = "auto"
    preferences: str = "auto"
    record_every: int = 1
    max_bytes_per_episode: int = DEFAULT_MAX_BYTES
    source: dict = field(default_factory=dict)

    def describe(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "obs_vector": self.obs_vector,
            "preferences": self.preferences,
            "record_every": int(self.record_every),
            "max_bytes_per_episode": int(self.max_bytes_per_episode),
            "source": {key: self.source.get(key, "default") for key in _DEFAULTS},
        }


@dataclass(frozen=True)
class DecisionContext:
    """Who produced an episode's decisions: training, callback evaluation, or evaluate."""

    mode: str
    source: str
    policy: str | None = None
    model: dict | None = None
    preference_source: Callable[[], np.ndarray | None] | None = None

    def __post_init__(self):
        if self.mode not in CONTEXT_MODES:
            raise ValueError(f"DecisionContext mode {self.mode!r} not in {CONTEXT_MODES}")
        if self.source not in CONTEXT_SOURCES:
            raise ValueError(
                f"DecisionContext source {self.source!r} not in {CONTEXT_SOURCES}")


@dataclass(frozen=True)
class DecisionRecording:
    """The single `decision_records=` value accepted by MeshRlEnv."""

    settings: DecisionRecordSettings
    context: DecisionContext


def _int_setting(key: str, value, minimum: int, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer >= {minimum}, got {value!r} ({source})")
    try:
        number = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{key} must be an integer >= {minimum}, got {value!r} ({source})") from exc
    if number < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}, got {number} ({source})")
    return number


def resolve_decision_records(*, enabled=None, obs_vector=None, preferences=None,
                             record_every=None,
                             max_bytes_per_episode=None) -> DecisionRecordSettings:
    """Validate decision-record settings before any simulator process exists."""
    given = {
        "enabled": enabled,
        "obs_vector": obs_vector,
        "preferences": preferences,
        "record_every": record_every,
        "max_bytes_per_episode": max_bytes_per_episode,
    }
    sources = {key: ("cli" if value is not None else "default")
               for key, value in given.items()}
    values = {key: (value if value is not None else _DEFAULTS[key])
              for key, value in given.items()}

    if not isinstance(values["enabled"], bool):
        raise ValueError(
            f"enabled must be a boolean, got {values['enabled']!r} ({sources['enabled']})")
    if not values["enabled"]:
        dependent = [key for key in _DEFAULTS
                     if key != "enabled" and given[key] is not None]
        if dependent:
            flags = ", ".join(f"{_FLAGS[key]} ({key})" for key in dependent)
            raise ValueError(
                f"{flags} requires {_FLAGS['enabled']} ({sources[dependent[0]]})")

    for key, modes in (("obs_vector", OBS_VECTOR_MODES),
                       ("preferences", PREFERENCE_MODES)):
        values[key] = str(values[key]).strip()
        if values[key] not in modes:
            raise ValueError(
                f"{key} {values[key]!r} is not valid ({sources[key]}); valid choices: "
                f"{list(modes)}")
    values["record_every"] = _int_setting(
        "record_every", values["record_every"], 1, sources["record_every"])
    values["max_bytes_per_episode"] = _int_setting(
        "max_bytes_per_episode", values["max_bytes_per_episode"], 0,
        sources["max_bytes_per_episode"])
    return DecisionRecordSettings(source=sources, **values)


def mask_sha256(mask) -> str:
    """SHA-256 of the mask as compact JSON integers."""
    # Cast to int: bool or int8 JSON would give a different digest than the published form.
    ints = [int(m) for m in mask]
    return hashlib.sha256(json.dumps(ints, separators=(",", ":")).encode("utf-8")).hexdigest()


def applied_action(requested, revalidated_slots, hold=HOLD_INDEX) -> list[int]:
    """Requested joint action with hold at every revalidated slot."""
    applied = [int(a) for a in requested]
    for slot in revalidated_slots:
        applied[int(slot)] = int(hold)
    return applied


def _slot_rows(values, slots: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != slots * _SLOT_ACTIONS:
        raise ValueError(
            f"preferences have {array.size} values, expected {slots * _SLOT_ACTIONS}")
    return array.reshape(slots, _SLOT_ACTIONS)


def masked_probs(per_slot: np.ndarray, mask) -> list[list[float]]:
    """Per-slot softmax over unmasked actions, 6 dp; masked actions are exactly 0.0."""
    flat_mask = np.asarray(mask).reshape(-1).astype(bool)
    rows = _slot_rows(per_slot, flat_mask.size // _SLOT_ACTIONS).astype(np.float64)
    allowed = flat_mask.reshape(rows.shape)
    result = []
    for logits, keep in zip(rows, allowed):
        probs = np.zeros(_SLOT_ACTIONS, dtype=np.float64)
        if keep.any():
            shifted = np.exp(logits[keep] - logits[keep].max())
            probs[keep] = shifted / shifted.sum()
        result.append([round(float(p), 6) if k else 0.0 for p, k in zip(probs, keep)])
    return result


def coverage_ranges(saved: list[int]) -> list[list[int]]:
    """Maximal contiguous runs of saved decisions."""
    ranges: list[list[int]] = []
    for decision in sorted(set(int(d) for d in saved)):
        if ranges and decision == ranges[-1][1] + 1:
            ranges[-1][1] = decision
        else:
            ranges.append([decision, decision])
    return ranges


def coverage_gaps(saved: list[int], expected: int) -> list[list[int]]:
    """Complement of the saved decisions within [1, expected]."""
    gaps: list[list[int]] = []
    start = 1
    for low, high in coverage_ranges([d for d in saved if 1 <= d <= expected]):
        if low > start:
            gaps.append([start, low - 1])
        start = high + 1
    if start <= expected:
        gaps.append([start, int(expected)])
    return gaps


def _steps_ref(decision: int, obs) -> dict:
    return {"file": TELEMETRY_FILE, "decision": int(decision), "obs_sha256": obs_sha256(obs)}


def reset_record(msg, obs, obs_dtype, steps_saved) -> dict:
    """Line 1: the reset observation and mask; no action or outcome."""
    mask = [int(m) for m in msg["mask"]]
    return {
        "type": "reset",
        "decision": int(msg["decision"]),
        "tick": int(msg["tick"]),
        "time_s": float(msg["time_s"]),
        "obs_dtype": obs_dtype,
        "obs_sha256": obs_sha256(obs, obs_dtype),
        "mask": mask,
        "mask_sha256": mask_sha256(mask),
        "steps_ref": _steps_ref(msg["decision"], obs) if steps_saved else None,
        "note": RESET_NOTE,
    }


def _preferences_field(values, mask) -> dict | None:
    if values is None:
        return None
    # Copy: the capture buffer belongs to the policy and may be reused by its next forward.
    rows = _slot_rows(np.array(values, dtype=np.float32, copy=True), len(mask) // _SLOT_ACTIONS)
    return {
        "representation": PREFERENCE_REPRESENTATION,
        "capture": PREFERENCE_CAPTURE,
        "per_slot": [[float(np.float32(v)) for v in row] for row in rows],
        "masked_probs": masked_probs(rows, mask),
    }


def decision_record(msg, pre, reward, obs_dtype, steps_saved, store_vector,
                    slot_node_ids) -> dict:
    """One decision: pre-action input n-1, requested/applied action, outcome n."""
    decision = int(msg["decision"])
    obs = np.asarray(pre["obs"])
    mask = [int(m) for m in pre["mask"]]
    requested = [int(a) for a in pre["requested"]]
    revalidated = [int(s) for s in msg["revalidated_slots"]]
    input_time = float(pre["time_s"])
    outcome_time = float(msg["time_s"])
    return {
        "type": "decision",
        "decision": decision,
        "targets": [node for node in slot_node_ids if node is not None],
        "input": {
            "source_decision": decision - 1,
            "tick": int(pre["tick"]),
            "time_s": input_time,
            "steps_ref": _steps_ref(decision - 1, obs) if steps_saved else None,
            "obs_dtype": obs_dtype,
            "obs_sha256": obs_sha256(obs, obs_dtype),
            "model_input_dtype": "float32",
            "model_input_conversion": MODEL_INPUT_CONVERSION,
            "model_input_sha256": obs_sha256(obs, "float32"),
            "obs_vector": (np.asarray(obs, dtype=obs_dtype).tolist()
                           if store_vector else None),
            "mask": mask,
            "mask_sha256": mask_sha256(mask),
        },
        "action": {
            "requested": requested,
            "revalidated_slots": revalidated,
            "applied": applied_action(requested, revalidated),
            "applied_status": "derived",
            "hold_index": HOLD_INDEX,
        },
        "outcome": {
            "tick": int(msg["tick"]),
            "time_s": outcome_time,
            "ticks_in_step": int(msg["ticks_in_step"]),
            "interval_s": {"start_exclusive": input_time, "end_inclusive": outcome_time},
            "done": bool(msg["done"]),
            "legacy_reward": float(msg["reward"]),
            "reward": reward_field(reward),
            "reward_schema_sha256": pre["reward_schema_sha256"],
        },
        "preferences": _preferences_field(pre.get("preferences"), mask),
    }


def _write_json_atomic(path, payload) -> None:
    """Write JSON via a temp file and os.replace; kept local for env/ layering."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with open(temp, "w") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _dumps(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_error(source) -> str | None:
    detail = getattr(getattr(source, "__self__", None), "error", None)
    return str(detail) if detail else None


class DecisionRecorder:
    """Writes one episode's decision records and manifest; callers isolate its failures."""

    def __init__(self, episode_dir, settings, context, *, episode, contract,
                 selection_describe, observation_schema, reward_schema):
        self._dir = Path(episode_dir)
        self._path = self._dir / DECISIONS_FILE
        self._manifest_path = self._dir / DECISIONS_MANIFEST
        self._settings = settings
        self._context = context
        self._obs_dtype = observation_schema["dtype"]
        self._reward_sha = reward_schema["sha256"]
        self._slot_node_ids = list(contract["slot_node_ids"])
        self._expected = int(contract["num_decisions"])
        self._capture_allowed = (settings.preferences == "auto"
                                 and context.mode == "evaluation")
        self._handle = None
        self._status = "writing"
        self._failed = False
        self._capped = False
        self._error: str | None = None
        self._bytes = 0
        self._records = 0
        self._saved: list[int] = []
        self._reset_written = False
        self._terminal: int | None = None
        self._prev: tuple[int, int, float] | None = None
        self._captured = False
        steps = selection_describe.get("telemetry") == "steps"
        is_model = (context.mode == "evaluation" and context.source == "evaluate"
                    and context.policy == "model")
        self._manifest = {
            "contract": "decision_record",
            "version": DECISION_RECORD_VERSION,
            "episode": {
                "dir_name": str(episode["dir_name"]),
                "index": int(episode["index"]),
                "seed": int(episode["seed"]),
                "seed_source": str(episode["seed_source"]),
                "mode": context.mode,
                "source": context.source,
                "policy": context.policy,
                "status": None,
                "rl_episode_manifest": "rl_episode.json",
                "steps_file": TELEMETRY_FILE if steps else None,
                "steps_telemetry_every": (int(selection_describe["telemetry_every"])
                                          if steps else None),
            },
            "settings": settings.describe(),
            "observation_schema": {
                "schema_id": observation_schema["schema_id"],
                "dtype": observation_schema["dtype"],
                "obs_dim": int(observation_schema["obs_dim"]),
                "sha256": observation_schema["sha256"],
            },
            "model_input": {
                "dtype": "float32",
                "conversion": PREPROCESS_NOTE,
                "lossless_from_obs": observation_schema["dtype"] == "float32",
            },
            "reward_schema_sha256": reward_schema["sha256"],
            "contract_identity": {
                "contract_id": contract["contract"],
                "node_ids": [str(node) for node in contract["node_ids"]],
                "slot_node_ids": list(contract["slot_node_ids"]),
                "action_meanings": list(contract["action_meanings"]),
                "hold_index": HOLD_INDEX,
                "tick_s": contract["tick_s"],
                "decision_interval_ticks": int(contract["decision_interval_ticks"]),
                "num_decisions": self._expected,
            },
            "model": dict(context.model) if is_model and context.model is not None else None,
            "preferences": {"captured": False, "representation": None, "capture": None,
                            "error": None},
            "coverage": None,
            "status": "writing",
            "jsonl_sha256": None,
            "bytes": 0,
            "error": None,
            "writer": {"library_versions": {"numpy": np.__version__},
                       "written_after_outcome": True},
        }
        try:
            self._handle = open(self._path, "w", encoding="utf-8")
        except Exception as exc:
            self._failed = True
            self._error = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
            self._finalize("failed")
            raise
        self._write_manifest()

    @property
    def status(self) -> str:
        return self._status

    def record_reset(self, msg, obs, steps_saved) -> None:
        """Write the mandatory reset line, or fail the sidecar if the cap cannot hold it."""
        line = _dumps(reset_record(msg, obs, self._obs_dtype, bool(steps_saved))) + "\n"
        size = len(line.encode("utf-8"))
        cap = self._settings.max_bytes_per_episode
        if cap > 0 and size > cap:
            raise ValueError(
                f"max_bytes_per_episode {cap} is smaller than the reset record ({size} bytes)")
        self._write_line(line)
        self._bytes += size
        self._records += 1
        self._reset_written = True
        self._prev = (int(msg["decision"]), int(msg["tick"]), float(msg["time_s"]))

    def record_decision(self, msg, pre, reward, steps_saved) -> None:
        """Append decision n after its outcome, subject to sampling and the byte cap."""
        if self._prev is None:
            raise ValueError("decision record before the reset record")
        prev_decision, prev_tick, prev_time = self._prev
        decision = int(msg["decision"])
        if decision != prev_decision + 1:
            raise ValueError(f"decision {decision} does not follow {prev_decision}")
        if prev_tick != int(msg["tick"]) - int(msg["ticks_in_step"]):
            raise ValueError(
                f"input tick {prev_tick} != outcome tick {msg['tick']} - ticks_in_step "
                f"{msg['ticks_in_step']}")
        self._prev = (decision, int(msg["tick"]), float(msg["time_s"]))
        self._terminal = decision
        due = decision % self._settings.record_every == 0 or bool(msg["done"])
        if self._capped or not due:
            return
        preferences = pre.get("preferences") if self._capture_allowed else None
        full = dict(pre, tick=prev_tick, time_s=prev_time, preferences=preferences,
                    reward_schema_sha256=self._reward_sha)
        store_vector = (not steps_saved) or self._settings.obs_vector == "always"
        record = decision_record(msg, full, reward, self._obs_dtype, bool(steps_saved),
                                 store_vector, self._slot_node_ids)
        line = _dumps(record) + "\n"
        size = len(line.encode("utf-8"))
        cap = self._settings.max_bytes_per_episode
        if cap > 0 and self._bytes + size > cap:
            self._capped = True
            return
        self._write_line(line)
        self._bytes += size
        self._records += 1
        self._saved.append(decision)
        if record["preferences"] is not None:
            self._captured = True

    def close(self, episode_status, stop_reason, episode_error) -> None:
        """Close the JSONL and write the final manifest."""
        if self._status != "writing":
            return
        self._close_handle()
        self._manifest["episode"]["status"] = episode_status
        if self._failed:
            status = "failed"
        elif self._capped:
            status = "capped"
        elif episode_status == "completed":
            status = "complete"
        else:
            status = "interrupted"
            self._error = (episode_error
                           or f"episode {episode_status} ({stop_reason})")[:_MAX_ERROR_CHARS]
        self._finalize(status)

    def fail(self, error: str) -> None:
        """Mark the sidecar failed, close the JSONL, and rewrite the manifest."""
        if self._status != "writing":
            return
        self._failed = True
        self._error = str(error)[:_MAX_ERROR_CHARS]
        self._close_handle()
        self._finalize("failed")

    def _write_line(self, line: str) -> None:
        if self._handle is None:
            raise ValueError(f"decision recorder for {self._path} is closed")
        self._handle.write(line)
        self._handle.flush()

    def _close_handle(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.close()

    def _coverage(self) -> dict:
        return {
            "decisions_expected": self._expected,
            "decision_0_retained": self._reset_written,
            "terminal_decision": self._terminal,
            "saved_ranges": coverage_ranges(self._saved),
            "gaps": coverage_gaps(self._saved, self._expected),
            "records": self._records,
        }

    def _preferences_block(self) -> dict:
        source = self._context.preference_source
        error = None
        if (self._capture_allowed and source is not None and not self._captured
                and self._saved):
            error = _source_error(source) or "preference source produced no capture"
        return {
            "captured": self._captured,
            "representation": PREFERENCE_REPRESENTATION if self._captured else None,
            "capture": PREFERENCE_CAPTURE if self._captured else None,
            "error": error,
        }

    def _write_manifest(self) -> None:
        self._manifest["coverage"] = self._coverage()
        self._manifest["preferences"] = self._preferences_block()
        self._manifest["error"] = self._error
        _write_json_atomic(self._manifest_path, self._manifest)

    def _finalize(self, status: str) -> None:
        self._status = status
        self._manifest["status"] = status
        # Hash the bytes on disk, not the in-memory counter, so the digest matches the file.
        if self._path.is_file():
            data = self._path.read_bytes()
            self._manifest["jsonl_sha256"] = hashlib.sha256(data).hexdigest()
            self._manifest["bytes"] = len(data)
        else:
            self._manifest["jsonl_sha256"] = None
            self._manifest["bytes"] = 0
        self._write_manifest()
