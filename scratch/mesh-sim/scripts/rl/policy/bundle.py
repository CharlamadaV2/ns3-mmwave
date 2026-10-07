"""Read a saved training run as a verified model bundle."""

import dataclasses
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from scripts.rl.cli_common import MANIFEST_NAME, sha256_file
from scripts.rl.env.observations import PRESETS
from scripts.rl.env.rewards import COMPONENTS
from scripts.rl.env.selection import TELEMETRY_MODES, RlSelection
from scripts.rl.policy.compat import BundleError

MANIFEST_VERSION = 4
CHECKPOINT_DIR = "checkpoints"
_SELECTION_KEYS = ("observation_preset", "reward_components", "reward_weights",
                   "telemetry", "telemetry_every")
_SCENARIO_SHA_KEYS = ("run_ini_sha256", "nodes_json_sha256", "buildings_json_sha256",
                      "jammers_json_sha256")


@dataclass(frozen=True)
class ModelBundle:
    """One verified model file with the manifest it came from."""

    run_dir: Path
    manifest: dict
    selection: str
    model_path: Path
    model_sha256: str
    num_timesteps: int | None

    def describe(self) -> dict:
        return {"run_dir": str(self.run_dir), "model_selection": self.selection,
                "model_path": str(self.model_path), "model_sha256": self.model_sha256,
                "num_timesteps": self.num_timesteps,
                "train_manifest_sha256": sha256_file(self.run_dir / MANIFEST_NAME)}


def read_manifest(run_dir) -> dict:
    """Load train_manifest.json; every problem is a BundleError."""
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        raise BundleError(f"No {MANIFEST_NAME} in {run_dir}")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BundleError(f"{path} does not contain a JSON object")
    return manifest


def _checkpoint_target(run_dir: Path, manifest: dict, model: str) -> tuple[Path, str, int]:
    relative = Path(model)
    if (relative.is_absolute() or len(relative.parts) != 2
            or relative.parts[0] != CHECKPOINT_DIR or relative.suffix != ".zip"):
        raise BundleError(
            f"--model {model!r} must be 'final', 'best', or "
            f"'{CHECKPOINT_DIR}/<name>.zip' relative to {run_dir}")
    target = (run_dir / relative).resolve()
    for entry in manifest.get("checkpoints") or []:
        if Path(entry["path"]).resolve() == target:
            return target, entry["sha256"], int(entry["num_timesteps"])
    raise BundleError(f"{model} is not listed in the manifest's retained checkpoints")


def _model_target(run_dir: Path, manifest: dict, model: str) -> tuple[Path, str, int | None]:
    if model == "final":
        path, digest = manifest.get("model_path"), manifest.get("model_sha256")
        if not path or not digest:
            raise BundleError(f"{run_dir} has no final model recorded")
        return Path(path), digest, None
    if model == "best":
        path, digest = manifest.get("best_model_path"), manifest.get("best_model_sha256")
        if not path or not digest:
            raise BundleError(
                f"{run_dir} has no best model; train with --eval-every-steps > 0")
        return Path(path), digest, None
    return _checkpoint_target(run_dir, manifest, model)


def read_bundle(run_dir, model: str = "final") -> ModelBundle:
    """Verify the manifest and the requested model file; no flag bypasses a digest."""
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)

    version = manifest.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise BundleError(
            f"train manifest version {version!r} is not {MANIFEST_VERSION}; "
            "retrain with current tooling")
    status = manifest.get("status")
    if status != "completed":
        raise BundleError(f"training run {run_dir} has status {status!r}, not 'completed'")
    control_mode = manifest.get("control_mode")
    if control_mode == "legacy":
        raise BundleError("legacy control mode is not supported by this loader")
    if control_mode != "centralized":
        raise BundleError(f"control mode {control_mode!r} is not supported by this loader")

    path, digest, num_timesteps = _model_target(run_dir, manifest, model)
    if not path.is_file():
        raise BundleError(f"model file is missing: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise BundleError(
            f"model file digest mismatch for {path}: recorded {digest}, found {actual}")
    selection = model if model in ("final", "best") else "checkpoint"
    return ModelBundle(run_dir, manifest, selection, path, digest, num_timesteps)


def policy_weights_sha256(model_path) -> str | None:
    """SHA-256 of the zip's policy.pth entry, stable across re-saves of identical weights."""
    with zipfile.ZipFile(model_path) as archive:
        if "policy.pth" not in archive.namelist():
            return None
        return hashlib.sha256(archive.read("policy.pth")).hexdigest()


def training_provenance(manifest: dict) -> dict:
    """Portable training facts an evaluation records: seeds, budgets, scenario digests."""
    evaluation = manifest.get("evaluation") or {}
    identity = manifest.get("scenario_identity") or {}
    return {
        "algorithm": manifest.get("algorithm"),
        "seed": manifest.get("seed"),
        "seed_source": manifest.get("seed_source"),
        "evaluation_seed": evaluation.get("seed"),
        "evaluation_episodes": evaluation.get("episodes"),
        "scenario_identity": {key: identity.get(key) for key in _SCENARIO_SHA_KEYS},
        "hyperparameters": dict(manifest.get("hyperparameters") or {}),
    }


def seed_roles(manifest: dict | None, model_selection: str | None,
               seeds) -> dict:
    """Which evaluation seeds collide with the training or model-selection seed."""
    held_out_seeds = [int(seed) for seed in seeds]
    roles = {"training_seed": None, "model_selection_seed": None,
             "selection_seed_used_for_model": model_selection == "best",
             "held_out_seeds": held_out_seeds, "overlap": [],
             "overlap_allowed": False, "held_out": None}
    if manifest is None:
        return roles

    training_seed = manifest.get("seed")
    evaluation = manifest.get("evaluation")
    selection_seed = (evaluation or {}).get("seed") if evaluation else None
    roles["training_seed"] = None if training_seed is None else int(training_seed)
    roles["model_selection_seed"] = None if selection_seed is None else int(selection_seed)
    reserved = {seed for seed in (roles["training_seed"], roles["model_selection_seed"])
                if seed is not None}
    roles["overlap"] = [seed for seed in held_out_seeds if seed in reserved]
    roles["held_out"] = not roles["overlap"]
    return roles


def selection_from_manifest(manifest: dict) -> RlSelection:
    """Rebuild the recorded policy selection, revalidating it against the registries."""
    saved = manifest.get("selection")
    if not isinstance(saved, dict):
        raise BundleError("train manifest has no recorded selection")

    preset = str(saved.get("observation_preset", ""))
    if preset not in PRESETS:
        raise BundleError(f"recorded observation_preset {preset!r} is not registered")
    components = tuple(str(name) for name in saved.get("reward_components") or ())
    unknown = [name for name in components if name not in COMPONENTS]
    if unknown:
        raise BundleError(f"recorded reward_components has unknown entries {unknown}")
    weights = tuple(float(w) for w in saved.get("reward_weights") or ())
    if len(weights) != len(components):
        raise BundleError(
            f"recorded reward_weights has {len(weights)} entries but reward_components "
            f"has {len(components)}")
    telemetry = str(saved.get("telemetry", "none"))
    if telemetry not in TELEMETRY_MODES:
        raise BundleError(f"recorded telemetry {telemetry!r} is not valid")
    every = int(saved.get("telemetry_every", 1))
    if every < 1:
        raise BundleError(f"recorded telemetry_every must be >= 1, got {every}")
    return RlSelection(preset, components, weights, telemetry, every,
                       {key: "manifest" for key in _SELECTION_KEYS})


def eval_selection(selection: RlSelection) -> RlSelection:
    """Force per-decision telemetry so evaluation can derive its own domain metrics."""
    source = dict(selection.source)
    source.update({"telemetry": "eval", "telemetry_every": "eval"})
    return dataclasses.replace(selection, telemetry="steps", telemetry_every=1,
                               source=source)


def load_model(bundle: ModelBundle, env, mask_fn):
    """Reload the verified model onto a masked env; only after check_compatibility."""
    from scripts.rl.agents.mask_ppo import MaskablePpoTrainer

    return MaskablePpoTrainer.load(str(bundle.model_path), env, mask_fn)
