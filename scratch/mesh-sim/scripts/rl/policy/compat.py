"""Ordered compatibility checks between a saved training manifest and a live env."""

from dataclasses import dataclass, field

from scripts.rl.env.observations import SchemaMismatchError, check_schema

STRUCTURAL_FIELDS = ("contract", "dimensions", "action_meanings", "max_controlled_nodes",
                     "num_mesh_nodes", "obs_dim", "mask_dim", "facts_schema")
SCENARIO_CONTRACT_FIELDS = ("slot_node_ids", "node_ids")
IDENTITY_DIGESTS = ("run_ini_sha256", "nodes_json_sha256", "buildings_json_sha256",
                    "jammers_json_sha256")
TRANSFER_NOTE = "matching checks do not imply transfer"


class CompatibilityError(ValueError):
    """A saved model cannot be evaluated against this environment."""


class BundleError(CompatibilityError):
    """The saved manifest or model file is missing, stale, or corrupt."""


class StructuralMismatchError(CompatibilityError):
    """The live init contract differs from the one the model was trained on."""

    def __init__(self, fields: list[str], detail: str):
        super().__init__(detail)
        self.fields = list(fields)


class RewardMismatchError(CompatibilityError):
    """The live reward differs from the saved one."""


class ScenarioMismatchError(CompatibilityError):
    """The live scenario identity differs from the saved one."""

    def __init__(self, differences: list[str], detail: str):
        super().__init__(detail)
        self.differences = list(differences)


@dataclass(frozen=True)
class CompatibilityReport:
    """Outcome of the ordered checks; `scenario` is "ok" or "overridden"."""

    warnings: list = field(default_factory=list)
    scenario: str = "ok"

    def describe(self) -> dict:
        return {"structural": "ok", "observation_schema": "ok", "reward": "ok",
                "scenario": self.scenario, "warnings": list(self.warnings),
                "note": TRANSFER_NOTE}


def _norm(value):
    return list(value) if isinstance(value, tuple) else value


def _difference(label: str, saved, live) -> str:
    return f"{label}: saved={saved!r} live={live!r}"


def _check_structural(saved: dict, live: dict) -> None:
    fields = [f for f in STRUCTURAL_FIELDS
              if _norm(saved.get(f)) != _norm(live.get(f))]
    if not fields:
        return
    detail = "; ".join(_difference(f, saved.get(f), live.get(f)) for f in fields)
    raise StructuralMismatchError(
        fields, f"Saved contract is incompatible with this run ({detail})")


def _check_reward(manifest: dict, saved_contract: dict, live_contract: dict,
                  live_reward: dict) -> None:
    saved = manifest.get("reward_schema") or {}
    live = live_reward or {}
    if saved.get("sha256") != live.get("sha256"):
        raise RewardMismatchError(
            "Saved reward differs from this run "
            f"({_difference('reward_schema.sha256', saved.get('sha256'), live.get('sha256'))})")
    authored_in_cpp = (saved.get("authority") == "cpp"
                       or "legacy" in list(saved.get("components") or []))
    if not authored_in_cpp:
        return
    fields = [f for f in ("reward_type", "reward_window")
              if saved_contract.get(f) != live_contract.get(f)]
    if fields:
        detail = "; ".join(
            _difference(f, saved_contract.get(f), live_contract.get(f)) for f in fields)
        raise RewardMismatchError(
            f"Saved reward depends on the simulator reward ({detail})")


def _scenario_differences(manifest: dict, saved_contract: dict, live_contract: dict,
                          live_identity: dict, live_band) -> list[str]:
    differences = []
    for label in SCENARIO_CONTRACT_FIELDS:
        if _norm(saved_contract.get(label)) != _norm(live_contract.get(label)):
            differences.append(
                _difference(label, saved_contract.get(label), live_contract.get(label)))
    saved_identity = manifest.get("scenario_identity") or {}
    for digest in IDENTITY_DIGESTS:
        if saved_identity.get(digest) != (live_identity or {}).get(digest):
            differences.append(_difference(f"scenario_identity.{digest}",
                                           saved_identity.get(digest),
                                           (live_identity or {}).get(digest)))
    if manifest.get("band") != live_band:
        differences.append(_difference("band", manifest.get("band"), live_band))
    return differences


def check_compatibility(manifest: dict, env, *, live_identity: dict, live_band,
                        allow_different_scenario: bool = False) -> CompatibilityReport:
    """Structural, then observation schema, then reward, then scenario identity."""
    saved_contract = manifest.get("contract") or {}
    live_contract = env.contract or {}
    _check_structural(saved_contract, live_contract)

    schema_warnings = check_schema(manifest.get("observation_schema") or {},
                                   env.observation_schema or {})
    _check_reward(manifest, saved_contract, live_contract, env.reward_schema)

    differences = list(schema_warnings) + _scenario_differences(
        manifest, saved_contract, live_contract, live_identity, live_band)
    if not differences:
        return CompatibilityReport([], "ok")
    if not allow_different_scenario:
        raise ScenarioMismatchError(
            differences,
            "Saved model was trained on a different scenario "
            f"({'; '.join(differences)}); pass --allow-different-scenario to proceed")
    return CompatibilityReport(differences, "overridden")
