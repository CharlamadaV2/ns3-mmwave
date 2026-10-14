"""Precedence, source attribution, and validation of RlSelection (selection.py).

Spec: src/rl/policy-inputs.md "Selection keys and flags": each key resolves
independently CLI > run.ini > default, both manifests record value and source,
and unknown/mismatched/legacy-mode selections fail before the simulator starts.
"""


import pytest

from scripts.rl.env.selection import resolve_selection

KEYS = ("observation_preset", "reward_components", "reward_weights", "telemetry",
        "telemetry_every")

FULL_INI = """[scenario]
seed = 1

[rl]
controlled_nodes = node-b, node-c
observation_preset = local_links_v1
reward_components = delivery_ratio, connectivity
reward_weights = 1.0, 0.5
telemetry = steps
telemetry_every = 2
"""
CENTRAL_ONLY = "[rl]\ncontrolled_nodes = node-b\n"


def ini(tmp_path, text, name="run.ini") -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


# --- Precedence and source per key -------------------------------------------------

@pytest.mark.parametrize("key,cli_value,attr,expected", [
    ("observation_preset", "geometry_v1", "observation_preset", "geometry_v1"),
    ("reward_components", "throughput_mbps,legacy", "reward_components",
     ("throughput_mbps", "legacy")),
    ("reward_weights", "2, 3", "reward_weights", (2.0, 3.0)),
    ("telemetry", "steps", "telemetry", "steps"),
    ("telemetry_every", 3, "telemetry_every", 3),
])
def test_cli_overrides_one_key_and_only_that_source_changes(tmp_path, key, cli_value,
                                                            attr, expected):
    selection = resolve_selection(ini(tmp_path, FULL_INI), **{key: cli_value})
    assert getattr(selection, attr) == expected
    sources = selection.describe()["source"]
    assert sources == {k: ("cli" if k == key else "run.ini") for k in KEYS}


@pytest.mark.parametrize("line,key", [
    ("observation_preset = service_v1", "observation_preset"),
    ("reward_components = legacy", "reward_components"),
    ("telemetry = steps", "telemetry"),
    ("telemetry = none", "telemetry"),
])
def test_single_ini_key_is_attributed_to_run_ini(tmp_path, line, key):
    selection = resolve_selection(ini(tmp_path, CENTRAL_ONLY + line + "\n"))
    sources = selection.describe()["source"]
    assert sources == {k: ("run.ini" if k == key else "default") for k in KEYS}


def test_cli_weights_combine_with_ini_components(tmp_path):
    text = CENTRAL_ONLY + "reward_components = delivery_ratio, connectivity\n"
    selection = resolve_selection(ini(tmp_path, text), reward_weights="0.25,4")
    assert selection.reward_weights == (0.25, 4.0)
    sources = selection.describe()["source"]
    assert (sources["reward_components"], sources["reward_weights"]) == ("run.ini", "cli")


def test_cli_telemetry_every_with_ini_steps(tmp_path):
    selection = resolve_selection(ini(tmp_path, CENTRAL_ONLY + "telemetry = steps\n"),
                                  telemetry_every=4)
    assert (selection.telemetry, selection.telemetry_every) == ("steps", 4)


def test_default_weights_are_one_per_component(tmp_path):
    selection = resolve_selection(ini(tmp_path, CENTRAL_ONLY),
                                  reward_components="legacy, connectivity,throughput_mbps")
    assert selection.reward_weights == (1.0, 1.0, 1.0)
    assert selection.describe()["source"]["reward_weights"] == "default"


# --- run.ini parsing of selection keys ---------------------------------------------

@pytest.mark.parametrize("line", ["observation_preset =", "observation_preset =   ",
                                  "observation_preset = ; none",
                                  "observation_preset = # none"])
def test_blank_or_comment_only_ini_value_counts_as_absent(tmp_path, line):
    selection = resolve_selection(ini(tmp_path, CENTRAL_ONLY + line + "\n"))
    assert selection.observation_preset == "raw_links_v1"
    assert selection.describe()["source"]["observation_preset"] == "default"


def test_inline_comments_in_list_values(tmp_path):
    text = CENTRAL_ONLY + ("reward_components = delivery_ratio, connectivity ; pair\n"
                           "reward_weights = 1.0, 0.5 # weights\n"
                           "telemetry = steps;inline\n"
                           "telemetry_every = 3 # every third\n")
    selection = resolve_selection(ini(tmp_path, text))
    assert selection.reward_components == ("delivery_ratio", "connectivity")
    assert selection.reward_weights == (1.0, 0.5)
    assert (selection.telemetry, selection.telemetry_every) == ("steps", 3)


# --- Invalid values ----------------------------------------------------------------


# --- Legacy mode ---------------------------------------------------------------------


# --- Description and helpers --------------------------------------------------------
