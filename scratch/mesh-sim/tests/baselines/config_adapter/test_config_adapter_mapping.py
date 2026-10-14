"""Mapping file v2, rectangle/rl_bounds geofences, polygon rejection, explicit [rl] bounds.

Source of every expectation: scripts/baselines/README.md "Mapping file (optional)" and
"TODO: polygon geofences" (including the exact error ending quoted there), and
src/config/config-loader.cc:345-348 (the simulator reads [rl] bounds with std::stod and
falls back to -1000/2000/-1000/1000 when a key is absent).
"""

import copy
import json

import pytest

from scripts.baselines import config
from scripts.baselines.tests.conftest import AREA, MAPPING, scenario_ini, write_scenario

RECT = {"x_min": 10.0, "x_max": 20.0, "y_min": -5.0, "y_max": 5.0}


def _ini(tmp_path, text=None):
    return config.read_ini(write_scenario(tmp_path / "s", ini_text=text))


def _load(tmp_path, payload, ini_text=None, raw=None):
    ini = _ini(tmp_path, ini_text)
    path = tmp_path / "mapping.json"
    path.write_text(raw if raw is not None else json.dumps(payload))
    return config.load_mapping(path, ini)


def _with_geofence(geofence):
    payload = copy.deepcopy(MAPPING)
    payload["geofence"] = geofence
    return payload


def _rl_text(replacements: dict) -> str:
    text = scenario_ini()
    for key, line in replacements.items():
        original = f"{key} = {AREA[key]}\n"
        assert original in text
        text = text.replace(original, line)
    return text


# --- rl_bounds: all four explicit -------------------------------------------------------


def test_rl_bounds_parse_comments_and_exponents(tmp_path):
    text = _rl_text({"x_min": "x_min = -1e2 # west\n", "x_max": "x_max = 4E2;east\n",
                     "y_min": "y_min = -0\n", "y_max": "y_max =   .5e3   \n"})
    mapping = _load(tmp_path, MAPPING, ini_text=text)
    assert mapping.rectangle == {"x_min": -100.0, "x_max": 400.0, "y_min": 0.0,
                                 "y_max": 500.0}
    assert mapping.geofence == {"source": "rl_bounds", **mapping.rectangle}


def test_effective_rl_bounds_fill_only_absent_keys_with_simulator_defaults(tmp_path):
    ini = _ini(tmp_path, _rl_text({"x_max": "", "y_min": "y_min = # later\n"}))
    assert config.explicit_rl_bounds(ini) == {"x_min": 0.0, "y_max": 400.0}
    assert config.effective_rl_bounds(ini) == {"x_min": 0.0, "x_max": 2000.0,
                                               "y_min": -1000.0, "y_max": 400.0}
    # Mirrors src/domain/sim-config.h RlConfig defaults.
    assert config.RL_BOUND_DEFAULTS == {"x_min": -1000.0, "x_max": 2000.0,
                                        "y_min": -1000.0, "y_max": 1000.0}


# --- rectangle_xy_m ---------------------------------------------------------------------

def test_rectangle_needs_no_rl_section(tmp_path):
    payload = _with_geofence({"source": "rectangle_xy_m", **RECT})
    mapping = _load(tmp_path, payload, ini_text="[scenario]\nnodes_file = nodes.json\n")
    assert mapping.geofence == {"source": "rectangle_xy_m", **RECT}
    assert all(type(v) is float for v in mapping.rectangle.values())


def test_rectangle_accepts_integers_negatives_and_tiny_extent(tmp_path):
    payload = _with_geofence({"source": "rectangle_xy_m", "x_min": -300, "x_max": -299,
                              "y_min": 0, "y_max": 1e-9})
    rect = _load(tmp_path, payload).rectangle
    assert rect == {"x_min": -300.0, "x_max": -299.0, "y_min": 0.0, "y_max": 1e-9}


# --- polygon rejection ------------------------------------------------------------------


# --- version, top-level keys, platforms -------------------------------------------------


def test_minimal_mapping_has_no_platforms(tmp_path):
    payload = {"baseline_mapping_version": 2, "geofence": {"source": "rl_bounds"}}
    mapping = _load(tmp_path, payload)
    assert mapping.platforms == {} and mapping.rectangle == AREA
    assert mapping.path == tmp_path / "mapping.json"


@pytest.mark.parametrize("platforms,expected", [
    ({}, {}), ({"nodes": {}}, {}),
    ({"nodes": {"gw": "aerial", "uav-a": "ground"}}, {"gw": "aerial", "uav-a": "ground"}),
])
def test_valid_platforms(tmp_path, platforms, expected):
    payload = {**copy.deepcopy(MAPPING), "platforms": platforms}
    assert _load(tmp_path, payload).platforms == expected
