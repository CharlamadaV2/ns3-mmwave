"""Strict [baseline] and mapping-file parsing."""

import copy
import json

import pytest

from scripts.baselines import config
from scripts.baselines.config import ConfigError
from scripts.baselines.tests.conftest import (AREA, MAPPING, SCENARIO_SECTIONS,
                                              baseline_section, scenario_ini, write_scenario)


def _ini(tmp_path, text):
    path = tmp_path / "run.ini"
    path.write_text(text)
    return path


def test_absent_section_means_none(tmp_path):
    cfg = config.load_baseline(_ini(tmp_path, SCENARIO_SECTIONS))
    assert cfg.algorithm == "none"
    assert cfg.present_keys == frozenset()


@pytest.mark.parametrize("line", ["algorithm = none", "algorithm =", "algorithm = # blank"])
def test_explicit_or_blank_none(tmp_path, line):
    cfg = config.load_baseline(_ini(tmp_path, f"[baseline]\n{line}\n"))
    assert cfg.algorithm == "none"
    config.require_method(cfg, "none")


@pytest.mark.parametrize("method", config.PLACEMENT_METHODS)
@pytest.mark.parametrize("objective", config.OBJECTIVES)
def test_methods_and_objectives(tmp_path, method, objective):
    ini = write_scenario(tmp_path, overrides={"algorithm": method, "objective": objective})
    cfg = config.load_baseline(ini)
    config.require_method(cfg, method)
    assert (cfg.algorithm, cfg.objective) == (method, objective)
    assert cfg.movable_nodes == ("uav-a", "uav-c")
    assert (cfg.seed, cfg.max_iterations) == (7, 60)
    assert cfg.application == "initial_positions"
    assert cfg.waypoint_policy == "reject"


def test_unknown_key(tmp_path):
    ini = write_scenario(tmp_path, overrides={"algoritm": "geometric"})
    with pytest.raises(ConfigError, match="unknown \\[baseline\\] key.*algoritm"):
        config.load_baseline(ini)


def test_keys_are_case_sensitive_like_the_simulator(tmp_path):
    with pytest.raises(ConfigError, match="Algorithm"):
        config.load_baseline(_ini(tmp_path, "[baseline]\nAlgorithm = geometric\n"))


@pytest.mark.parametrize("header", ("[ baseline ]", "[baseline ]", "[ baseline]"))
def test_padded_baseline_header_is_rejected(tmp_path, header):
    with pytest.raises(ConfigError, match="surrounding whitespace; use '\\[baseline\\]'"):
        config.load_baseline(_ini(tmp_path, f"{header}\nalgorithm = geometric\n"))


@pytest.mark.parametrize("key,value", [
    ("algorithm", "rl"), ("objective", "speed"), ("application", "periodic"),
    ("waypoint_policy", "clip"), ("seed", "-1"), ("seed", "1.5"), ("seed", "abc"),
    ("max_iterations", "0"), ("max_iterations", "ten"),
])
def test_unknown_or_invalid_value(tmp_path, key, value):
    ini = write_scenario(tmp_path, overrides={key: value})
    with pytest.raises(ConfigError, match=f"baseline.{key}"):
        config.load_baseline(ini)


def test_duplicate_section_rejected(tmp_path):
    text = scenario_ini() + "\n[baseline]\nobjective = balanced\n"
    with pytest.raises(ConfigError, match="already exists"):
        config.load_baseline(_ini(tmp_path, text))


def test_duplicate_key_rejected(tmp_path):
    text = scenario_ini(extra="").replace("objective = coverage",
                                          "objective = coverage\nobjective = balanced")
    with pytest.raises(ConfigError, match="already exists"):
        config.load_baseline(_ini(tmp_path, text))


def test_duplicate_key_elsewhere_rejected(tmp_path):
    text = scenario_ini().replace("tick_s = 0.1", "tick_s = 0.1\ntick_s = 0.2")
    with pytest.raises(ConfigError):
        config.load_baseline(_ini(tmp_path, text))


def test_colon_assignment_rejected(tmp_path):
    text = scenario_ini().replace("objective = coverage", "objective: coverage")
    with pytest.raises(ConfigError):
        config.load_baseline(_ini(tmp_path, text))


def test_continuation_line_rejected(tmp_path):
    text = scenario_ini().replace("objective = coverage", "objective = coverage\n  balanced")
    with pytest.raises(ConfigError, match="indented"):
        config.load_baseline(_ini(tmp_path, text))


def test_default_section_is_ordinary(tmp_path):
    text = "[DEFAULT]\nfoo = 1\n\n" + baseline_section()
    cfg = config.load_baseline(_ini(tmp_path, text))
    assert cfg.algorithm == "geometric"


@pytest.mark.parametrize("value", ["uav-a,,uav-c", "uav-a, uav-a", " , ", "all, uav-a",
                                   "uav-a, all"])
def test_bad_movable_ids(tmp_path, value):
    ini = write_scenario(tmp_path, overrides={"movable_nodes": value})
    with pytest.raises(ConfigError, match="movable_nodes"):
        config.load_baseline(ini)


@pytest.mark.parametrize("value", ["all", " all ", "all # every node"])
def test_movable_all(tmp_path, value):
    cfg = config.load_baseline(write_scenario(tmp_path, overrides={"movable_nodes": value}))
    assert cfg.movable_nodes == ("all",) and cfg.movable_all


@pytest.mark.parametrize("key,hint", [
    ("gateway_node_id", "removed: baselines are gateway-free"),
    ("rf_config", "removed: candidates are scored by the simulator channel"),
])
def test_removed_keys_name_a_migration_hint(tmp_path, key, hint):
    ini = write_scenario(tmp_path, overrides={key: "gw"})
    with pytest.raises(ConfigError) as err:
        config.load_baseline(ini)
    assert f"unknown [baseline] key(s): {key} ({hint}" in str(err.value)


def test_traffic_gateway_key_is_unrelated(tmp_path):
    ini = write_scenario(tmp_path, ini_text=scenario_ini(extra="\n[traffic]\n"
                                                         "gateway_node_id = gw\n"))
    assert config.load_baseline(ini).algorithm == "geometric"


def test_numeric_defaults(tmp_path):
    cfg = config.load_baseline(write_scenario(tmp_path))
    assert (cfg.aerial_fixed_cost_m2, cfg.ground_fixed_cost_m2) == (150000.0, 150000.0)
    assert (cfg.aerial_cost_m2_per_m, cfg.ground_cost_m2_per_m) == (500.0, 100.0)
    assert (cfg.aerial_max_displacement_m, cfg.ground_max_displacement_m) == (None, None)
    assert (cfg.candidate_grid_cells, cfg.coverage_grid_cells) == (400, 400)
    assert cfg.grid_min_resolution_m == 5.0
    assert (cfg.coverage_probe_height_m, cfg.coverage_probe_rx_gain_dbi) == (1.5, None)
    assert (cfg.coverage_sinr_db, cfg.balanced_core_fraction) == (-6.7, 0.5)


@pytest.mark.parametrize("key,value,expected", [
    ("aerial_fixed_cost_m2", "0", 0.0), ("ground_fixed_cost_m2", "1.5e5", 150000.0),
    ("aerial_cost_m2_per_m", "0.0", 0.0), ("ground_cost_m2_per_m", "250", 250.0),
    ("aerial_max_displacement_m", "0.5", 0.5), ("ground_max_displacement_m", "300", 300.0),
    ("candidate_grid_cells", "1", 1), ("coverage_grid_cells", "900", 900),
    ("grid_min_resolution_m", "0.25", 0.25), ("coverage_probe_height_m", "0", 0.0),
    ("coverage_probe_rx_gain_dbi", "0", 0.0), ("coverage_probe_rx_gain_dbi", "7.5", 7.5),
    ("coverage_sinr_db", "-10", -10.0), ("coverage_sinr_db", "+3.5", 3.5),
    ("balanced_core_fraction", "0", 0.0), ("balanced_core_fraction", "1", 1.0),
])
def test_numeric_key_bounds_accept(tmp_path, key, value, expected):
    cfg = config.load_baseline(write_scenario(tmp_path, overrides={key: value}))
    assert getattr(cfg, key) == expected
    assert type(getattr(cfg, key)) is type(expected)


@pytest.mark.parametrize("key,value", [
    ("aerial_fixed_cost_m2", "-1"), ("ground_fixed_cost_m2", "inf"),
    ("aerial_cost_m2_per_m", "-0.5"), ("ground_cost_m2_per_m", "nan"),
    ("aerial_max_displacement_m", "0"), ("ground_max_displacement_m", "-5"),
    ("candidate_grid_cells", "0"), ("coverage_grid_cells", "2.5"),
    ("candidate_grid_cells", "-3"), ("grid_min_resolution_m", "0"),
    ("coverage_probe_height_m", "-0.1"), ("coverage_probe_rx_gain_dbi", "-1"),
    ("coverage_probe_rx_gain_dbi", "inf"), ("coverage_sinr_db", "nan"),
    ("coverage_sinr_db", "1_0"), ("balanced_core_fraction", "1.01"),
    ("balanced_core_fraction", "-0.1"), ("aerial_fixed_cost_m2", "lots"),
])
def test_numeric_key_bounds_reject(tmp_path, key, value):
    ini = write_scenario(tmp_path, overrides={key: value})
    with pytest.raises(ConfigError, match=f"baseline.{key}"):
        config.load_baseline(ini)


def test_zero_cap_is_rejected_with_hint(tmp_path):
    ini = write_scenario(tmp_path, overrides={"aerial_max_displacement_m": "0"})
    with pytest.raises(ConfigError, match="omit the key for no cap"):
        config.load_baseline(ini)


@pytest.mark.parametrize("missing", ["seed", "max_iterations"])
def test_optimization_requires_seed_and_iterations(tmp_path, missing):
    ini = write_scenario(tmp_path, overrides={"algorithm": "optimization"}, drop=(missing,))
    cfg = config.load_baseline(ini)
    with pytest.raises(ConfigError, match=f"requires \\[baseline\\] {missing}"):
        config.require_method(cfg, "optimization")
    config.require_method(cfg, "geometric")


@pytest.mark.parametrize("missing", ["objective", "movable_nodes"])
def test_active_method_required_keys(tmp_path, missing):
    cfg = config.load_baseline(write_scenario(tmp_path, drop=(missing,)))
    with pytest.raises(ConfigError, match=missing):
        config.require_method(cfg, "geometric")


def test_mapping_file_is_optional(tmp_path):
    cfg = config.load_baseline(write_scenario(tmp_path, drop=("mapping_file",)))
    config.require_method(cfg, "geometric")
    assert cfg.mapping_file is None


def test_required_keys_follow_effective_method(tmp_path):
    text = SCENARIO_SECTIONS + "\n[baseline]\nalgorithm = none\n"
    cfg = config.load_baseline(_ini(tmp_path, text))
    config.require_method(cfg, "none")
    with pytest.raises(ConfigError, match="objective"):
        config.require_method(cfg, "geometric")


def test_relative_paths_resolve_against_ini(tmp_path):
    ini = write_scenario(tmp_path / "scenario",
                         overrides={"mapping_file": "../shared/map.json"})
    cfg = config.load_baseline(ini)
    assert cfg.mapping_file == ini.resolve().parent / "../shared/map.json"


def test_path_with_spaces_and_absolute_path(tmp_path):
    ini = write_scenario(tmp_path / "scenario",
                         overrides={"mapping_file": "my maps/mapping file.json"})
    cfg = config.load_baseline(ini)
    assert cfg.mapping_file.name == "mapping file.json"
    assert cfg.mapping_file.parent.name == "my maps"
    absolute = tmp_path / "abs dir" / "map.json"
    cfg = config.load_baseline(write_scenario(tmp_path / "other",
                                              overrides={"mapping_file": str(absolute)}))
    assert cfg.mapping_file == absolute


def test_inline_comments_stripped(tmp_path):
    ini = write_scenario(tmp_path, overrides={"algorithm": "optimization  # chosen",
                                              "objective": "balanced ; note",
                                              "movable_nodes": "uav-a, uav-c # two"})
    cfg = config.load_baseline(ini)
    assert (cfg.algorithm, cfg.objective) == ("optimization", "balanced")
    assert cfg.movable_nodes == ("uav-a", "uav-c")


def test_rl_bounds_and_controlled_nodes(tmp_path):
    ini = config.read_ini(write_scenario(tmp_path))
    assert config.explicit_rl_bounds(ini) == {"x_min": 0.0, "x_max": 400.0,
                                              "y_min": 0.0, "y_max": 400.0}
    assert config.rl_controlled_nodes(ini) == ("uav-a", "uav-b", "uav-c", "walker")
    partial = config.read_ini(_ini(tmp_path, "[rl]\nx_min = 5\n"))
    assert config.explicit_rl_bounds(partial) == {"x_min": 5.0}
    assert config.effective_rl_bounds(partial) == {**config.RL_BOUND_DEFAULTS, "x_min": 5.0}
    assert config.rl_controlled_nodes(partial) is None


def _mapping(tmp_path, payload, ini_text=None):
    ini = config.read_ini(write_scenario(tmp_path / "s", ini_text=ini_text))
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(payload))
    return path, ini


def test_mapping_valid(tmp_path):
    mapping = config.load_mapping(*_mapping(tmp_path, MAPPING))
    assert mapping.geofence == {"source": "rl_bounds", "x_min": 0.0, "x_max": 400.0,
                                "y_min": 0.0, "y_max": 400.0}
    assert mapping.platforms == {}


def test_mapping_platforms(tmp_path):
    payload = copy.deepcopy(MAPPING)
    payload["platforms"] = {"nodes": {"walker": "aerial", "uav-a": "ground"}}
    mapping = config.load_mapping(*_mapping(tmp_path, payload))
    assert mapping.platforms == {"walker": "aerial", "uav-a": "ground"}
    del payload["platforms"]
    assert config.load_mapping(*_mapping(tmp_path / "b", payload)).platforms == {}


def test_absent_mapping_defaults_to_rl_bounds(tmp_path):
    ini = config.read_ini(write_scenario(tmp_path))
    mapping = config.load_mapping(None, ini)
    assert mapping.path is None and mapping.platforms == {}
    assert mapping.geofence == {"source": "rl_bounds", **AREA}
    partial = config.read_ini(write_scenario(
        tmp_path / "p", ini_text=scenario_ini().replace("y_min = 0.0\n", "")))
    with pytest.raises(ConfigError, match="y_min.*loader defaults are not used"):
        config.load_mapping(None, partial)


def test_mapping_v1_gets_a_migration_message(tmp_path):
    payload = {"baseline_mapping_version": 1,
               "origin": {"lat": 10.0, "lon": 20.0, "synthetic": True},
               "ground_datum": "z_is_agl_m", "geofence": {"source": "rl_bounds"},
               "radios": {"default": ["meshradio"]}}
    with pytest.raises(ConfigError) as err:
        config.load_mapping(*_mapping(tmp_path, payload))
    message = str(err.value)
    assert "baseline_mapping_version 1 is no longer read; migrate to version 2" in message
    assert "delete origin, ground_datum and radios" in message


@pytest.mark.parametrize("key,value", [
    ("origin", {"lat": 0.0, "lon": 0.0, "synthetic": True}),
    ("ground_datum", "z_is_agl_m"),
    ("radios", {"default": ["meshradio"]}),
])
def test_mapping_v2_rejects_retired_keys(tmp_path, key, value):
    payload = {**copy.deepcopy(MAPPING), key: value}
    with pytest.raises(ConfigError, match=f"{key} removed in mapping version 2"):
        config.load_mapping(*_mapping(tmp_path, payload))


def test_mapping_rectangle(tmp_path):
    payload = copy.deepcopy(MAPPING)
    payload["geofence"] = {"source": "rectangle_xy_m", "x_min": 10, "x_max": 20,
                           "y_min": -5, "y_max": 5}
    mapping = config.load_mapping(*_mapping(tmp_path, payload))
    assert mapping.rectangle == {"x_min": 10.0, "x_max": 20.0, "y_min": -5.0, "y_max": 5.0}


def test_rl_bounds_requires_explicit_keys(tmp_path):
    text = scenario_ini().replace("y_max = 400.0\n", "")
    with pytest.raises(ConfigError, match="y_max.*loader defaults are not used"):
        config.load_mapping(*_mapping(tmp_path, MAPPING, ini_text=text))


@pytest.mark.parametrize("geofence", [
    {"source": "polygon_xy_m", "vertices": [[0, 0], [1, 0], [0, 1]]},
    {"source": "rectangle_xy_m", "x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1,
     "holes": [[[0.2, 0.2], [0.3, 0.2], [0.2, 0.3]]]},
    {"source": "geojson", "coordinates": [[[0, 0], [1, 0], [0, 1], [0, 0]]]},
    {"source": "polygon"},
    {"points": [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}, {"lat": 1, "lon": 0}]},
    [{"lat": 0, "lon": 0}, {"lat": 0, "lon": 1}, {"lat": 1, "lon": 0}],
])
def test_polygon_geofence_rejected_with_rectangle_limit(tmp_path, geofence):
    payload = copy.deepcopy(MAPPING)
    payload["geofence"] = geofence
    with pytest.raises(ConfigError) as err:
        config.load_mapping(*_mapping(tmp_path, payload))
    message = str(err.value)
    assert "only axis-aligned rectangle" in message
    assert "polygon geofences are a documented TODO" in message
    assert "bounding rectangle is never substituted" in message


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(baseline_mapping_version=3), "baseline_mapping_version must be 2"),
    (lambda m: m.update(baseline_mapping_version=True), "baseline_mapping_version must be 2"),
    (lambda m: m.pop("baseline_mapping_version"), "baseline_mapping_version must be 2"),
    (lambda m: m.update(extra=1), "unknown key"),
    (lambda m: m.pop("geofence"), "missing key"),
    (lambda m: m.update(geofence={"source": "circle"}), "geofence.source"),
    (lambda m: m.update(geofence={"source": "rectangle_xy_m", "x_min": 5, "x_max": 5,
                                  "y_min": 0, "y_max": 1}), "x_min < x_max"),
    (lambda m: m.update(geofence={"source": "rectangle_xy_m", "x_min": 0}), "missing key"),
    (lambda m: m.update(geofence={"source": "rl_bounds", "x_min": 0}), "unknown key"),
    (lambda m: m.update(platforms={"nodes": {"gw": "sea"}}), "platforms.nodes.gw"),
    (lambda m: m.update(platforms={"nodes": ["gw"]}), "platforms.nodes must be an object"),
    (lambda m: m.update(platforms={"default": "aerial"}), "unknown key"),
])
def test_mapping_invalid(tmp_path, mutate, match):
    payload = copy.deepcopy(MAPPING)
    mutate(payload)
    with pytest.raises(ConfigError, match=match):
        config.load_mapping(*_mapping(tmp_path, payload))


def test_mapping_missing_or_malformed(tmp_path):
    ini = config.read_ini(write_scenario(tmp_path / "s"))
    with pytest.raises(ConfigError, match="not found"):
        config.load_mapping(tmp_path / "absent.json", ini)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        config.load_mapping(bad, ini)
    listing = tmp_path / "list.json"
    listing.write_text("[]")
    with pytest.raises(ConfigError, match="must be a JSON object"):
        config.load_mapping(listing, ini)
