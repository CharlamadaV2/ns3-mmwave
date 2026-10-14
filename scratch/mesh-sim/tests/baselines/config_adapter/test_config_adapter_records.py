"""nodes.json loading, node records, datum, ownership, evaluation region, identity, band.

Sources: scripts/baselines/README.md ("Mapping file" platform rules, "z is metres above
flat ground", "Hold executor contract", "Channel scoring" planning seed and run_id),
src/domain/node-spec.h (node_type defaults to "drone"; random_walk bounds default to
+-100), and src/config/rl-control.cc tokenize() (controlled_nodes tokens are trimmed and
empty tokens dropped).
"""


import pytest

from scripts.baselines import adapter, config
from scripts.baselines.config import ConfigError
from scripts.baselines.tests.conftest import AREA, NODES, scenario_ini, write_scenario
from scripts.sim_support import parse_seed_spec

ROSTER = [node["id"] for node in NODES]
CONTROLLED_LINE = "controlled_nodes = uav-a, uav-b, uav-c, walker"


def _records(tmp_path, nodes=None, mapping=None, controlled=None, **overrides):
    ini = write_scenario(tmp_path / "s", overrides=overrides or None, nodes=nodes,
                         mapping=mapping)
    cfg = config.load_baseline(ini)
    loaded = config.load_mapping(cfg.mapping_file, config.read_ini(ini))
    return adapter.build_records(adapter.load_nodes(ini.parent / "nodes.json"), cfg,
                                 loaded, controlled)


def _by_id(records):
    return {r.id: r for r in records}


def _node(node_id, **fields):
    return {"id": node_id, **fields}


# --- load_nodes -------------------------------------------------------------------------


# --- build_records: platforms, mobility, positions --------------------------------------

def test_absent_node_type_defaults_to_drone_and_aerial(tmp_path):
    nodes = [_node("a", position={"x": 1.0, "y": 2.0, "z": 3.0}),
             _node("b", node_type="vehicle")]
    by_id = _by_id(_records(tmp_path, nodes=nodes, movable_nodes="a"))
    assert (by_id["a"].node_type, by_id["a"].platform, by_id["a"].platform_source) == (
        "drone", "aerial", "node_type")
    assert by_id["a"].mobility == "fixed"
    # Absent position means (0, 0, 0), as start_position documents.
    assert (by_id["b"].x, by_id["b"].y, by_id["b"].z) == (0.0, 0.0, 0.0)


def test_partial_random_walk_bounds_use_simulator_defaults(tmp_path):
    nodes = [_node("a", mobility="random_walk", random_walk={"bounds": {"x_max": 400.0}}),
             _node("b", mobility="random_walk", random_walk={"speed_mps": 1.0}),
             _node("c", mobility="random_walk", random_walk={"bounds": None})]
    by_id = _by_id(_records(tmp_path, nodes=nodes, movable_nodes="a"))
    assert by_id["a"].random_walk_bounds == {"x_min": -100.0, "x_max": 400.0,
                                             "y_min": -100.0, "y_max": 100.0}
    assert by_id["b"].random_walk_bounds == adapter.RANDOM_WALK_DEFAULTS
    assert by_id["c"].random_walk_bounds == adapter.RANDOM_WALK_DEFAULTS


# --- datum ------------------------------------------------------------------------------

def test_datum_accepts_ground_level(tmp_path):
    nodes = [_node("a", position={"x": 0, "y": 0, "z": 0}), _node("b", position={"z": 0.0})]
    adapter.check_datum(_records(tmp_path, nodes=nodes, movable_nodes="a"))


def test_datum_uses_the_first_waypoint_for_waypoint_nodes(tmp_path):
    ok = [_node("a", mobility="waypoint", position={"z": -5.0},
                waypoints=[{"t": 0, "x": 1, "y": 1, "z": 2.0}])]
    adapter.check_datum(_records(tmp_path / "ok", nodes=ok, movable_nodes="a"))
    bad = [_node("a", mobility="waypoint", position={"z": 5.0},
                 waypoints=[{"t": 0, "x": 1, "y": 1, "z": -2.0}])]
    with pytest.raises(ConfigError, match="below the ground datum"):
        adapter.check_datum(_records(tmp_path / "bad", nodes=bad, movable_nodes="a"))


# --- [rl] controlled_nodes tokens and ownership -----------------------------------------

def _controlled(tmp_path, line):
    text = scenario_ini().replace(CONTROLLED_LINE, line)
    return config.rl_controlled_nodes(config.read_ini(write_scenario(tmp_path,
                                                                     ini_text=text)))


@pytest.mark.parametrize("line,expected", [
    ("controlled_nodes =   uav-c ,uav-a  # two", ("uav-c", "uav-a")),
    ("controlled_nodes = uav-a,,uav-c,", ("uav-a", "uav-c")),
    ("controlled_nodes = ,uav-a", ("uav-a",)),
    ("controlled_nodes = all ; every node", ("all",)),
    ("controlled_nodes =", ()),
    ("controlled_nodes = # none", ()),
])
def test_controlled_tokens_match_cpp_tokenize(tmp_path, line, expected):
    assert _controlled(tmp_path, line) == expected


def test_ownership_through_ini_with_whitespace_and_empty_tokens(tmp_path):
    controlled = _controlled(tmp_path / "c", "controlled_nodes =  uav-c ,, uav-a , ")
    cfg = config.load_baseline(write_scenario(tmp_path / "m", overrides={
        "movable_nodes": "  uav-a,uav-c  # same set"}))
    assert adapter.check_ownership(cfg, controlled, ROSTER) == ["uav-c", "uav-a"]


def test_ownership_all_against_single_node_roster(tmp_path):
    cfg = config.load_baseline(write_scenario(tmp_path, overrides={"movable_nodes": "all"}))
    assert adapter.check_ownership(cfg, ("solo",), ["solo"]) == ["solo"]
    assert adapter.check_ownership(cfg, ("all",), ["solo"]) == ["solo"]


# --- evaluation region ------------------------------------------------------------------

@pytest.mark.parametrize("rect,expected", [
    (dict(AREA), dict(AREA)),
    ({"x_min": -10.0, "x_max": 410.0, "y_min": -10.0, "y_max": 410.0}, dict(AREA)),
    ({"x_min": 399.0, "x_max": 500.0, "y_min": 399.0, "y_max": 500.0},
     {"x_min": 399.0, "x_max": 400.0, "y_min": 399.0, "y_max": 400.0}),
])
def test_evaluation_region_intersections(rect, expected):
    assert adapter.evaluation_region(rect, dict(AREA)) == expected


# --- planning identity ------------------------------------------------------------------

def _scenario_ini(tmp_path, seed_line="seed = 1\n", run_id_line=""):
    text = scenario_ini().replace("seed = 1\n", seed_line + run_id_line)
    return config.read_ini(write_scenario(tmp_path, ini_text=text))


@pytest.mark.parametrize("spec,expected", [("5-7", 5), ("7,5-6", 7), ("0", 0),
                                           (" 3 , 1", 3)])
def test_planning_seed_is_the_first_expanded_simulation_seed(tmp_path, spec, expected):
    ini = _scenario_ini(tmp_path)
    assert adapter.planning_identity(ini, parse_seed_spec(spec)) == (expected, 1)


@pytest.mark.parametrize("line,expected", [("seed = 9 # c\n", 9), ("seed = 0\n", 0),
                                           ("seed =\n", 42), ("", 42),
                                           ("seed = ; later\n", 42), ("seed = 007\n", 7)])
def test_planning_seed_falls_back_to_scenario_seed(tmp_path, line, expected):
    assert adapter.planning_identity(_scenario_ini(tmp_path, line), None)[0] == expected
    assert adapter.planning_identity(_scenario_ini(tmp_path, line), [])[0] == expected


@pytest.mark.parametrize("line,expected", [("run_id = 0\n", 0), ("run_id = 12 # r\n", 12),
                                           ("run_id =\n", 1), ("", 1)])
def test_run_id_from_scenario(tmp_path, line, expected):
    ini = _scenario_ini(tmp_path, run_id_line=line)
    assert adapter.planning_identity(ini, [3]) == (3, expected)


# --- band -------------------------------------------------------------------------------

@pytest.mark.parametrize("extra,cli,expected", [
    ("\n[channel]\nband =   sub-6   # jammers\n", None, ("sub-6", "run.ini")),
    ("\n[channel]\nband =\n", None, ("mmwave", "default")),
    ("\n[channel]\nband = ; none\n", None, ("mmwave", "default")),
    ("\n[channel]\nfrequency_ghz = 28\n", None, ("mmwave", "default")),
    ("\n[channel]\nband = sub-6\n", "", ("sub-6", "run.ini")),
    ("", "", ("mmwave", "default")),
])
def test_band_resolution_edges(tmp_path, extra, cli, expected):
    ini = config.read_ini(write_scenario(tmp_path, ini_text=scenario_ini(extra=extra)))
    assert adapter.resolve_band(ini, cli) == expected


# --- penalties --------------------------------------------------------------------------
