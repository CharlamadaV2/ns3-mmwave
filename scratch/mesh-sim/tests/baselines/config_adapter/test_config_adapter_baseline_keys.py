"""Boundary values for every [baseline] key, choices, movable_nodes, and required keys.

Source of every expectation: the key table in scripts/baselines/README.md ("The
[baseline] section") and src/config/run-ini-reference.md: costs >= 0, displacement caps
> 0 (zero invalid), grid cells integer >= 1, grid_min_resolution_m > 0, probe height
>= 0, probe gain finite >= 0, coverage_sinr_db finite, balanced_core_fraction in [0, 1],
seed integer >= 0, max_iterations integer >= 1; unknown values are rejected; a blank
value means the key is absent.
"""


import pytest

from scripts.baselines import config
from scripts.baselines.tests.conftest import write_scenario

INT_KEYS = ["candidate_grid_cells", "coverage_grid_cells"]


def _load(tmp_path, **overrides):
    return config.load_baseline(write_scenario(tmp_path, overrides=overrides))


# --- strings that are never a finite number, for every float key -----------------------


# --- integer keys -----------------------------------------------------------------------

@pytest.mark.parametrize("key", INT_KEYS)
@pytest.mark.parametrize("value,expected", [("1", 1), ("01", 1), (" 7 ", 7),
                                            ("400 # cells", 400), ("12;x", 12),
                                            ("1000000", 1000000)])
def test_grid_cells_accept_plain_integers(tmp_path, key, value, expected):
    got = getattr(_load(tmp_path, **{key: value}), key)
    assert type(got) is int and got == expected


@pytest.mark.parametrize("value,expected", [("0", 0), ("00", 0), ("007", 7),
                                            ("4294967296", 4294967296), (" 3 # s", 3)])
def test_seed_accepts_non_negative_integers(tmp_path, value, expected):
    assert _load(tmp_path, seed=value).seed == expected


@pytest.mark.parametrize("value,expected", [("1", 1), ("01", 1), ("999999", 999999)])
def test_max_iterations_accepts_positive_integers(tmp_path, value, expected):
    assert _load(tmp_path, max_iterations=value).max_iterations == expected


# --- choice keys ------------------------------------------------------------------------


@pytest.mark.parametrize("key,value,expected", [
    ("algorithm", "  optimization  ", "optimization"),
    ("objective", "resilience;x", "resilience"),
    ("application", "initial_positions # only one", "initial_positions"),
    ("waypoint_policy", "translate\t; shift path", "translate"),
])
def test_choice_values_trim_and_strip_comments(tmp_path, key, value, expected):
    assert getattr(_load(tmp_path, **{key: value}), key) == expected


# --- removed and unknown keys -----------------------------------------------------------


# --- movable_nodes ----------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("uav-a,uav-c", ("uav-a", "uav-c")),
    ("  uav-a  ,\tuav-c  ", ("uav-a", "uav-c")),
    ("uav-c, uav-a", ("uav-c", "uav-a")),
    ("walker", ("walker",)),
    ("uav a, b", ("uav a", "b")),
    ("ALL", ("ALL",)),
    ("All", ("All",)),
])
def test_movable_lists(tmp_path, value, expected):
    cfg = _load(tmp_path, movable_nodes=value)
    assert cfg.movable_nodes == expected
    assert not cfg.movable_all  # only lower-case `all` means the whole roster


# --- require_method ---------------------------------------------------------------------


def test_geometric_needs_neither_seed_nor_max_iterations(tmp_path):
    cfg = config.load_baseline(write_scenario(tmp_path, drop=("seed", "max_iterations")))
    config.require_method(cfg, "geometric")
    assert (cfg.seed, cfg.max_iterations) == (None, None)
