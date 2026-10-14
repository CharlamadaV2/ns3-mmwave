"""Edge cases of scripts/stats.py: sample_stats sizes, t-table df rounding, NaN input.

Contract (scripts/stats.py Doxygen, scripts/CLAUDE.md): t_critical_95(n) uses
df = n - 1, returns 0.0 when n <= 1, and otherwise the value of the largest
tabulated df not above n - 1 (beyond 120 -> 1.980), so intervals are never too
narrow. sample_stats returns None stats and n=0 for an empty list, std=0.0 and
no ci95 for one value, and ci95 = t * std / sqrt(n) when use_ci and n > 1.
"""

import math

import pytest

from scripts import stats

# Two-sided 95% Student-t quantiles (t_{0.975, df}) from standard tables, to 4 dp.
_REFERENCE_T = {1: 12.7062, 2: 4.3027, 5: 2.5706, 10: 2.2281, 30: 2.0423,
                31: 2.0395, 35: 2.0301, 39: 2.0227, 40: 2.0211, 45: 2.0141,
                59: 2.0010, 60: 2.0003, 61: 1.9996, 100: 1.9840, 119: 1.9801,
                120: 1.9799, 121: 1.9798, 200: 1.9719, 1000: 1.9623}


# t_critical_95 ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 0, -1, -100])
def test_t_critical_is_zero_without_degrees_of_freedom(n):
    assert stats.t_critical_95(n) == 0.0


@pytest.mark.parametrize("n, expected", [
    (31, 2.042),   # df 30, last dense row
    (32, 2.042),   # df 31 -> rounds down to 30
    (40, 2.042),   # df 39 -> 30
    (41, 2.021),   # df 40 exact
    (60, 2.021),   # df 59 -> 40
    (61, 2.000),   # df 60 exact
    (120, 2.000),  # df 119 -> 60
    (121, 1.980),  # df 120 exact
    (122, 1.980),  # df 121 -> 120
    (10 ** 6, 1.980),
])
def test_t_critical_rounds_df_down_between_table_rows(n, expected):
    assert stats.t_critical_95(n) == expected


@pytest.mark.parametrize("df", sorted(_REFERENCE_T))
def test_t_critical_is_never_narrower_than_the_true_quantile(df):
    # Table entries are rounded to 3 dp, so allow that rounding (5e-4) only.
    assert stats.t_critical_95(df + 1) >= _REFERENCE_T[df] - 5e-4


# sample_stats ----------------------------------------------------------------------

def test_sample_stats_of_an_empty_list_is_all_none():
    for use_ci in (False, True):
        assert stats.sample_stats([], use_ci) == {"mean": None, "std": None, "min": None,
                                                  "max": None, "n": 0}


def test_sample_stats_of_one_value_has_zero_std_and_no_ci():
    result = stats.sample_stats([3.5], use_ci=True)
    assert result == {"mean": 3.5, "std": 0.0, "min": 3.5, "max": 3.5, "n": 1}
    assert "ci95" not in result


def test_sample_stats_of_two_values_uses_df_one():
    result = stats.sample_stats([0.0, 2.0], use_ci=True)
    assert result["mean"] == pytest.approx(1.0)
    assert result["std"] == pytest.approx(math.sqrt(2.0))
    # ci95 = 12.706 * sqrt(2) / sqrt(2)
    assert result["ci95"] == pytest.approx(12.706)


def test_sample_stats_of_identical_values_has_a_zero_width_interval():
    result = stats.sample_stats([0.7] * 5, use_ci=True)
    assert result["std"] == 0.0
    assert result["ci95"] == 0.0
    assert result["min"] == result["max"] == pytest.approx(0.7)


def test_sample_stats_omits_ci_unless_requested():
    assert "ci95" not in stats.sample_stats([1.0, 2.0, 3.0])
    assert "ci95" not in stats.sample_stats([1.0, 2.0, 3.0], use_ci=False)


@pytest.mark.parametrize("n", [2, 3, 31, 32, 41, 121, 130])
def test_sample_stats_ci_matches_the_shared_t_value(n):
    values = [float(i % 7) - 2.5 for i in range(n)]
    result = stats.sample_stats(values, use_ci=True)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    assert result["n"] == n
    assert result["std"] == pytest.approx(std)
    assert result["ci95"] == pytest.approx(stats.t_critical_95(n) * std / math.sqrt(n))


def test_sample_stats_accepts_ints_and_negative_values_and_leaves_input_alone():
    values = [-3, 1, 5]
    result = stats.sample_stats(values, use_ci=True)
    assert values == [-3, 1, 5]
    assert (result["mean"], result["min"], result["max"]) == (1.0, -3, 5)
    assert result["std"] == pytest.approx(4.0)


def test_sample_stats_is_order_independent_for_finite_values():
    forward = stats.sample_stats([0.1, 0.9, 0.4, 0.4], use_ci=True)
    backward = stats.sample_stats([0.4, 0.4, 0.9, 0.1], use_ci=True)
    assert forward == pytest.approx(backward)
