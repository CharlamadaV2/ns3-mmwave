"""Shared sample statistics and t critical values for 95% confidence intervals."""

import math
from typing import Any

T_TABLE_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
              19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
              25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
              40: 2.021, 60: 2.000, 120: 1.980}


## @fn t_critical_95
# @brief Look up the two-tailed 95% Student t critical value.
#
# @param n  Number of observations (degrees of freedom = n - 1).
# @return The table value for the largest tabulated df not above n - 1 (df beyond 120 uses 1.980); 0.0 if n <= 1.
def t_critical_95(n: int) -> float:
    """Return the two-tailed t critical value for a 95% CI with *n* observations."""
    df = n - 1
    if df <= 0:
        return 0.0
    # largest tabulated df <= df keeps intervals from being too narrow
    return T_TABLE_95[max(k for k in T_TABLE_95 if k <= df)]


## @fn sample_stats
# @brief Summarise a list of numbers.
#
# @param values  Observations (any length).
# @param use_ci  If True and n > 1, add a `ci95` half-width.
# @return Dict with `mean`, `std` (sample, n-1), `min`, `max`, `n`; optionally `ci95`
#         (`t * std / sqrt(n)`). For an empty list the stats are None and `n` is 0.
#
# With one value, `std` is 0.0 and no `ci95` is added.
def sample_stats(values: list[float], use_ci: bool = False) -> dict[str, Any]:
    """Compute mean, sample std, min, max, n, and optionally 95% CI."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)

    result: dict[str, Any] = {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
    }
    if use_ci and n > 1:
        result["ci95"] = t_critical_95(n) * std / math.sqrt(n)
    return result
