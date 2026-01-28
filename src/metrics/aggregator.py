"""
Aggregation of metrics across multiple simulation runs.
"""

from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def aggregate_simulation_metrics(
    runs_metrics: List[Dict[str, Any]],
) -> Dict[str, List[Any]]:
    """
    Aggregate metrics from multiple simulation runs.

    Args:
        runs_metrics: List of metric dictionaries from individual runs

    Returns:
        Dictionary where each key maps to a list of values across runs
    """
    aggregated: Dict[str, List[Any]] = defaultdict(list)

    for run_metrics in runs_metrics:
        for key, value in run_metrics.items():
            aggregated[key].append(value)

    return dict(aggregated)


def calculate_confidence_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple:
    """
    Calculate confidence interval for an array of values.

    Args:
        values: Array of values from multiple runs
        confidence: Confidence level (default 95%)

    Returns:
        (mean, ci_lower, ci_upper) tuple
    """
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)

    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0

    # t-distribution critical value approximation for 95% CI
    z = 1.96 if confidence == 0.95 else 1.645

    ci_width = z * (std / np.sqrt(n))

    return (mean, mean - ci_width, mean + ci_width)


def aggregate_time_series(
    series_list: List[Any], min_length: int = 2
) -> tuple:
    """
    Aggregate time series data across runs with CI.

    Args:
        series_list: List of pandas Series or arrays
        min_length: Minimum length for valid aggregation

    Returns:
        (mean_curve, ci_lower, ci_upper, n_timesteps) tuple
    """
    if not series_list:
        return (None, None, None, 0)

    # Get values arrays
    arrays = []
    for s in series_list:
        if hasattr(s, "values"):
            arrays.append(s.values)
        else:
            arrays.append(np.array(s))

    # Find minimum length
    lengths = [len(a) for a in arrays]
    min_len = min(lengths)

    if min_len < min_length:
        return (None, None, None, 0)

    # Truncate to common length
    matrix = np.array([a[:min_len] for a in arrays])

    # Calculate statistics
    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.zeros_like(mean_curve)

    n_runs = len(arrays)
    ci_width = 1.96 * (std_curve / np.sqrt(n_runs))

    return (mean_curve, mean_curve - ci_width, mean_curve + ci_width, min_len)


def summarize_aggregated_metrics(
    aggregated: Dict[str, List[Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Create summary statistics from aggregated metrics.

    Args:
        aggregated: Dictionary of metric lists

    Returns:
        Dictionary with mean, std, n for each scalar metric
    """
    summary = {}

    for key, values in aggregated.items():
        # Filter to scalar values
        scalars = [
            v
            for v in values
            if isinstance(v, (int, float, np.number)) and not np.isnan(v)
        ]

        if scalars:
            summary[key] = {
                "mean": float(np.mean(scalars)),
                "std": float(np.std(scalars, ddof=1)) if len(scalars) > 1 else 0.0,
                "n": len(scalars),
            }

    return summary
