"""
JSON export utilities with NumPy/Pandas serialization.
"""

from pathlib import Path
from typing import Any, Dict
import json
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NumpyPandasEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and Pandas types."""

    def default(self, obj: Any) -> Any:
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # NumPy scalars
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Pandas types
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")

        # NaN handling
        if pd.isna(obj):
            return None

        return super().default(obj)


def make_json_serializable(data: Any) -> Any:
    """
    Recursively convert complex data structures to JSON-serializable types.

    Args:
        data: Data structure to convert

    Returns:
        JSON-serializable version of the data
    """
    # NumPy arrays
    if isinstance(data, np.ndarray):
        return make_json_serializable(data.tolist())

    # Pandas structures
    if isinstance(data, pd.Series):
        return make_json_serializable(data.to_dict())
    if isinstance(data, pd.DataFrame):
        return make_json_serializable(data.to_dict(orient="list"))

    # Collections
    if isinstance(data, dict):
        return {str(k): make_json_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [make_json_serializable(item) for item in data]

    # NumPy scalars
    if isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    if isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)

    # Temporal types
    if isinstance(data, (pd.Timestamp, pd.Timedelta)):
        return str(data)

    # NaN/None
    if np.isscalar(data) and pd.isna(data):
        return None

    # Already serializable
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        return str(data)


def export_metrics_json(
    metrics: Dict[str, Any],
    output_path: Path,
    exclude_arrays: bool = True,
) -> None:
    """
    Export metrics dictionary to JSON file.

    Args:
        metrics: Metrics dictionary
        output_path: Output file path
        exclude_arrays: If True, exclude large arrays (only scalar metrics)
    """
    if exclude_arrays:
        # Filter to scalar metrics only
        filtered = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                filtered[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                filtered[key] = float(value)
            # Skip arrays, Series, DataFrames
    else:
        filtered = make_json_serializable(metrics)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            filtered,
            f,
            indent=2,
            cls=NumpyPandasEncoder,
        )

    logger.debug(f"Exported metrics to {output_path}")
