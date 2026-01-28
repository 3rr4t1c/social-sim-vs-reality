"""
Temporal comparison visualization.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import logging

from .base import save_figure, close_figure

logger = logging.getLogger(__name__)


def plot_temporal_comparison(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (12, 6),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    confidence_alpha: float = 0.25,
    grid_alpha: float = 0.3,
    rolling_window_days: int = 7,
    output_format: str = "pdf",
) -> Optional[Path]:
    """
    Plot temporal activity with mean +/- 95% CI for simulations.

    Shows rolling average of actions per user over time,
    with real data as solid line and simulations with confidence bands.

    Args:
        real_metrics: Metrics dictionary for real data
        sim_metrics_agg: Aggregated simulation metrics
        output_dir: Directory for output file
        figure_size: Figure dimensions
        color_real: Color for real data line
        color_palette: Colors for simulations
        confidence_alpha: Transparency for CI band
        grid_alpha: Grid line transparency
        rolling_window_days: Window size (for label)
        output_format: Output format ("pdf", "png")

    Returns:
        Path to saved figure, or None if no data
    """
    if "rolling_actions_per_user" not in real_metrics:
        logger.warning("Skipping temporal plot: missing rolling_actions_per_user")
        return None

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    fig, ax = plt.subplots(figsize=figure_size)

    # Real data
    real_series = real_metrics["rolling_actions_per_user"]
    real_days = (real_series.index - real_series.index.min()).days

    ax.plot(
        real_days,
        real_series.values,
        label="Real Data",
        color=color_real,
        linewidth=2.5,
        zorder=10,
    )

    # Simulations
    for idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        runs_series = metrics_dict.get("rolling_actions_per_user", [])

        if not runs_series:
            continue

        # Align: truncate to minimum length
        min_length = min(len(s) for s in runs_series if hasattr(s, "__len__"))
        if min_length < 2:
            continue

        # Stack into matrix
        data_matrix = np.array(
            [
                s.values[:min_length] if hasattr(s, "values") else s[:min_length]
                for s in runs_series
            ]
        )

        # Statistics
        mean_curve = np.mean(data_matrix, axis=0)
        std_curve = np.std(data_matrix, axis=0, ddof=1)
        n_runs = len(runs_series)

        # 95% CI
        ci_width = 1.96 * (std_curve / np.sqrt(n_runs))

        sim_days = np.arange(min_length)
        color = color_palette[idx % len(color_palette)]

        # Clean simulation name for label
        display_name = sim_id.replace("_simulation", "").replace("_", " ")

        # Mean line
        ax.plot(
            sim_days,
            mean_curve,
            label=f"{display_name} (n={n_runs})",
            color=color,
            linewidth=2,
        )

        # CI band
        ax.fill_between(
            sim_days,
            mean_curve - ci_width,
            mean_curve + ci_width,
            color=color,
            alpha=confidence_alpha,
        )

    ax.set_xlabel("Days since start", fontsize=12)
    ax.set_ylabel(f"Actions per user ({rolling_window_days}-day rolling avg)", fontsize=12)
    ax.set_title("Temporal Activity: Real vs Simulations", fontsize=14, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=grid_alpha)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_path = output_dir / "temporal_comparison"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: temporal_comparison.{output_format}")
    return output_path.with_suffix(f".{output_format}")
