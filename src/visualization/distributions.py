"""
Distribution comparison visualizations.

New graphs:
- User Activity Distribution (4 panels): Active-only vs Including zeros, Real vs Synthetic
- Total Activity Distribution (4 panels): Actions per timestep and Active users per timestep

Each simulation gets its own graph file.
Synthetic data is averaged across runs (not concatenated).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from .base import save_figure, close_figure

logger = logging.getLogger(__name__)


def _get_time_granularity_label(time_binning: str) -> str:
    """
    Convert pandas time binning code to human-readable label.

    Args:
        time_binning: Pandas frequency string (e.g., "D", "W", "H", "3D")

    Returns:
        Human-readable label (e.g., "1 day", "1 week", "3 days")
    """
    # Parse the frequency string
    import re
    match = re.match(r"(\d*)([A-Za-z]+)", time_binning)
    if not match:
        return time_binning

    num_str, unit = match.groups()
    num = int(num_str) if num_str else 1

    unit_map = {
        "D": ("day", "days"),
        "W": ("week", "weeks"),
        "H": ("hour", "hours"),
        "T": ("minute", "minutes"),
        "MIN": ("minute", "minutes"),
        "S": ("second", "seconds"),
        "M": ("month", "months"),
        "Y": ("year", "years"),
    }

    unit_upper = unit.upper()
    if unit_upper in unit_map:
        singular, plural = unit_map[unit_upper]
        return f"{num} {singular if num == 1 else plural}"

    return time_binning


def _average_series_across_runs(series_list: List[pd.Series]) -> pd.Series:
    """
    Average multiple Series across runs, aligning by index.

    Args:
        series_list: List of pandas Series with same index type

    Returns:
        Series with averaged values
    """
    if not series_list:
        return pd.Series(dtype=float)

    # Filter out empty series
    valid_series = [s for s in series_list if s is not None and len(s) > 0]
    if not valid_series:
        return pd.Series(dtype=float)

    if len(valid_series) == 1:
        return valid_series[0]

    # Combine all series into a DataFrame and compute mean
    df = pd.concat(valid_series, axis=1)
    return df.mean(axis=1)


def plot_user_activity_distribution(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (14, 10),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    output_format: str = "pdf",
    use_log_scale: bool = True,
    bins: int = 50,
    time_binning: str = "D",
) -> List[Path]:
    """
    Plot User Activity Distribution in 4 panels - one graph per simulation.

    Layout:
    +----------------------------------+----------------------------------+
    |  REAL - Active only              |  SYNTHETIC - Active only         |
    +----------------------------------+----------------------------------+
    |  REAL - Including zeros          |  SYNTHETIC - Including zeros     |
    +----------------------------------+----------------------------------+

    Synthetic data is averaged across runs (by user), not concatenated.

    Args:
        real_metrics: Metrics dictionary for real data
        sim_metrics_agg: Aggregated simulation metrics
        output_dir: Directory for output file
        figure_size: Figure dimensions
        color_real: Color for real data
        color_palette: Colors for simulations
        output_format: Output format
        use_log_scale: Use log scale for y-axis
        bins: Number of histogram bins
        time_binning: Time granularity for labels

    Returns:
        List of paths to saved figures
    """
    # Check required data (indexed versions)
    if "mean_actions_active_only_indexed" not in real_metrics:
        # Fallback to non-indexed if indexed not available
        if "mean_actions_active_only" not in real_metrics:
            logger.warning("Skipping user activity distribution: missing data")
            return []

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    output_paths = []
    time_label = _get_time_granularity_label(time_binning)

    # Get real data
    real_active_only = real_metrics.get("mean_actions_active_only", np.array([]))
    real_with_zeros = real_metrics.get("mean_actions_with_zeros", np.array([]))

    # Create one graph per simulation
    for sim_idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        # Get indexed series for averaging across runs
        active_indexed_list = metrics_dict.get("mean_actions_active_only_indexed", [])
        zeros_indexed_list = metrics_dict.get("mean_actions_with_zeros_indexed", [])

        # Average across runs
        if active_indexed_list:
            syn_active_series = _average_series_across_runs(active_indexed_list)
            syn_active_only = syn_active_series.values if len(syn_active_series) > 0 else np.array([])
        else:
            syn_active_only = np.array([])

        if zeros_indexed_list:
            syn_zeros_series = _average_series_across_runs(zeros_indexed_list)
            syn_with_zeros = syn_zeros_series.values if len(syn_zeros_series) > 0 else np.array([])
        else:
            syn_with_zeros = np.array([])

        if len(syn_active_only) == 0 and len(syn_with_zeros) == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=figure_size)
        sim_color = color_palette[sim_idx % len(color_palette)]
        display_name = sim_id.replace("_simulation", "").replace("_", " ").title()

        # Helper function to plot histogram with counts
        def plot_hist(ax, data, color, title, xlabel=f"Mean actions per user per {time_label}"):
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=12, fontweight="bold")
                return

            # Filter out invalid values
            data = np.asarray(data)
            data = data[np.isfinite(data)]
            if len(data) == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=12, fontweight="bold")
                return

            ax.hist(
                data,
                bins=bins,
                density=False,
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Add statistics text
            mean_val = np.mean(data)
            median_val = np.median(data)
            stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nN: {len(data):,}"
            ax.text(
                0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)

            if use_log_scale:
                ax.set_yscale("log")

        # Plot each panel
        plot_hist(axes[0, 0], real_active_only, color_real, "Real Data - Active Only")
        plot_hist(axes[0, 1], syn_active_only, sim_color, f"{display_name} - Active Only")
        plot_hist(axes[1, 0], real_with_zeros, color_real, "Real Data - Including Zeros")
        plot_hist(axes[1, 1], syn_with_zeros, sim_color, f"{display_name} - Including Zeros")

        plt.suptitle(
            f"User Activity Distribution: Real vs {display_name}\n(Mean actions per user per {time_label})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        # Save with simulation name
        safe_name = sim_id.replace(" ", "_").lower()
        output_path = output_dir / f"user_activity_{safe_name}"
        save_figure(fig, output_path, fmt=output_format)
        close_figure(fig)

        output_paths.append(output_path.with_suffix(f".{output_format}"))
        logger.info(f"Generated: user_activity_{safe_name}.{output_format}")

    return output_paths


def plot_total_activity_distribution(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (14, 10),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    output_format: str = "pdf",
    use_log_scale: bool = False,
    bins: int = 50,
    time_binning: str = "D",
) -> List[Path]:
    """
    Plot Total Activity Distribution in 4 panels - one graph per simulation.

    Layout:
    +----------------------------------+----------------------------------+
    |  REAL - Total actions/timestep   |  SYNTHETIC - Total actions       |
    +----------------------------------+----------------------------------+
    |  REAL - Active users/timestep    |  SYNTHETIC - Active users        |
    +----------------------------------+----------------------------------+

    Synthetic data is CONCATENATED across runs (not averaged) and plotted
    with density=True to show probability distribution.

    Args:
        real_metrics: Metrics dictionary for real data
        sim_metrics_agg: Aggregated simulation metrics
        output_dir: Directory for output file
        figure_size: Figure dimensions
        color_real: Color for real data
        color_palette: Colors for simulations
        output_format: Output format
        use_log_scale: Use log scale for y-axis
        bins: Number of histogram bins
        time_binning: Time granularity for labels

    Returns:
        List of paths to saved figures
    """
    # Check required data (use indexed versions which respect exclude_partial_days)
    if "total_actions_per_timestep_indexed" not in real_metrics:
        logger.warning("Skipping total activity distribution: missing data")
        return []

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    output_paths = []
    time_label = _get_time_granularity_label(time_binning)

    # Get real data (use indexed versions which have partial days excluded if configured)
    real_total_indexed = real_metrics.get("total_actions_per_timestep_indexed")
    real_users_indexed = real_metrics.get("active_users_per_timestep_indexed")
    real_total_actions = real_total_indexed.values if real_total_indexed is not None else np.array([])
    real_active_users = real_users_indexed.values if real_users_indexed is not None else np.array([])
    real_n_days = len(real_total_actions)

    # Create one graph per simulation
    for sim_idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        # Get indexed Series and CONCATENATE across runs (indexed versions have partial days excluded)
        total_list = metrics_dict.get("total_actions_per_timestep_indexed", [])
        users_list = metrics_dict.get("active_users_per_timestep_indexed", [])

        # Count runs
        n_runs = len([t for t in total_list if t is not None and len(t) > 0])

        # Concatenate all runs (extract values from Series)
        if total_list:
            syn_total_actions = np.concatenate([
                s.values if hasattr(s, 'values') else np.asarray(s)
                for s in total_list if s is not None and len(s) > 0
            ])
        else:
            syn_total_actions = np.array([])

        if users_list:
            syn_active_users = np.concatenate([
                s.values if hasattr(s, 'values') else np.asarray(s)
                for s in users_list if s is not None and len(s) > 0
            ])
        else:
            syn_active_users = np.array([])

        if len(syn_total_actions) == 0 and len(syn_active_users) == 0:
            continue

        # Calculate days per run (assuming all runs have same duration)
        syn_n_days = len(syn_total_actions) // n_runs if n_runs > 0 else 0

        fig, axes = plt.subplots(2, 2, figsize=figure_size)
        sim_color = color_palette[sim_idx % len(color_palette)]
        display_name = sim_id.replace("_simulation", "").replace("_", " ").title()

        # Helper function to plot histogram with DENSITY
        def plot_hist(ax, data, color, title, xlabel, n_days, n_runs=1):
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=12, fontweight="bold")
                return

            # Filter out invalid values
            data = np.asarray(data)
            data = data[np.isfinite(data)]
            if len(data) == 0:
                ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title, fontsize=12, fontweight="bold")
                return

            ax.hist(
                data,
                bins=bins,
                density=True,  # Use density for probability
                alpha=0.7,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )

            # Add statistics text (no N, show days/runs instead)
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            if n_runs > 1:
                stats_text = f"Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}\nDays: {n_days}, Runs: {n_runs}"
            else:
                stats_text = f"Mean: {mean_val:.1f}\nMedian: {median_val:.1f}\nStd: {std_val:.1f}\nDays: {n_days}"
            ax.text(
                0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(alpha=0.3)

            if use_log_scale:
                ax.set_yscale("log")

        # Plot each panel
        plot_hist(
            axes[0, 0], real_total_actions, color_real,
            "Real Data - Total Actions", f"Actions per {time_label}",
            n_days=real_n_days, n_runs=1
        )
        plot_hist(
            axes[0, 1], syn_total_actions, sim_color,
            f"{display_name} - Total Actions", f"Actions per {time_label}",
            n_days=syn_n_days, n_runs=n_runs
        )
        plot_hist(
            axes[1, 0], real_active_users, color_real,
            "Real Data - Active Users", f"Active users per {time_label}",
            n_days=real_n_days, n_runs=1
        )
        plot_hist(
            axes[1, 1], syn_active_users, sim_color,
            f"{display_name} - Active Users", f"Active users per {time_label}",
            n_days=syn_n_days, n_runs=n_runs
        )

        plt.suptitle(
            f"Total Activity Distribution: Real vs {display_name}\n(Probability density per {time_label})",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        # Save with simulation name
        safe_name = sim_id.replace(" ", "_").lower()
        output_path = output_dir / f"total_activity_{safe_name}"
        save_figure(fig, output_path, fmt=output_format)
        close_figure(fig)

        output_paths.append(output_path.with_suffix(f".{output_format}"))
        logger.info(f"Generated: total_activity_{safe_name}.{output_format}")

    return output_paths


def plot_inter_event_time(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (10, 6),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    output_format: str = "pdf",
    max_hours: float = 168,
    **kwargs,
) -> Optional[Path]:
    """Plot inter-event time distribution."""
    if "inter_event_times" not in real_metrics:
        logger.warning("Skipping IET distribution: missing inter_event_times")
        return None

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    fig, ax = plt.subplots(figsize=figure_size)

    real_iet = real_metrics["inter_event_times"]
    real_iet = real_iet[(real_iet > 0) & (real_iet <= max_hours)]

    ax.hist(
        real_iet,
        bins=50,
        density=False,
        alpha=0.7,
        label="Real Data",
        color=color_real,
        edgecolor="white",
    )

    for idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        iet_list = metrics_dict.get("inter_event_times", [])
        if not iet_list:
            continue

        combined = np.concatenate([i for i in iet_list if i is not None])
        combined = combined[(combined > 0) & (combined <= max_hours)]
        if len(combined) == 0:
            continue

        color = color_palette[idx % len(color_palette)]
        display_name = sim_id.replace("_simulation", "").replace("_", " ")

        ax.hist(
            combined,
            bins=50,
            density=False,
            alpha=0.5,
            label=display_name,
            color=color,
            edgecolor="white",
        )

    ax.set_xlabel("Inter-event time (hours)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Inter-Event Time Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "inter_event_time"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: inter_event_time.{output_format}")
    return output_path.with_suffix(f".{output_format}")


def plot_response_time_boxplot(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (10, 6),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    output_format: str = "pdf",
    **kwargs,
) -> Optional[Path]:
    """Plot response time distribution as boxplot."""
    if "response_time_array" not in real_metrics:
        logger.warning("Skipping response time boxplot: missing response_time_array")
        return None

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    fig, ax = plt.subplots(figsize=figure_size)

    data = [real_metrics["response_time_array"]]
    labels = ["Real"]
    colors = [color_real]

    for idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        rt_list = metrics_dict.get("response_time_array", [])
        if not rt_list:
            continue

        combined = np.concatenate([r for r in rt_list if r is not None])
        if len(combined) == 0:
            continue

        data.append(combined)
        labels.append(sim_id.replace("_simulation", "").replace("_", " "))
        colors.append(color_palette[idx % len(color_palette)])

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Response time (hours)", fontsize=12)
    ax.set_title("Response Time Distribution", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = output_dir / "response_time_boxplot"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: response_time_boxplot.{output_format}")
    return output_path.with_suffix(f".{output_format}")


def plot_quality_kde(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (10, 6),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    output_format: str = "pdf",
    **kwargs,
) -> Optional[Path]:
    """Plot KDE of extra/quality feature."""
    if "extra_distribution" not in real_metrics:
        logger.warning("Skipping quality KDE: missing extra_distribution")
        return None

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    fig, ax = plt.subplots(figsize=figure_size)

    real_extra = real_metrics["extra_distribution"]
    if real_extra.ndim > 1:
        real_extra = real_extra[:, 0]

    ax.hist(
        real_extra,
        bins=50,
        density=False,
        alpha=0.7,
        label="Real Data",
        color=color_real,
        edgecolor="white",
    )

    for idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        extra_list = metrics_dict.get("extra_distribution", [])
        if not extra_list:
            continue

        combined = np.concatenate(
            [
                e[:, 0] if e.ndim > 1 else e
                for e in extra_list
                if e is not None and len(e) > 0
            ]
        )
        if len(combined) == 0:
            continue

        color = color_palette[idx % len(color_palette)]
        display_name = sim_id.replace("_simulation", "").replace("_", " ")

        ax.hist(
            combined,
            bins=50,
            density=False,
            alpha=0.5,
            label=display_name,
            color=color,
            edgecolor="white",
        )

    ax.set_xlabel("Quality/Extra feature value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Quality Distribution", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "quality_kde"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: quality_kde.{output_format}")
    return output_path.with_suffix(f".{output_format}")


# Deprecated functions
def plot_activity_cdf(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    figure_size: tuple = (10, 6),
    color_real: str = "#2E3440",
    color_palette: List[str] = None,
    output_format: str = "pdf",
    **kwargs,
) -> Optional[Path]:
    """[DEPRECATED] Use plot_user_activity_distribution instead."""
    logger.warning("plot_activity_cdf is deprecated. Use plot_user_activity_distribution instead.")

    if "activity_per_user" not in real_metrics:
        return None

    if color_palette is None:
        color_palette = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

    fig, ax = plt.subplots(figsize=figure_size)

    real_activity = np.sort(real_metrics["activity_per_user"])
    real_cdf = np.arange(1, len(real_activity) + 1) / len(real_activity)

    ax.plot(real_activity, real_cdf, label="Real Data", color=color_real, linewidth=2.5, zorder=10)

    for idx, (sim_id, metrics_dict) in enumerate(sim_metrics_agg.items()):
        activities = metrics_dict.get("activity_per_user", [])
        if not activities:
            continue

        combined = np.concatenate([a for a in activities if a is not None])
        if len(combined) == 0:
            continue

        sorted_activity = np.sort(combined)
        cdf = np.arange(1, len(sorted_activity) + 1) / len(sorted_activity)

        color = color_palette[idx % len(color_palette)]
        display_name = sim_id.replace("_simulation", "").replace("_", " ")

        ax.plot(sorted_activity, cdf, label=display_name, color=color, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Number of actions per user", fontsize=12)
    ax.set_ylabel("Cumulative probability", fontsize=12)
    ax.set_title("Activity Distribution (CDF)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    output_path = output_dir / "activity_cdf"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: activity_cdf.{output_format}")
    return output_path.with_suffix(f".{output_format}")


def plot_cascade_distribution(
    real_metrics: Dict[str, Any],
    sim_metrics_agg: Dict[str, Dict[str, List[Any]]],
    output_dir: Path,
    **kwargs,
) -> Optional[Path]:
    """[DEPRECATED] Cascade metrics have been removed."""
    logger.warning("plot_cascade_distribution is deprecated. Cascade metrics have been removed.")
    return None
