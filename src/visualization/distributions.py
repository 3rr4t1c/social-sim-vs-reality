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


# =============================================================================
# POST VS RESHARE PLOTS
# =============================================================================

def plot_action_type_fractions(
    real_metrics: Dict[str, Any],
    sim_metrics: Dict[str, List[Dict[str, Any]]],
    sim_name: str,
    output_dir: Path,
    figure_size: tuple = (8, 6),
    color_real: str = "#2E3440",
    color_sim: str = "#0077BB",
    grid_alpha: float = 0.3,
    output_format: str = "pdf",
) -> Optional[Path]:
    """
    Plot bar chart comparing post vs reshare fractions between real and simulated data.

    Args:
        real_metrics: Metrics dictionary for real data
        sim_metrics: Dictionary mapping simulation names to lists of run metrics
        sim_name: Name of the simulation to plot
        output_dir: Directory for output file
        figure_size: Figure dimensions
        color_real: Color for real data bars
        color_sim: Color for simulation bars
        grid_alpha: Grid line transparency
        output_format: Output format ("pdf", "png")

    Returns:
        Path to saved figure, or None if no data
    """
    if "post_fraction" not in real_metrics or "reshare_fraction" not in real_metrics:
        logger.warning("Skipping action type fractions plot: missing fraction metrics")
        return None

    if sim_name not in sim_metrics or not sim_metrics[sim_name]:
        logger.warning(f"Skipping action type fractions plot: no data for {sim_name}")
        return None

    # Get real fractions
    real_post = real_metrics["post_fraction"]
    real_reshare = real_metrics["reshare_fraction"]

    # Aggregate simulation fractions (mean across runs)
    sim_runs = sim_metrics[sim_name]
    sim_post_values = [r.get("post_fraction", 0) for r in sim_runs if "post_fraction" in r]
    sim_reshare_values = [r.get("reshare_fraction", 0) for r in sim_runs if "reshare_fraction" in r]

    if not sim_post_values:
        return None

    sim_post_mean = np.mean(sim_post_values)
    sim_post_std = np.std(sim_post_values, ddof=1) if len(sim_post_values) > 1 else 0
    sim_reshare_mean = np.mean(sim_reshare_values)
    sim_reshare_std = np.std(sim_reshare_values, ddof=1) if len(sim_reshare_values) > 1 else 0

    # Create plot
    fig, ax = plt.subplots(figsize=figure_size)

    x = np.array([0, 1])  # Two groups: Posts, Reshares
    width = 0.35

    # Real data bars
    real_values = [real_post, real_reshare]
    bars_real = ax.bar(x - width/2, real_values, width, label="Real", color=color_real)

    # Simulation bars with error bars
    sim_values = [sim_post_mean, sim_reshare_mean]
    sim_errors = [sim_post_std, sim_reshare_std]
    bars_sim = ax.bar(x + width/2, sim_values, width, label=f"Simulated (n={len(sim_runs)})",
                      color=color_sim, yerr=sim_errors, capsize=5)

    # Labels and formatting
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title(f"Action Type Distribution: Real vs {sim_name.replace('_simulation', '').replace('_', ' ')}",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Posts", "Reshares"], fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=grid_alpha)
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, val in zip(bars_real, real_values):
        ax.annotate(f"{val:.2%}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    for bar, val in zip(bars_sim, sim_values):
        ax.annotate(f"{val:.2%}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    safe_name = sim_name.replace(" ", "_").lower()
    output_path = output_dir / f"action_type_fractions_{safe_name}"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: action_type_fractions_{safe_name}.{output_format}")
    return output_path.with_suffix(f".{output_format}")


def plot_post_vs_reshare_scatter(
    real_metrics: Dict[str, Any],
    sim_metrics: Dict[str, List[Dict[str, Any]]],
    sim_name: str,
    output_dir: Path,
    figure_size: tuple = (14, 6),
    color_real: str = "#2E3440",
    color_sim: str = "#0077BB",
    grid_alpha: float = 0.3,
    max_points: int = 5000,
    output_format: str = "png",
    point_alpha: float = 0.5,
    point_size: int = 20,
    time_binning: str = "D",
    clip_percentile: Optional[int] = 99,
) -> Optional[Path]:
    """
    Plot scatter of mean posts vs mean reshares per user per time unit.

    Creates two side-by-side panels: Real data | Simulated data.
    Each point represents one user.

    Args:
        real_metrics: Metrics dictionary for real data
        sim_metrics: Dictionary mapping simulation names to lists of run metrics
        sim_name: Name of the simulation to plot
        output_dir: Directory for output file
        figure_size: Figure dimensions
        color_real: Color for real data points
        color_sim: Color for simulation points
        grid_alpha: Grid line transparency
        max_points: Maximum points to display (samples if exceeded)
        output_format: Output format ("pdf", "png")
        point_alpha: Point transparency
        point_size: Point size (uniform for all points)
        time_binning: Time binning setting for axis labels
        clip_percentile: Percentile to clip axes at (None = no clipping)

    Returns:
        Path to saved figure, or None if no data
    """
    # Map time_binning to readable labels
    time_unit_labels = {
        "D": "day",
        "W": "week",
        "H": "hour",
        "h": "hour",
        "1D": "day",
        "1W": "week",
        "1H": "hour",
        "1h": "hour",
    }
    time_unit = time_unit_labels.get(time_binning, "time unit")

    if "user_post_reshare_means" not in real_metrics:
        logger.warning("Skipping post vs reshare scatter: missing user_post_reshare_means")
        return None

    if sim_name not in sim_metrics or not sim_metrics[sim_name]:
        logger.warning(f"Skipping post vs reshare scatter: no data for {sim_name}")
        return None

    real_df = real_metrics["user_post_reshare_means"].copy()
    if real_df.empty:
        return None

    # Aggregate simulation data across runs (same users, average their means)
    sim_runs = sim_metrics[sim_name]
    sim_dfs = [r.get("user_post_reshare_means") for r in sim_runs
               if r.get("user_post_reshare_means") is not None and not r.get("user_post_reshare_means").empty]

    if not sim_dfs:
        return None

    # Concatenate all runs and group by user_id to average
    sim_combined = pd.concat(sim_dfs, ignore_index=True)
    agg_cols = {"mean_posts": "mean", "mean_reshares": "mean"}
    sim_df = sim_combined.groupby("user_id").agg(agg_cols).reset_index()

    # Sample if too many points
    if len(real_df) > max_points:
        real_df = real_df.sample(n=max_points, random_state=42)
    if len(sim_df) > max_points:
        sim_df = sim_df.sample(n=max_points, random_state=42)

    # Calculate axis limits
    if clip_percentile is not None:
        all_posts = np.concatenate([real_df["mean_posts"].values, sim_df["mean_posts"].values])
        all_reshares = np.concatenate([real_df["mean_reshares"].values, sim_df["mean_reshares"].values])
        max_post = np.percentile(all_posts, clip_percentile)
        max_reshare = np.percentile(all_reshares, clip_percentile)
        max_val = max(max_post, max_reshare) * 1.05

        # Count points outside visible range
        real_outside = ((real_df["mean_posts"] > max_val) | (real_df["mean_reshares"] > max_val)).sum()
        sim_outside = ((sim_df["mean_posts"] > max_val) | (sim_df["mean_reshares"] > max_val)).sum()
    else:
        max_post = max(real_df["mean_posts"].max(), sim_df["mean_posts"].max()) * 1.1
        max_reshare = max(real_df["mean_reshares"].max(), sim_df["mean_reshares"].max()) * 1.1
        max_val = max(max_post, max_reshare)
        real_outside = 0
        sim_outside = 0

    # Calculate actual max values (before clipping)
    real_max_posts = real_df["mean_posts"].max()
    real_max_reshares = real_df["mean_reshares"].max()
    sim_max_posts = sim_df["mean_posts"].max()
    sim_max_reshares = sim_df["mean_reshares"].max()

    # Create plot with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)

    # Real data panel
    ax1.scatter(real_df["mean_posts"], real_df["mean_reshares"],
                alpha=point_alpha, s=point_size, color=color_real, edgecolors="none")
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
    ax1.set_xlabel(f"Mean posts per {time_unit}", fontsize=11)
    ax1.set_ylabel(f"Mean reshares per {time_unit}", fontsize=11)
    ax1.set_title("Real Data", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.grid(alpha=grid_alpha)
    ax1.set_aspect("equal", adjustable="box")

    # Stats box for real data
    real_stats = f"N: {len(real_df):,} users"
    if clip_percentile is not None and real_outside > 0:
        real_stats += f"\nClipped: {real_outside} (p{clip_percentile})"
    # Calculate actual visible max (max of points within axis limits)
    real_visible = real_df[(real_df["mean_posts"] <= max_val) & (real_df["mean_reshares"] <= max_val)]
    real_vis_posts = real_visible["mean_posts"].max() if not real_visible.empty else real_max_posts
    real_vis_reshares = real_visible["mean_reshares"].max() if not real_visible.empty else real_max_reshares
    # Show visible max and absolute max if different
    if real_max_posts > real_vis_posts * 1.01:  # More than 1% difference
        real_stats += f"\nMax posts/{time_unit}: {real_vis_posts:.2f} (abs: {real_max_posts:.2f})"
    else:
        real_stats += f"\nMax posts/{time_unit}: {real_max_posts:.2f}"
    if real_max_reshares > real_vis_reshares * 1.01:  # More than 1% difference
        real_stats += f"\nMax reshares/{time_unit}: {real_vis_reshares:.2f} (abs: {real_max_reshares:.2f})"
    else:
        real_stats += f"\nMax reshares/{time_unit}: {real_max_reshares:.2f}"
    ax1.text(
        0.97, 0.97, real_stats,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Simulation panel
    display_name = sim_name.replace("_simulation", "").replace("_", " ")
    ax2.scatter(sim_df["mean_posts"], sim_df["mean_reshares"],
                alpha=point_alpha, s=point_size, color=color_sim, edgecolors="none")
    ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
    ax2.set_xlabel(f"Mean posts per {time_unit}", fontsize=11)
    ax2.set_ylabel(f"Mean reshares per {time_unit}", fontsize=11)
    ax2.set_title(f"Simulated: {display_name} (n={len(sim_runs)} runs)", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.grid(alpha=grid_alpha)
    ax2.set_aspect("equal", adjustable="box")

    # Stats box for simulation data
    sim_stats = f"N: {len(sim_df):,} users"
    if clip_percentile is not None and sim_outside > 0:
        sim_stats += f"\nClipped: {sim_outside} (p{clip_percentile})"
    # Calculate actual visible max (max of points within axis limits)
    sim_visible = sim_df[(sim_df["mean_posts"] <= max_val) & (sim_df["mean_reshares"] <= max_val)]
    sim_vis_posts = sim_visible["mean_posts"].max() if not sim_visible.empty else sim_max_posts
    sim_vis_reshares = sim_visible["mean_reshares"].max() if not sim_visible.empty else sim_max_reshares
    # Show visible max and absolute max if different
    if sim_max_posts > sim_vis_posts * 1.01:  # More than 1% difference
        sim_stats += f"\nMax posts/{time_unit}: {sim_vis_posts:.2f} (abs: {sim_max_posts:.2f})"
    else:
        sim_stats += f"\nMax posts/{time_unit}: {sim_max_posts:.2f}"
    if sim_max_reshares > sim_vis_reshares * 1.01:  # More than 1% difference
        sim_stats += f"\nMax reshares/{time_unit}: {sim_vis_reshares:.2f} (abs: {sim_max_reshares:.2f})"
    else:
        sim_stats += f"\nMax reshares/{time_unit}: {sim_max_reshares:.2f}"
    ax2.text(
        0.97, 0.97, sim_stats,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    safe_name = sim_name.replace(" ", "_").lower()
    output_path = output_dir / f"post_vs_reshare_scatter_{safe_name}"
    save_figure(fig, output_path, fmt=output_format)
    close_figure(fig)

    logger.info(f"Generated: post_vs_reshare_scatter_{safe_name}.{output_format}")
    return output_path.with_suffix(f".{output_format}")
