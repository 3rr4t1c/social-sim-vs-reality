"""Visualization modules for analysis plots."""

from .base import setup_matplotlib, get_color_palette, save_figure
from .temporal import plot_temporal_comparison
from .distributions import (
    plot_user_activity_distribution,
    plot_total_activity_distribution,
    plot_activity_cdf,
    plot_cascade_distribution,
    plot_inter_event_time,
    plot_response_time_boxplot,
    plot_quality_kde,
    plot_action_type_fractions,
    plot_post_vs_reshare_scatter,
)

__all__ = [
    "setup_matplotlib",
    "get_color_palette",
    "save_figure",
    "plot_temporal_comparison",
    "plot_user_activity_distribution",
    "plot_total_activity_distribution",
    "plot_activity_cdf",
    "plot_cascade_distribution",
    "plot_inter_event_time",
    "plot_response_time_boxplot",
    "plot_quality_kde",
    "plot_action_type_fractions",
    "plot_post_vs_reshare_scatter",
]
