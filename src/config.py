"""
Configuration loader and validation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Predefined color palettes
PALETTES = {
    "colorblind": [
        "#0077BB",  # Blue (safe for all)
        "#EE7733",  # Orange
        "#009988",  # Teal
        "#CC3311",  # Red
        "#33BBEE",  # Cyan
        "#EE3377",  # Magenta
        "#BBBBBB",  # Grey
    ],
    "default": [
        "#5E81AC",  # Blue
        "#BF616A",  # Red
        "#A3BE8C",  # Green
        "#EBCB8B",  # Yellow
        "#B48EAD",  # Purple
        "#88C0D0",  # Cyan
        "#D08770",  # Orange
        "#8FBCBB",  # Teal
    ],
}


@dataclass
class Config:
    """Validated configuration object."""

    # Paths
    data_dir: Path
    output_dir: Path
    real_data_pattern: str
    synthetic_dir_pattern: str
    cache_dir: Path

    # Processing
    chunk_size: int
    use_caching: bool

    # Temporal
    rolling_window_days: int
    time_binning: str
    min_active_users: int

    # Metrics
    percentiles: List[int]
    top_users_percent: int
    response_time_max_hours: int
    analyze_extra_features: bool
    include_orphaned_users: bool
    exclude_partial_days: bool

    # Visualization
    figure_size: Tuple[int, int]
    figure_size_wide: Tuple[int, int]
    dpi: int
    save_dpi: int
    color_palette: List[str]
    color_real: str
    format_temporal: str
    format_distribution: str
    format_scatter: str
    max_scatter_points: int
    scatter_method: str
    rasterize_scatter: bool
    scatter_point_size_min: int
    scatter_point_size_max: int
    scatter_point_alpha: float
    scatter_clip_percentile: int  # Clip axes at this percentile (None = no clip)
    scatter_show_size_legend: bool  # Show legend explaining point sizes
    scatter_active_only: bool  # If True, mean computed only on active days
    confidence_alpha: float
    grid_alpha: float
    use_log_scale_user_activity: bool
    use_log_scale_total_activity: bool

    # Plots
    plots: Dict[str, bool]

    # Export
    save_figures: bool
    save_tables: bool
    latex_format: str
    save_json_metrics: bool

    # Converter
    converter_start_date: str
    converter_quality_scale: float

    # Derived paths
    real_data_dir: Path = field(init=False)
    synthetic_data_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths and create directories."""
        self.real_data_dir = self.data_dir / "real"
        self.synthetic_data_dir = self.data_dir / "synthetic"
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.metrics_dir = self.output_dir / "metrics"

        # Create directories
        for dir_path in [
            self.output_dir,
            self.figures_dir,
            self.tables_dir,
            self.metrics_dir,
            self.cache_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """
    Load configuration from settings.py.

    Returns:
        Validated Config object
    """
    import settings

    # Resolve color palette
    palette_setting = getattr(settings, "COLOR_PALETTE", "colorblind")
    if isinstance(palette_setting, str):
        color_palette = PALETTES.get(palette_setting, PALETTES["colorblind"])
    else:
        color_palette = list(palette_setting)

    config = Config(
        # Paths
        data_dir=Path(settings.DATA_DIR),
        output_dir=Path(settings.OUTPUT_DIR),
        real_data_pattern=settings.REAL_DATA_PATTERN,
        synthetic_dir_pattern=settings.SYNTHETIC_DIR_PATTERN,
        cache_dir=Path(settings.CACHE_DIR),
        # Processing
        chunk_size=settings.CHUNK_SIZE,
        use_caching=settings.USE_CACHING,
        # Temporal
        rolling_window_days=settings.ROLLING_WINDOW_DAYS,
        time_binning=settings.TIME_BINNING,
        min_active_users=settings.MIN_ACTIVE_USERS,
        # Metrics
        percentiles=list(settings.PERCENTILES),
        top_users_percent=settings.TOP_USERS_PERCENT,
        response_time_max_hours=settings.RESPONSE_TIME_MAX_HOURS,
        analyze_extra_features=settings.ANALYZE_EXTRA_FEATURES,
        include_orphaned_users=getattr(settings, "INCLUDE_ORPHANED_USERS", False),
        exclude_partial_days=getattr(settings, "EXCLUDE_PARTIAL_DAYS", True),
        # Visualization
        figure_size=tuple(settings.FIGURE_SIZE),
        figure_size_wide=tuple(settings.FIGURE_SIZE_WIDE),
        dpi=settings.DPI,
        save_dpi=settings.SAVE_DPI,
        color_palette=color_palette,
        color_real=settings.COLOR_REAL,
        format_temporal=settings.FORMAT_TEMPORAL,
        format_distribution=settings.FORMAT_DISTRIBUTION,
        format_scatter=settings.FORMAT_SCATTER,
        max_scatter_points=settings.MAX_SCATTER_POINTS,
        scatter_method=settings.SCATTER_METHOD,
        rasterize_scatter=settings.RASTERIZE_SCATTER,
        scatter_point_size_min=getattr(settings, "SCATTER_POINT_SIZE_MIN", 10),
        scatter_point_size_max=getattr(settings, "SCATTER_POINT_SIZE_MAX", 200),
        scatter_point_alpha=getattr(settings, "SCATTER_POINT_ALPHA", 0.5),
        scatter_clip_percentile=getattr(settings, "SCATTER_CLIP_PERCENTILE", 99),
        scatter_show_size_legend=getattr(settings, "SCATTER_SHOW_SIZE_LEGEND", True),
        scatter_active_only=getattr(settings, "SCATTER_ACTIVE_ONLY", True),
        confidence_alpha=settings.CONFIDENCE_ALPHA,
        grid_alpha=settings.GRID_ALPHA,
        use_log_scale_user_activity=getattr(settings, "USE_LOG_SCALE_USER_ACTIVITY", True),
        use_log_scale_total_activity=getattr(settings, "USE_LOG_SCALE_TOTAL_ACTIVITY", False),
        # Plots
        plots=dict(settings.PLOTS),
        # Export
        save_figures=settings.SAVE_FIGURES,
        save_tables=settings.SAVE_TABLES,
        latex_format=settings.LATEX_FORMAT,
        save_json_metrics=settings.SAVE_JSON_METRICS,
        # Converter
        converter_start_date=settings.CONVERTER_START_DATE,
        converter_quality_scale=settings.CONVERTER_QUALITY_SCALE,
    )

    logger.info(f"Configuration loaded: data_dir={config.data_dir}")
    return config
