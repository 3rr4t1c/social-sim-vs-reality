"""
Social Media Simulator Validation Configuration
================================================
Modify this file to customize the analysis.

Usage:
    - Modify the variables below
    - Run: python analyze.py
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
REAL_DATA_PATTERN = "*_dataset.csv"  # Pattern for real data files
SYNTHETIC_DIR_PATTERN = "*_simulation"  # Pattern for simulation folders

# =============================================================================
# PROCESSING
# =============================================================================
CHUNK_SIZE = 500_000
USE_CACHING = True
CACHE_DIR = Path(".cache")

# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================
ROLLING_WINDOW_DAYS = 7
TIME_BINNING = "D"  # "D"=daily, "W"=weekly, "H"=hourly
MIN_ACTIVE_USERS = 5

# =============================================================================
# METRICS
# =============================================================================
PERCENTILES = [25, 50, 75, 90, 95, 99]
TOP_USERS_PERCENT = 10
RESPONSE_TIME_MAX_HOURS = 168  # 7 days

ANALYZE_EXTRA_FEATURES = True

# Include "orphaned" users (without actions in the period) in activity calculations
# Default: False (considers only users with at least 1 action)
INCLUDE_ORPHANED_USERS = False

# Exclude first and last day (often partial) from distribution plots
# Default: True (excludes partial days for more accurate visualizations)
EXCLUDE_PARTIAL_DAYS = True

# =============================================================================
# VISUALIZATION
# =============================================================================
FIGURE_SIZE = (12, 6)
FIGURE_SIZE_WIDE = (14, 6)
DPI = 120
SAVE_DPI = 300

# Color palette: "colorblind", "default", or custom list of hex colors
# Options:
#   - "colorblind": palette optimized for deuteranopia/protanopia
#   - "default": Nord theme palette
#   - Custom list: ["#0077BB", "#EE7733", ...]
COLOR_PALETTE = "colorblind"

# Real data color (always dark gray for contrast)
COLOR_REAL = "#2E3440"

# Output format by plot type
# "pdf" for vector (publication), "png" for raster (preview/scatter)
FORMAT_TEMPORAL = "pdf"
FORMAT_DISTRIBUTION = "pdf"
FORMAT_SCATTER = "png"  # Recommended png for thousands of points

# Scatter plot settings
MAX_SCATTER_POINTS = 5000
SCATTER_METHOD = "auto"  # "scatter", "hexbin", "auto"
RASTERIZE_SCATTER = True  # Rasterize points in PDFs for reduced file size

# Post vs Reshare scatter point sizing (based on mean actions per time unit)
SCATTER_POINT_SIZE_MIN = 10  # Size for least active users
SCATTER_POINT_SIZE_MAX = 200  # Size for most active users
SCATTER_POINT_ALPHA = 0.5  # Point transparency
SCATTER_CLIP_PERCENTILE = 99.9  # Clip axes at this percentile (None = no clip)
SCATTER_SHOW_SIZE_LEGEND = True  # Show legend explaining point sizes
SCATTER_ACTIVE_ONLY = True  # If True, mean is computed only on active days (not influenced by dataset duration)

# Style
CONFIDENCE_ALPHA = 0.25  # Confidence interval band transparency
GRID_ALPHA = 0.3

# Logarithmic scale for Y-axis in distribution plots
USE_LOG_SCALE_USER_ACTIVITY = True  # For user_activity_distribution (default: enabled)
USE_LOG_SCALE_TOTAL_ACTIVITY = (
    False  # For total_activity_distribution (default: disabled)
)

# =============================================================================
# PLOTS TO GENERATE
# =============================================================================
# Set to True to enable, False to disable
PLOTS = {
    "temporal_comparison": True,  # Temporal evolution with CI (DEFAULT)
    "user_activity_distribution": True,  # User activity distribution (4 panels)
    "total_activity_distribution": True,  # Total activity distribution (4 panels)
    "action_type_fractions": True,  # Post vs reshare fractions bar plot
    "post_vs_reshare_scatter": True,  # User mean posts vs reshares scatter
    "inter_event_time": False,  # Inter-event time distribution
    "response_time_boxplot": False,  # Response time boxplot
    "quality_kde": False,  # Quality/extra feature KDE
}

# =============================================================================
# EXPORT
# =============================================================================
SAVE_FIGURES = True
SAVE_TABLES = True
LATEX_FORMAT = "booktabs"  # "booktabs" (elegant) or "simple"
SAVE_JSON_METRICS = True

# =============================================================================
# SIMULATOR DATA CONVERSION
# =============================================================================
CONVERTER_START_DATE = "2020-12-20 01:25:21+00:00"
CONVERTER_QUALITY_SCALE = 100.0  # Multiplies quality (0-1) by this factor
