"""
Base visualization utilities.
"""

import warnings
import logging
from pathlib import Path
from typing import List, Optional

# Suppress matplotlib warnings BEFORE import
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt

# Suppress font manager logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Predefined color palettes
PALETTES = {
    "colorblind": [
        "#0077BB",  # Blue
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


def setup_matplotlib(
    dpi: int = 120,
    save_dpi: int = 300,
    grid_alpha: float = 0.3,
) -> None:
    """
    Configure matplotlib for publication-quality figures.

    Args:
        dpi: Display DPI
        save_dpi: Output file DPI
        grid_alpha: Grid transparency
    """
    plt.rcParams.update(
        {
            # Font sizes
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            # DPI
            "figure.dpi": dpi,
            "savefig.dpi": save_dpi,
            # Grid
            "axes.grid": True,
            "grid.alpha": grid_alpha,
            # PDF settings for editable text
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Better defaults
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def get_color_palette(palette_name: str = "colorblind") -> List[str]:
    """
    Get color palette by name.

    Args:
        palette_name: "colorblind", "default", or a list of hex colors

    Returns:
        List of hex color strings
    """
    if isinstance(palette_name, list):
        return palette_name

    return PALETTES.get(palette_name, PALETTES["colorblind"])


def save_figure(
    fig: plt.Figure,
    filepath: Path,
    fmt: str = "pdf",
    dpi: int = 300,
) -> None:
    """
    Save figure with suppressed warnings.

    Args:
        fig: Matplotlib figure
        filepath: Output path (without extension)
        fmt: Format ("pdf", "png", etc.)
        dpi: Output DPI (mainly for raster formats)
    """
    output_path = filepath.with_suffix(f".{fmt}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(
            output_path,
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )

    logger.debug(f"Saved: {output_path.name}")


def close_figure(fig: plt.Figure) -> None:
    """Close figure to free memory."""
    plt.close(fig)
