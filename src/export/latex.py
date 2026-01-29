"""
LaTeX table export utilities.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_latex_table(
    real_metrics: Dict[str, Any],
    sim_summaries: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
    latex_format: str = "booktabs",
) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Generate summary comparison table and export as CSV and LaTeX.

    Args:
        real_metrics: Metrics dictionary for real data
        sim_summaries: Summarized simulation metrics {sim_id: {metric: {mean, std, n}}}
        output_dir: Directory for output files
        latex_format: "booktabs" for elegant tables, "simple" for basic

    Returns:
        (DataFrame, csv_path, tex_path) tuple
    """
    # Define metrics to include in table
    # Updated: removed cascade metrics, added new activity metrics
    metric_labels = [
        ("total_actions", "Total Actions"),
        ("total_users", "Total Users"),
        ("duration_days", "Duration (days)"),
        ("orphan_rate", "Orphan Rate"),
        ("post_fraction", "Post Fraction"),
        ("reshare_fraction", "Reshare Fraction"),
        ("unique_targets_ratio", "Unique Targets Ratio"),
        ("actions_per_user_median", "Actions/User (median)"),
        ("actions_per_user_p90", "Actions/User (p90)"),
        ("active_user_ratio", "Active User Ratio"),
        ("gini_coefficient", "Gini Coefficient"),
        ("burstiness", "Burstiness"),
        ("response_time_median", "Resp. Time (h, median)"),
    ]

    rows = []

    # Real data row
    row_real = {"Dataset": "Real Data"}
    for key, label in metric_labels:
        value = real_metrics.get(key)
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            # Use appropriate formatting based on metric type
            if key in ("total_actions", "total_users"):
                row_real[label] = f"{int(value):,}"
            elif key == "duration_days":
                row_real[label] = f"{value:.1f}"
            else:
                row_real[label] = f"{value:.3f}"
        else:
            row_real[label] = "-"
    rows.append(row_real)

    # Simulation rows
    for sim_id, summary in sim_summaries.items():
        display_name = sim_id.replace("_simulation", "").replace("_", " ")
        row_sim = {"Dataset": display_name}

        for key, label in metric_labels:
            if key in summary:
                mean = summary[key]["mean"]
                std = summary[key]["std"]
                n = summary[key]["n"]

                # Use appropriate formatting based on metric type
                if key in ("total_actions", "total_users"):
                    if n > 1:
                        row_sim[label] = f"{int(mean):,} ± {int(std):,}"
                    else:
                        row_sim[label] = f"{int(mean):,}"
                elif key == "duration_days":
                    if n > 1:
                        row_sim[label] = f"{mean:.1f} ± {std:.1f}"
                    else:
                        row_sim[label] = f"{mean:.1f}"
                else:
                    if n > 1:
                        row_sim[label] = f"{mean:.3f} ± {std:.3f}"
                    else:
                        row_sim[label] = f"{mean:.3f}"
            else:
                row_sim[label] = "-"

        rows.append(row_sim)

    df = pd.DataFrame(rows)

    # Export CSV
    csv_path = output_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path.name}")

    # Export LaTeX
    tex_path = output_dir / "summary_table.tex"

    if latex_format == "booktabs":
        latex_str = _to_booktabs_latex(df)
    else:
        latex_str = df.to_latex(index=False, escape=True)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    logger.info(f"Saved: {tex_path.name}")

    return df, csv_path, tex_path


def _to_booktabs_latex(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to booktabs-style LaTeX table.

    Args:
        df: DataFrame to convert

    Returns:
        LaTeX string with booktabs formatting
    """
    # Escape LaTeX special characters
    def escape_latex(s):
        if not isinstance(s, str):
            return str(s)
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\^{}",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s

    # Build column spec
    n_cols = len(df.columns)
    col_spec = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of Real and Simulated Metrics}",
        r"\label{tab:comparison}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    headers = " & ".join(escape_latex(str(col)) for col in df.columns)
    lines.append(f"{headers} \\\\")
    lines.append(r"\midrule")

    # Data rows
    for _, row in df.iterrows():
        values = " & ".join(escape_latex(str(val)) for val in row.values)
        lines.append(f"{values} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
