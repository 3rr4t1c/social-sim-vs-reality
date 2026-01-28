"""
Main validation pipeline orchestration.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import logging

from .config import Config, load_config
from .data import (
    load_real_data,
    load_simulation_data,
    discover_simulations,
    convert_simulator_output,
    auto_convert_all_simulations,
)
from .data.loader import invalidate_cache_for_simulations
from .metrics import calculate_metrics, aggregate_simulation_metrics
from .metrics.aggregator import summarize_aggregated_metrics
from .visualization import (
    setup_matplotlib,
    plot_temporal_comparison,
    plot_activity_cdf,
    plot_cascade_distribution,
    plot_inter_event_time,
    plot_response_time_boxplot,
    plot_quality_kde,
    plot_user_activity_distribution,
    plot_total_activity_distribution,
)
from .export import generate_latex_table, export_metrics_json

logger = logging.getLogger(__name__)


def list_plots() -> None:
    """Print available plot types."""
    print("\nAvailable plots:")
    print("  temporal        - Temporal activity comparison with CI")
    print("  user_activity   - User activity distribution (4 panels)")
    print("  total_activity  - Total activity distribution (4 panels)")
    print("  iet             - Inter-event time distribution")
    print("  response_time   - Response time boxplot")
    print("  quality         - Quality/extra feature KDE")
    print("\nDeprecated (kept for compatibility):")
    print("  activity_cdf    - [Use user_activity instead]")
    print("  cascade         - [Removed from analysis]")
    print("\nUse: python analyze.py --plots temporal user_activity total_activity")


def run_conversion(
    input_dir: str,
    output_dir: str,
    verbose: bool = False,
) -> int:
    """
    Run data conversion from simulator format.

    Args:
        input_dir: Directory with raw simulator output
        output_dir: Directory for converted files
        verbose: Print progress

    Returns:
        Number of files converted
    """
    config = load_config()

    return convert_simulator_output(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        start_date=config.converter_start_date,
        quality_scale=config.converter_quality_scale,
        verbose=verbose,
    )


def run_validation(
    plots: Optional[List[str]] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> None:
    """
    Run the complete validation pipeline.

    Args:
        plots: List of specific plots to generate (None = use settings.py)
        use_cache: Whether to use caching
        verbose: Enable verbose logging
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = load_config()
    config.use_caching = use_cache

    # Setup matplotlib
    setup_matplotlib(
        dpi=config.dpi,
        save_dpi=config.save_dpi,
        grid_alpha=config.grid_alpha,
    )

    logger.info("=" * 70)
    logger.info("SOCIAL MEDIA SIMULATOR VALIDATION")
    logger.info("=" * 70)

    # =========================================================================
    # 1. LOAD REAL DATA
    # =========================================================================
    logger.info("\n[1/4] Loading real data...")

    try:
        real_df = load_real_data(
            config.real_data_dir,
            pattern=config.real_data_pattern,
            cache_dir=config.cache_dir if use_cache else None,
            use_cache=use_cache,
        )

        real_metrics = calculate_metrics(
            real_df,
            dataset_name="Real Data",
            rolling_window_days=config.rolling_window_days,
            time_binning=config.time_binning,
            min_active_users=config.min_active_users,
            percentiles=config.percentiles,
            top_users_percent=config.top_users_percent,
            response_time_max_hours=config.response_time_max_hours,
            analyze_extra=config.analyze_extra_features,
            include_orphaned=config.include_orphaned_users,
            exclude_partial_days=config.exclude_partial_days,
        )

        # Save metrics
        if config.save_json_metrics:
            export_metrics_json(
                real_metrics,
                config.metrics_dir / "real_metrics.json",
            )

        logger.info(f"  Real data: {len(real_df):,} actions")

    except FileNotFoundError as e:
        logger.error(f"CRITICAL: {e}")
        return

    # =========================================================================
    # 2. AUTO-CONVERT RAW FILES & LOAD SIMULATIONS
    # =========================================================================
    logger.info("\n[2/4] Processing simulations...")

    # Auto-convert any raw simulator files found in simulation folders
    converted_count, affected_sims = auto_convert_all_simulations(
        config.synthetic_data_dir,
        pattern=config.synthetic_dir_pattern,
        start_date=config.converter_start_date,
        quality_scale=config.converter_quality_scale,
    )

    # Invalidate cache for simulations that had new files converted
    if converted_count > 0 and use_cache and config.cache_dir:
        invalidate_cache_for_simulations(config.cache_dir, affected_sims)

    sim_metrics_agg: Dict[str, Dict[str, List[Any]]] = {}
    sim_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    simulations = discover_simulations(
        config.synthetic_data_dir,
        pattern=config.synthetic_dir_pattern,
    )

    for sim_name, sim_dir in simulations:
        logger.info(f"\n  Processing {sim_name}...")

        runs = load_simulation_data(
            sim_dir,
            sim_name,
            cache_dir=config.cache_dir if use_cache else None,
            use_cache=use_cache,
        )

        if not runs:
            continue

        # Calculate metrics for each run
        runs_metrics = []
        for run_name, run_df in runs.items():
            try:
                run_metrics = calculate_metrics(
                    run_df,
                    dataset_name=f"{sim_name}/{run_name}",
                    rolling_window_days=config.rolling_window_days,
                    time_binning=config.time_binning,
                    min_active_users=config.min_active_users,
                    percentiles=config.percentiles,
                    top_users_percent=config.top_users_percent,
                    response_time_max_hours=config.response_time_max_hours,
                    analyze_extra=config.analyze_extra_features,
                    include_orphaned=config.include_orphaned_users,
                    exclude_partial_days=config.exclude_partial_days,
                )
                runs_metrics.append(run_metrics)
                logger.info(f"    {run_name}: OK")
            except Exception as e:
                logger.error(f"    {run_name}: FAILED - {e}")

        if runs_metrics:
            # Aggregate metrics
            aggregated = aggregate_simulation_metrics(runs_metrics)
            sim_metrics_agg[sim_name] = aggregated

            # Summarize for table
            sim_summaries[sim_name] = summarize_aggregated_metrics(aggregated)

            # Save summary
            if config.save_json_metrics:
                export_metrics_json(
                    {k: v for k, v in sim_summaries[sim_name].items()},
                    config.metrics_dir / f"{sim_name}_summary.json",
                    exclude_arrays=False,
                )

            logger.info(f"  {sim_name}: {len(runs_metrics)} run(s) processed")

    # =========================================================================
    # 3. GENERATE VISUALIZATIONS
    # =========================================================================
    logger.info("\n[3/4] Generating visualizations...")

    # Determine which plots to generate
    if plots:
        # Map short names to full names
        plot_mapping = {
            "temporal": "temporal_comparison",
            "user_activity": "user_activity_distribution",
            "total_activity": "total_activity_distribution",
            "activity_cdf": "activity_cdf",
            "cdf": "activity_cdf",
            "cascade": "cascade_distribution",
            "iet": "inter_event_time",
            "response_time": "response_time_boxplot",
            "quality": "quality_kde",
        }
        plots_to_generate = {
            plot_mapping.get(p, p): True for p in plots
        }
    else:
        plots_to_generate = config.plots

    plot_kwargs = dict(
        real_metrics=real_metrics,
        sim_metrics_agg=sim_metrics_agg,
        output_dir=config.figures_dir,
        figure_size=config.figure_size,
        color_real=config.color_real,
        color_palette=config.color_palette,
    )

    if plots_to_generate.get("temporal_comparison"):
        plot_temporal_comparison(
            **plot_kwargs,
            confidence_alpha=config.confidence_alpha,
            grid_alpha=config.grid_alpha,
            rolling_window_days=config.rolling_window_days,
            output_format=config.format_temporal,
        )

    if plots_to_generate.get("user_activity_distribution"):
        plot_user_activity_distribution(
            **plot_kwargs,
            output_format=config.format_distribution,
            use_log_scale=config.use_log_scale_user_activity,
            time_binning=config.time_binning,
        )

    if plots_to_generate.get("total_activity_distribution"):
        plot_total_activity_distribution(
            **plot_kwargs,
            output_format=config.format_distribution,
            use_log_scale=config.use_log_scale_total_activity,
            time_binning=config.time_binning,
        )

    if plots_to_generate.get("activity_cdf"):
        plot_activity_cdf(
            **plot_kwargs,
            output_format=config.format_distribution,
        )

    if plots_to_generate.get("cascade_distribution"):
        plot_cascade_distribution(
            **plot_kwargs,
            output_format=config.format_distribution,
        )

    if plots_to_generate.get("inter_event_time"):
        plot_inter_event_time(
            **plot_kwargs,
            output_format=config.format_distribution,
        )

    if plots_to_generate.get("response_time_boxplot"):
        plot_response_time_boxplot(
            **plot_kwargs,
            output_format=config.format_distribution,
        )

    if plots_to_generate.get("quality_kde"):
        plot_quality_kde(
            **plot_kwargs,
            output_format=config.format_distribution,
        )

    # =========================================================================
    # 4. GENERATE TABLES
    # =========================================================================
    logger.info("\n[4/4] Generating summary table...")

    if config.save_tables:
        df_summary, csv_path, tex_path = generate_latex_table(
            real_metrics=real_metrics,
            sim_summaries=sim_summaries,
            output_dir=config.tables_dir,
            latex_format=config.latex_format,
        )

        # Print table
        print("\n" + "=" * 70)
        print("SUMMARY TABLE")
        print("=" * 70)
        print(df_summary.to_string(index=False))
        print("=" * 70)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Real data: {len(real_df):,} actions")
    logger.info(f"Simulations analyzed: {len(sim_metrics_agg)}")

    for sim_name in sim_metrics_agg:
        n_runs = len(sim_metrics_agg[sim_name].get("orphan_rate", []))
        logger.info(f"  - {sim_name}: {n_runs} run(s)")

    logger.info(f"\nOutputs saved to: {config.output_dir}")
    logger.info(f"  - Figures: {config.figures_dir}")
    logger.info(f"  - Tables: {config.tables_dir}")
    logger.info(f"  - Metrics: {config.metrics_dir}")
