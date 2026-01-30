"""
Metrics calculation for social media datasets.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import json
import logging

from .activity import (
    calculate_activity_metrics,
    calculate_activity_metrics_indexed,
    calculate_action_type_fractions,
    calculate_user_post_reshare_means,
)

logger = logging.getLogger(__name__)


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient (measure of inequality).

    Gini = 0: perfect equality
    Gini = 1: maximum inequality

    Args:
        values: Array of values (e.g., action counts per user)

    Returns:
        Gini coefficient in [0, 1]
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)

    total = np.sum(sorted_values)
    if total == 0:
        return 0.0

    return float(np.sum((2 * index - n - 1) * sorted_values) / (n * total))


def calculate_metrics(
    df: pd.DataFrame,
    dataset_name: str,
    rolling_window_days: int = 7,
    time_binning: str = "D",
    min_active_users: int = 5,
    percentiles: List[int] = None,
    top_users_percent: int = 10,
    response_time_max_hours: int = 168,
    analyze_extra: bool = True,
    include_orphaned: bool = False,
    exclude_partial_days: bool = False,
    scatter_active_only: bool = True,
) -> Dict[str, Any]:
    """
    Calculate validation metrics for a dataset.

    Metrics calculated:
        - Basic stats: total_actions, total_users, duration_days
        - orphan_rate: Percentage of broken target references
        - rolling_actions_per_user: Time series of activity
        - gini_coefficient: Activity inequality
        - burstiness: Temporal pattern regularity
        - response_time_*: Response time statistics
        - reshare_ratio: Percentage of actions that are reshares
        - unique_targets_ratio: Percentage of posts that get reshared
        - actions_per_user_*: Median and percentiles
        - active_user_ratio: Percentage of users who are active
        - Activity distributions for graphing

    Args:
        df: Preprocessed DataFrame
        dataset_name: Identifier for logging
        rolling_window_days: Window for rolling average
        time_binning: Pandas resample frequency ("D", "H", "W")
        min_active_users: Minimum users per time bin
        percentiles: Percentiles to calculate
        top_users_percent: Percentage for top user contribution
        response_time_max_hours: Max valid response time
        analyze_extra: Whether to analyze extra feature
        include_orphaned: Include orphaned users in calculations
        exclude_partial_days: Exclude first/last day from total activity distributions

    Returns:
        Dictionary of calculated metrics
    """
    if percentiles is None:
        percentiles = [25, 50, 75, 90, 95, 99]

    metrics: Dict[str, Any] = {}

    # =========================================================================
    # BASIC STATS
    # =========================================================================
    metrics["total_actions"] = len(df)
    metrics["total_users"] = df["author_id"].nunique()

    # Duration in days
    if not df.empty:
        duration = df["timestamp"].max() - df["timestamp"].min()
        metrics["duration_days"] = float(duration.total_seconds() / 86400)
    else:
        metrics["duration_days"] = 0.0

    # =========================================================================
    # ORPHAN RATE
    # =========================================================================
    all_action_ids = set(df["action_id"].unique())
    referenced_targets = set(
        df.loc[df["target_action_id"] != "", "target_action_id"].unique()
    )
    orphans = referenced_targets - all_action_ids

    metrics["orphan_rate"] = (
        len(orphans) / len(referenced_targets) if referenced_targets else 0.0
    )

    # =========================================================================
    # RESHARE METRICS
    # =========================================================================
    reshares = df[df["action_type"] == "reshare"]
    posts = df[df["action_type"] == "post"]

    metrics["reshare_ratio"] = len(reshares) / len(df) if len(df) > 0 else 0.0

    # Unique targets ratio: % of posts that receive at least 1 reshare
    if len(posts) > 0:
        posts_with_reshares = df[df["target_action_id"] != ""]["target_action_id"].nunique()
        metrics["unique_targets_ratio"] = posts_with_reshares / len(posts)
    else:
        metrics["unique_targets_ratio"] = 0.0

    # Post vs Reshare fractions
    fractions = calculate_action_type_fractions(df)
    metrics["post_fraction"] = fractions["post_fraction"]
    metrics["reshare_fraction"] = fractions["reshare_fraction"]

    # =========================================================================
    # TEMPORAL METRICS
    # =========================================================================
    df_indexed = df.set_index("timestamp")
    df_resampled = df_indexed.resample(time_binning)

    actions_per_bin = df_resampled.size()
    active_users_per_bin = df_resampled["author_id"].nunique()

    valid_bins = active_users_per_bin >= min_active_users

    if valid_bins.any():
        actions_per_user = actions_per_bin / active_users_per_bin
        apu_valid = actions_per_user[valid_bins]

        # Rolling average
        rolling_apu = apu_valid.rolling(window=rolling_window_days, min_periods=1).mean()
        metrics["rolling_actions_per_user"] = rolling_apu
        metrics["actions_per_user_mean"] = float(apu_valid.mean())

        # Active user ratio
        total_users = df["author_id"].nunique()
        active_ratio = active_users_per_bin / total_users
        metrics["active_users_ratio_mean"] = float(active_ratio.mean())

    # =========================================================================
    # RESPONSE TIME
    # =========================================================================
    posts_lookup = df[["action_id", "timestamp"]].rename(
        columns={"timestamp": "post_ts", "action_id": "target_id"}
    )

    interactions = df[df["target_action_id"] != ""].merge(
        posts_lookup, left_on="target_action_id", right_on="target_id", how="inner"
    )

    if not interactions.empty:
        time_diff = interactions["timestamp"] - interactions["post_ts"]
        interactions["response_time_hours"] = time_diff.dt.total_seconds() / 3600

        valid_responses = interactions[
            (interactions["response_time_hours"] >= 0)
            & (interactions["response_time_hours"] <= response_time_max_hours)
        ]["response_time_hours"]

        if not valid_responses.empty:
            metrics["response_time_mean"] = float(valid_responses.mean())
            metrics["response_time_median"] = float(valid_responses.median())
            metrics["response_time_array"] = valid_responses.values

            for p in percentiles:
                metrics[f"response_time_p{p}"] = float(np.percentile(valid_responses, p))

    # =========================================================================
    # USER BEHAVIOR (Activity Distribution)
    # =========================================================================
    user_counts = df["author_id"].value_counts()
    metrics["activity_per_user"] = user_counts.values

    # Gini coefficient
    metrics["gini_coefficient"] = calculate_gini_coefficient(user_counts.values)

    # Actions per user stats
    metrics["actions_per_user_median"] = float(user_counts.median())
    metrics["actions_per_user_p90"] = float(np.percentile(user_counts.values, 90))

    # Top users contribution
    top_n = max(1, int(len(user_counts) * (top_users_percent / 100)))
    top_contribution = user_counts.iloc[:top_n].sum() / user_counts.sum()
    metrics[f"top_{top_users_percent}pct_contribution"] = float(top_contribution)

    # Active user ratio (users with at least 1 action / total unique users in dataset)
    # Note: in real data, all users have at least 1 action by definition
    # This metric is more meaningful when comparing with orphaned users
    metrics["active_user_ratio"] = 1.0  # All users in the dataset are active

    # =========================================================================
    # BURSTINESS (Inter-Event Times)
    # =========================================================================
    # Sample for performance
    if len(user_counts) > 50_000:
        sample_users = user_counts.sample(n=50_000, random_state=42).index
        df_sample = df[df["author_id"].isin(sample_users)].copy()
    else:
        df_sample = df.copy()

    df_sorted = df_sample.sort_values(["author_id", "timestamp"])
    df_sorted["prev_timestamp"] = df_sorted.groupby("author_id")["timestamp"].shift(1)

    time_diff = df_sorted["timestamp"] - df_sorted["prev_timestamp"]
    iet_hours = (time_diff.dt.total_seconds() / 3600).dropna()

    if not iet_hours.empty:
        metrics["inter_event_times"] = iet_hours.values
        mean_iet = iet_hours.mean()
        std_iet = iet_hours.std()

        if (std_iet + mean_iet) > 0:
            burstiness = (std_iet - mean_iet) / (std_iet + mean_iet)
            metrics["burstiness"] = float(burstiness)

        for p in [50, 75, 90]:
            metrics[f"inter_event_time_p{p}"] = float(np.percentile(iet_hours, p))

    # =========================================================================
    # ACTIVITY DISTRIBUTIONS (for new graphs)
    # =========================================================================
    # Non-indexed versions (for backward compatibility)
    activity_metrics = calculate_activity_metrics(
        df,
        window_size=time_binning,
        include_orphaned=include_orphaned,
    )
    metrics.update(activity_metrics)

    # Indexed versions (for cross-run aggregation in visualizations)
    activity_metrics_indexed = calculate_activity_metrics_indexed(
        df,
        window_size=time_binning,
        include_orphaned=include_orphaned,
        exclude_partial_days=exclude_partial_days,
    )
    metrics["mean_actions_active_only_indexed"] = activity_metrics_indexed["mean_actions_active_only"]
    metrics["mean_actions_with_zeros_indexed"] = activity_metrics_indexed["mean_actions_with_zeros"]
    metrics["total_actions_per_timestep_indexed"] = activity_metrics_indexed["total_actions_per_timestep"]
    metrics["active_users_per_timestep_indexed"] = activity_metrics_indexed["active_users_per_timestep"]

    # =========================================================================
    # USER POST VS RESHARE MEANS (for scatter plot)
    # =========================================================================
    user_post_reshare = calculate_user_post_reshare_means(
        df, window_size=time_binning, active_only=scatter_active_only
    )
    metrics["user_post_reshare_means"] = user_post_reshare

    # =========================================================================
    # EXTRA FEATURES
    # =========================================================================
    if analyze_extra and "extra" in df.columns:
        _calculate_extra_metrics(df, metrics, dataset_name)

    logger.info(f"Calculated metrics for {dataset_name}")
    return metrics


def _calculate_extra_metrics(
    df: pd.DataFrame, metrics: Dict[str, Any], dataset_name: str
) -> None:
    """Calculate metrics for the 'extra' feature column."""

    def parse_extra(value):
        if pd.isna(value) or value == "":
            return None
        try:
            if isinstance(value, str):
                parsed = json.loads(value)
            else:
                parsed = value

            if isinstance(parsed, (list, tuple)):
                return np.array(parsed, dtype=float)
            else:
                return np.array([float(parsed)], dtype=float)
        except (ValueError, TypeError, json.JSONDecodeError):
            try:
                return np.array([float(value)], dtype=float)
            except (ValueError, TypeError):
                return None

    extras_parsed = df["extra"].apply(parse_extra).dropna()

    if extras_parsed.empty:
        logger.debug(f"No valid extra values in {dataset_name}")
        return

    try:
        extras_matrix = np.vstack(extras_parsed.values)
        n_samples, n_features = extras_matrix.shape

        metrics["extra_distribution"] = extras_matrix

        if n_features == 1:
            metrics["extra_mean"] = float(np.mean(extras_matrix[:, 0]))
            metrics["extra_std"] = float(np.std(extras_matrix[:, 0]))
        else:
            metrics["extra_n_features"] = n_features
            for i in range(n_features):
                metrics[f"extra_f{i}_mean"] = float(np.mean(extras_matrix[:, i]))
                metrics[f"extra_f{i}_std"] = float(np.std(extras_matrix[:, i]))

        logger.debug(f"Extra features in {dataset_name}: {n_features}D, {n_samples} samples")

    except ValueError:
        all_values = np.concatenate(extras_parsed.values)
        metrics["extra_distribution"] = all_values
        metrics["extra_mean"] = float(np.mean(all_values))
        metrics["extra_std"] = float(np.std(all_values))
        logger.warning(f"Flattened extra features in {dataset_name}")
