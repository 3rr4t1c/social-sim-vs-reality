"""
Temporal activity calculation functions.

Based on user-provided script for calculating per-user and aggregate activity.
"""

from typing import Dict, Set, List, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_user_set(df: pd.DataFrame, include_orphaned: bool = False) -> Set[str]:
    """
    Extract the set of all users (active authors and optionally orphaned targets).

    Args:
        df: DataFrame containing action data
        include_orphaned: If True, includes users who only appear as targets
                         (reshared but original action is missing)

    Returns:
        Set of all user IDs
    """
    # Get active users (at least one action)
    active_users = df["author_id"].unique()
    active_users = set(active_users[~pd.isna(active_users)])

    if include_orphaned:
        # Get orphaned users (appear as target but never as author)
        orphaned_users = df["target_author_id"].dropna().unique()
        orphaned_users = set(orphaned_users)
        orphaned_users = orphaned_users - active_users
        return active_users.union(orphaned_users)

    return active_users


def get_temporal_activity_matrix(
    df: pd.DataFrame,
    window_size: str = "1D",
    include_orphaned: bool = False,
) -> pd.DataFrame:
    """
    Calculate temporal activity matrix for all users.

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Size of temporal window (e.g., '1D', '1W', '1H')
        include_orphaned: Include users who only appear as targets

    Returns:
        DataFrame with index=timestamps, columns=user_ids, values=action_counts
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Get complete user set
    all_users = get_user_set(df, include_orphaned=include_orphaned)

    # Define global timeline
    start_date = df["timestamp"].min().floor(window_size)
    end_date = df["timestamp"].max().floor(window_size)
    full_timeline = pd.date_range(start=start_date, end=end_date, freq=window_size)

    # Group and count
    grouped = df.groupby(
        [pd.Grouper(key="timestamp", freq=window_size), "author_id"]
    ).size()

    # Transform to matrix
    activity_matrix = grouped.unstack(level="author_id", fill_value=0)

    # Align to full timeline and all users
    activity_matrix = activity_matrix.reindex(index=full_timeline, fill_value=0)
    activity_matrix = activity_matrix.reindex(columns=list(all_users), fill_value=0)

    return activity_matrix


def calculate_mean_actions_per_user_active_only(
    df: pd.DataFrame,
    window_size: str = "1D",
) -> np.ndarray:
    """
    Calculate mean actions per user considering only timesteps when user is active.

    For each user: mean = total_actions / timesteps_with_actions

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Temporal window size

    Returns:
        Array of mean actions per user (one value per user)
    """
    if df.empty:
        return np.array([])

    activity_matrix = get_temporal_activity_matrix(
        df, window_size=window_size, include_orphaned=False
    )

    if activity_matrix.empty:
        return np.array([])

    means = []
    for user_id in activity_matrix.columns:
        user_activity = activity_matrix[user_id]
        active_timesteps = user_activity[user_activity > 0]

        if len(active_timesteps) > 0:
            mean_actions = active_timesteps.mean()
            means.append(mean_actions)

    return np.array(means)


def calculate_mean_actions_per_user_with_zeros(
    df: pd.DataFrame,
    window_size: str = "1D",
    include_orphaned: bool = False,
) -> np.ndarray:
    """
    Calculate mean actions per user including zero-activity timesteps.

    For each user: count starts from their first action until end of dataset.

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Temporal window size
        include_orphaned: Include users who only appear as targets

    Returns:
        Array of mean actions per user (one value per user)
    """
    if df.empty:
        return np.array([])

    activity_matrix = get_temporal_activity_matrix(
        df, window_size=window_size, include_orphaned=include_orphaned
    )

    if activity_matrix.empty:
        return np.array([])

    means = []
    for user_id in activity_matrix.columns:
        user_activity = activity_matrix[user_id]

        # Find first non-zero timestep for this user
        non_zero_indices = user_activity[user_activity > 0].index
        if len(non_zero_indices) == 0:
            # Orphaned user with no actions
            if include_orphaned:
                means.append(0.0)
            continue

        first_active_idx = non_zero_indices[0]

        # Get activity from first action onwards
        user_activity_from_first = user_activity.loc[first_active_idx:]

        if len(user_activity_from_first) > 0:
            mean_actions = user_activity_from_first.mean()
            means.append(mean_actions)

    return np.array(means)


def calculate_total_actions_per_timestep(
    df: pd.DataFrame,
    window_size: str = "1D",
) -> np.ndarray:
    """
    Calculate total actions in each timestep.

    Args:
        df: DataFrame with timestamp column
        window_size: Temporal window size

    Returns:
        Array of total actions per timestep
    """
    if df.empty:
        return np.array([])

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Group by timestep and count
    counts = df.groupby(pd.Grouper(key="timestamp", freq=window_size)).size()

    # Fill missing timesteps with 0
    start_date = df["timestamp"].min().floor(window_size)
    end_date = df["timestamp"].max().floor(window_size)
    full_timeline = pd.date_range(start=start_date, end=end_date, freq=window_size)

    counts = counts.reindex(full_timeline, fill_value=0)

    return counts.values


def calculate_active_users_per_timestep(
    df: pd.DataFrame,
    window_size: str = "1D",
) -> np.ndarray:
    """
    Calculate number of unique active users in each timestep.

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Temporal window size

    Returns:
        Array of active user counts per timestep
    """
    if df.empty:
        return np.array([])

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Group by timestep and count unique users
    counts = df.groupby(pd.Grouper(key="timestamp", freq=window_size))["author_id"].nunique()

    # Fill missing timesteps with 0
    start_date = df["timestamp"].min().floor(window_size)
    end_date = df["timestamp"].max().floor(window_size)
    full_timeline = pd.date_range(start=start_date, end=end_date, freq=window_size)

    counts = counts.reindex(full_timeline, fill_value=0)

    return counts.values


def calculate_activity_metrics(
    df: pd.DataFrame,
    window_size: str = "1D",
    include_orphaned: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Calculate all activity-related metrics for a dataset.

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Temporal window size
        include_orphaned: Include orphaned users in "with zeros" calculation

    Returns:
        Dictionary with:
            - mean_actions_active_only: Array of means (only active timesteps)
            - mean_actions_with_zeros: Array of means (including zeros from first action)
            - total_actions_per_timestep: Array of total actions per timestep
            - active_users_per_timestep: Array of active user counts per timestep
    """
    return {
        "mean_actions_active_only": calculate_mean_actions_per_user_active_only(
            df, window_size=window_size
        ),
        "mean_actions_with_zeros": calculate_mean_actions_per_user_with_zeros(
            df, window_size=window_size, include_orphaned=include_orphaned
        ),
        "total_actions_per_timestep": calculate_total_actions_per_timestep(
            df, window_size=window_size
        ),
        "active_users_per_timestep": calculate_active_users_per_timestep(
            df, window_size=window_size
        ),
    }


# =============================================================================
# INDEXED VERSIONS (for cross-run aggregation)
# =============================================================================

def calculate_mean_actions_per_user_active_only_indexed(
    df: pd.DataFrame,
    window_size: str = "1D",
) -> pd.Series:
    """
    Calculate mean actions per user (active only) with user_id index.

    Returns:
        Series with index=user_id, values=mean_actions
    """
    if df.empty:
        return pd.Series(dtype=float)

    activity_matrix = get_temporal_activity_matrix(
        df, window_size=window_size, include_orphaned=False
    )

    if activity_matrix.empty:
        return pd.Series(dtype=float)

    results = {}
    for user_id in activity_matrix.columns:
        user_activity = activity_matrix[user_id]
        active_timesteps = user_activity[user_activity > 0]

        if len(active_timesteps) > 0:
            results[user_id] = active_timesteps.mean()

    return pd.Series(results, dtype=float)


def calculate_mean_actions_per_user_with_zeros_indexed(
    df: pd.DataFrame,
    window_size: str = "1D",
    include_orphaned: bool = False,
) -> pd.Series:
    """
    Calculate mean actions per user (including zeros) with user_id index.

    Returns:
        Series with index=user_id, values=mean_actions
    """
    if df.empty:
        return pd.Series(dtype=float)

    activity_matrix = get_temporal_activity_matrix(
        df, window_size=window_size, include_orphaned=include_orphaned
    )

    if activity_matrix.empty:
        return pd.Series(dtype=float)

    results = {}
    for user_id in activity_matrix.columns:
        user_activity = activity_matrix[user_id]

        non_zero_indices = user_activity[user_activity > 0].index
        if len(non_zero_indices) == 0:
            if include_orphaned:
                results[user_id] = 0.0
            continue

        first_active_idx = non_zero_indices[0]
        user_activity_from_first = user_activity.loc[first_active_idx:]

        if len(user_activity_from_first) > 0:
            results[user_id] = user_activity_from_first.mean()

    return pd.Series(results, dtype=float)


def calculate_total_actions_per_timestep_indexed(
    df: pd.DataFrame,
    window_size: str = "1D",
    exclude_partial_days: bool = False,
) -> pd.Series:
    """
    Calculate total actions per timestep with timestamp index.

    Args:
        df: DataFrame with timestamp column
        window_size: Temporal window size
        exclude_partial_days: If True, exclude first and last timestep (often partial)

    Returns:
        Series with index=timestamp, values=action_count
    """
    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    counts = df.groupby(pd.Grouper(key="timestamp", freq=window_size)).size()

    start_date = df["timestamp"].min().floor(window_size)
    end_date = df["timestamp"].max().floor(window_size)
    full_timeline = pd.date_range(start=start_date, end=end_date, freq=window_size)

    counts = counts.reindex(full_timeline, fill_value=0)

    # Exclude first and last timestep if requested (often partial days)
    if exclude_partial_days and len(counts) > 2:
        counts = counts.iloc[1:-1]

    return counts.astype(float)


def calculate_active_users_per_timestep_indexed(
    df: pd.DataFrame,
    window_size: str = "1D",
    exclude_partial_days: bool = False,
) -> pd.Series:
    """
    Calculate active users per timestep with timestamp index.

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Temporal window size
        exclude_partial_days: If True, exclude first and last timestep (often partial)

    Returns:
        Series with index=timestamp, values=user_count
    """
    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    counts = df.groupby(pd.Grouper(key="timestamp", freq=window_size))["author_id"].nunique()

    start_date = df["timestamp"].min().floor(window_size)
    end_date = df["timestamp"].max().floor(window_size)
    full_timeline = pd.date_range(start=start_date, end=end_date, freq=window_size)

    counts = counts.reindex(full_timeline, fill_value=0)

    # Exclude first and last timestep if requested (often partial days)
    if exclude_partial_days and len(counts) > 2:
        counts = counts.iloc[1:-1]

    return counts.astype(float)


def calculate_activity_metrics_indexed(
    df: pd.DataFrame,
    window_size: str = "1D",
    include_orphaned: bool = False,
    exclude_partial_days: bool = False,
) -> Dict[str, pd.Series]:
    """
    Calculate all activity metrics with proper indexing for cross-run aggregation.

    Args:
        df: DataFrame with timestamp and author_id columns
        window_size: Temporal window size
        include_orphaned: Include orphaned users in "with zeros" calculation
        exclude_partial_days: Exclude first/last timestep from total activity metrics

    Returns:
        Dictionary of Series with proper indices (user_id or timestamp)
    """
    return {
        "mean_actions_active_only": calculate_mean_actions_per_user_active_only_indexed(
            df, window_size=window_size
        ),
        "mean_actions_with_zeros": calculate_mean_actions_per_user_with_zeros_indexed(
            df, window_size=window_size, include_orphaned=include_orphaned
        ),
        "total_actions_per_timestep": calculate_total_actions_per_timestep_indexed(
            df, window_size=window_size, exclude_partial_days=exclude_partial_days
        ),
        "active_users_per_timestep": calculate_active_users_per_timestep_indexed(
            df, window_size=window_size, exclude_partial_days=exclude_partial_days
        ),
    }


# =============================================================================
# POST VS RESHARE METRICS
# =============================================================================

def calculate_action_type_fractions(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fraction of posts and reshares.

    Args:
        df: DataFrame with action_type column

    Returns:
        Dictionary with post_fraction and reshare_fraction
    """
    if df.empty:
        return {"post_fraction": 0.0, "reshare_fraction": 0.0}

    total = len(df)
    posts = len(df[df["action_type"] == "post"])
    reshares = len(df[df["action_type"] == "reshare"])

    return {
        "post_fraction": posts / total if total > 0 else 0.0,
        "reshare_fraction": reshares / total if total > 0 else 0.0,
    }


def calculate_user_post_reshare_means(
    df: pd.DataFrame,
    window_size: str = "1D",
) -> pd.DataFrame:
    """
    Calculate mean posts and reshares per time unit for each user.

    Uses the temporal activity matrix to properly compute means from the first
    non-zero action for each user.

    For each user, computes:
    - mean_posts: average posts per time unit (from first action to end)
    - mean_reshares: average reshares per time unit (from first action to end)
    - mean_total_actions: average total actions per time unit (mean_posts + mean_reshares)

    Args:
        df: DataFrame with timestamp, author_id, action_type columns
        window_size: Temporal window size (default: daily)

    Returns:
        DataFrame with columns [user_id, mean_posts, mean_reshares, mean_total_actions]
    """
    if df.empty:
        return pd.DataFrame(columns=["user_id", "mean_posts", "mean_reshares", "mean_total_actions"])

    # Get activity matrix for ALL actions (to determine first action per user)
    all_activity_matrix = get_temporal_activity_matrix(
        df, window_size=window_size, include_orphaned=False
    )

    if all_activity_matrix.empty:
        return pd.DataFrame(columns=["user_id", "mean_posts", "mean_reshares", "mean_total_actions"])

    # Split by action type and get separate matrices
    posts_df = df[df["action_type"] == "post"]
    reshares_df = df[df["action_type"] == "reshare"]

    posts_matrix = get_temporal_activity_matrix(
        posts_df, window_size=window_size, include_orphaned=False
    ) if not posts_df.empty else pd.DataFrame()

    reshares_matrix = get_temporal_activity_matrix(
        reshares_df, window_size=window_size, include_orphaned=False
    ) if not reshares_df.empty else pd.DataFrame()

    results = []

    for user_id in all_activity_matrix.columns:
        user_total_activity = all_activity_matrix[user_id]

        # Find first non-zero index (first action)
        non_zero_mask = user_total_activity > 0
        if not non_zero_mask.any():
            continue

        first_action_idx = user_total_activity[non_zero_mask].index[0]

        # Count timesteps from first action to end
        active_period = user_total_activity.loc[first_action_idx:]
        n_timesteps = len(active_period)

        if n_timesteps == 0:
            n_timesteps = 1

        # Get posts in active period
        if not posts_matrix.empty and user_id in posts_matrix.columns:
            user_posts = posts_matrix[user_id].loc[first_action_idx:] if first_action_idx in posts_matrix.index else posts_matrix[user_id]
            total_posts = user_posts.sum()
        else:
            total_posts = 0

        # Get reshares in active period
        if not reshares_matrix.empty and user_id in reshares_matrix.columns:
            user_reshares = reshares_matrix[user_id].loc[first_action_idx:] if first_action_idx in reshares_matrix.index else reshares_matrix[user_id]
            total_reshares = user_reshares.sum()
        else:
            total_reshares = 0

        mean_posts = total_posts / n_timesteps
        mean_reshares = total_reshares / n_timesteps

        results.append({
            "user_id": user_id,
            "mean_posts": mean_posts,
            "mean_reshares": mean_reshares,
            "mean_total_actions": mean_posts + mean_reshares,
        })

    return pd.DataFrame(results)
