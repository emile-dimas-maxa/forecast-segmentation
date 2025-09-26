"""Amount clipping transformation to filter out small transactions."""

import time

import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig


def clip_small_amounts(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """
    Clip amounts below threshold to zero.

    This transformation helps filter out very small transactions that might be
    noise in the data or represent minimal business impact.

    Args:
        df: Input DataFrame with 'amount' column
        config: Configuration object containing amount_clipping_threshold

    Returns:
        DataFrame with amounts below threshold set to 0
    """
    start_time = time.time()
    initial_rows = len(df)

    logger.debug("Starting amount clipping transformation")
    logger.debug("Input shape: {} rows × {} columns", initial_rows, len(df.columns))

    if config is None:
        config = SegmentationConfig()

    df = df.copy()

    # Check if amount column exists
    if "amount" not in df.columns:
        logger.warning("No 'amount' column found - skipping amount clipping")
        return df

    threshold = config.amount_clipping_threshold
    logger.debug("Using amount clipping threshold: {}", threshold)

    # Store original amounts before any modifications
    original_amounts = df["amount"].copy()
    original_amount_sum = df["amount"].sum()

    # Count transactions below threshold before clipping
    below_threshold_mask = df["amount"] < threshold
    transactions_below_threshold = below_threshold_mask.sum()
    total_amount_clipped = df.loc[below_threshold_mask, "amount"].sum()

    # Log statistics before clipping
    logger.debug(
        "Transactions below threshold {}: {} ({:.1f}%)",
        threshold,
        transactions_below_threshold,
        100 * transactions_below_threshold / len(df) if len(df) > 0 else 0,
    )
    logger.debug("Total amount to be clipped: {:.2f}", total_amount_clipped)

    # Perform the clipping
    df.loc[below_threshold_mask, "amount"] = 0.0
    new_amount_sum = df["amount"].sum()

    # Add clipping indicator columns for analysis
    df["was_amount_clipped"] = below_threshold_mask.astype(int)
    df["original_amount"] = original_amounts

    # Log results
    amount_reduction = original_amount_sum - new_amount_sum
    amount_reduction_pct = (amount_reduction / original_amount_sum * 100) if original_amount_sum > 0 else 0

    logger.debug("Amount clipping completed:")
    logger.debug(
        "  - Transactions clipped: {} ({:.1f}%)",
        transactions_below_threshold,
        100 * transactions_below_threshold / len(df) if len(df) > 0 else 0,
    )
    logger.debug("  - Total amount clipped: {:.2f} ({:.1f}%)", amount_reduction, amount_reduction_pct)
    logger.debug("  - Original total amount: {:.2f}", original_amount_sum)
    logger.debug("  - New total amount: {:.2f}", new_amount_sum)

    # Entity-level statistics
    entity_stats = (
        df.groupby("dim_value")
        .agg({"was_amount_clipped": "sum", "amount": "count"})
        .rename(columns={"amount": "total_transactions"})
    )
    entity_stats["clipping_rate"] = entity_stats["was_amount_clipped"] / entity_stats["total_transactions"] * 100

    # Log entity-level summary
    avg_clipping_rate = entity_stats["clipping_rate"].mean()
    max_clipping_rate = entity_stats["clipping_rate"].max()
    entities_with_clipping = (entity_stats["was_amount_clipped"] > 0).sum()

    logger.debug("Entity-level clipping statistics:")
    logger.debug("  - Entities affected: {} / {}", entities_with_clipping, len(entity_stats))
    logger.debug("  - Average clipping rate per entity: {:.1f}%", avg_clipping_rate)
    logger.debug("  - Maximum clipping rate per entity: {:.1f}%", max_clipping_rate)

    # Generate detailed impact analysis using the original data
    original_df = df.copy()
    original_df["amount"] = original_amounts  # Restore original amounts for analysis
    impact_analysis = analyze_clipping_impact(original_df, config)

    # Store impact analysis as metadata in the dataframe
    df.attrs["clipping_impact_analysis"] = impact_analysis
    df.attrs["clipping_threshold"] = threshold
    df.attrs["total_amount_clipped"] = amount_reduction
    df.attrs["clipping_rate_pct"] = (transactions_below_threshold / len(df) * 100) if len(df) > 0 else 0

    # Log detailed impact analysis
    if len(impact_analysis) > 0:
        logger.info("Detailed clipping impact analysis by entity:")
        for _, row in impact_analysis.iterrows():
            logger.info(
                "  - {}: {}/{} transactions clipped ({:.1f}%), amount reduced by {:.2f} ({:.1f}%)",
                row["dim_value"],
                row["transactions_below_threshold"],
                row["total_transactions"],
                row["clipping_rate_pct"],
                row["amount_below_threshold"],
                row["amount_loss_pct"],
            )

        # Log summary statistics
        total_entities = len(impact_analysis)
        entities_affected = (impact_analysis["transactions_below_threshold"] > 0).sum()
        avg_entity_loss = impact_analysis["amount_loss_pct"].mean()
        max_entity_loss = impact_analysis["amount_loss_pct"].max()

        logger.info(
            "Impact summary: {}/{} entities affected, avg {:.1f}% loss per entity, max {:.1f}% loss",
            entities_affected,
            total_entities,
            avg_entity_loss,
            max_entity_loss,
        )

    elapsed_time = time.time() - start_time
    logger.debug("Amount clipping completed in {:.2f}s - {} rows × {} columns", elapsed_time, len(df), len(df.columns))

    return df


def analyze_clipping_impact(df: pd.DataFrame, config: SegmentationConfig = None) -> pd.DataFrame:
    """
    Analyze the impact of amount clipping without actually performing it.

    This function provides insights into what would happen if clipping were applied,
    useful for threshold tuning and impact assessment.

    Args:
        df: Input DataFrame with 'amount' column
        config: Configuration object containing amount_clipping_threshold

    Returns:
        DataFrame with clipping impact analysis by entity
    """
    if config is None:
        config = SegmentationConfig()

    if "amount" not in df.columns:
        logger.warning("No 'amount' column found - cannot analyze clipping impact")
        return pd.DataFrame()

    threshold = config.amount_clipping_threshold

    # Analyze impact by entity
    impact_analysis = (
        df.groupby("dim_value")
        .agg({"amount": ["count", "sum", lambda x: (x < threshold).sum(), lambda x: x[x < threshold].sum()]})
        .round(2)
    )

    # Flatten column names
    impact_analysis.columns = ["total_transactions", "total_amount", "transactions_below_threshold", "amount_below_threshold"]

    # Calculate percentages
    impact_analysis["clipping_rate_pct"] = (
        impact_analysis["transactions_below_threshold"] / impact_analysis["total_transactions"] * 100
    ).round(1)
    impact_analysis["amount_loss_pct"] = (impact_analysis["amount_below_threshold"] / impact_analysis["total_amount"] * 100).round(
        1
    )

    # Calculate impact metrics
    impact_analysis["remaining_amount"] = impact_analysis["total_amount"] - impact_analysis["amount_below_threshold"]
    impact_analysis["threshold"] = threshold

    return impact_analysis.reset_index()
