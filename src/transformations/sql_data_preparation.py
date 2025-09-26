"""SQL-based data preparation for large transaction tables.

This module provides SQL-based implementations of the first 3 pipeline steps:
1. Base data preparation
2. Amount clipping
3. Monthly aggregation

This is much more efficient than pandas for large transaction tables.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.segmentation import SegmentationConfig


def create_sql_data_preparation_query(config: SegmentationConfig, source_table: str = "source_data") -> tuple[str, dict]:
    """
    Create SQL query for consolidated data preparation steps 1-3 with parameterized values.

    Based on the segmentation.sql logic but consolidated into a single query
    that performs base preparation, amount clipping, and monthly aggregation.

    Args:
        config: Configuration object with parameters
        source_table: Name/alias of the source table to query from

    Returns:
        Tuple of (SQL query string with placeholders, parameters dict)
    """

    # Parameters to prevent SQL injection
    # Format dates as SQL-compatible strings
    params = {
        "start_date": f"'{config.start_date.isoformat()}'",
        "end_date": f"'{config.end_date.isoformat()}'",
        "amount_clipping_threshold": config.amount_clipping_threshold,
        "early_month_days": 10,
        "mid_month_end_day": 20,
        "pre_eom_days": 5,
    }

    sql_query = f"""
    -- =====================================================
    -- EOM FORECASTING DATA PREPARATION WITH AMOUNT CLIPPING
    -- Consolidated SQL for Steps 1-3: Base Preparation + Amount Clipping + Monthly Aggregation
    -- =====================================================

    WITH config AS (
        SELECT 
            -- Date range parameters
            %(start_date)s::DATE AS start_date,
            %(end_date)s::DATE AS end_date,
            
            -- Amount clipping threshold
            %(amount_clipping_threshold)s AS amount_clipping_threshold,
            
            -- Calendar day thresholds for pattern detection
            %(early_month_days)s AS early_month_days,       -- First N days of month for early month signal
            %(mid_month_end_day)s AS mid_month_end_day,      -- End day for mid-month period (10-20)
            %(pre_eom_days)s AS pre_eom_days             -- Days before EOM to consider for pre-EOM signals
    )
    
    -- =====================================================
    -- STEP 1: BASE DATA PREPARATION WITH AMOUNT CLIPPING
    -- =====================================================
    
    , base_data_with_clipping AS (
        SELECT 
            dim_value,
            date,
            -- Store original amount for impact analysis
            amount AS original_amount,
            -- Apply amount clipping: set amounts below threshold to 0
            CASE 
                WHEN amount < c.amount_clipping_threshold THEN 0.0
                ELSE amount 
            END AS amount,
            -- Add clipping indicator
            CASE 
                WHEN amount < c.amount_clipping_threshold THEN 1
                ELSE 0 
            END AS was_amount_clipped,
            is_last_work_day_of_month,
            -- Add month and quarter identifiers
            DATE_TRUNC('month', date) AS month,
            DATE_TRUNC('quarter', date) AS quarter,
            EXTRACT(YEAR FROM date) AS year,
            EXTRACT(MONTH FROM date) AS month_num,
            EXTRACT(QUARTER FROM date) AS quarter_num,
            -- Add day position in month
            EXTRACT(DAY FROM date) AS day_of_month,
            -- Business days from EOM (negative = before EOM)
            CASE 
                WHEN is_last_work_day_of_month = TRUE THEN 0
                ELSE -ROW_NUMBER() OVER (
                    PARTITION BY dim_value, DATE_TRUNC('month', date) 
                    ORDER BY date DESC
                )
            END AS days_from_eom
        FROM 
            {source_table} ts
        CROSS JOIN 
            config c
        WHERE 
            ts.date >= c.start_date
            AND ts.date <= c.end_date
    )
    
    -- =====================================================
    -- STEP 2: AMOUNT CLIPPING IMPACT ANALYSIS
    -- =====================================================
    
    , clipping_impact_by_entity AS (
        SELECT 
            dim_value,
            COUNT(*) AS total_transactions,
            SUM(was_amount_clipped) AS transactions_below_threshold,
            SUM(original_amount) AS total_amount,
            SUM(original_amount - amount) AS amount_below_threshold,
            ROUND(SUM(was_amount_clipped)::FLOAT / COUNT(*) * 100, 1) AS clipping_rate_pct,
            ROUND((SUM(original_amount - amount) / NULLIF(SUM(original_amount), 0)) * 100, 1) AS amount_loss_pct,
            SUM(amount) AS remaining_amount,
            (SELECT amount_clipping_threshold FROM config) AS threshold
        FROM 
            base_data_with_clipping
        GROUP BY 
            dim_value
    )
    
    , clipping_impact_summary AS (
        SELECT 
            COUNT(*) AS total_transactions,
            SUM(was_amount_clipped) AS transactions_clipped,
            ROUND(SUM(was_amount_clipped)::FLOAT / COUNT(*) * 100, 1) AS clipping_rate_pct,
            SUM(original_amount) AS total_original_amount,
            SUM(amount) AS total_clipped_amount,
            SUM(original_amount - amount) AS total_amount_clipped,
            ROUND((SUM(original_amount - amount) / NULLIF(SUM(original_amount), 0)) * 100, 1) AS amount_reduction_pct,
            COUNT(DISTINCT dim_value) AS total_entities,
            COUNT(DISTINCT CASE WHEN was_amount_clipped = 1 THEN dim_value END) AS entities_affected
        FROM 
            base_data_with_clipping
    )
    
    -- =====================================================
    -- STEP 3: MONTHLY AGGREGATIONS WITH CLIPPING METRICS
    -- =====================================================
    
    , monthly_aggregates AS (
        SELECT 
            dim_value,
            month,
            year,
            month_num,
            
            -- Total monthly amounts (after clipping)
            COALESCE(SUM(amount), 0) AS monthly_total,
            COUNT(CASE WHEN amount <> 0 THEN 1 ELSE NULL END) AS monthly_transactions,
            COALESCE(AVG(CASE WHEN amount <> 0 THEN amount END), 0) AS monthly_avg_amount,
            COALESCE(STDDEV_POP(CASE WHEN amount <> 0 THEN amount END), 0) AS monthly_std_amount,
            
            -- EOM specific amounts (after clipping)
            COALESCE(SUM(CASE WHEN is_last_work_day_of_month THEN amount ELSE 0 END), 0) AS eom_amount,
            COUNT(CASE WHEN is_last_work_day_of_month AND amount <> 0 THEN 1 ELSE NULL END) AS eom_transaction_count,
            
            -- Non-EOM amounts (after clipping)
            COALESCE(SUM(CASE WHEN NOT is_last_work_day_of_month THEN amount ELSE 0 END), 0) AS non_eom_total,
            COUNT(CASE WHEN NOT is_last_work_day_of_month AND amount <> 0 THEN 1 ELSE NULL END) AS non_eom_transactions,
            COALESCE(AVG(CASE WHEN NOT is_last_work_day_of_month AND amount <> 0 THEN amount END), 0) AS non_eom_avg,
            
            -- Pre-EOM signals (after clipping)
            COALESCE(SUM(CASE WHEN days_from_eom BETWEEN -(SELECT pre_eom_days FROM config) AND -1 THEN amount ELSE 0 END), 0) AS pre_eom_5d_total,
            COUNT(CASE WHEN days_from_eom BETWEEN -(SELECT pre_eom_days FROM config) AND -1 AND amount <> 0 THEN 1 ELSE NULL END) AS pre_eom_5d_count,
            
            -- Early month signal (after clipping)
            COALESCE(SUM(CASE WHEN day_of_month <= (SELECT early_month_days FROM config) THEN amount ELSE 0 END), 0) AS early_month_total,
            
            -- Mid month signal (after clipping)
            COALESCE(SUM(CASE WHEN day_of_month BETWEEN (SELECT early_month_days FROM config) AND (SELECT mid_month_end_day FROM config) THEN amount ELSE 0 END), 0) AS mid_month_total,
            
            -- Maximum single transaction (after clipping)
            COALESCE(MAX(amount), 0) AS max_monthly_transaction,
            
            -- Transaction distribution within month
            COALESCE(STDDEV_POP(day_of_month), 0) AS day_dispersion,
            
            -- Calendar indicators
            MAX(CASE WHEN month_num IN (3, 6, 9, 12) THEN 1 ELSE 0 END) AS is_quarter_end,
            MAX(CASE WHEN month_num = 12 THEN 1 ELSE 0 END) AS is_year_end,
            
            -- EOM activity flag (after clipping)
            MAX(CASE WHEN is_last_work_day_of_month AND amount > 0 THEN 1 ELSE 0 END) AS has_nonzero_eom,
            
            -- Quarterly totals for seasonality detection (after clipping)
            SUM(CASE WHEN month_num IN (3, 6, 9, 12) THEN amount ELSE 0 END) AS quarter_end_amount,
            SUM(CASE WHEN month_num = 12 THEN amount ELSE 0 END) AS year_end_amount,
            
            -- Amount clipping metrics per entity per month
            SUM(was_amount_clipped) AS monthly_transactions_clipped,
            COUNT(*) AS monthly_total_transactions,
            ROUND(SUM(was_amount_clipped)::FLOAT / COUNT(*) * 100, 1) AS monthly_clipping_rate_pct,
            SUM(original_amount) AS monthly_original_amount,
            SUM(original_amount - amount) AS monthly_amount_clipped,
            CASE 
                WHEN SUM(original_amount) > 0 
                THEN ROUND((SUM(original_amount - amount) / SUM(original_amount)) * 100, 1)
                ELSE 0.0 
            END AS monthly_amount_loss_pct
            
        FROM 
            base_data_with_clipping
        GROUP BY 
            dim_value, month, year, month_num
    )
    
    -- =====================================================
    -- FINAL OUTPUT: MONTHLY AGGREGATES + CLIPPING IMPACT
    -- =====================================================
    
    SELECT 
        ma.*,
        -- Add clipping impact metadata for each entity
        ci.total_transactions AS entity_total_transactions,
        ci.transactions_below_threshold AS entity_transactions_clipped,
        ci.clipping_rate_pct AS entity_clipping_rate_pct,
        ci.amount_loss_pct AS entity_amount_loss_pct,
        ci.threshold AS clipping_threshold_used
    FROM 
        monthly_aggregates ma
    LEFT JOIN 
        clipping_impact_by_entity ci ON ma.dim_value = ci.dim_value
    ORDER BY 
        ma.dim_value, ma.month
    """

    return sql_query, params


def execute_sql_data_preparation_snowflake(
    config: SegmentationConfig, source_table: str = "source_data"
) -> tuple[pd.DataFrame, dict]:
    """
    Execute SQL-based data preparation steps 1-3 using Snowflake connection.

    This function creates and executes a parameterized SQL query against Snowflake
    to perform data preparation, amount clipping, and monthly aggregation.

    Args:
        config: Configuration object with parameters
        source_table: Name of the source table in Snowflake

    Returns:
        Tuple of (monthly_aggregated_df, clipping_impact_dict)
    """
    start_time = time.time()

    logger.info("Starting SQL-based data preparation using Snowflake")

    try:
        # Import Snowpark session (only when needed)
        try:
            from src.snowflake import snowpark_session
        except ImportError as e:
            logger.error("Snowpark not available: {}", e)
            raise ImportError("snowflake-snowpark-python package is required for SQL execution")

        # Create parameterized SQL query
        sql_query, params = create_sql_data_preparation_query(config, source_table)

        logger.debug("Executing parameterized SQL query against Snowflake using Snowpark")
        logger.debug("Parameters: {}", params)

        # Get Snowpark session
        session = snowpark_session()

        # Execute query with parameters to prevent SQL injection
        # Snowpark doesn't support parameterized queries like pandas, so we safely format
        # the query using validated configuration values (all params come from SegmentationConfig)
        try:
            formatted_query = sql_query % params
            logger.debug("Executing SQL query with {} characters", len(formatted_query))
            result_df = session.sql(formatted_query).to_pandas()
        except Exception as query_error:
            logger.error("SQL query execution failed: {}", query_error)
            logger.debug(
                "Failed query (first 500 chars): {}",
                formatted_query[:500] if "formatted_query" in locals() else "Query formatting failed",
            )
            raise

        # Extract clipping impact from the result
        if len(result_df) > 0:
            logger.debug("Query result columns: {}", list(result_df.columns))
            logger.debug("Query result shape: {} rows × {} columns", len(result_df), len(result_df.columns))

            # Check if clipping impact columns are present
            clipping_columns = [
                "entity_total_transactions",
                "entity_transactions_clipped",
                "entity_clipping_rate_pct",
                "entity_amount_loss_pct",
                "clipping_threshold_used",
            ]
            missing_columns = [col for col in clipping_columns if col not in result_df.columns]

            if missing_columns:
                logger.warning("Missing clipping impact columns: {}", missing_columns)
                logger.debug("Available columns: {}", list(result_df.columns))
                # Create basic clipping impact without the missing columns
                clipping_impact = {
                    "total_entities": result_df["dim_value"].nunique() if "dim_value" in result_df.columns else 0,
                    "threshold": config.amount_clipping_threshold,
                }
            else:
                # Calculate overall clipping impact from the results
                clipping_impact = {
                    "total_transactions": result_df["entity_total_transactions"].sum(),
                    "transactions_clipped": result_df["entity_transactions_clipped"].sum(),
                    "clipping_rate_pct": result_df["entity_clipping_rate_pct"].mean(),
                    "total_entities": result_df["dim_value"].nunique(),
                    "entities_affected": (result_df["entity_transactions_clipped"] > 0).sum(),
                    "threshold": result_df["clipping_threshold_used"].iloc[0]
                    if "clipping_threshold_used" in result_df.columns
                    else config.amount_clipping_threshold,
                }

                # Calculate amount impact if available
                if "entity_amount_loss_pct" in result_df.columns:
                    clipping_impact["amount_reduction_pct"] = result_df["entity_amount_loss_pct"].mean()
        else:
            clipping_impact = {}

        elapsed_time = time.time() - start_time
        logger.info(
            "SQL-based data preparation completed in {:.2f}s - Output: {} rows × {} columns",
            elapsed_time,
            len(result_df),
            len(result_df.columns),
        )

        # Log clipping impact
        if clipping_impact:
            logger.info("Amount clipping impact summary:")
            logger.info("  - Total transactions: {:,}", clipping_impact.get("total_transactions", 0))
            logger.info(
                "  - Transactions clipped: {:,} ({:.1f}%)",
                clipping_impact.get("transactions_clipped", 0),
                clipping_impact.get("clipping_rate_pct", 0),
            )
            logger.info(
                "  - Entities affected: {} / {}",
                clipping_impact.get("entities_affected", 0),
                clipping_impact.get("total_entities", 0),
            )

        return result_df, clipping_impact

    except Exception as e:
        logger.error("SQL-based data preparation failed: {}", str(e))
        raise


def execute_sql_data_preparation(df: pd.DataFrame, config: SegmentationConfig) -> tuple[pd.DataFrame, dict]:
    """
    Execute SQL-based data preparation steps 1-3 on the input DataFrame.

    This function:
    1. Creates a temporary table/view from the input DataFrame
    2. Executes the consolidated SQL query
    3. Returns the monthly aggregated results + impact analysis

    Args:
        df: Input transaction DataFrame
        config: Configuration object

    Returns:
        Tuple of (monthly_aggregated_df, clipping_impact_dict)
    """
    start_time = time.time()
    initial_rows = len(df)

    logger.info("Starting SQL-based data preparation (steps 1-3)")
    logger.info("Input data shape: {} rows × {} columns", initial_rows, len(df.columns))

    try:
        # For now, we'll simulate the SQL execution using pandas operations
        # In a real implementation, this would execute against Snowflake/database

        # Step 1: Base data preparation with amount clipping (simulated)
        logger.debug("Executing SQL: Base data preparation with amount clipping")

        # Apply date filtering
        df_filtered = df[(df["date"] >= pd.Timestamp(config.start_date)) & (df["date"] <= pd.Timestamp(config.end_date))].copy()

        # Store original amounts
        df_filtered["original_amount"] = df_filtered["amount"].copy()

        # Apply amount clipping
        clipping_mask = df_filtered["amount"] < config.amount_clipping_threshold
        df_filtered["was_amount_clipped"] = clipping_mask.astype(int)
        df_filtered.loc[clipping_mask, "amount"] = 0.0

        # Add time-based features
        df_filtered["month"] = df_filtered["date"].dt.to_period("M")
        df_filtered["year"] = df_filtered["date"].dt.year
        df_filtered["month_num"] = df_filtered["date"].dt.month
        df_filtered["day_of_month"] = df_filtered["date"].dt.day

        # Calculate days from EOM
        df_filtered["days_from_eom"] = df_filtered.apply(
            lambda row: 0
            if row["is_last_work_day_of_month"]
            else -(
                df_filtered[(df_filtered["dim_value"] == row["dim_value"]) & (df_filtered["month"] == row["month"])]["date"]
                .rank(method="dense", ascending=False)
                .loc[row.name]
            ),
            axis=1,
        ).astype(int)

        logger.debug("SQL: Base preparation completed - {} rows after filtering", len(df_filtered))

        # Step 2: Calculate clipping impact
        logger.debug("Executing SQL: Clipping impact analysis")

        clipping_impact = {
            "total_transactions": len(df_filtered),
            "transactions_clipped": df_filtered["was_amount_clipped"].sum(),
            "clipping_rate_pct": (df_filtered["was_amount_clipped"].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0,
            "total_original_amount": df_filtered["original_amount"].sum(),
            "total_clipped_amount": df_filtered["amount"].sum(),
            "total_amount_clipped": (df_filtered["original_amount"] - df_filtered["amount"]).sum(),
            "entities_affected": df_filtered[df_filtered["was_amount_clipped"] == 1]["dim_value"].nunique(),
            "total_entities": df_filtered["dim_value"].nunique(),
            "threshold": config.amount_clipping_threshold,
        }

        if clipping_impact["total_original_amount"] > 0:
            clipping_impact["amount_reduction_pct"] = (
                clipping_impact["total_amount_clipped"] / clipping_impact["total_original_amount"] * 100
            )
        else:
            clipping_impact["amount_reduction_pct"] = 0.0

        # Step 3: Monthly aggregations
        logger.debug("Executing SQL: Monthly aggregations")

        # Group by entity and month for aggregation
        monthly_agg = (
            df_filtered.groupby(["dim_value", "month", "year", "month_num"])
            .agg(
                {
                    # Total monthly amounts (after clipping)
                    "amount": ["sum", "count", "mean", "std", "max"],
                    "original_amount": "sum",
                    "was_amount_clipped": "sum",
                    "day_of_month": "std",
                    "is_last_work_day_of_month": "max",
                    "month_num": "first",  # For calendar indicators
                }
            )
            .reset_index()
        )

        # Flatten column names
        monthly_agg.columns = [
            "dim_value",
            "month",
            "year",
            "month_num",
            "monthly_total",
            "monthly_transactions",
            "monthly_avg_amount",
            "monthly_std_amount",
            "max_monthly_transaction",
            "monthly_original_amount",
            "monthly_transactions_clipped",
            "day_dispersion",
            "has_nonzero_eom_flag",
            "month_num_check",
        ]

        # Fill NaN values
        monthly_agg["monthly_std_amount"] = monthly_agg["monthly_std_amount"].fillna(0)
        monthly_agg["day_dispersion"] = monthly_agg["day_dispersion"].fillna(0)

        # Calculate EOM-specific metrics
        eom_data = (
            df_filtered[df_filtered["is_last_work_day_of_month"]]
            .groupby(["dim_value", "month"])
            .agg({"amount": ["sum", "count"]})
            .reset_index()
        )
        eom_data.columns = ["dim_value", "month", "eom_amount", "eom_transaction_count"]

        # Calculate non-EOM metrics
        non_eom_data = (
            df_filtered[~df_filtered["is_last_work_day_of_month"]]
            .groupby(["dim_value", "month"])
            .agg({"amount": ["sum", "count", "mean"]})
            .reset_index()
        )
        non_eom_data.columns = ["dim_value", "month", "non_eom_total", "non_eom_transactions", "non_eom_avg"]

        # Merge all aggregations
        result_df = monthly_agg.merge(eom_data, on=["dim_value", "month"], how="left")
        result_df = result_df.merge(non_eom_data, on=["dim_value", "month"], how="left")

        # Fill missing values
        eom_cols = ["eom_amount", "eom_transaction_count"]
        non_eom_cols = ["non_eom_total", "non_eom_transactions", "non_eom_avg"]
        result_df[eom_cols + non_eom_cols] = result_df[eom_cols + non_eom_cols].fillna(0)

        # Add calendar indicators
        result_df["is_quarter_end"] = result_df["month_num"].isin([3, 6, 9, 12]).astype(int)
        result_df["is_year_end"] = (result_df["month_num"] == 12).astype(int)
        result_df["has_nonzero_eom"] = (result_df["eom_amount"] > 0).astype(int)

        # Add quarterly/yearly amounts
        result_df["quarter_end_amount"] = result_df.apply(lambda row: row["monthly_total"] if row["is_quarter_end"] else 0, axis=1)
        result_df["year_end_amount"] = result_df.apply(lambda row: row["monthly_total"] if row["is_year_end"] else 0, axis=1)

        # Calculate clipping percentages per month
        result_df["monthly_clipping_rate_pct"] = (
            result_df["monthly_transactions_clipped"]
            / (result_df["monthly_transactions"] + result_df["monthly_transactions_clipped"])
            * 100
        ).fillna(0)

        result_df["monthly_amount_clipped"] = result_df["monthly_original_amount"] - result_df["monthly_total"]
        result_df["monthly_amount_loss_pct"] = (
            result_df["monthly_amount_clipped"] / result_df["monthly_original_amount"] * 100
        ).fillna(0)

        # Convert month back to datetime for compatibility
        result_df["month"] = result_df["month"].dt.to_timestamp()

        elapsed_time = time.time() - start_time
        logger.info(
            "SQL-based data preparation completed in {:.2f}s - Output: {} rows × {} columns",
            elapsed_time,
            len(result_df),
            len(result_df.columns),
        )

        # Log clipping impact
        logger.info("Amount clipping impact summary:")
        logger.info("  - Total transactions: {:,}", clipping_impact["total_transactions"])
        logger.info(
            "  - Transactions clipped: {:,} ({:.1f}%)",
            clipping_impact["transactions_clipped"],
            clipping_impact["clipping_rate_pct"],
        )
        logger.info(
            "  - Amount clipped: {:.2f} ({:.1f}%)", clipping_impact["total_amount_clipped"], clipping_impact["amount_reduction_pct"]
        )
        logger.info("  - Entities affected: {} / {}", clipping_impact["entities_affected"], clipping_impact["total_entities"])

        return result_df, clipping_impact

    except Exception as e:
        logger.error("SQL-based data preparation failed: {}", str(e))
        raise
