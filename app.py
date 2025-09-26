#!/usr/bin/env python3
"""EOM Forecasting Pipeline Application with intermediate results support."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config.segmentation import SegmentationConfig
from src.config.source import SourceConfig
from src.data.timeseries import fetch_timeseries_data
from src.snowflake import snowpark_session
from src.transformations.pipeline import EOMForecastingPipeline, run_pipeline


def setup_logging(log_level: str = "INFO", log_file: str | None = None):
    """Configure logging for the application."""
    logger.remove()  # Remove default handler

    # Console logging
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        colorize=True,
    )

    # File logging if specified
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
        )


def run_with_intermediate_results(
    df,
    config: SegmentationConfig,
    output_dir: str,
    target_month: str | None = None,
    list_results: bool = False,
    load_step: int | None = None,
    use_sql_preparation: bool = True,
    snowflake_source_table: str | None = None,
) -> None:
    """Run pipeline with intermediate results functionality.

    Args:
        df: Input DataFrame (can be None when using Snowflake SQL preparation)
    """

    # Create pipeline with intermediate saving and SQL options
    pipeline = EOMForecastingPipeline(
        config=config,
        save_intermediate=True,
        output_dir=output_dir,
        use_sql_preparation=use_sql_preparation,
        snowflake_source_table=snowflake_source_table,
    )

    # Check if we should load an existing intermediate result
    if load_step is not None:
        logger.info("Loading intermediate result from step {}", load_step)
        loaded_df = pipeline.load_intermediate_result(load_step)
        if loaded_df is not None:
            logger.info("Successfully loaded step {} data: {} rows × {} columns", load_step, len(loaded_df), len(loaded_df.columns))
            print("\nLoaded Data Summary:")
            print(f"Shape: {loaded_df.shape}")
            print(f"Columns: {list(loaded_df.columns)}")
            return
        else:
            logger.error("Failed to load intermediate result for step {}", load_step)
            return

    # List existing results if requested
    if list_results:
        logger.info("Listing existing intermediate results...")
        saved_files = pipeline.list_intermediate_results()
        if saved_files:
            print(f"\nFound {len(saved_files)} intermediate result files:")
            for filepath in saved_files:
                try:
                    size_kb = filepath.stat().st_size / 1024
                    print(f"  - {filepath.name} ({size_kb:.1f} KB)")
                except Exception:
                    print(f"  - {filepath.name}")
        else:
            print("No intermediate result files found.")
        return

    # Run the full pipeline
    logger.info("Starting EOM forecasting pipeline with intermediate results saving")
    logger.info("Output directory: {}", Path(output_dir).absolute())

    try:
        result = pipeline.transform(df, target_month)

        logger.success("Pipeline completed successfully!")
        print("\nFinal Result Summary:")
        print(f"Shape: {result.shape}")
        print(f"Columns: {len(result.columns)}")

        # Show sample of key columns if available
        key_columns = ["dim_value", "eom_pattern", "combined_priority_score", "recommendation"]
        available_key_columns = [col for col in key_columns if col in result.columns]

        if available_key_columns:
            print("\nSample of key results:")
            print(result[available_key_columns].head())

        # List saved intermediate files
        saved_files = pipeline.list_intermediate_results()
        print(f"\nIntermediate results saved: {len(saved_files)} files")

    except Exception as e:
        logger.error("Pipeline failed: {}", e)
        raise


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="EOM Forecasting Pipeline with intermediate results support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RECOMMENDED: Production with Snowflake SQL (processes 100M+ rows in database)
  python app.py --snowflake-source-table int__t__cad_core_banking_regular_time_series_recorded
  
  # Development: SQL-based preparation with pandas simulation (default behavior)
  python app.py --save-intermediate --log-level DEBUG
  
  # Production with intermediate results (handles massive datasets efficiently)
  python app.py --save-intermediate --snowflake-source-table my_table --output-dir results/
  
  # Custom target month with SQL preparation
  python app.py --target-month 2025-01-01 --snowflake-source-table my_table
  
  # Legacy: pandas-only approach (NOT recommended for large datasets)
  python app.py --disable-sql-preparation
  
  # List existing intermediate results
  python app.py --list-results --output-dir results/
  
  # Load and inspect a specific intermediate step
  python app.py --load-step 5 --output-dir results/
        """,
    )

    # Pipeline options
    parser.add_argument(
        "--target-month", type=str, default="2025-07-01", help="Target month for final output filtering (format: YYYY-MM-DD)"
    )

    # Intermediate results options
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results at each pipeline step")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory to save intermediate results",
    )

    parser.add_argument("--list-results", action="store_true", help="List existing intermediate result files and exit")

    parser.add_argument("--load-step", type=int, choices=range(0, 11), help="Load and display a specific intermediate step (0-10)")

    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set the logging level")

    parser.add_argument("--log-file", type=str, help="Optional log file path")

    # Data source options
    parser.add_argument(
        "--skip-data-fetch", action="store_true", help="Skip data fetching (for testing with intermediate results only)"
    )

    # SQL preparation options (SQL-based is DEFAULT and RECOMMENDED)
    parser.add_argument(
        "--snowflake-source-table",
        type=str,
        help="Snowflake source table name for SQL-based preparation (RECOMMENDED for production - processes data directly in database)",
        default="maxa_snbx.daniel_data_private.int__t__cad_core_banking_regular_time_series_recorded",
    )

    parser.add_argument(
        "--disable-sql-preparation",
        action="store_true",
        help="Disable SQL-based preparation and use legacy pandas-only approach (NOT recommended for large datasets)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    logger.info("Starting EOM Forecasting Pipeline Application")
    logger.info("Arguments: {}", vars(args))

    try:
        # Handle intermediate results operations that don't need data
        if (args.list_results or args.load_step is not None) and args.skip_data_fetch:
            logger.info("Skipping data fetch for intermediate results operation")
            config = SegmentationConfig()
            run_with_intermediate_results(
                None,
                config,
                args.output_dir,
                list_results=args.list_results,
                load_step=args.load_step,
                use_sql_preparation=not args.disable_sql_preparation,
                snowflake_source_table=args.snowflake_source_table,
            )
            return

        # Determine if we need to fetch raw transaction data
        use_sql_preparation = not args.disable_sql_preparation
        snowflake_source_table = args.snowflake_source_table

        # Skip data fetch if using Snowflake SQL preparation (SQL handles data directly)
        if use_sql_preparation and snowflake_source_table:
            logger.info("Using Snowflake SQL-based preparation - skipping raw data fetch")
            logger.info("SQL query will process data directly from table: {}", snowflake_source_table)
            df = None  # No DataFrame needed for SQL execution
        elif not args.skip_data_fetch:
            logger.info("Fetching raw transaction data from Snowflake...")
            session = snowpark_session()
            source_config = SourceConfig()
            df = fetch_timeseries_data(session, source_config.transaction_table_name)
            logger.info("Data fetched: {} rows × {} columns", len(df), len(df.columns))
        else:
            logger.warning("Skipping data fetch - using empty DataFrame for pandas simulation")
            import pandas as pd

            df = pd.DataFrame()

        # Setup configuration
        config = SegmentationConfig()

        # Log SQL preparation mode (settings already determined above)
        if use_sql_preparation:
            if snowflake_source_table:
                logger.info("Mode: Snowflake SQL-based preparation (OPTIMAL for large datasets)")
            else:
                logger.info("Mode: Pandas simulation of SQL-based preparation (development/testing)")
                logger.info("💡 TIP: For production with large datasets, use --snowflake-source-table TABLE_NAME")
        else:
            logger.info("Mode: Traditional pandas-only preparation (legacy mode)")
            logger.warning("⚠️  Pandas-only mode not recommended for large datasets")

        # Run pipeline based on options
        if args.save_intermediate or args.list_results or args.load_step is not None:
            run_with_intermediate_results(
                df,
                config,
                args.output_dir,
                args.target_month,
                list_results=args.list_results,
                load_step=args.load_step,
                use_sql_preparation=use_sql_preparation,
                snowflake_source_table=snowflake_source_table,
            )
        else:
            # Standard pipeline run
            logger.info("Running standard pipeline (no intermediate results)")
            result = run_pipeline(
                df,
                config,
                args.target_month,
                save_intermediate=False,
                output_dir=None,
                use_sql_preparation=use_sql_preparation,
                snowflake_source_table=snowflake_source_table,
            )

            logger.success("Pipeline completed successfully!")
            print(f"\nResult shape: {result.shape}")
            print(result.head() if len(result) > 0 else "No results returned")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Application failed: {}", e)
        if args.log_level == "DEBUG":
            logger.exception("Full traceback:")
        sys.exit(1)

    logger.success("Application completed successfully!")


if __name__ == "__main__":
    main()
