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
) -> None:
    """Run pipeline with intermediate results functionality."""

    # Create pipeline with intermediate saving
    pipeline = EOMForecastingPipeline(config=config, save_intermediate=True, output_dir=output_dir)

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
        print(f"\nFinal Result Summary:")
        print(f"Shape: {result.shape}")
        print(f"Columns: {len(result.columns)}")

        # Show sample of key columns if available
        key_columns = ["dim_value", "eom_pattern", "combined_priority_score", "recommendation"]
        available_key_columns = [col for col in key_columns if col in result.columns]

        if available_key_columns:
            print(f"\nSample of key results:")
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
  # Basic pipeline run
  python app.py
  
  # Run with intermediate results saving
  python app.py --save-intermediate --output-dir results/
  
  # Run with custom target month and debug logging
  python app.py --target-month 2025-01-01 --log-level DEBUG
  
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

    parser.add_argument("--load-step", type=int, choices=range(0, 10), help="Load and display a specific intermediate step (0-9)")

    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set the logging level")

    parser.add_argument("--log-file", type=str, help="Optional log file path")

    # Data source options
    parser.add_argument(
        "--skip-data-fetch", action="store_true", help="Skip data fetching (for testing with intermediate results only)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    logger.info("Starting EOM Forecasting Pipeline Application")
    logger.info("Arguments: {}", vars(args))

    try:
        # Handle intermediate results operations that don't need data
        if args.list_results or args.load_step is not None:
            if args.skip_data_fetch:
                logger.info("Skipping data fetch for intermediate results operation")
                config = SegmentationConfig()
                run_with_intermediate_results(
                    None, config, args.output_dir, list_results=args.list_results, load_step=args.load_step
                )
                return

        # Fetch data from Snowflake
        if not args.skip_data_fetch:
            logger.info("Connecting to Snowflake and fetching data...")
            session = snowpark_session()
            source_config = SourceConfig()
            df = fetch_timeseries_data(session, source_config.transaction_table_name)
            logger.info("Data fetched: {} rows × {} columns", len(df), len(df.columns))
        else:
            logger.warning("Skipping data fetch - using empty DataFrame")
            import pandas as pd

            df = pd.DataFrame()

        # Setup configuration
        config = SegmentationConfig()

        # Run pipeline based on options
        if args.save_intermediate or args.list_results or args.load_step is not None:
            run_with_intermediate_results(
                df, config, args.output_dir, args.target_month, list_results=args.list_results, load_step=args.load_step
            )
        else:
            # Standard pipeline run
            logger.info("Running standard pipeline (no intermediate results)")
            result = run_pipeline(df, config, args.target_month)

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
