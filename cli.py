#!/usr/bin/env python3
"""
Radio Forecast Segmentation CLI
Command-line interface for running segmentation and forecasting pipelines
"""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

import typer
from loguru import logger
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

from src.forecast.config import ForecastingConfig, ForecastMethod, EvaluationLevel
from src.forecast.pipeline import ForecastingPipeline
from src.segmentation.config import SegmentationConfig
from src.segmentation.pipeline import SegmentationPipeline
from src.snowflake import snowpark_session

# Initialize Typer app and Rich console
app = typer.Typer(
    name="radio-forecast-segmentation",
    help="Radio Forecast Segmentation CLI - Run segmentation and forecasting pipelines",
    rich_markup_mode="rich",
)
console = Console()

# Subcommands
segmentation_app = typer.Typer(name="segmentation", help="Segmentation pipeline commands")
forecast_app = typer.Typer(name="forecast", help="Forecasting pipeline commands")
app.add_typer(segmentation_app, name="segmentation")
app.add_typer(forecast_app, name="forecast")


def get_snowflake_session() -> Session:
    try:
        return snowpark_session()
    except Exception as e:
        rprint(f"[red]Error getting Snowflake session: {e}[/red]")
        raise typer.Exit(1) from e


@segmentation_app.command("run")
def run_segmentation(
    # Configuration parameters
    config_file: Path | None = typer.Option(None, "--config", help="Path to JSON configuration file"),
    start_date: str = typer.Option("2022-01-01", "--start-date", help="Analysis start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end-date", help="Analysis end date (YYYY-MM-DD), defaults to current date"),
    source_table: str = typer.Option(
        "maxa_dev.data_private.int__t__cad_core_banking_regular_time_series_recorded", "--source-table", help="Source table name"
    ),
    min_months_history: int = typer.Option(3, "--min-months", help="Minimum months of history required"),
    rolling_window_months: int = typer.Option(12, "--rolling-window", help="Rolling window for feature calculation"),
    min_transactions: int = typer.Option(6, "--min-transactions", help="Minimum non-zero transactions to include series"),
    # Output options
    output_table: str | None = typer.Option(
        None, "--output-table", help="Output table name (if not specified, uses default naming)"
    ),
    save_to_file: Path | None = typer.Option(None, "--save-to-file", help="Save results to local file (CSV)"),
    # Processing options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run the segmentation pipeline"""

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    rprint("[bold blue]🔍 Starting Radio Forecast Segmentation Pipeline[/bold blue]")

    # Load configuration
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config_data = json.load(f)
        config = SegmentationConfig(**config_data)
        rprint(f"[green]✓[/green] Loaded configuration from {config_file}")
    else:
        # Create configuration from command line parameters
        config_params = {
            "start_date": datetime.strptime(start_date, "%Y-%m-%d").date(),
            "source_table": source_table,
            "min_months_history": min_months_history,
            "rolling_window_months": rolling_window_months,
            "min_transactions": min_transactions,
        }

        if end_date:
            config_params["end_date"] = datetime.strptime(end_date, "%Y-%m-%d").date()

        config = SegmentationConfig(**config_params)
        rprint("[green]✓[/green] Created configuration from command line parameters")

    # Display configuration summary
    table = Table(title="Segmentation Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Start Date", str(config.start_date))
    table.add_row("End Date", str(config.end_date) if config.end_date else "Current Date")
    table.add_row("Source Table", config.source_table)
    table.add_row("Min Months History", str(config.min_months_history))
    table.add_row("Rolling Window", str(config.rolling_window_months))
    table.add_row("Min Transactions", str(config.min_transactions))

    console.print(table)

    # Get Snowflake session
    session = get_snowflake_session()

    try:
        # Run segmentation pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running segmentation pipeline...", total=None)

            result_df = SegmentationPipeline.run_full_pipeline(session, config)

            progress.update(task, description="Pipeline completed successfully!")

        # Get result summary
        row_count = result_df.count()
        rprint(f"[green]✓[/green] Segmentation completed successfully! Generated {row_count:,} rows")

        # Save results if requested
        if output_table:
            with console.status(f"[bold green]Saving to table {output_table}..."):
                result_df.write.save_as_table(output_table, mode="overwrite")
            rprint(f"[green]✓[/green] Results saved to table: {output_table}")

        if save_to_file:
            with console.status(f"[bold green]Saving to file {save_to_file}..."):
                # Convert to Pandas and save
                pandas_df = result_df.to_pandas()
                pandas_df.to_csv(save_to_file, index=False)
            rprint(f"[green]✓[/green] Results saved to file: {save_to_file}")

        # Show sample of results
        rprint("\n[bold]Sample Results:[/bold]")
        sample_df = result_df.limit(5).to_pandas()
        console.print(sample_df.to_string(index=False))

    except Exception as e:
        rprint(f"[red]Error running segmentation pipeline: {str(e)}[/red]")
        raise typer.Exit(1) from e

    finally:
        session.close()
        rprint("[dim]Snowflake session closed[/dim]")


@forecast_app.command("run-all")
def run_all_forecasting_methods(
    # Configuration parameters
    config_file: Path | None = typer.Option(None, "--config", help="Path to JSON configuration file"),
    segmented_table: str | None = typer.Option(None, "--segmented-table", help="Table with segmented data (required)"),
    train_start: str | None = typer.Option(None, "--train-start", help="Training start date (YYYY-MM-DD) (required)"),
    train_end: str | None = typer.Option(None, "--train-end", help="Training end date (YYYY-MM-DD) (required)"),
    test_start: str | None = typer.Option(None, "--test-start", help="Test start date (YYYY-MM-DD) (required)"),
    test_end: str | None = typer.Option(None, "--test-end", help="Test end date (YYYY-MM-DD) (required)"),
    # Forecasting parameters
    forecast_horizon: int = typer.Option(1, "--horizon", help="Number of months to forecast ahead"),
    min_history: int = typer.Option(12, "--min-history", help="Minimum months of history required"),
    # Output options
    output_prefix: str = typer.Option("forecast_backtest_all", "--output-prefix", help="Prefix for output tables"),
    save_forecasts: bool = typer.Option(True, "--save-forecasts/--no-save-forecasts", help="Save individual forecasts"),
    save_errors: bool = typer.Option(True, "--save-errors/--no-save-errors", help="Save error metrics"),
    show_detailed_results: bool = typer.Option(True, "--detailed/--summary", help="Show detailed results breakdown"),
    # Processing options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run ALL forecasting methods and evaluate at all levels (dim_value, segment, overall)"""

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    rprint("[bold blue]📈 Starting Comprehensive Forecasting Evaluation - ALL METHODS[/bold blue]")

    # Validate required parameters
    if not segmented_table:
        rprint("[red]Error: --segmented-table is required[/red]")
        raise typer.Exit(1)

    if not all([train_start, train_end, test_start, test_end]):
        rprint("[red]Error: All date parameters (--train-start, --train-end, --test-start, --test-end) are required[/red]")
        raise typer.Exit(1)

    # Load configuration or create with ALL methods
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config_data = json.load(f)
        config = ForecastingConfig(**config_data)
        # Override to use ALL methods
        config.methods_to_test = list(ForecastMethod)
        rprint(f"[green]✓[/green] Loaded configuration from {config_file} and set to use ALL methods")
    else:
        # Create configuration with ALL methods
        try:
            config_params = {
                "train_start_date": datetime.strptime(train_start, "%Y-%m-%d").date(),
                "train_end_date": datetime.strptime(train_end, "%Y-%m-%d").date(),
                "test_start_date": datetime.strptime(test_start, "%Y-%m-%d").date(),
                "test_end_date": datetime.strptime(test_end, "%Y-%m-%d").date(),
                "forecast_horizon": forecast_horizon,
                "min_history_months": min_history,
                "methods_to_test": list(ForecastMethod),  # ALL METHODS
                "evaluation_levels": list(EvaluationLevel),  # ALL LEVELS
                "output_table_prefix": output_prefix,
                "save_forecasts": save_forecasts,
                "save_errors": save_errors,
            }

            config = ForecastingConfig(**config_params)
            rprint("[green]✓[/green] Created configuration to test ALL forecasting methods")

        except ValueError as e:
            rprint(f"[red]Error in configuration: {str(e)}[/red]")
            raise typer.Exit(1) from e

    # Display configuration summary
    table = Table(title="Comprehensive Forecasting Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Segmented Table", segmented_table)
    table.add_row("Train Period", f"{config.train_start_date} to {config.train_end_date}")
    table.add_row("Test Period", f"{config.test_start_date} to {config.test_end_date}")
    table.add_row("Forecast Horizon", str(config.forecast_horizon))
    table.add_row("Min History", str(config.min_history_months))
    table.add_row("Methods", f"ALL ({len(config.methods_to_test)} methods)")
    table.add_row("Evaluation Levels", "dim_value, segment (credit/debit/net), overall")
    table.add_row("Output Prefix", config.output_table_prefix)

    console.print(table)

    # Show methods being tested
    methods_table = Table(title="Methods Being Tested")
    methods_table.add_column("Method", style="cyan")
    methods_table.add_column("Description", style="white")

    method_descriptions = {
        "zero": "Always predicts zero",
        "naive": "Uses the last observed value",
        "moving_average": "Simple moving average",
        "weighted_moving_average": "Weighted moving average with exponential decay",
        "ets": "Exponential smoothing (Error, Trend, Seasonality)",
        "arima": "AutoRegressive Integrated Moving Average",
        "sarima": "Seasonal ARIMA",
        "xgboost_individual": "XGBoost trained on individual time series",
        "xgboost_global": "XGBoost trained on all time series",
        "croston": "Croston's method for intermittent demand",
        "ensemble": "Ensemble of multiple methods",
        "segment_aggregate": "Forecast at segment level then distribute",
    }

    for method in config.methods_to_test:
        description = method_descriptions.get(method.value, "No description available")
        methods_table.add_row(method.value, description)

    console.print(methods_table)

    # Get Snowflake session
    session = get_snowflake_session()

    try:
        # Load segmented data
        with console.status(f"[bold green]Loading segmented data from {segmented_table}..."):
            segmented_df = session.table(segmented_table)
            row_count = segmented_df.count()

        rprint(f"[green]✓[/green] Loaded {row_count:,} rows from segmented data")

        # Run comprehensive forecasting pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running comprehensive forecasting evaluation...", total=None)

            forecasts_df, dim_value_errors_df, aggregated_errors_df = ForecastingPipeline.run_backtesting(
                session, config, segmented_df
            )

            progress.update(task, description="Forecasting evaluation completed successfully!")

        rprint(f"[green]✓[/green] Comprehensive forecasting evaluation completed successfully!")

        # Display detailed results if requested
        if show_detailed_results:
            _display_comprehensive_results(console, forecasts_df, dim_value_errors_df, aggregated_errors_df)

        # Show summary statistics
        _display_method_comparison_summary(console, aggregated_errors_df)

    except Exception as e:
        rprint(f"[red]Error running comprehensive forecasting evaluation: {str(e)}[/red]")
        raise typer.Exit(1) from e

    finally:
        session.close()
        rprint("[dim]Snowflake session closed[/dim]")


@forecast_app.command("run")
def run_forecasting(
    # Configuration parameters
    config_file: Path | None = typer.Option(None, "--config", help="Path to JSON configuration file"),
    segmented_table: str | None = typer.Option(None, "--segmented-table", help="Table with segmented data (required)"),
    train_start: str | None = typer.Option(None, "--train-start", help="Training start date (YYYY-MM-DD) (required)"),
    train_end: str | None = typer.Option(None, "--train-end", help="Training end date (YYYY-MM-DD) (required)"),
    test_start: str | None = typer.Option(None, "--test-start", help="Test start date (YYYY-MM-DD) (required)"),
    test_end: str | None = typer.Option(None, "--test-end", help="Test end date (YYYY-MM-DD) (required)"),
    # Forecasting parameters
    methods: list[str] = typer.Option(
        ["naive", "moving_average", "arima"], "--method", help="Forecasting methods to test (can be used multiple times)"
    ),
    forecast_horizon: int = typer.Option(1, "--horizon", help="Number of months to forecast ahead"),
    min_history: int = typer.Option(12, "--min-history", help="Minimum months of history required"),
    # Output options
    output_prefix: str = typer.Option("forecast_backtest", "--output-prefix", help="Prefix for output tables"),
    save_forecasts: bool = typer.Option(True, "--save-forecasts/--no-save-forecasts", help="Save individual forecasts"),
    save_errors: bool = typer.Option(True, "--save-errors/--no-save-errors", help="Save error metrics"),
    # Processing options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run the forecasting pipeline with backtesting"""

    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    rprint("[bold blue]📈 Starting Radio Forecast Pipeline[/bold blue]")

    # Validate required parameters
    if not segmented_table:
        rprint("[red]Error: --segmented-table is required[/red]")
        raise typer.Exit(1)

    if not all([train_start, train_end, test_start, test_end]):
        rprint("[red]Error: All date parameters (--train-start, --train-end, --test-start, --test-end) are required[/red]")
        raise typer.Exit(1)

    # Load configuration
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            config_data = json.load(f)
        config = ForecastingConfig(**config_data)
        rprint(f"[green]✓[/green] Loaded configuration from {config_file}")
    else:
        # Create configuration from command line parameters
        try:
            config_params = {
                "train_start_date": datetime.strptime(train_start, "%Y-%m-%d").date(),
                "train_end_date": datetime.strptime(train_end, "%Y-%m-%d").date(),
                "test_start_date": datetime.strptime(test_start, "%Y-%m-%d").date(),
                "test_end_date": datetime.strptime(test_end, "%Y-%m-%d").date(),
                "forecast_horizon": forecast_horizon,
                "min_history_months": min_history,
                "methods_to_test": [ForecastMethod(method) for method in methods],
                "output_table_prefix": output_prefix,
                "save_forecasts": save_forecasts,
                "save_errors": save_errors,
            }

            config = ForecastingConfig(**config_params)
            rprint("[green]✓[/green] Created configuration from command line parameters")

        except ValueError as e:
            rprint(f"[red]Error in configuration: {str(e)}[/red]")
            raise typer.Exit(1)

    # Display configuration summary
    table = Table(title="Forecasting Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Segmented Table", segmented_table)
    table.add_row("Train Period", f"{config.train_start_date} to {config.train_end_date}")
    table.add_row("Test Period", f"{config.test_start_date} to {config.test_end_date}")
    table.add_row("Forecast Horizon", str(config.forecast_horizon))
    table.add_row("Min History", str(config.min_history_months))
    table.add_row("Methods", ", ".join([method.value for method in config.methods_to_test]))
    table.add_row("Output Prefix", config.output_table_prefix)

    console.print(table)

    # Get Snowflake session
    session = get_snowflake_session()

    try:
        # Load segmented data
        with console.status(f"[bold green]Loading segmented data from {segmented_table}..."):
            segmented_df = session.table(segmented_table)
            row_count = segmented_df.count()

        rprint(f"[green]✓[/green] Loaded {row_count:,} rows from segmented data")

        # Run forecasting pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running forecasting pipeline...", total=None)

            results = ForecastingPipeline.run_backtesting(session, config, segmented_df)

            progress.update(task, description="Forecasting completed successfully!")

        rprint(f"[green]✓[/green] Forecasting completed successfully!")

        # Display results summary
        if isinstance(results, dict):
            rprint("\n[bold]Results Summary:[/bold]")
            for key, value in results.items():
                if hasattr(value, "count"):
                    count = value.count()
                    rprint(f"  {key}: {count:,} rows")
                else:
                    rprint(f"  {key}: {value}")

    except Exception as e:
        rprint(f"[red]Error running forecasting pipeline: {str(e)}[/red]")
        raise typer.Exit(1)

    finally:
        session.close()
        rprint("[dim]Snowflake session closed[/dim]")


@app.command("config")
def show_config_templates():
    """Show configuration file templates for segmentation and forecasting"""

    rprint("[bold blue]Configuration Templates[/bold blue]\n")

    # Segmentation config template
    rprint("[bold]Segmentation Configuration Template:[/bold]")
    segmentation_config = SegmentationConfig()
    segmentation_dict = segmentation_config.model_dump()

    # Convert date objects to strings for JSON serialization
    for key, value in segmentation_dict.items():
        if isinstance(value, date):
            segmentation_dict[key] = value.isoformat()

    rprint("```json")
    rprint(json.dumps(segmentation_dict, indent=2))
    rprint("```\n")

    # Forecasting config template
    rprint("[bold]Forecasting Configuration Template:[/bold]")
    forecasting_config = ForecastingConfig(
        train_start_date=date(2023, 1, 1),
        train_end_date=date(2023, 12, 31),
        test_start_date=date(2024, 1, 1),
        test_end_date=date(2024, 3, 31),
    )
    forecasting_dict = forecasting_config.model_dump()

    # Convert date objects to strings for JSON serialization
    for key, value in forecasting_dict.items():
        if isinstance(value, date):
            forecasting_dict[key] = value.isoformat()

    rprint("```json")
    rprint(json.dumps(forecasting_dict, indent=2))
    rprint("```\n")

    rprint("[dim]Save either template to a JSON file and use with --config flag[/dim]")


@app.command("methods")
def list_forecast_methods():
    """List available forecasting methods"""

    rprint("[bold blue]Available Forecasting Methods[/bold blue]\n")

    table = Table()
    table.add_column("Method", style="cyan")
    table.add_column("Description", style="white")

    method_descriptions = {
        "zero": "Always predicts zero",
        "naive": "Uses the last observed value",
        "moving_average": "Simple moving average",
        "weighted_moving_average": "Weighted moving average with exponential decay",
        "ets": "Exponential smoothing (Error, Trend, Seasonality)",
        "arima": "AutoRegressive Integrated Moving Average",
        "sarima": "Seasonal ARIMA",
        "xgboost_individual": "XGBoost trained on individual time series",
        "xgboost_global": "XGBoost trained on all time series",
        "croston": "Croston's method for intermittent demand",
        "ensemble": "Ensemble of multiple methods",
        "segment_aggregate": "Forecast at segment level then distribute",
    }

    for method in ForecastMethod:
        description = method_descriptions.get(method.value, "No description available")
        table.add_row(method.value, description)

    console.print(table)


@app.callback()
def main(version: bool = typer.Option(False, "--version", help="Show version information")):
    """
    Radio Forecast Segmentation CLI

    A comprehensive tool for running segmentation and forecasting pipelines on time series data.
    """
    if version:
        rprint("[bold blue]Radio Forecast Segmentation CLI[/bold blue]")
        rprint("Version: 1.0.0")
        rprint("Built with Typer and Rich")
        raise typer.Exit()


def _display_comprehensive_results(console: Console, forecasts_df, dim_value_errors_df, aggregated_errors_df):
    """Display comprehensive results breakdown"""

    rprint("\n[bold blue]📊 Comprehensive Results Breakdown[/bold blue]")

    # Convert to pandas for easier display
    try:
        # Sample forecasts
        rprint("\n[bold]Sample Forecasts:[/bold]")
        sample_forecasts = forecasts_df.limit(10).to_pandas()
        console.print(sample_forecasts.to_string(index=False))

        # Dim value errors summary
        rprint("\n[bold]Dim Value Level Errors (Top 10 by MAE):[/bold]")
        dim_value_sample = dim_value_errors_df.order_by(F.col("mae").desc()).limit(10).to_pandas()
        console.print(dim_value_sample.to_string(index=False))

        # Aggregated errors summary
        rprint("\n[bold]Aggregated Errors Summary:[/bold]")
        agg_sample = aggregated_errors_df.limit(20).to_pandas()
        console.print(agg_sample.to_string(index=False))

    except Exception as e:
        rprint(f"[yellow]Warning: Could not display detailed results: {str(e)}[/yellow]")


def _display_method_comparison_summary(console: Console, aggregated_errors_df):
    """Display method comparison summary"""

    rprint("\n[bold blue]🏆 Method Performance Comparison[/bold blue]")

    try:
        # Convert to pandas for analysis
        errors_pd = aggregated_errors_df.to_pandas()

        # Overall performance by method
        if "method" in errors_pd.columns and "mae" in errors_pd.columns:
            method_performance = (
                errors_pd.groupby("method")
                .agg({"mae": "mean", "rmse": "mean", "mape": "mean", "directional_accuracy": "mean"})
                .round(4)
                .sort_values("mae")
            )

            # Create performance table
            perf_table = Table(title="Overall Method Performance (Ranked by MAE)")
            perf_table.add_column("Rank", style="bold cyan")
            perf_table.add_column("Method", style="cyan")
            perf_table.add_column("MAE", style="green")
            perf_table.add_column("RMSE", style="green")
            perf_table.add_column("MAPE", style="green")
            perf_table.add_column("Dir. Accuracy", style="green")

            for rank, (method, row) in enumerate(method_performance.iterrows(), 1):
                perf_table.add_row(
                    str(rank),
                    method,
                    f"{row['mae']:.4f}",
                    f"{row['rmse']:.4f}",
                    f"{row['mape']:.2f}%",
                    f"{row['directional_accuracy']:.2f}%",
                )

            console.print(perf_table)

        # Performance by evaluation level
        if "evaluation_level" in errors_pd.columns:
            rprint("\n[bold]Performance by Evaluation Level:[/bold]")
            level_performance = errors_pd.groupby(["evaluation_level", "method"]).agg({"mae": "mean"}).round(4).reset_index()

            for level in level_performance["evaluation_level"].unique():
                level_data = level_performance[level_performance["evaluation_level"] == level].sort_values("mae")

                level_table = Table(title=f"{level.upper()} Level Performance")
                level_table.add_column("Rank", style="bold cyan")
                level_table.add_column("Method", style="cyan")
                level_table.add_column("MAE", style="green")

                for rank, (_, row) in enumerate(level_data.iterrows(), 1):
                    level_table.add_row(str(rank), row["method"], f"{row['mae']:.4f}")

                console.print(level_table)

        # Best method recommendations
        rprint("\n[bold green]🎯 Recommendations:[/bold green]")

        if "method" in errors_pd.columns and "mae" in errors_pd.columns:
            best_overall = method_performance.index[0]
            best_mae = method_performance.loc[best_overall, "mae"]
            best_directional = method_performance.sort_values("directional_accuracy", ascending=False).index[0]
            best_dir_acc = method_performance.loc[best_directional, "directional_accuracy"]

            rprint(f"• [bold]Best Overall Method (MAE):[/bold] {best_overall} (MAE: {best_mae:.4f})")
            rprint(f"• [bold]Best Directional Accuracy:[/bold] {best_directional} ({best_dir_acc:.2f}%)")

            # Show top 3 methods
            top_3 = method_performance.head(3)
            rprint(f"• [bold]Top 3 Methods:[/bold] {', '.join(top_3.index.tolist())}")

    except Exception as e:
        rprint(f"[yellow]Warning: Could not generate method comparison: {str(e)}[/yellow]")


@forecast_app.command("auto")
def auto_forecast_compare(
    # Data parameters
    segmented_table: str = typer.Option(..., "--segmented-table", help="Table with segmented data"),
    train_start: str = typer.Option(..., "--train-start", help="Training start date (YYYY-MM-DD)"),
    train_end: str = typer.Option(..., "--train-end", help="Training end date (YYYY-MM-DD)"),
    test_start: str = typer.Option(..., "--test-start", help="Test start date (YYYY-MM-DD)"),
    test_end: str = typer.Option(..., "--test-end", help="Test end date (YYYY-MM-DD)"),
    # Configuration
    config_file: Path | None = typer.Option(None, "--config", help="Path to configuration file"),
    output_prefix: str = typer.Option(None, "--output-prefix", help="Output prefix (auto-generated if not provided)"),
    # Processing options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Run automated forecasting comparison using the standalone script"""

    rprint("[bold blue]🤖 Running Automated Forecasting Comparison[/bold blue]")

    # Generate output prefix if not provided
    if not output_prefix:
        from datetime import datetime

        output_prefix = f"auto_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build command for the automated script
    script_path = Path(__file__).parent / "scripts" / "auto_forecast_compare.py"

    cmd_parts = [
        "python",
        str(script_path),
        "--segmented-table",
        segmented_table,
        "--train-start",
        train_start,
        "--train-end",
        train_end,
        "--test-start",
        test_start,
        "--test-end",
        test_end,
        "--output-prefix",
        output_prefix,
    ]

    if config_file:
        cmd_parts.extend(["--config", str(config_file)])

    if verbose:
        cmd_parts.append("--verbose")

    # Show what will be executed
    rprint(f"[dim]Executing: {' '.join(cmd_parts)}[/dim]")

    # Run the automated script
    import subprocess

    try:
        result = subprocess.run(cmd_parts, check=True, capture_output=False)
        rprint("[green]✓[/green] Automated forecasting comparison completed successfully!")
    except subprocess.CalledProcessError as e:
        rprint(f"[red]Error running automated comparison: {e}[/red]")
        raise typer.Exit(1) from e
    except FileNotFoundError:
        rprint(f"[red]Error: Could not find automated script at {script_path}[/red]")
        raise typer.Exit(1)


@forecast_app.command("compare")
def compare_methods(
    # Data parameters
    results_table: str = typer.Option(..., "--results-table", help="Table with forecasting results"),
    # Display options
    metric: str = typer.Option("mae", "--metric", help="Metric to use for comparison (mae, rmse, mape, directional_accuracy)"),
    top_n: int = typer.Option(10, "--top-n", help="Show top N methods"),
    by_level: bool = typer.Option(True, "--by-level/--overall", help="Show breakdown by evaluation level"),
):
    """Compare forecasting methods from saved results"""

    rprint("[bold blue]📊 Forecasting Methods Comparison[/bold blue]")

    # Get Snowflake session
    session = get_snowflake_session()

    try:
        # Load results
        with console.status(f"[bold green]Loading results from {results_table}..."):
            results_df = session.table(results_table)
            row_count = results_df.count()

        rprint(f"[green]✓[/green] Loaded {row_count:,} rows from results table")

        # Display comparison
        _display_method_comparison_summary(console, results_df)

    except Exception as e:
        rprint(f"[red]Error loading results: {str(e)}[/red]")
        raise typer.Exit(1) from e

    finally:
        session.close()
        rprint("[dim]Snowflake session closed[/dim]")


if __name__ == "__main__":
    app()
