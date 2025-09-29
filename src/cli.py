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

from src.forecast.config import ForecastingConfig, ForecastMethod
from src.forecast.pipeline import ForecastingPipeline
from src.segmentation.config import SegmentationConfig
from src.segmentation.pipeline import SegmentationPipeline

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


def get_snowflake_session(
    account: str | None = None,
    user: str | None = None,
    password: str | None = None,
    warehouse: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    role: str | None = None,
) -> Session:
    """Create and return a Snowflake session"""

    # Try to get connection parameters from environment variables if not provided
    connection_params = {
        "account": account or os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": user or os.getenv("SNOWFLAKE_USER"),
        "password": password or os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": warehouse or os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": database or os.getenv("SNOWFLAKE_DATABASE"),
        "schema": schema or os.getenv("SNOWFLAKE_SCHEMA"),
        "role": role or os.getenv("SNOWFLAKE_ROLE"),
    }

    # Remove None values
    connection_params = {k: v for k, v in connection_params.items() if v is not None}

    if not connection_params.get("account"):
        rprint("[red]Error: Snowflake account is required. Set SNOWFLAKE_ACCOUNT environment variable or use --account flag[/red]")
        raise typer.Exit(1)

    if not connection_params.get("user"):
        rprint("[red]Error: Snowflake user is required. Set SNOWFLAKE_USER environment variable or use --user flag[/red]")
        raise typer.Exit(1)

    if not connection_params.get("password"):
        rprint(
            "[red]Error: Snowflake password is required. Set SNOWFLAKE_PASSWORD environment variable or use --password flag[/red]"
        )
        raise typer.Exit(1)

    try:
        with console.status("[bold green]Connecting to Snowflake..."):
            session = Session.builder.configs(connection_params).create()

        rprint(f"[green]✓[/green] Connected to Snowflake as {connection_params['user']}")
        return session

    except Exception as e:
        rprint(f"[red]Error connecting to Snowflake: {str(e)}[/red]")
        raise typer.Exit(1) from e


@segmentation_app.command("run")
def run_segmentation(
    # Configuration parameters
    config_file: Path | None = typer.Option(None, "--config", help="Path to JSON configuration file"),
    start_date: str = typer.Option("2022-01-01", "--start-date", help="Analysis start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end-date", help="Analysis end date (YYYY-MM-DD), defaults to current date"),
    source_table: str = typer.Option(
        "int__t__cad_core_banking_regular_time_series_recorded", "--source-table", help="Source table name"
    ),
    min_months_history: int = typer.Option(3, "--min-months", help="Minimum months of history required"),
    rolling_window_months: int = typer.Option(12, "--rolling-window", help="Rolling window for feature calculation"),
    min_transactions: int = typer.Option(6, "--min-transactions", help="Minimum non-zero transactions to include series"),
    # Output options
    output_table: str | None = typer.Option(
        None, "--output-table", help="Output table name (if not specified, uses default naming)"
    ),
    save_to_file: Path | None = typer.Option(None, "--save-to-file", help="Save results to local file (CSV)"),
    # Snowflake connection parameters
    account: str | None = typer.Option(None, "--account", help="Snowflake account"),
    user: str | None = typer.Option(None, "--user", help="Snowflake user"),
    password: str | None = typer.Option(None, "--password", help="Snowflake password"),
    warehouse: str | None = typer.Option(None, "--warehouse", help="Snowflake warehouse"),
    database: str | None = typer.Option(None, "--database", help="Snowflake database"),
    schema: str | None = typer.Option(None, "--schema", help="Snowflake schema"),
    role: str | None = typer.Option(None, "--role", help="Snowflake role"),
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
    session = get_snowflake_session(account, user, password, warehouse, database, schema, role)

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
    # Snowflake connection parameters
    account: str | None = typer.Option(None, "--account", help="Snowflake account"),
    user: str | None = typer.Option(None, "--user", help="Snowflake user"),
    password: str | None = typer.Option(None, "--password", help="Snowflake password"),
    warehouse: str | None = typer.Option(None, "--warehouse", help="Snowflake warehouse"),
    database: str | None = typer.Option(None, "--database", help="Snowflake database"),
    schema: str | None = typer.Option(None, "--schema", help="Snowflake schema"),
    role: str | None = typer.Option(None, "--role", help="Snowflake role"),
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
    session = get_snowflake_session(account, user, password, warehouse, database, schema, role)

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


if __name__ == "__main__":
    app()
