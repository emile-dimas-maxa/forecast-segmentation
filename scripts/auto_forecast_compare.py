#!/usr/bin/env python3
"""
Automated Forecasting and Comparison Script
Runs all forecasting methods and generates comprehensive comparison reports
"""

import os
import sys
import json
import argparse
from datetime import datetime, date
from pathlib import Path

# Add src to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from rich import print as rprint
from loguru import logger
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F

from forecast.config import ForecastingConfig, ForecastMethod, EvaluationLevel
from forecast.pipeline import ForecastingPipeline


class AutoForecastCompare:
    """Automated forecasting and comparison runner"""

    def __init__(self, config_file: str = None):
        self.console = Console()
        self.config_file = config_file
        self.session = None

    def get_snowflake_session(self) -> Session:
        """Create Snowflake session from environment variables"""
        connection_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
        }

        # Remove None values
        connection_params = {k: v for k, v in connection_params.items() if v is not None}

        required_params = ["account", "user", "password"]
        missing_params = [param for param in required_params if not connection_params.get(param)]

        if missing_params:
            rprint(
                f"[red]Error: Missing required environment variables: {', '.join([f'SNOWFLAKE_{p.upper()}' for p in missing_params])}[/red]"
            )
            sys.exit(1)

        try:
            with self.console.status("[bold green]Connecting to Snowflake..."):
                session = Session.builder.configs(connection_params).create()

            rprint(f"[green]✓[/green] Connected to Snowflake as {connection_params['user']}")
            return session

        except Exception as e:
            rprint(f"[red]Error connecting to Snowflake: {str(e)}[/red]")
            sys.exit(1)

    def load_config(
        self,
        segmented_table: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        output_prefix: str = "auto_forecast",
    ) -> ForecastingConfig:
        """Load or create forecasting configuration"""

        if self.config_file and Path(self.config_file).exists():
            with open(self.config_file, "r") as f:
                config_data = json.load(f)
            config = ForecastingConfig(**config_data)
            rprint(f"[green]✓[/green] Loaded configuration from {self.config_file}")
        else:
            # Create default configuration with ALL methods
            config_params = {
                "train_start_date": datetime.strptime(train_start, "%Y-%m-%d").date(),
                "train_end_date": datetime.strptime(train_end, "%Y-%m-%d").date(),
                "test_start_date": datetime.strptime(test_start, "%Y-%m-%d").date(),
                "test_end_date": datetime.strptime(test_end, "%Y-%m-%d").date(),
                "forecast_horizon": 1,
                "min_history_months": 12,
                "methods_to_test": list(ForecastMethod),  # ALL METHODS
                "evaluation_levels": list(EvaluationLevel),  # ALL LEVELS
                "output_table_prefix": output_prefix,
                "save_forecasts": True,
                "save_errors": True,
            }

            config = ForecastingConfig(**config_params)
            rprint("[green]✓[/green] Created configuration to test ALL forecasting methods")

        # Always override to use ALL methods for comprehensive comparison
        config.methods_to_test = list(ForecastMethod)
        config.evaluation_levels = list(EvaluationLevel)

        return config

    def run_comprehensive_forecast(
        self,
        segmented_table: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        output_prefix: str = "auto_forecast",
    ):
        """Run comprehensive forecasting evaluation"""

        rprint("[bold blue]🚀 Starting Automated Comprehensive Forecasting Evaluation[/bold blue]")

        # Load configuration
        config = self.load_config(segmented_table, train_start, train_end, test_start, test_end, output_prefix)

        # Display configuration
        self.display_config_summary(config, segmented_table)

        # Get Snowflake session
        self.session = self.get_snowflake_session()

        try:
            # Load segmented data
            with self.console.status(f"[bold green]Loading segmented data from {segmented_table}..."):
                segmented_df = self.session.table(segmented_table)
                row_count = segmented_df.count()

            rprint(f"[green]✓[/green] Loaded {row_count:,} rows from segmented data")

            # Run comprehensive forecasting
            with self.console.status("[bold green]Running comprehensive forecasting evaluation..."):
                forecasts_df, dim_value_errors_df, aggregated_errors_df = ForecastingPipeline.run_backtesting(
                    self.session, config, segmented_df
                )

            rprint(f"[green]✓[/green] Comprehensive forecasting evaluation completed!")

            # Generate comparison report
            self.generate_comparison_report(forecasts_df, dim_value_errors_df, aggregated_errors_df, output_prefix)

            return forecasts_df, dim_value_errors_df, aggregated_errors_df

        except Exception as e:
            rprint(f"[red]Error running comprehensive forecasting: {str(e)}[/red]")
            raise

        finally:
            if self.session:
                self.session.close()
                rprint("[dim]Snowflake session closed[/dim]")

    def display_config_summary(self, config: ForecastingConfig, segmented_table: str):
        """Display configuration summary"""

        table = Table(title="Automated Forecasting Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Segmented Table", segmented_table)
        table.add_row("Train Period", f"{config.train_start_date} to {config.train_end_date}")
        table.add_row("Test Period", f"{config.test_start_date} to {config.test_end_date}")
        table.add_row("Forecast Horizon", str(config.forecast_horizon))
        table.add_row("Methods", f"ALL ({len(config.methods_to_test)} methods)")
        table.add_row("Evaluation Levels", "dim_value, segment (credit/debit/net), overall")
        table.add_row("Output Prefix", config.output_table_prefix)

        self.console.print(table)

        # Show all methods being tested
        methods_table = Table(title="All Methods Being Tested")
        methods_table.add_column("Method", style="cyan")
        methods_table.add_column("Type", style="yellow")

        method_types = {
            "zero": "Baseline",
            "naive": "Baseline",
            "moving_average": "Statistical",
            "weighted_moving_average": "Statistical",
            "ets": "Statistical",
            "arima": "Statistical",
            "sarima": "Statistical",
            "xgboost_individual": "Machine Learning",
            "xgboost_global": "Machine Learning",
            "croston": "Specialized",
            "ensemble": "Ensemble",
            "segment_aggregate": "Hierarchical",
        }

        for method in config.methods_to_test:
            method_type = method_types.get(method.value, "Unknown")
            methods_table.add_row(method.value, method_type)

        self.console.print(methods_table)

    def generate_comparison_report(self, forecasts_df, dim_value_errors_df, aggregated_errors_df, output_prefix: str):
        """Generate comprehensive comparison report"""

        rprint("\n[bold blue]📊 Generating Comprehensive Comparison Report[/bold blue]")

        try:
            # Convert to pandas for analysis
            errors_pd = aggregated_errors_df.to_pandas()

            # Overall method performance
            if "method" in errors_pd.columns and "mae" in errors_pd.columns:
                method_performance = (
                    errors_pd.groupby("method")
                    .agg({"mae": "mean", "rmse": "mean", "mape": "mean", "directional_accuracy": "mean"})
                    .round(4)
                    .sort_values("mae")
                )

                # Display overall rankings
                self.display_overall_rankings(method_performance)

                # Display performance by evaluation level
                self.display_level_performance(errors_pd)

                # Display recommendations
                self.display_recommendations(method_performance)

                # Save detailed report to file
                self.save_detailed_report(method_performance, errors_pd, output_prefix)

        except Exception as e:
            rprint(f"[yellow]Warning: Could not generate full comparison report: {str(e)}[/yellow]")

    def display_overall_rankings(self, method_performance):
        """Display overall method rankings"""

        rprint("\n[bold green]🏆 Overall Method Rankings[/bold green]")

        # Overall performance table
        perf_table = Table(title="Overall Method Performance (Ranked by MAE)")
        perf_table.add_column("Rank", style="bold cyan")
        perf_table.add_column("Method", style="cyan")
        perf_table.add_column("MAE", style="green")
        perf_table.add_column("RMSE", style="green")
        perf_table.add_column("MAPE", style="green")
        perf_table.add_column("Dir. Accuracy", style="green")
        perf_table.add_column("Score", style="bold yellow")

        for rank, (method, row) in enumerate(method_performance.iterrows(), 1):
            # Calculate composite score (lower is better for MAE, RMSE, MAPE; higher is better for directional accuracy)
            score = (row["mae"] + row["rmse"] + row["mape"] / 100) / 3 - row["directional_accuracy"] / 100

            perf_table.add_row(
                str(rank),
                method,
                f"{row['mae']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mape']:.2f}%",
                f"{row['directional_accuracy']:.2f}%",
                f"{score:.4f}",
            )

        self.console.print(perf_table)

    def display_level_performance(self, errors_pd):
        """Display performance by evaluation level"""

        if "evaluation_level" not in errors_pd.columns:
            return

        rprint("\n[bold]Performance by Evaluation Level:[/bold]")

        for level in ["dim_value", "segment", "overall"]:
            level_data = errors_pd[errors_pd["evaluation_level"] == level]
            if level_data.empty:
                continue

            level_performance = level_data.groupby("method").agg({"mae": "mean"}).round(4).sort_values("mae")

            level_table = Table(title=f"{level.upper()} Level Performance (Top 5)")
            level_table.add_column("Rank", style="bold cyan")
            level_table.add_column("Method", style="cyan")
            level_table.add_column("MAE", style="green")

            for rank, (method, row) in enumerate(level_performance.head(5).iterrows(), 1):
                level_table.add_row(str(rank), method, f"{row['mae']:.4f}")

            self.console.print(level_table)

    def display_recommendations(self, method_performance):
        """Display method recommendations"""

        rprint("\n[bold green]🎯 Automated Recommendations:[/bold green]")

        # Best overall method
        best_overall = method_performance.index[0]
        best_mae = method_performance.loc[best_overall, "mae"]

        # Best directional accuracy
        best_directional = method_performance.sort_values("directional_accuracy", ascending=False).index[0]
        best_dir_acc = method_performance.loc[best_directional, "directional_accuracy"]

        # Most balanced method (good across all metrics)
        method_performance["balanced_score"] = (
            method_performance["mae"].rank()
            + method_performance["rmse"].rank()
            + method_performance["mape"].rank()
            + method_performance["directional_accuracy"].rank(ascending=False)
        ) / 4
        best_balanced = method_performance.sort_values("balanced_score").index[0]

        rprint(f"• [bold]Best Overall (Lowest MAE):[/bold] {best_overall} (MAE: {best_mae:.4f})")
        rprint(f"• [bold]Best Directional Accuracy:[/bold] {best_directional} ({best_dir_acc:.2f}%)")
        rprint(f"• [bold]Most Balanced:[/bold] {best_balanced}")

        # Top 3 recommendations
        top_3 = method_performance.head(3).index.tolist()
        rprint(f"• [bold]Top 3 Recommendations:[/bold] {', '.join(top_3)}")

        # Method categories
        statistical_methods = ["moving_average", "weighted_moving_average", "arima", "sarima", "ets"]
        ml_methods = ["xgboost_individual", "xgboost_global"]

        best_statistical = (
            method_performance.loc[method_performance.index.intersection(statistical_methods)].index[0]
            if any(m in method_performance.index for m in statistical_methods)
            else None
        )

        best_ml = (
            method_performance.loc[method_performance.index.intersection(ml_methods)].index[0]
            if any(m in method_performance.index for m in ml_methods)
            else None
        )

        if best_statistical:
            rprint(f"• [bold]Best Statistical Method:[/bold] {best_statistical}")
        if best_ml:
            rprint(f"• [bold]Best ML Method:[/bold] {best_ml}")

    def save_detailed_report(self, method_performance, errors_pd, output_prefix: str):
        """Save detailed report to file"""

        report_file = f"reports/{output_prefix}_comparison_report.txt"
        os.makedirs("reports", exist_ok=True)

        with open(report_file, "w") as f:
            f.write("AUTOMATED FORECASTING COMPARISON REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("OVERALL METHOD RANKINGS:\n")
            f.write("-" * 30 + "\n")
            for rank, (method, row) in enumerate(method_performance.iterrows(), 1):
                f.write(
                    f"{rank:2d}. {method:20s} MAE: {row['mae']:.4f} RMSE: {row['rmse']:.4f} MAPE: {row['mape']:.2f}% Dir.Acc: {row['directional_accuracy']:.2f}%\n"
                )

            f.write(f"\nDetailed report saved to: {report_file}")

        rprint(f"[green]✓[/green] Detailed report saved to: {report_file}")


def main():
    """Main function for command line usage"""

    parser = argparse.ArgumentParser(description="Automated Forecasting and Comparison")
    parser.add_argument("--segmented-table", required=True, help="Table with segmented data")
    parser.add_argument("--train-start", required=True, help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--train-end", required=True, help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--test-start", required=True, help="Test start date (YYYY-MM-DD)")
    parser.add_argument("--test-end", required=True, help="Test end date (YYYY-MM-DD)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-prefix", default="auto_forecast", help="Output prefix for tables and reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Create and run automated comparison
    auto_compare = AutoForecastCompare(config_file=args.config)

    try:
        auto_compare.run_comprehensive_forecast(
            segmented_table=args.segmented_table,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            output_prefix=args.output_prefix,
        )

        rprint("\n[bold green]🎉 Automated forecasting comparison completed successfully![/bold green]")

    except Exception as e:
        rprint(f"[red]Error in automated comparison: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
