"""
EOM Forecasting Segmentation Pipeline
Modular transformation functions for Snowpark DataFrames
"""

from loguru import logger
from snowflake.snowpark import DataFrame, Session

from src.segmentation.config import SegmentationConfig
from src.segmentation.transformation.data_preparation import load_source_data, prepare_base_data
from src.segmentation.transformation.aggregation import create_monthly_aggregates, apply_eom_clipping
from src.segmentation.transformation.rolling_features import calculate_rolling_features
from src.segmentation.transformation.portfolio_metrics import calculate_portfolio_metrics
from src.segmentation.transformation.pattern_metrics import calculate_pattern_metrics
from src.segmentation.transformation.importance_classification import classify_importance
from src.segmentation.transformation.eom_pattern_classification import (
    calculate_eom_smooth_scores,
    calculate_pattern_distances,
    calculate_pattern_probabilities,
    classify_eom_patterns,
)
from src.segmentation.transformation.general_pattern_classification import classify_general_patterns
from src.segmentation.transformation.final_classification import create_final_classification
from src.segmentation.transformation.growth_metrics import calculate_growth_metrics
from src.segmentation.transformation.output_formatting import select_final_columns


class SegmentationPipeline:
    """Pipeline for EOM forecasting segmentation using static methods"""

    @staticmethod
    def run_full_pipeline(session: Session, config: SegmentationConfig, source_df: DataFrame | None = None) -> DataFrame:
        """
        Run the complete segmentation pipeline

        Args:
            session: Snowpark session
            config: Configuration object
            source_df: Optional source DataFrame. If not provided, reads from config.source_table

        Returns:
            Final segmented DataFrame
        """
        logger.info("=" * 80)
        logger.info("Starting Full Segmentation Pipeline")
        logger.info(f"Configuration: start_date={config.start_date}, end_date={config.end_date}")
        logger.info("=" * 80)

        # Step 1: Load and prepare base data
        df = load_source_data(session, config) if source_df is None else source_df

        df = prepare_base_data(config, df)

        # Step 2: Monthly aggregations
        df = create_monthly_aggregates(config, df)

        # Step 2.5: Apply EOM clipping if configured
        if config.daily_amount_clip_threshold is not None:
            df = apply_eom_clipping(config, df)

        # Step 3: Rolling features
        df = calculate_rolling_features(config, df)

        # Step 4: Portfolio metrics
        df = calculate_portfolio_metrics(config, df)

        # Step 5: Pattern metrics
        df = calculate_pattern_metrics(config, df)

        # Step 6: Importance classification
        df = classify_importance(config, df)

        # Step 7: EOM pattern classification (smooth scoring)
        df = calculate_eom_smooth_scores(config, df)
        df = calculate_pattern_distances(config, df)
        df = calculate_pattern_probabilities(config, df)
        df = classify_eom_patterns(config, df)

        # Step 8: General pattern classification
        df = classify_general_patterns(config, df)

        # Step 9: Final classification and recommendations
        df = create_final_classification(config, df)

        # Step 10: Growth metrics
        df = calculate_growth_metrics(config, df)

        # Final selection and filtering
        df = select_final_columns(config, df)

        logger.info("=" * 80)
        logger.info("Completed Full Segmentation Pipeline Successfully")
        logger.info("=" * 80)

        return df
