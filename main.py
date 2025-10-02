from src.segmentation.pipeline import SegmentationPipeline
from src.features.feature_pipeline import run_feature_pipeline

from pathlib import Path
import pandas as pd
from src.snowflake import snowpark_session
from dotenv import load_dotenv

from src.splitter import TimeSeriesBacktest

output_dir = "outputs"
segmentation_file_name = "segmentation_df"
feature_file_name = "feature_df"


def main():
    # create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    load_dotenv()
    session = snowpark_session()
    if Path(f"{output_dir}/{segmentation_file_name}.csv").exists():
        segmentation_df = pd.read_csv(f"{output_dir}/{segmentation_file_name}.csv").rename(columns=str.upper)
        segmentation_df = session.create_dataframe(segmentation_df)
    else:
        segmentation_df = SegmentationPipeline.run_full_pipeline(
            session=session,
            source_table="maxa_dev.data_private.int__t__cad_core_banking_regular_time_series_recorded",
            start_date="2022-01-01",
            end_date="2025-09-01",
        )

        segmentation_df.to_pandas().rename(columns=str.lower).to_csv(f"{output_dir}/{segmentation_file_name}.csv", index=False)

    # Run feature pipeline
    feature_df = run_feature_pipeline(
        df=segmentation_df,
        forecast_month="2025-09-01",
    )

    # Run forecasting pipeline
    feature_df = feature_df.to_pandas().rename(columns=str.lower)

    feature_df.to_csv(f"{output_dir}/{feature_file_name}.csv", index=False)

    # model = SegmentedForecastModel(
    #     segment_col="segment",
    #     target_col="sales",
    #     dimensions=["region", "product"],
    #     model_mapping={
    #         "region": {"type": "arima", "params": {"order": (1, 1, 1)}},
    #     },
    # )

    # for train_idx, test_idx in splitter.split(feature_df):
    #     train_df = feature_df.iloc[train_idx]
    #     test_df = feature_df.iloc[test_idx]

    #     print(train_df.shape, test_df.shape)


if __name__ == "__main__":
    main()
