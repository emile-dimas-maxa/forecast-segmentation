from src.snowflake import snowpark_session
from src.data.timeseries import fetch_timeseries_data
from src.transformations.pipeline import run_pipeline
from src.config.segmentation import SegmentationConfig
from src.config.source import SourceConfig


def main():
    session = snowpark_session()
    df = fetch_timeseries_data(session, SourceConfig.timeseries_table_name)
    config = SegmentationConfig()
    result = run_pipeline(df, config)
    print(result)


if __name__ == "__main__":
    main()
