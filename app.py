from src.snowflake import snowpark_session
from src.data.timeseries import fetch_timeseries_data
from src.transformations.pipeline import run_pipeline
from src.config.segmentation import SegmentationConfig
from src.config.source import SourceConfig


def main():
    session = snowpark_session()
    source_config = SourceConfig()
    df = fetch_timeseries_data(session, source_config.transaction_table_name)
    config = SegmentationConfig(source_config=source_config)
    result = run_pipeline(df, config)
    print(result)


if __name__ == "__main__":
    main()
