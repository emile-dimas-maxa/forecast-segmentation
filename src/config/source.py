from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    timeseries_table_name: str = Field(default="MAXA_SNBX.SEGMENTATION_ANALYSIS.ALL_TIMESERIES")
    transaction_table_name: str = Field(default="MAXA_SNBX.DANIEL_DATA_MART.INT__T__CAD_CORE_BANKING_REGULAR_TIME_SERIES_RECORDED")
