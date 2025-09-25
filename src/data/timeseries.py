from snowflake.snowpark import Session
import pandas as pd


def fetch_timeseries_data(session: Session, table_name: str) -> pd.DataFrame:
    return session.table(table_name).to_pandas().rename(columns=str.lower)
