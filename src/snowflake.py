import os
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from dotenv import load_dotenv


def create_snowpark_session() -> Session:
    # creds = _make_snowpark_creds(credentials_file)
    creds = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "client_session_keep_alive": True,
        # "network_timeout": 600,
        "authenticator": "username_password_mfa",  # or "snowflake"
    }
    return Session.builder.configs(creds).create()


def snowpark_session() -> Session:
    load_dotenv(".env", override=True)
    try:
        return get_active_session()
    except Exception:
        session = create_snowpark_session()
        schema = os.getenv("SNOWFLAKE_SCHEMA")
        if schema:
            session.use_schema(schema)
        return session
