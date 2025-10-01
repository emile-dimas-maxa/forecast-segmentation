"""
Data loading components for Streamlit app
"""

import streamlit as st
import pandas as pd

from src.streamlit.data_utils import (
    load_csv_data,
    load_snowflake_data,
    validate_data_format,
    prepare_classification_data,
    SNOWPARK_AVAILABLE,
)


def render_sample_data_loader() -> pd.DataFrame | None:
    """Render sample data generation interface"""
    st.info("Generate synthetic EOM time series data for demonstration")
    col1, col2 = st.columns(2)
    with col1:
        n_series = st.number_input("Number of series", 10, 500, 100)
    with col2:
        n_months = st.number_input("Number of months", 6, 60, 24)

    if st.button("Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            from src.streamlit.data_utils import generate_sample_timeseries_data

            return generate_sample_timeseries_data(n_series=n_series, n_months=n_months)
    return None


def render_csv_loader() -> pd.DataFrame | None:
    """Render CSV file upload interface"""
    st.info("Upload a CSV file with your EOM time series data")

    # Show required format
    with st.expander("📋 Required CSV Format", expanded=False):
        st.markdown("""
        **Required columns:**
        - `dim_value`: Series identifier (string)
        - `forecast_month`: Date/time column (YYYY-MM-DD format)
        - `target_eom_amount`: End-of-month amount (numeric)
        
        **Optional columns** (will be calculated if missing):
        - `monthly_volume`, `eom_concentration`, `eom_frequency`, `eom_cv`
        - `months_of_history`, `has_eom_history`, `has_nonzero_eom`
        """)

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = load_csv_data(uploaded_file)
            st.success(f"Loaded CSV with {len(df)} rows")
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    return None


def render_snowflake_loader() -> pd.DataFrame | None:
    """Render Snowflake connection interface"""
    if not SNOWPARK_AVAILABLE:
        st.error("Snowpark is not available. Install with: `pip install snowflake-snowpark-python`")
        return None

    st.info("Connect to Snowflake and load EOM time series data")

    with st.expander("🔐 Snowflake Connection", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            account = st.text_input("Account", help="Your Snowflake account identifier")
            user = st.text_input("User", help="Your Snowflake username")
            warehouse = st.text_input("Warehouse", help="Snowflake warehouse name")
        with col2:
            password = st.text_input("Password", type="password", help="Your Snowflake password")
            database = st.text_input("Database", help="Database name")
            schema = st.text_input("Schema", help="Schema name")

    query_option = st.radio("Data source:", ["Table", "Custom Query"])

    if query_option == "Table":
        table_name = st.text_input("Table Name", help="Full table name (e.g., DATABASE.SCHEMA.TABLE)")
        query = None
    else:
        query = st.text_area("SQL Query", height=100, help="Custom SQL query to fetch EOM data")
        table_name = None

    if st.button("Load from Snowflake"):
        if not all([account, user, password, warehouse, database, schema]):
            st.error("Please fill in all connection parameters")
            return None

        if query_option == "Table" and not table_name:
            st.error("Please specify table name")
            return None

        if query_option == "Custom Query" and not query:
            st.error("Please provide SQL query")
            return None

        connection_params = {
            "account": account,
            "user": user,
            "password": password,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }

        try:
            with st.spinner("Loading from Snowflake..."):
                df = load_snowflake_data(connection_params, table_name, query)
            st.success(f"Loaded {len(df)} rows from Snowflake")
            return df
        except Exception as e:
            st.error(f"Error loading from Snowflake: {e}")
    return None


def render_data_validation(df: pd.DataFrame) -> bool:
    """Render data validation interface"""
    st.subheader("📋 Data Validation")

    # Validate data format
    is_valid, issues = validate_data_format(df)

    if issues:
        for issue in issues:
            if "Missing required columns" in issue:
                st.error(issue)
            else:
                st.warning(issue)

    if is_valid:
        st.success("✅ Data format is valid!")

        # Show data preview
        with st.expander("👀 Data Preview", expanded=False):
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Series count:** {df['dim_value'].nunique()}")
            st.write(f"**Date range:** {df['forecast_month'].min()} to {df['forecast_month'].max()}")

        return True
    else:
        st.error("❌ Data format is invalid. Please fix the issues above.")
        return False


def render_data_loader_interface():
    """Main data loading interface"""
    st.subheader("📊 Data Loading Options")

    # Add a back button
    if st.button("⬅️ Back to Main App"):
        st.session_state.show_data_loader = False
        st.rerun()

    data_source = st.radio(
        "Choose data source:",
        ["Sample Data", "CSV File", "Snowflake Table"],
        help="Select how you want to load your EOM time series data",
    )

    df = None

    # Render appropriate loader based on selection
    if data_source == "Sample Data":
        df = render_sample_data_loader()
    elif data_source == "CSV File":
        df = render_csv_loader()
    elif data_source == "Snowflake Table":
        df = render_snowflake_loader()

    # Validate and process loaded data
    if df is not None:
        if render_data_validation(df):
            if st.button("✅ Use This Data"):
                with st.spinner("Preparing data for classification..."):
                    # Prepare classification data (latest month per series)
                    classification_data = prepare_classification_data(df)

                    st.session_state.raw_data = df
                    st.session_state.classification_data = classification_data
                    st.session_state.data_loaded = True

                    st.success(f"✅ Data loaded successfully! {len(classification_data)} time series ready for classification")
                    st.rerun()
