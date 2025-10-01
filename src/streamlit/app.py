"""
Streamlit App for EOM Pattern Classification - Refactored Version
Interactive archetype tuning and human labeling interface
"""

import streamlit as st

from src.streamlit.components.data_loader import render_data_loader_interface
from src.streamlit.data_utils import load_human_labels
from src.streamlit.pages.archetype_tuning import render_archetype_tuning_page
from src.streamlit.pages.human_labeling import render_human_labeling_page


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "classification_data" not in st.session_state:
        st.session_state.classification_data = None
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "human_labels" not in st.session_state:
        st.session_state.human_labels = load_human_labels()
    if "current_series_idx" not in st.session_state:
        st.session_state.current_series_idx = 0
    if "show_data_loader" not in st.session_state:
        st.session_state.show_data_loader = False


def render_sidebar():
    """Render the sidebar navigation and data management"""
    st.sidebar.title("🎯 EOM Classification Tool")

    # Page navigation
    page = st.sidebar.radio(
        "Choose a page:", ["🎛️ Archetype Tuning", "🏷️ Human Labeling"], help="Navigate between different features of the app"
    )

    # Data management section
    st.sidebar.header("📊 Data Management")

    if not st.session_state.data_loaded:
        if st.sidebar.button("📥 Load Data", help="Load your EOM time series data"):
            st.session_state.show_data_loader = True
    else:
        st.sidebar.success("✅ Data loaded successfully")

        # Show data statistics
        n_series = len(st.session_state.classification_data)
        n_labeled = len(st.session_state.human_labels)

        st.sidebar.metric("Time Series", n_series)
        st.sidebar.metric("Labeled Series", n_labeled)

        if st.sidebar.button("🔄 Change Data Source", help="Load different data"):
            st.session_state.data_loaded = False
            st.session_state.show_data_loader = True
            st.rerun()

    return page


# Page configuration - must be first
st.set_page_config(page_title="EOM Pattern Classification Tool", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Handle data loading interface
    if st.session_state.get("show_data_loader", False):
        render_data_loader_interface()
    else:
        # Route to appropriate page
        if selected_page == "🎛️ Archetype Tuning":
            render_archetype_tuning_page()
        elif selected_page == "🏷️ Human Labeling":
            render_human_labeling_page()


if __name__ == "__main__":
    main()
