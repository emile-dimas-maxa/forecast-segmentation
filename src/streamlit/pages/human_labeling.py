"""
Human labeling page
"""

import streamlit as st

from src.streamlit.data_utils import load_human_labels
from src.streamlit.components.labeling import render_labeling_interface, render_evaluation_metrics


def render_human_labeling_page():
    """Render the human labeling page"""
    st.title("🏷️ Human Labeling Interface")

    # Check if data is loaded
    if not st.session_state.get("data_loaded", False):
        st.warning("⚠️ No data loaded. Please load data first using the sidebar.")
        if st.button("🔄 Load Sample Data"):
            from src.streamlit.data_utils import generate_sample_timeseries_data, prepare_classification_data

            raw_data = generate_sample_timeseries_data()
            classification_data = prepare_classification_data(raw_data)

            st.session_state.raw_data = raw_data
            st.session_state.classification_data = classification_data
            st.session_state.data_loaded = True

            st.success(f"Loaded {len(classification_data)} time series for classification")
            st.rerun()
        return

    classification_data = st.session_state.classification_data
    raw_data = st.session_state.raw_data

    # Initialize session state for labeling
    if "current_series_idx" not in st.session_state:
        st.session_state.current_series_idx = 0

    if "human_labels" not in st.session_state:
        st.session_state.human_labels = load_human_labels()

    # Check if we have model predictions
    if "eom_pattern" not in classification_data.columns:
        st.warning("⚠️ No model predictions found. Please run classification first in the Archetype Tuning page.")
        return

    current_idx = st.session_state.current_series_idx
    human_labels = st.session_state.human_labels

    # Render labeling interface
    new_idx, updated_labels, labels_changed = render_labeling_interface(classification_data, raw_data, current_idx, human_labels)

    # Update session state if needed
    if labels_changed:
        st.session_state.human_labels = updated_labels

    # Evaluation metrics
    st.divider()
    render_evaluation_metrics(classification_data, human_labels)
