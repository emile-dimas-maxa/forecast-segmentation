"""
Archetype tuning page
"""

import streamlit as st

from src.segmentation.pandas_classification import PandasEOMClassifier
from src.segmentation.config import SegmentationConfig
from src.streamlit.components.archetype_controls import render_archetype_controls
from src.streamlit.components.visualization import (
    create_archetype_visualization,
    render_classification_results,
    render_detailed_results_table,
)


def render_archetype_tuning_page():
    """Render the archetype tuning page"""
    st.title("🎛️ EOM Archetype Tuning")

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

    # Create side-by-side layout for controls and visualization
    controls_col, viz_col = st.columns([3, 2])

    with controls_col:
        # Archetype configuration controls
        archetype_config = render_archetype_controls()

    with viz_col:
        # 3D visualization of archetypes
        st.subheader("📊 Archetype Visualization")
        fig_3d = create_archetype_visualization(archetype_config)
        st.plotly_chart(fig_3d, use_container_width=True)

    # Automatically run classification with current archetype config
    with st.spinner("Running EOM classification..."):
        config = SegmentationConfig()
        classifier = PandasEOMClassifier(config, archetype_config)
        classified_data = classifier.run_full_classification(classification_data.copy())

    # Show classification results
    render_classification_results(classified_data)

    # Detailed results table
    render_detailed_results_table(classified_data)
