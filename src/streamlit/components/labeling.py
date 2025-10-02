"""
Human labeling components for Streamlit app
"""

import pandas as pd
import streamlit as st

from src.streamlit.data_utils import (
    get_timeseries_plot_data,
    save_human_labels,
    load_human_labels,
    calculate_evaluation_metrics,
)
from src.streamlit.components.visualization import create_timeseries_chart, create_probability_chart


# EOM pattern options for labeling
EOM_PATTERN_OPTIONS = [
    "CONTINUOUS_STABLE",
    "CONTINUOUS_VOLATILE",
    "INTERMITTENT_ACTIVE",
    "INTERMITTENT_DORMANT",
    "RARE_RECENT",
    "RARE_STALE",
    "UNKNOWN",
]


def render_navigation_controls(current_idx: int, total_series: int) -> tuple[int, bool]:
    """Render navigation controls and return new index and whether to save"""
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=current_idx <= 0):
            return max(0, current_idx - 1), False

    with col2:
        if st.button("➡️ Next", disabled=current_idx >= total_series - 1):
            return min(total_series - 1, current_idx + 1), False

    with col3:
        new_idx = st.number_input(
            f"Series Index (0-{total_series - 1})",
            min_value=0,
            max_value=total_series - 1,
            value=current_idx,
            key="series_idx_input",
        )
        if new_idx != current_idx:
            return new_idx, False

    with col4:
        st.write(f"**{current_idx + 1}** of **{total_series}**")

    return current_idx, False


def render_pattern_buttons(current_human_label: str | None) -> str | None:
    """Render pattern selection buttons and return selected pattern"""
    st.markdown("**Select the correct EOM pattern for this time series:**")

    # Create buttons in a grid layout
    cols = st.columns(4)  # 4 columns for better layout
    selected_pattern = None

    for i, pattern in enumerate(EOM_PATTERN_OPTIONS):
        col_idx = i % 4
        with cols[col_idx]:
            # Highlight button if it matches current label
            button_type = "primary" if current_human_label == pattern else "secondary"
            if st.button(pattern, key=f"pattern_{pattern}", type=button_type):
                selected_pattern = pattern

    return selected_pattern


def render_series_info(series_data: pd.DataFrame, summary: dict, predicted_pattern: str, confidence: float):
    """Render series information and statistics"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Months", summary["total_months"])
        st.metric("Avg Monthly Volume", f"{summary['avg_monthly_volume']:.0f}")

    with col2:
        st.metric("Avg EOM Amount", f"{summary['avg_eom_amount']:.0f}")
        st.metric("EOM Frequency", f"{summary['eom_frequency']:.1%}")

    with col3:
        st.metric("Volatility (CV)", f"{summary['volatility_cv']:.2f}")
        st.metric("Pattern (True)", summary.get("eom_pattern_primary", "Unknown"))


def render_prediction_info(predicted_pattern: str, confidence: float):
    """Render model prediction information"""
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Predicted Pattern:** {predicted_pattern}")
    with col2:
        st.write(f"**Confidence:** {confidence:.1%}")


def render_labeling_interface(
    classification_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    current_idx: int,
    human_labels: dict,
) -> tuple[int, dict, bool]:
    """Render the complete labeling interface"""

    total_series = len(classification_data)
    current_series = classification_data.iloc[current_idx]["dim_value"]

    # Navigation controls
    st.subheader(f"🏷️ Human Labeling - Series: {current_series}")
    new_idx, _ = render_navigation_controls(current_idx, total_series)

    # If index changed, update session state
    if new_idx != current_idx:
        st.session_state.current_series_idx = new_idx
        st.rerun()

    # Get current series data
    series_data, summary = get_timeseries_plot_data(raw_data, current_series)
    current_row = classification_data.iloc[current_idx]

    # Get model predictions
    predicted_pattern = current_row["eom_pattern_primary"]
    confidence = current_row["eom_pattern_confidence"]

    # Get current human label
    current_human_label = human_labels.get(current_series)

    # Pattern selection buttons
    selected_pattern = render_pattern_buttons(current_human_label)

    # Handle pattern selection
    labels_changed = False
    if selected_pattern:
        human_labels[current_series] = selected_pattern
        save_human_labels(human_labels)
        labels_changed = True

        # Auto-advance to next series
        if current_idx < total_series - 1:
            st.session_state.current_series_idx = current_idx + 1
            st.rerun()

    # Charts side by side
    chart_col, prob_col = st.columns([2, 1])

    with chart_col:
        # EOM time series chart
        fig = create_timeseries_chart(series_data, current_series)
        st.plotly_chart(fig, use_container_width=True)

    with prob_col:
        # Pattern probabilities
        prob_cols = [col for col in classification_data.columns if col.endswith("_probability")]
        if prob_cols:
            prob_data = []
            for col in prob_cols:
                pattern = col.replace("_probability", "").upper()
                prob_data.append({"Pattern": pattern, "Probability": current_row[col]})

            prob_df = pd.DataFrame(prob_data).sort_values("Probability", ascending=True)
            fig_probs = create_probability_chart(prob_df)
            st.plotly_chart(fig_probs, use_container_width=True)

        # Prediction info below probability chart
        render_prediction_info(predicted_pattern, confidence)

    # Series information
    render_series_info(series_data, summary, predicted_pattern, confidence)

    return new_idx, human_labels, labels_changed


def render_evaluation_metrics(classification_data: pd.DataFrame, human_labels: dict):
    """Render evaluation metrics"""
    if not human_labels:
        st.info("No human labels available yet. Start labeling to see evaluation metrics.")
        return

    st.subheader("📊 Evaluation Metrics")

    # Calculate metrics
    metrics = calculate_evaluation_metrics(classification_data, human_labels)

    # Display overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{metrics['macro_precision']:.1%}")
    with col3:
        st.metric("Recall", f"{metrics['macro_recall']:.1%}")
    with col4:
        st.metric("F1 Score", f"{metrics['macro_f1']:.1%}")

    # Show labeled count
    st.write(f"**Labeled Series:** {metrics['n_labeled']} / {len(classification_data)}")

    # Per-class metrics
    if metrics["class_metrics"]:
        st.subheader("Per-Class Metrics")
        class_df = pd.DataFrame(metrics["class_metrics"]).T
        st.dataframe(class_df.round(3), use_container_width=True)
