"""
Visualization components for Streamlit app
"""

import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go


from src.segmentation.pandas_classification import ArchetypeConfig


def create_archetype_visualization(archetype_config: ArchetypeConfig):
    """Create 3D visualization of archetypes"""

    # Archetype data
    archetypes = {
        "CONTINUOUS_STABLE": archetype_config.continuous_stable,
        "CONTINUOUS_VOLATILE": archetype_config.continuous_volatile,
        "INTERMITTENT_ACTIVE": archetype_config.intermittent_active,
        "INTERMITTENT_DORMANT": archetype_config.intermittent_dormant,
        "RARE_RECENT": archetype_config.rare_recent,
        "RARE_STALE": archetype_config.rare_stale,
    }

    # Create 3D scatter plot
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, (name, (reg, stab, rec)) in enumerate(archetypes.items()):
        fig.add_trace(
            go.Scatter3d(
                x=[reg],
                y=[stab],
                z=[rec],
                mode="markers+text",
                marker=dict(size=15, color=colors[i % len(colors)]),
                text=[name],
                textposition="top center",
                name=name,
                hovertemplate=f"<b>{name}</b><br>"
                + f"Regularity: {reg}<br>"
                + f"Stability: {stab}<br>"
                + f"Recency: {rec}<extra></extra>",
            )
        )

    fig.update_layout(
        title="EOM Pattern Archetypes (Centroids)",
        scene=dict(
            xaxis_title="Regularity Score",
            yaxis_title="Stability Score",
            zaxis_title="Recency Score",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
            zaxis=dict(range=[0, 100]),
        ),
        height=600,
        showlegend=True,
    )

    return fig


def create_pattern_distribution_chart(classified_data: pd.DataFrame):
    """Create EOM pattern distribution chart"""
    pattern_counts = classified_data["eom_pattern_primary"].value_counts()

    fig_dist = px.bar(
        x=pattern_counts.index,
        y=pattern_counts.values,
        title="EOM Pattern Distribution",
        labels={"x": "EOM Pattern", "y": "Count"},
        color=pattern_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_dist.update_layout(xaxis_tickangle=-45)
    return fig_dist


def create_importance_distribution_chart(classified_data: pd.DataFrame):
    """Create EOM importance distribution pie chart"""
    importance_counts = classified_data["eom_importance_tier"].value_counts()
    fig_imp = px.pie(values=importance_counts.values, names=importance_counts.index, title="EOM Importance Distribution")
    return fig_imp


def create_timeseries_chart(series_data: pd.DataFrame, current_series: str):
    """Create EOM time series chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series_data["forecast_month"],
            y=series_data["target_eom_amount"],
            mode="lines+markers",
            name="EOM Amount",
            line=dict(color="red", width=3),
            marker=dict(size=6),
        )
    )

    fig.update_layout(
        height=400,
        title=f"EOM Pattern for {current_series}",
        xaxis_title="Month",
        yaxis_title="EOM Amount",
        showlegend=False,
        hovermode="x unified",
    )

    return fig


def create_probability_chart(prob_df: pd.DataFrame):
    """Create pattern probabilities chart"""
    fig_probs = px.bar(prob_df, x="Probability", y="Pattern", orientation="h", title="Pattern Probabilities", height=280)
    fig_probs.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig_probs


def render_classification_results(classified_data: pd.DataFrame):
    """Render classification results visualization"""
    st.subheader("EOM Classification Results")

    # Show EOM pattern distribution
    fig_dist = create_pattern_distribution_chart(classified_data)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Show EOM importance distribution
    fig_imp = create_importance_distribution_chart(classified_data)
    st.plotly_chart(fig_imp, use_container_width=True)


def render_detailed_results_table(classified_data: pd.DataFrame):
    """Render detailed classification results table"""
    st.subheader("Detailed Classification Results")

    # Filter and display options
    col1, col2, col3 = st.columns(3)
    with col1:
        pattern_filter = st.multiselect(
            "Filter by Pattern",
            options=classified_data["eom_pattern_primary"].unique(),
            default=classified_data["eom_pattern_primary"].unique(),
        )
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
    with col3:
        max_rows = st.number_input("Max Rows to Display", 10, 1000, 50)

    # Apply filters
    filtered_data = classified_data[
        (classified_data["eom_pattern_primary"].isin(pattern_filter))
        & (classified_data["eom_pattern_confidence"] >= min_confidence)
    ].head(max_rows)

    # Display table
    display_cols = [
        "dim_value",
        "eom_pattern_primary",
        "eom_pattern_confidence",
        "regularity_score",
        "stability_score",
        "recency_score",
        "eom_importance_tier",
    ]

    st.dataframe(filtered_data[display_cols].round(3), use_container_width=True, height=400)

    return filtered_data
