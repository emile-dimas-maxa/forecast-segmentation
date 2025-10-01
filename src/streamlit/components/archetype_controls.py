"""
Archetype configuration controls for Streamlit app
"""

import streamlit as st
from src.segmentation.pandas_classification import ArchetypeConfig


def render_archetype_info():
    """Render information about archetype dimensions"""
    st.markdown("Adjust the centroids for each EOM pattern. Each pattern is defined by three dimensions:")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info("**Regularity**: How consistently the pattern occurs")
    with col_info2:
        st.info("**Stability**: How predictable the pattern magnitude is")
    with col_info3:
        st.info("**Recency**: How recently the pattern was observed")


def render_continuous_patterns_tab(archetype_config: ArchetypeConfig) -> ArchetypeConfig:
    """Render continuous patterns configuration tab"""
    st.markdown("**Continuous patterns occur regularly with predictable timing**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### CONTINUOUS_STABLE")
        cs_reg = st.slider(
            "Regularity",
            0,
            100,
            archetype_config.continuous_stable[0],
            key="cs_reg",
            help="High regularity - occurs consistently",
        )
        cs_stab = st.slider(
            "Stability",
            0,
            100,
            archetype_config.continuous_stable[1],
            key="cs_stab",
            help="High stability - predictable amounts",
        )
        cs_rec = st.slider(
            "Recency", 0, 100, archetype_config.continuous_stable[2], key="cs_rec", help="Medium recency - ongoing pattern"
        )
        archetype_config.continuous_stable = (cs_reg, cs_stab, cs_rec)
    
    with col2:
        st.markdown("##### CONTINUOUS_VOLATILE")
        cv_reg = st.slider(
            "Regularity",
            0,
            100,
            archetype_config.continuous_volatile[0],
            key="cv_reg",
            help="High regularity - occurs consistently",
        )
        cv_stab = st.slider(
            "Stability",
            0,
            100,
            archetype_config.continuous_volatile[1],
            key="cv_stab",
            help="Low stability - unpredictable amounts",
        )
        cv_rec = st.slider(
            "Recency", 0, 100, archetype_config.continuous_volatile[2], key="cv_rec", help="Medium recency - ongoing pattern"
        )
        archetype_config.continuous_volatile = (cv_reg, cv_stab, cv_rec)
    
    return archetype_config


def render_intermittent_patterns_tab(archetype_config: ArchetypeConfig) -> ArchetypeConfig:
    """Render intermittent patterns configuration tab"""
    st.markdown("**Intermittent patterns occur sporadically with gaps**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### INTERMITTENT_ACTIVE")
        ia_reg = st.slider(
            "Regularity", 0, 100, archetype_config.intermittent_active[0], key="ia_reg", help="Medium regularity - some gaps"
        )
        ia_stab = st.slider(
            "Stability",
            0,
            100,
            archetype_config.intermittent_active[1],
            key="ia_stab",
            help="Medium stability - moderate predictability",
        )
        ia_rec = st.slider(
            "Recency", 0, 100, archetype_config.intermittent_active[2], key="ia_rec", help="High recency - recently active"
        )
        archetype_config.intermittent_active = (ia_reg, ia_stab, ia_rec)
    
    with col2:
        st.markdown("##### INTERMITTENT_DORMANT")
        id_reg = st.slider(
            "Regularity", 0, 100, archetype_config.intermittent_dormant[0], key="id_reg", help="Medium regularity - some gaps"
        )
        id_stab = st.slider(
            "Stability",
            0,
            100,
            archetype_config.intermittent_dormant[1],
            key="id_stab",
            help="Medium stability - moderate predictability",
        )
        id_rec = st.slider(
            "Recency", 0, 100, archetype_config.intermittent_dormant[2], key="id_rec", help="Low recency - not recently active"
        )
        archetype_config.intermittent_dormant = (id_reg, id_stab, id_rec)
    
    return archetype_config


def render_rare_patterns_tab(archetype_config: ArchetypeConfig) -> ArchetypeConfig:
    """Render rare patterns configuration tab"""
    st.markdown("**Rare patterns occur infrequently and unpredictably**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### RARE_RECENT")
        rr_reg = st.slider(
            "Regularity", 0, 100, archetype_config.rare_recent[0], key="rr_reg", help="Low regularity - very infrequent"
        )
        rr_stab = st.slider(
            "Stability",
            0,
            100,
            archetype_config.rare_recent[1],
            key="rr_stab",
            help="Any stability - less important for rare patterns",
        )
        rr_rec = st.slider(
            "Recency", 0, 100, archetype_config.rare_recent[2], key="rr_rec", help="High recency - happened recently"
        )
        archetype_config.rare_recent = (rr_reg, rr_stab, rr_rec)
    
    with col2:
        st.markdown("##### RARE_STALE")
        rs_reg = st.slider(
            "Regularity", 0, 100, archetype_config.rare_stale[0], key="rs_reg", help="Low regularity - very infrequent"
        )
        rs_stab = st.slider(
            "Stability",
            0,
            100,
            archetype_config.rare_stale[1],
            key="rs_stab",
            help="Any stability - less important for rare patterns",
        )
        rs_rec = st.slider(
            "Recency", 0, 100, archetype_config.rare_stale[2], key="rs_rec", help="Low recency - hasn't happened recently"
        )
        archetype_config.rare_stale = (rs_reg, rs_stab, rs_rec)
    
    return archetype_config


def render_archetype_controls() -> ArchetypeConfig:
    """Render the complete archetype configuration interface"""
    st.subheader("🎛️ Archetype Configuration")
    render_archetype_info()

    # Organize archetypes in tabs for better UX
    tab1, tab2, tab3 = st.tabs(["📈 Continuous Patterns", "🔄 Intermittent Patterns", "⚡ Rare Patterns"])
    
    archetype_config = ArchetypeConfig()
    
    with tab1:
        archetype_config = render_continuous_patterns_tab(archetype_config)
    
    with tab2:
        archetype_config = render_intermittent_patterns_tab(archetype_config)
    
    with tab3:
        archetype_config = render_rare_patterns_tab(archetype_config)
    
    return archetype_config
