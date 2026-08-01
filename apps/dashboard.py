import streamlit as st
import pandas as pd
import plotly.express as px
from eateot import EATEOT_TRACK_PROFILES

st.set_page_config(page_title="AI Cognitive Degradation Lab", layout="wide")
st.title("🧠 EATEOT Neural Degradation Diagnostic Suite")

# Sidebar Controls
st.sidebar.header("Degradation Parameters")
track = st.sidebar.selectbox("Select Track Profile", list(EATEOT_TRACK_PROFILES.keys()))
decay_mult = st.sidebar.slider("Decay Multiplier", 0.1, 3.0, 1.0, 0.1)
subnetwork = st.sidebar.radio("Target Sub-Network", ["all", "attn", "mlp", "norm"])

# Run Test Button
if st.sidebar.button("🧪 Run Neural IQ Battery"):
    st.subheader(f"Diagnostic Results: Track {track} ({decay_mult}x Decay on [{subnetwork.upper()}])")

    # 1. Collect benchmark results from your engine
    # (Extract domain scores into a dictionary: {'Set Logic': 0, 'Relational': 50, ...})
    results = {
        "Categorical": 100,
        "Numerical": 85,
        "Syllogism": 70,
        "Relational Memory": 40,
        "Abstract Set Logic": 0
    }

    df = pd.DataFrame(list(results.items()), columns=["Domain", "Score"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🕸️ Cognitive Footprint (Radar Chart)")
        fig_radar = px.line_polar(df, r='Score', theta='Domain', line_close=True, range_r=[0,100])
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("### 📊 Domain Accuracy Breakdown")
        fig_bar = px.bar(df, x='Domain', y='Score', color='Score', range_y=[0,100])
        st.plotly_chart(fig_bar, use_container_width=True)
