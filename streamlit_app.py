import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from src.game import LewisGame
import json
from datetime import datetime

st.set_page_config(page_title="Lewis Signaling Game — Emergent Communication Playground", layout="wide")
st.title("🧬 Lewis Signaling Game")
st.markdown("**Multi-agent REINFORCE • Watch languages emerge • Now with export & presets**")

tab1, tab2, tab3, tab4 = st.tabs(["📖 What emergent languages teach us", "🎮 Interactive Dashboard", "📊 Results", "🏆 My Languages Gallery"])

# ... (your existing one-pager tab1 stays exactly the same) ...
with tab1:
    st.markdown("""### What emergent languages teach us about **AI alignment & communication** ...""")  # (same as before)

with tab2:
    st.sidebar.header("Simulation Parameters")
    # ... existing sliders ...

    col_preset, col_run = st.columns([1,3])
    with col_preset:
        preset = st.selectbox("Quick Presets", ["Custom", "Classic Lewis (2 states)", "Noisy Biology", "High-Vocab Challenge", "3-Agent Extension"])
        if preset != "Custom":
            st.info(f"Loaded {preset}")

    if st.button("▶️ Run Full Simulation", type="primary", use_container_width=True):
        # ... existing simulation logic (kept) ...

        # NEW: Save results for export
        results = {
            "timestamp": datetime.now().isoformat(),
            "params": {"n_states": n_states, "n_signals": n_signals, "lr": lr, "episodes": n_episodes, "seed": seed},
            "final_sender": game.sender.tolist(),
            "final_receiver": game.receiver.tolist(),
            "avg_reward": float(np.mean(rewards_history)),
            "coherence": float(coherence[-1]) if 'coherence' in locals() else None
        }

        st.download_button(
            label="💾 Download My Language (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"lewis_language_{seed}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

        # ... rest of your viz (heatmaps + plots) ...

with tab4:
    st.write("Your saved languages will appear here in future versions (local storage coming).")

st.caption("Made with ❤️ for NullLabTests • Pure NumPy + Streamlit • v2 with export & presets")
