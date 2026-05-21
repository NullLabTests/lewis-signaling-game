import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.game import LewisGame

st.set_page_config(page_title="Lewis Signaling Game — Emergent Communication Playground", layout="wide")
st.title("🧬 Lewis Signaling Game")
st.markdown("**Multi-agent REINFORCE simulation • Watch a shared language emerge from scratch**")

# ================== ONE-PAGER TAB ==================
tab1, tab2, tab3 = st.tabs(["📖 What emergent languages teach us", "🎮 Interactive Dashboard", "📊 Results"])

with tab1:
    st.markdown("""
    ### What emergent languages teach us about **AI alignment & communication**

    In the **Lewis Signaling Game**, two agents (sender + receiver) have **no pre-coded language**.  
    They only discover meaning through trial-and-error + rewards.

    - Sender sees a hidden state → sends a signal  
    - Receiver sees only the signal → picks an action  
    - Both get rewarded **only** if action matches the true state

    Over episodes, **a shared symbolic language emerges purely from interaction**.

    This is one of the simplest models of:
    - **Emergent communication** in AI
    - **Grounded meaning** (symbols get meaning from successful coordination)
    - **Alignment without explicit programming** (no central designer)
    - Real-world parallels in molecular biology, ant pheromones, cell signaling

    **Why it matters for modern AI**:
    - Shows how agents can invent protocols without human supervision
    - Highlights risks/rewards of emergent behavior in multi-agent systems
    - Foundation for scalable oversight, interpretability, and safe AGI communication

    *Built as a minimal NumPy REINFORCE demo — pure self-organization.*
    """)

# ================== DASHBOARD TAB ==================
with tab2:
    st.sidebar.header("Simulation Parameters")

    n_states = st.sidebar.slider("Number of States (hidden world)", 2, 8, 3)
    n_signals = st.sidebar.slider("Vocabulary Size (signals)", 2, 12, 3)
    lr = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
    n_episodes = st.sidebar.slider("Number of Episodes", 500, 10000, 2000, step=500)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    if st.button("▶️ Run Full Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Training {n_episodes} episodes..."):
            game = LewisGame(n_states=n_states, n_signals=n_signals, lr=lr, seed=seed)

            rewards_history = []
            sender_history = []
            receiver_history = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for episode in range(n_episodes):
                state = game.rng.integers(n_states)
                signal, action, reward = game.step(state)
                game.update(state, signal, action, reward)

                rewards_history.append(reward)

                if episode % 100 == 0 or episode == n_episodes - 1:
                    sender_history.append(game.sender.copy())
                    receiver_history.append(game.receiver.copy())

                progress_bar.progress((episode + 1) / n_episodes)
                if episode % 200 == 0:
                    status_text.text(f"Episode {episode:,} • Avg reward so far: {np.mean(rewards_history):.3f}")

            st.success("✅ Simulation complete!")

            # Live signaling viz
            st.subheader("Live Agent Signaling Visualization")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Sender Policy: P(signal | state)")
                fig_sender = go.Figure(data=go.Heatmap(
                    z=game.sender,
                    x=[f"Signal {i}" for i in range(n_signals)],
                    y=[f"State {i}" for i in range(n_states)],
                    colorscale="Viridis"
                ))
                fig_sender.update_layout(height=400)
                st.plotly_chart(fig_sender, use_container_width=True)

            with col2:
                st.caption("Receiver Policy: P(action | signal)")
                fig_receiver = go.Figure(data=go.Heatmap(
                    z=game.receiver,
                    x=[f"Action {i}" for i in range(n_states)],
                    y=[f"Signal {i}" for i in range(n_signals)],
                    colorscale="Plasma"
                ))
                fig_receiver.update_layout(height=400)
                st.plotly_chart(fig_receiver, use_container_width=True)

            # Convergence plots
            st.subheader("Convergence Plots")
            col3, col4 = st.columns(2)
            with col3:
                fig_reward = go.Figure()
                fig_reward.add_trace(go.Scatter(
                    y=np.convolve(rewards_history, np.ones(100)/100, mode='valid'),
                    mode='lines', name='Rolling Avg Reward'
                ))
                fig_reward.update_layout(title="Reward Convergence", xaxis_title="Episode", yaxis_title="Avg Reward")
                st.plotly_chart(fig_reward, use_container_width=True)

            with col4:
                coherence = []
                for s, r in zip(sender_history, receiver_history):
                    coherence.append(np.max(s, axis=1).mean() * np.max(r, axis=1).mean())
                fig_coh = go.Figure(data=go.Scatter(y=coherence, mode='lines', name='Language Coherence'))
                fig_coh.update_layout(title="Emerging Language Coherence", xaxis_title="Snapshot", yaxis_title="Score")
                st.plotly_chart(fig_coh, use_container_width=True)

            st.info("🎉 A shared language has emerged! The heatmaps show the learned conventions.")

with tab3:
    st.write("Raw results would go here if you want to expand later.")

st.caption("Made with ❤️ for NullLabTests • Pure NumPy + Streamlit • Run locally or deploy to Hugging Face / Streamlit Community Cloud")
