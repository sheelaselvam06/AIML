import streamlit as st
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# Set page config
st.set_page_config(page_title="Frozen Lake RL", layout="wide")

# Title
st.title("🎮 Frozen Lake Reinforcement Learning")
st.markdown("---")

# Sidebar for controls
st.sidebar.header("🎛️ Controls")

# Environment setup
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏔️ Environment Setup")
    
    # Environment parameters
    is_slippery = st.sidebar.checkbox("Make Slippery", value=False)
    map_size = st.sidebar.selectbox("Map Size", ["4x4", "8x8"], index=0)
    
    # Create environment
    if map_size == "4x4":
        env_name = "FrozenLake-v1"
        grid_size = 4
    else:
        env_name = "FrozenLake8x8-v1"
        grid_size = 8

    # Only create new environment if settings changed
    if 'env_settings' not in st.session_state or st.session_state.env_settings != (env_name, is_slippery):
        env = gym.make(env_name, is_slippery=is_slippery, render_mode="rgb_array")
        st.session_state.env = env
        st.session_state.env_settings = (env_name, is_slippery)
        st.session_state.state = env.reset()[0]
        st.session_state.episode = 1
        st.session_state.steps = 0
        st.session_state.total_reward = 0
        st.session_state.history = []
    else:
        env = st.session_state.env

    # Reset environment
    if st.button("🔄 Reset Environment"):
        st.session_state.state = env.reset()[0]
        st.session_state.episode = 1
        st.session_state.steps = 0
        st.session_state.total_reward = 0
        st.session_state.history = []
        st.rerun()

with col2:
    st.subheader("📊 Environment Info")
    st.write(f"**States**: {env.observation_space.n}")
    st.write(f"**Actions**: {env.action_space.n}")
    st.write(f"**Slippery**: {is_slippery}")
    st.write(f"**Map Size**: {map_size}")

# Main content
st.markdown("---")

# Action selection
st.subheader("🎯 Take Action")
col1, col2, col3, col4 = st.columns(4)

action_map = {0: "⬅️ Left", 1: "⬇️ Down", 2: "➡️ Right", 3: "⬆️ Up"}
selected_action = None

with col1:
    if st.button("⬅️ Left", key="left"):
        selected_action = 0
with col2:
    if st.button("⬇️ Down", key="down"):
        selected_action = 1
with col3:
    if st.button("➡️ Right", key="right"):
        selected_action = 2
with col4:
    if st.button("⬆️ Up", key="up"):
        selected_action = 3

# Store state in session
if 'state' not in st.session_state:
    st.session_state.state = env.reset()[0]
    st.session_state.episode = 1
    st.session_state.steps = 0
    st.session_state.total_reward = 0
    st.session_state.history = []
    st.session_state.env = env

# Use the stored environment
env = st.session_state.env

# Take action if selected
if selected_action is not None:
    result = env.step(selected_action)
    observation, reward, terminated, truncated, info = result
    
    st.session_state.state = observation
    st.session_state.total_reward += reward
    st.session_state.steps += 1
    
    # Record action
    st.session_state.history.append({
        'episode': st.session_state.episode,
        'step': st.session_state.steps,
        'action': action_map[selected_action],
        'state': observation,
        'reward': reward,
        'terminated': terminated
    })
    
    # Reset if episode ended
    if terminated or truncated:
        st.session_state.episode += 1
        st.session_state.steps = 0
        env.reset()
        st.success(f"Episode {st.session_state.episode - 1} completed! Total reward: {st.session_state.total_reward}")
        st.session_state.total_reward = 0

# Display environment
st.subheader("🏞️ Current State")
col1, col2 = st.columns([1, 2])

with col1:
    # Render environment
    rgb_array = env.render()
    st.image(rgb_array, caption="Frozen Lake Environment", use_column_width=True)

with col2:
    # Current stats
    st.subheader("📈 Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Episode", st.session_state.episode)
        st.metric("Steps", st.session_state.steps)
    
    with col2:
        st.metric("Total Reward", st.session_state.total_reward)
        st.metric("Current State", st.session_state.state)

# Action explanation
st.subheader("🎮 Action Legend")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("⬅️ **Left**: Move left")
with col2:
    st.info("⬇️ **Down**: Move down")
with col3:
    st.info("➡️ **Right**: Move right")
with col4:
    st.info("⬆️ **Up**: Move up")

# History
if st.session_state.history:
    st.subheader("📜 Action History")
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.history)
    
    # Show last 10 actions
    if len(df) > 10:
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)
    
    # Download history
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download History",
        data=csv,
        file_name="frozen_lake_history.csv",
        mime="text/csv"
    )

# Environment description
st.markdown("---")
st.subheader("🎯 Game Description")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **🏔️ Frozen Lake Rules:**
    - **S** = Start position
    - **F** = Frozen ice (safe)
    - **H** = Hole (game over)
    - **G** = Goal (reward +1)
    
    **🎯 Objective:**
    Reach the goal without falling in holes!
    """)

with col2:
    st.markdown("""
    **🎮 Controls:**
    - Use arrow buttons to move
    - Each move costs 0 reward
    - Goal gives +1 reward
    - Hole gives 0 reward (episode ends)
    
    **⚙️ Settings:**
    - **Slippery**: Random movement (harder)
    - **Non-slippery**: Deterministic (easier)
    """)

# Auto-play option
st.markdown("---")
st.subheader("🤖 Auto Play")

if st.button("🎲 Random Agent Play"):
    st.write("Running random agent for 10 episodes...")
    
    total_rewards = []
    episodes_data = []
    
    for episode in range(10):
        state = env.reset()[0]
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 50:
            action = env.action_space.sample()  # Random action
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        episodes_data.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': steps,
            'success': episode_reward > 0
        })
    
    # Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Success Rate", f"{sum(ep['success'] for ep in episodes_data)}/10")
    
    with col2:
        st.metric("Average Reward", f"{np.mean(total_rewards):.2f}")
    
    with col3:
        st.metric("Average Steps", f"{np.mean([ep['steps'] for ep in episodes_data]):.1f}")
    
    # Show episodes
    episodes_df = pd.DataFrame(episodes_data)
    st.dataframe(episodes_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("🤖 **Reinforcement Learning Demo** - Built with Streamlit & Gymnasium")
