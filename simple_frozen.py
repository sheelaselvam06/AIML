import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

print("🎮 Frozen Lake Reinforcement Learning Demo")
print("=" * 50)

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# Environment info
print(f"📊 Environment Info:")
print(f"   States: {env.observation_space.n}")
print(f"   Actions: {env.action_space.n}")
print(f"   Action Map: 0=Left, 1=Down, 2=Right, 3=Up")

# Reset environment
state, info = env.reset()
print(f"🏁 Starting State: {state}")

# Define a simple path to goal
# Path: Right, Right, Down, Down, Right, Right
actions = [2, 2, 1, 1, 2, 2]

print(f"\n🚀 Taking actions: {actions}")
print("-" * 50)

for i, action in enumerate(actions):
    # Take action
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Action name
    action_names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    action_name = action_names[action]
    
    print(f"Step {i+1}: {action_name} (action={action})")
    print(f"   State: {state} → {next_state}")
    print(f"   Reward: {reward}")
    print(f"   Done: {terminated or truncated}")
    
    state = next_state
    
    if terminated or truncated:
        if reward > 0:
            print("🎉 GOAL REACHED! Success!")
        else:
            print("❌ Fell in a hole! Game Over!")
        break

# Save final state visualization
plt.figure(figsize=(6, 6))
rgb_array = env.render()
plt.imshow(rgb_array)
plt.title("Frozen Lake - Final State")
plt.axis('off')
plt.savefig('frozen_lake_simple.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n📸 Final state saved as 'frozen_lake_simple.png'")
print(f"🏁 Final State: {state}")
print(f"🎯 Total Reward: {reward if 'reward' in locals() else 0}")

env.close()
print("\n✅ Demo completed!")
