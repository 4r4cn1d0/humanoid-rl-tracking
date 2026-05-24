#!/usr/bin/env python3
"""Check actual robot base position in the environment."""
import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)
env = gym.make("FetchReachDense-v3")
obs, _ = env.reset()

# Access the MuJoCo model
model = env.unwrapped.model
data = env.unwrapped.data

# Get robot base position
base_body_id = model.body("robot0:base_link").id
base_pos = data.xpos[base_body_id]

# Get joint positions
slide0 = data.qpos[model.joint("robot0:slide0").id]
slide1 = data.qpos[model.joint("robot0:slide1").id]
slide2 = data.qpos[model.joint("robot0:slide2").id]

print("=== ACTUAL ROBOT POSITIONS ===")
print(f"robot0:slide0 (X offset): {slide0:.4f}")
print(f"robot0:slide1 (Y offset): {slide1:.4f}")
print(f"robot0:slide2 (Z offset): {slide2:.4f}")
print(f"\nRobot base_link body position: {base_pos}")
print(f"\nGripper position: {obs['achieved_goal']}")
print(f"Goal position: {obs['desired_goal']}")

# Get table position
print(f"\n=== TABLE INFO ===")
table_body_id = model.body("table0").id
table_pos = data.xpos[table_body_id]
print(f"Table body position: {table_pos}")

env.close()
