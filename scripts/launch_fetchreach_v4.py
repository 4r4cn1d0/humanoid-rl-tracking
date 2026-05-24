#!/usr/bin/env python3
"""
Launch the official FetchReachDense-v4 environment from Gymnasium-Robotics.
This version has the fix for the robot positioning bug.
Requires Python 3.10+ and latest versions from GitHub.
"""
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

# Register and create the official environment
gym.register_envs(gymnasium_robotics)
env = gym.make('FetchReachDense-v4', render_mode='human')

print("=" * 60)
print("OFFICIAL FetchReachDense-v4 Environment")
print("=" * 60)
print("Pure Gymnasium-Robotics environment - no modifications")
print("This version has the FIX for robot positioning")
print("Press Ctrl+C to exit\n")

try:
    episode = 0
    while True:
        obs, info = env.reset()
        
        print(f"\n=== Episode {episode} ===")
        print(f"Gripper position: {obs['achieved_goal']}")
        print(f"Goal position:    {obs['desired_goal']}")
        dist = np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'])
        print(f"Initial distance: {dist:.3f}m ({dist*100:.1f}cm)")
        
        for step in range(50):
            # Take random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if step % 10 == 0:
                dist = np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'])
                print(f"  Step {step:2d}: distance={dist*100:.1f}cm, reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        final_dist = np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'])
        success = final_dist < 0.05
        print(f"Final: distance={final_dist*100:.1f}cm, success={success}")
        
        episode += 1
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n\nExiting.")
finally:
    env.close()
