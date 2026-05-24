#!/usr/bin/env python3
"""
Visualize the Fetch robot workspace and distances.
Shows:
- Robot base position
- Gripper start position
- Table position and size
- Reachable workspace boundaries
- Sample goal positions
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    print("=== Fetch Robot Workspace Visualization ===\n")
    
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchReachDense-v3", render_mode="human", max_episode_steps=200)
    
    print("ROBOT CONFIGURATION:")
    print("-" * 60)
    
    # Reset to get initial state
    obs, _ = env.reset(seed=42)
    
    # Get key positions
    gripper_pos = obs["achieved_goal"]
    goal_pos = obs["desired_goal"]
    
    print(f"Robot base (MODIFIED):     [1.05, 0.75, 0.0] m (at table back edge, centered)")
    print(f"Gripper start position:    {gripper_pos} m")
    print(f"Sample goal position:      {goal_pos} m")
    print(f"\nTable (ORIGINAL):")
    print(f"  Position:                [1.3, 0.75, 0.2] m")
    print(f"  Size (half-extents):     [0.25, 0.35, 0.2] m")
    print(f"  Full dimensions:         [0.50, 0.70, 0.40] m (W x D x H)")
    print(f"  Surface height:          0.40 m")
    print(f"  X range:                 [1.05, 1.55] m")
    print(f"  Y range:                 [0.40, 1.10] m")
    
    print(f"\nDISTANCES:")
    print("-" * 60)
    
    # Distance from base to gripper
    base_pos = np.array([1.05, 0.75, 0.0])
    dist_base_to_gripper = np.linalg.norm(gripper_pos - base_pos)
    print(f"Base to gripper start:     {dist_base_to_gripper:.3f} m")
    
    # Distance from base to table center
    table_center = np.array([1.3, 0.75, 0.40])
    dist_base_to_table = np.linalg.norm(table_center[:2] - base_pos[:2])
    print(f"Base to table center (XY): {dist_base_to_table:.3f} m (0.25m forward)")
    
    # Distance gripper to goal
    dist_gripper_to_goal = np.linalg.norm(goal_pos - gripper_pos)
    print(f"Gripper to goal (initial): {dist_gripper_to_goal:.3f} m")
    
    print(f"\nWORKSPACE BOUNDARIES:")
    print("-" * 60)
    print(f"Goal sampling range:       ±0.15 m from gripper start")
    print(f"Approximate X range:       [1.19, 1.49] m")
    print(f"Approximate Y range:       [0.60, 0.90] m")
    print(f"Approximate Z range:       [0.40, 0.70] m")
    print(f"Success threshold:         0.05 m (5 cm)")
    
    print(f"\nCONTROL PARAMETERS:")
    print("-" * 60)
    print(f"Control frequency:         25 Hz")
    print(f"Action scaling:            0.05 (actions multiplied by 0.05)")
    print(f"Max displacement per step: 0.05 m (5 cm)")
    print(f"Episode length:            50 steps (2 seconds)")
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VISUALIZATION")
    print("=" * 60)
    print("The MuJoCo viewer is now open.")
    print("- Red sphere = goal position")
    print("- Robot will hold position (zero action)")
    print("- Press Ctrl+C to exit\n")
    
    try:
        episode = 0
        while True:
            obs, _ = env.reset(seed=episode)
            gripper_pos = obs["achieved_goal"]
            goal_pos = obs["desired_goal"]
            dist = np.linalg.norm(goal_pos - gripper_pos)
            
            print(f"\nEpisode {episode}:")
            print(f"  Gripper: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
            print(f"  Goal:    [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
            print(f"  Distance: {dist:.3f} m ({dist*100:.1f} cm)")
            print(f"  Goal offset from gripper start:")
            offset = goal_pos - gripper_pos
            print(f"    ΔX: {offset[0]:+.3f} m, ΔY: {offset[1]:+.3f} m, ΔZ: {offset[2]:+.3f} m")
            
            # Hold position for 3 seconds to observe
            for step in range(75):  # 3 seconds at 25 Hz
                obs, _, terminated, truncated, _ = env.step(np.zeros(4))
                env.render()
                time.sleep(0.04)
                
                if terminated or truncated:
                    break
            
            episode += 1
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\nExiting visualization.")
    finally:
        env.close()
        
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
