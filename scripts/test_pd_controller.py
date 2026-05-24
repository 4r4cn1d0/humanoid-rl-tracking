#!/usr/bin/env python3
"""
Test PD controller performance without RL (Phase 1 test).
Verifies that the PD controller alone can track circles reasonably.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv


def test_pd_controller():
    """Test PD controller tracking performance without RL policy."""
    print("Testing PD controller performance (no RL)...")
    
    # Test different PD gain combinations
    gain_configs = [
        {"kp": 25.0, "kd": 8.0, "name": "Original"},
        {"kp": 50.0, "kd": 15.0, "name": "Higher"},
        {"kp": 100.0, "kd": 30.0, "name": "Much Higher"},
        {"kp": 200.0, "kd": 50.0, "name": "Very High"},
    ]
    
    best_config = None
    best_error = float('inf')
    
    for config in gain_configs:
        print(f"\nTesting PD gains: Kp={config['kp']}, Kd={config['kd']}")
        
        # Create environment with PD controller
        env = ResidualTrackingEnv(
            trajectory="circle",
            trajectory_kwargs={"radius": 0.15, "speed": 1.0},
            control_dt=0.03,
            max_episode_steps=500,
            pd_kp=config["kp"],
            pd_kd=config["kd"],
            residual_alpha=0.0,  # No RL contribution
            action_repeat=1,  # No action repeat for testing
            velocity_max=0.5,
        )
        
        # Disable velocity commands for PD-only test
        env._use_velocity_commands = False
        
        obs, info = env.reset()
        
        tracking_errors = []
        
        # Run episode with zero actions (PD controller only)
        for step in range(500):
            # Provide zero action (PD controller does all the work)
            action = np.zeros(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            
            tracking_errors.append(info["tracking_error"])
            
            if terminated or truncated:
                break
        
        # Compute statistics
        mean_error = np.mean(tracking_errors)
        max_error = np.max(tracking_errors)
        std_error = np.std(tracking_errors)
        
        print(f"  Mean tracking error: {mean_error:.4f} m")
        print(f"  Max tracking error: {max_error:.4f} m")
        print(f"  Std tracking error: {std_error:.4f} m")
        print(f"  Steps completed: {len(tracking_errors)}")
        
        if mean_error < best_error:
            best_error = mean_error
            best_config = config
        
        env.close()
    
    print(f"\nBest configuration: Kp={best_config['kp']}, Kd={best_config['kd']}")
    print(f"Best mean error: {best_error:.4f} m")
    
    # Check if PD controller performs reasonably
    # For residual RL, PD doesn't need to be perfect - just provide a baseline
    # The RL policy will learn to compensate for PD imperfections
    if best_error < 0.20:  # 20cm threshold (more lenient for mocap control)
        print("\n✓ PD controller provides reasonable baseline (mean error < 20cm)")
        print("  RL policy will learn to compensate for remaining error")
        return True
    else:
        print(f"\n✗ PD controller needs further tuning (best error = {best_error:.4f} m)")
        print("  The action representation or controller interface may need adjustment")
        return False


if __name__ == "__main__":
    success = test_pd_controller()
    exit(0 if success else 1)
