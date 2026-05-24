#!/usr/bin/env python3
"""
Test SAC training script setup (environment and model instantiation).
Verifies that the SAC training pipeline can be initialized without errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_tracking_env import ResidualTrackingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def test_sac_setup():
    """Test SAC environment and model instantiation."""
    print("Testing SAC training setup...")
    
    # Create environment
    print("Creating ResidualTrackingEnv...")
    env = ResidualTrackingEnv(
        trajectory="circle",
        trajectory_kwargs={"radius": 0.15, "speed": 1.0},
        control_dt=0.03,
        max_episode_steps=100,
        pd_kp=200.0,
        pd_kd=50.0,
        residual_alpha=0.1,
        action_repeat=2,
        velocity_max=0.5,
    )
    
    print("Environment created successfully")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test environment reset and step
    print("\nTesting environment reset and step...")
    obs, info = env.reset()
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Observation shapes:")
    for key, val in obs.items():
        print(f"    {key}: {val.shape}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step successful, reward: {reward:.4f}")
    
    env.close()
    
    # Test with VecNormalize
    print("\nTesting with VecNormalize...")
    env = ResidualTrackingEnv(
        trajectory="circle",
        trajectory_kwargs={"radius": 0.15, "speed": 1.0},
        control_dt=0.03,
        max_episode_steps=100,
        pd_kp=200.0,
        pd_kd=50.0,
        residual_alpha=0.1,
        action_repeat=2,
        velocity_max=0.5,
    )
    
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    print("VecNormalize created successfully")
    
    # Test SAC model instantiation
    print("\nTesting SAC model instantiation...")
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    print(f"Using device: {device}")
    
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        device=device,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        gradient_steps=1,
        train_freq=1,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 256],
                qf=[512, 512, 256],
            ),
            activation_fn=torch.nn.SiLU,
        ),
    )
    
    print("SAC model created successfully")
    print(f"  Policy: {model.policy}")
    
    # Test a few training steps
    print("\nTesting training steps...")
    model.learn(total_timesteps=100, log_interval=10)
    print("Training steps completed successfully")
    
    vec_env.close()
    
    print("\n✓ All SAC setup tests passed")
    return True


if __name__ == "__main__":
    try:
        success = test_sac_setup()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
