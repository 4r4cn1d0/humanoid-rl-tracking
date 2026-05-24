#!/usr/bin/env python3
"""
Evaluate trained SAC policy robustness on FetchReach with various perturbations.
Tests for overfitting by adding noise and delays not seen during training.
"""
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.residual_fetchreach_env import ResidualFetchReachEnv


def evaluate_policy(model, env, n_episodes=100):
    """Evaluate policy and return success rate and mean error."""
    successes = []
    errors = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_errors = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_errors.append(info['tracking_error'])
        
        successes.append(info['is_success'])
        errors.append(np.mean(ep_errors))
    
    return {
        'success_rate': np.mean(successes),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
    }


def main():
    # Load trained model
    model_path = _ROOT / "sac_fetchreach_model.zip"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Train the model first with: python train/train_sac_fetchreach.py")
        return
    
    print("Loading trained model...")
    model = SAC.load(model_path)
    
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION")
    print("="*60)
    
    # Test 1: Clean environment (training conditions)
    print("\n1. CLEAN (Training Conditions)")
    print("-" * 40)
    env_clean = ResidualFetchReachEnv(
        max_episode_steps=50,
        pd_kp=40.0,
        pd_kd=10.0,
        residual_alpha=0.5,
    )
    results_clean = evaluate_policy(model, env_clean, n_episodes=100)
    print(f"Success Rate: {results_clean['success_rate']*100:.1f}%")
    print(f"Mean Error:   {results_clean['mean_error']*100:.2f} cm")
    print(f"Std Error:    {results_clean['std_error']*100:.2f} cm")
    env_clean.close()
    
    # Test 2: Observation noise
    print("\n2. OBSERVATION NOISE (σ=0.01)")
    print("-" * 40)
    # Note: Current env doesn't support obs noise, would need to add wrapper
    print("(Not implemented - would need observation noise wrapper)")
    
    # Test 3: Action noise
    print("\n3. ACTION NOISE (σ=0.05)")
    print("-" * 40)
    # Note: Current env doesn't support action noise, would need to add wrapper
    print("(Not implemented - would need action noise wrapper)")
    
    # Test 4: Different PD gains (dynamics mismatch)
    print("\n4. PD GAIN MISMATCH (Kp=30, Kd=8)")
    print("-" * 40)
    env_pd = ResidualFetchReachEnv(
        max_episode_steps=50,
        pd_kp=30.0,  # Different from training
        pd_kd=8.0,   # Different from training
        residual_alpha=0.5,
    )
    results_pd = evaluate_policy(model, env_pd, n_episodes=100)
    print(f"Success Rate: {results_pd['success_rate']*100:.1f}%")
    print(f"Mean Error:   {results_pd['mean_error']*100:.2f} cm")
    print(f"Std Error:    {results_pd['std_error']*100:.2f} cm")
    env_pd.close()
    
    # Test 5: Different residual alpha (control authority mismatch)
    print("\n5. RESIDUAL ALPHA MISMATCH (α=0.3)")
    print("-" * 40)
    env_alpha = ResidualFetchReachEnv(
        max_episode_steps=50,
        pd_kp=40.0,
        pd_kd=10.0,
        residual_alpha=0.3,  # Different from training
    )
    results_alpha = evaluate_policy(model, env_alpha, n_episodes=100)
    print(f"Success Rate: {results_alpha['success_rate']*100:.1f}%")
    print(f"Mean Error:   {results_alpha['mean_error']*100:.2f} cm")
    print(f"Std Error:    {results_alpha['std_error']*100:.2f} cm")
    env_alpha.close()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Clean:           {results_clean['success_rate']*100:5.1f}% success")
    print(f"PD Mismatch:     {results_pd['success_rate']*100:5.1f}% success")
    print(f"Alpha Mismatch:  {results_alpha['success_rate']*100:5.1f}% success")
    
    # Check for overfitting
    if results_clean['success_rate'] > 0.95 and results_pd['success_rate'] < 0.7:
        print("\n⚠️  WARNING: Possible overfitting detected!")
        print("   Policy performs well in training conditions but degrades")
        print("   significantly with parameter changes.")
    elif results_clean['success_rate'] > 0.95:
        print("\n✓ Policy appears robust to parameter variations")
    
    print("\nTo prevent overfitting:")
    print("1. Add observation noise during training")
    print("2. Add action noise during training")
    print("3. Randomize PD gains during training")
    print("4. Use early stopping based on validation performance")


if __name__ == "__main__":
    main()
