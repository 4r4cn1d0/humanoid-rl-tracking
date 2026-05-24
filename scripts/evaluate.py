#!/usr/bin/env python3
"""
Evaluation script for 3D end-effector tracking challenge.
Tests trained policy on various trajectories with clean conditions.
Generates plots and performance metrics.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.pure_sac_env import PureSACEnv


def evaluate_trajectory(model, trajectory_type, speed, n_episodes=10, render=False):
    """Evaluate policy on a specific trajectory."""
    
    # Determine trajectory kwargs based on type
    if trajectory_type in ['circle', 'figure8']:
        traj_kwargs = {
            'speed': speed,
            'radius': 0.15,
            'center': [1.34, 0.75, 0.53]
        }
    else:  # spline
        traj_kwargs = {'speed': speed}
    
    env = PureSACEnv(
        render_mode='human' if render else None,
        trajectory=trajectory_type,
        trajectory_kwargs=traj_kwargs,
        control_dt=0.05,
        max_episode_steps=200,
        obs_noise_std=0.0,  # Clean conditions
        action_noise_std=0.0,
        action_delay=0,
        success_threshold=0.03,
    )
    
    successes = []
    errors = []
    all_positions = []
    all_targets = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_errors = []
        ep_positions = []
        ep_targets = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_errors.append(info['tracking_error'])
            ep_positions.append(obs['achieved_goal'].copy())
            ep_targets.append(obs['desired_goal'].copy())
            
            done = terminated or truncated
        
        successes.append(info['is_success'])
        errors.extend(ep_errors)
        all_positions.append(np.array(ep_positions))
        all_targets.append(np.array(ep_targets))
    
    env.close()
    
    return {
        'success_rate': np.mean(successes) * 100,
        'mean_error': np.mean(errors) * 100,  # cm
        'std_error': np.std(errors) * 100,
        'max_error': np.max(errors) * 100,
        'positions': all_positions,
        'targets': all_targets,
    }


def plot_results(results, trajectory_type, speed, output_dir='outputs'):
    """Generate evaluation plots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot 1: XY trajectory (first episode)
    ax = axes[0, 0]
    pos = results['positions'][0]
    tgt = results['targets'][0]
    ax.plot(tgt[:, 0], tgt[:, 1], 'b-', label='Target', linewidth=2.5, alpha=0.8)
    ax.plot(pos[:, 0], pos[:, 1], 'r--', label='Actual', linewidth=2)
    ax.scatter(pos[0, 0], pos[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(pos[-1, 0], pos[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title(f'{trajectory_type.capitalize()} Trajectory (XY Plane)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Tracking error over time
    ax = axes[0, 1]
    errors_cm = np.linalg.norm(results['positions'][0] - results['targets'][0], axis=1) * 100
    ax.plot(errors_cm, 'r-', linewidth=2, label='Tracking Error')
    ax.axhline(3.0, color='g', linestyle='--', linewidth=2, label='Success Threshold (3cm)')
    ax.fill_between(range(len(errors_cm)), 0, 3.0, alpha=0.2, color='green')
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Tracking Error (cm)', fontsize=11)
    ax.set_title('Tracking Error Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    ax = axes[1, 0]
    all_errors = []
    for pos, tgt in zip(results['positions'], results['targets']):
        all_errors.extend(np.linalg.norm(pos - tgt, axis=1) * 100)
    ax.hist(all_errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(3.0, color='g', linestyle='--', linewidth=2.5, label='Success Threshold')
    ax.axvline(results['mean_error'], color='r', linestyle='-', linewidth=2, label=f'Mean: {results["mean_error"]:.2f}cm')
    ax.set_xlabel('Tracking Error (cm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Distribution (All Episodes)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
╔══════════════════════════════════════╗
║     EVALUATION SUMMARY               ║
╚══════════════════════════════════════╝

Trajectory:     {trajectory_type.capitalize()}
Speed:          {speed} m/s
Episodes:       10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics:

  Success Rate:   {results['success_rate']:6.1f}%
  Mean Error:     {results['mean_error']:6.2f} cm
  Std Error:      {results['std_error']:6.2f} cm
  Max Error:      {results['max_error']:6.2f} cm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Conditions:     Clean (no noise/delay)
Threshold:      3.0 cm
    """
    ax.text(0.05, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    filename = f'{output_dir}/eval_{trajectory_type}_speed{speed:.1f}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot: {filename}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained SAC policy')
    parser.add_argument('--model', type=str, default='pure_sac_model.zip',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per trajectory')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes (slower)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Load trained model
    print(f"\n{'='*60}")
    print("3D END-EFFECTOR TRACKING - EVALUATION")
    print(f"{'='*60}\n")
    print(f"Loading model: {args.model}")
    
    try:
        model = SAC.load(args.model)
        print("✓ Model loaded successfully\n")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{args.model}' not found!")
        print("  Train a model first using: python train/train_pure_sac.py")
        return
    
    # Test configurations
    tests = [
        ('circle', 0.6),
        ('circle', 0.9),
        ('circle', 1.2),
        ('spline', 0.6),
        ('spline', 0.9),
        ('figure8', 0.6),
        ('figure8', 0.9),
    ]
    
    print(f"{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for traj_type, speed in tests:
        print(f"Evaluating {traj_type:8s} @ speed {speed:.1f} m/s...")
        results = evaluate_trajectory(model, traj_type, speed, 
                                     n_episodes=args.episodes,
                                     render=args.render)
        
        print(f"  Success Rate: {results['success_rate']:5.1f}%")
        print(f"  Mean Error:   {results['mean_error']:5.2f} cm")
        print(f"  Max Error:    {results['max_error']:5.2f} cm")
        
        plot_results(results, traj_type, speed, args.output_dir)
        all_results.append((traj_type, speed, results))
        print()
    
    # Print summary table
    print(f"{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}\n")
    print(f"{'Trajectory':<12} {'Speed':<8} {'Success':<10} {'Mean Err':<10} {'Max Err':<10}")
    print(f"{'-'*60}")
    for traj_type, speed, results in all_results:
        print(f"{traj_type:<12} {speed:<8.1f} {results['success_rate']:>6.1f}%   "
              f"{results['mean_error']:>6.2f} cm  {results['max_error']:>6.2f} cm")
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Plots saved to '{args.output_dir}/'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
