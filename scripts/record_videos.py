#!/usr/bin/env python3
"""
Record videos of trained policy tracking various trajectories.
"""

import sys
from pathlib import Path
import imageio
from stable_baselines3 import SAC

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.pure_sac_env import PureSACEnv


def record_episode(model, trajectory_type, speed, filename, max_steps=200):
    """Record one episode as video."""
    
    # Determine trajectory kwargs
    if trajectory_type in ['circle', 'figure8']:
        traj_kwargs = {
            'speed': speed,
            'radius': 0.15,
            'center': [1.34, 0.75, 0.53]
        }
    else:  # spline
        traj_kwargs = {'speed': speed}
    
    env = PureSACEnv(
        render_mode='rgb_array',
        trajectory=trajectory_type,
        trajectory_kwargs=traj_kwargs,
        control_dt=0.05,
        max_episode_steps=max_steps,
        obs_noise_std=0.0,
        action_noise_std=0.0,
        action_delay=0,
        success_threshold=0.03,
    )
    
    frames = []
    obs, _ = env.reset()
    done = False
    step = 0
    
    print(f"  Recording {trajectory_type} @ {speed} m/s...", end='', flush=True)
    
    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    
    env.close()
    
    if frames:
        imageio.mimsave(filename, frames, fps=20)
        print(f" ✓ Saved ({len(frames)} frames)")
    else:
        print(f" ✗ No frames captured")
    
    return len(frames) > 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Record videos of trained policy')
    parser.add_argument('--model', type=str, default='pure_sac_model.zip',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='outputs/videos',
                        help='Output directory for videos')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("VIDEO RECORDING")
    print(f"{'='*60}\n")
    print(f"Loading model: {args.model}")
    
    try:
        model = SAC.load(args.model)
        print("✓ Model loaded successfully\n")
    except FileNotFoundError:
        print(f"✗ Error: Model file '{args.model}' not found!")
        return
    
    # Videos to record
    videos = [
        ('circle', 0.6, 'circle_slow.mp4'),
        ('circle', 1.2, 'circle_fast.mp4'),
        ('spline', 0.9, 'spline_medium.mp4'),
        ('figure8', 0.9, 'figure8_medium.mp4'),
    ]
    
    print("Recording videos:")
    print(f"{'-'*60}\n")
    
    success_count = 0
    for traj_type, speed, filename in videos:
        filepath = output_dir / filename
        if record_episode(model, traj_type, speed, str(filepath)):
            success_count += 1
    
    print(f"\n{'-'*60}")
    print(f"Recorded {success_count}/{len(videos)} videos successfully")
    print(f"Videos saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
