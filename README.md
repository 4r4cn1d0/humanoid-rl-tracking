# 3D End-Effector Trajectory Tracking with Pure SAC

A reinforcement learning system that trains a Franka Panda arm (via FetchReachDense-v4) to track time-varying 3D trajectories using **Soft Actor-Critic (SAC)** with a **performance-gated curriculum**. No hand-tuned controllers — the policy learns everything from scratch.

---

## Results

**Best checkpoint: `best_model.zip` (175k training steps)**

| Condition | RMSE | Mean Error | Steps <3cm | Steps <5cm |
|---|---|---|---|---|
| Circle (clean) | 1.17 cm | 0.86 cm | **98.5%** | 99.0% |
| Figure-8 (clean) | 1.93 cm | 1.78 cm | **94.8%** | 99.5% |
| Spline (clean) | 3.53 cm | 3.01 cm | 68.3% | 91.5% |
| Circle + obs noise | 2.08 cm | 1.84 cm | **92.0%** | 98.8% |
| Circle + action noise | 1.18 cm | 0.88 cm | **98.5%** | 99.0% |
| Circle + 1-step delay | 1.45 cm | 0.99 cm | **98.5%** | 98.5% |

> All metrics use `control_dt=0.05`, `max_episode_steps=200`, `success_threshold=3cm` — exact training parameters.

---

## Evaluation Plots

### Summary: Tracking Error by Condition
<img src="outputs/best_model_eval/plots/tracking_error.png" width="600"/>

### Summary: Step-Level Success Rate by Condition
<img src="outputs/best_model_eval/plots/success_rate.png" width="600"/>

### Checkpoint Scan — Finding the Best Model
<img src="outputs/best_model_eval/plots/checkpoint_scan.png" width="600"/>

> Performance was strong from 100k–600k steps, then collapsed when the curriculum pushed into harder spline stages (curriculum forgetting). Best checkpoint: **175k steps**.

---

### Per-Condition Results

Each condition shows: **trajectory path XY** (target vs achieved) | **tracking error + action smoothness over time**

---

#### Circle — Clean
<table><tr>
<td><img src="outputs/best_model_eval/circle_clean/ep0_path_xy.png" width="300"/></td>
<td><img src="outputs/best_model_eval/circle_clean/ep0_error_smoothness.png" width="380"/></td>
</tr></table>

RMSE: **1.17 cm** | Mean error: **0.86 cm** | Steps <3cm: **98.5%**

---

#### Figure-8 — Clean
<table><tr>
<td><img src="outputs/best_model_eval/figure8_clean/ep0_path_xy.png" width="300"/></td>
<td><img src="outputs/best_model_eval/figure8_clean/ep0_error_smoothness.png" width="380"/></td>
</tr></table>

RMSE: **1.93 cm** | Mean error: **1.78 cm** | Steps <3cm: **94.8%**

---

#### Spline — Clean
<table><tr>
<td><img src="outputs/best_model_eval/spline_clean/ep0_path_xy.png" width="300"/></td>
<td><img src="outputs/best_model_eval/spline_clean/ep0_error_smoothness.png" width="380"/></td>
</tr></table>

RMSE: **3.53 cm** | Mean error: **3.01 cm** | Steps <3cm: **68.3%**

---

#### Circle — Observation Noise (σ=0.01)
<table><tr>
<td><img src="outputs/best_model_eval/circle_obs_noise/ep0_path_xy.png" width="300"/></td>
<td><img src="outputs/best_model_eval/circle_obs_noise/ep0_error_smoothness.png" width="380"/></td>
</tr></table>

RMSE: **2.08 cm** | Mean error: **1.84 cm** | Steps <3cm: **92.0%**

---

#### Circle — Action Noise (σ=0.02)
<table><tr>
<td><img src="outputs/best_model_eval/circle_action_noise/ep0_path_xy.png" width="300"/></td>
<td><img src="outputs/best_model_eval/circle_action_noise/ep0_error_smoothness.png" width="380"/></td>
</tr></table>

RMSE: **1.18 cm** | Mean error: **0.88 cm** | Steps <3cm: **98.5%**

---

#### Circle — 1-Step Action Delay
<table><tr>
<td><img src="outputs/best_model_eval/circle_delay_1/ep0_path_xy.png" width="300"/></td>
<td><img src="outputs/best_model_eval/circle_delay_1/ep0_error_smoothness.png" width="380"/></td>
</tr></table>

RMSE: **1.45 cm** | Mean error: **0.99 cm** | Steps <3cm: **98.5%**

---

## Demo Videos

| Condition | GIF |
|---|---|
| Circle (clean) | <img src="outputs/best_model_eval/gifs/circle_clean.gif" width="240"/> |
| Figure-8 (clean) | <img src="outputs/best_model_eval/gifs/figure8_clean.gif" width="240"/> |
| Spline (clean) | <img src="outputs/best_model_eval/gifs/spline_clean.gif" width="240"/> |
| Circle + obs noise | <img src="outputs/best_model_eval/gifs/circle_obs_noise.gif" width="240"/> |
| Circle + action noise | <img src="outputs/best_model_eval/gifs/circle_action_noise.gif" width="240"/> |
| Circle + 1-step delay | <img src="outputs/best_model_eval/gifs/circle_delay_1.gif" width="240"/> |

Full MP4s in `outputs/best_model_eval/videos/`.

---

## Quick Start

```bash
# 1. Create and activate environment
python3.11 -m venv venv_py311
source venv_py311/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.2

# 3. Evaluate the best model
python scripts/eval_suite.py \
  --model best_model.zip \
  --pure-sac \
  --no-track-orientation \
  --control-dt 0.05 \
  --max-episode-steps 200 \
  --max-steps 200 \
  --episodes 20 \
  --output-root outputs/my_eval \
  --record-videos

# 4. Scan all checkpoints to find best
python scripts/scan_checkpoints.py

# 5. Generate summary plots
python scripts/plot_best_eval.py
```

---

## Project Structure

```
.
├── best_model.zip                  ← Best trained model (175k steps)
├── pure_sac_model.zip              ← Final model (1M steps, NOT best)
├── requirements.txt
├── README.md
│
├── envs/
│   ├── tracking_env.py             ← Base FetchReachDense wrapper
│   └── pure_sac_env.py             ← Pure SAC env (adds target_vel, phase_encoding)
│
├── train/
│   ├── train_pure_sac.py           ← Main training script
│   └── callbacks.py                ← RewardLogging + PerformanceGatedCurriculum
│
├── utils/
│   ├── trajectories/
│   │   ├── circle.py               ← Circular trajectory
│   │   ├── figure_eight.py         ← Figure-8 trajectory
│   │   └── spline.py               ← Waypoint spline trajectory
│   └── orientation_utils.py
│
├── scripts/
│   ├── eval_suite.py               ← Run all 6 eval conditions
│   ├── eval_policy.py              ← Single-condition eval + video
│   ├── scan_checkpoints.py         ← Find best checkpoint
│   └── plot_best_eval.py           ← Generate summary plots
│
├── experiments/
│   └── pure_sac.yaml               ← Training config (curriculum, hyperparams)
│
├── checkpoints/                    ← All 200 checkpoints (every 5k steps)
├── outputs/
│   └── best_model_eval/
│       ├── plots/                  ← tracking_error.png, success_rate.png, checkpoint_scan.png
│       └── videos/                 ← 6 MP4 videos
└── archive/                        ← Old dev files (PPO, residual RL, debug scripts)
```

---

## Full Project History & Design Decisions

### The Problem

Train a robot arm to track a time-varying 3D target position using reinforcement learning. The arm is a Franka Panda (7-DOF) simulated in MuJoCo via `FetchReachDense-v4`. The action space is 4D mocap position deltas. The challenge requires handling:
- Multiple trajectory shapes (circle, figure-8, spline)
- Observation noise
- Action noise
- Action delay

---

### Phase 1: PPO — Why It Failed

The first attempt used **PPO (Proximal Policy Optimization)**. PPO is an on-policy algorithm — it discards all experience after each update and must collect fresh data every iteration. For this task that meant:

- **Sample inefficiency**: needed 500k–1M steps to see meaningful learning (10–20 hours on M4 MacBook Air at ~12 FPS)
- **No replay**: couldn't reuse the expensive MuJoCo simulation data
- **Exploration**: entropy bonus decreases as training progresses, making it easy to get stuck in local optima
- **Continuous control**: PPO is designed for discrete action spaces (games, text); continuous control is possible but SAC is purpose-built for it

PPO was abandoned after it failed to break 20% step-success on the circle trajectory after 100k steps.

---

### Phase 2: Residual RL (PD + SAC) — Why It Was Removed

The second approach added a **PD operational-space controller** as a baseline:

$$u_{\text{total}} = u_{\text{pd}} + \alpha \cdot u_{\text{rl}}$$

where $u_{\text{pd}} = K_p \cdot e + K_d \cdot \dot{e}$ is the PD term with gains $K_p=200$, $K_d=50$, and $\alpha \in [0,1]$ is a residual scaling factor annealed from 0.1 → 1.0 during training.

**Why it was removed:**
- The PD gains (Kp=200, Kd=50) required manual tuning per trajectory type and speed
- The residual scaling factor α needed its own annealing schedule
- The architecture added complexity without a clear performance ceiling — if the PD was already doing most of the work, what was SAC actually learning?
- Most importantly: the task asked for a pure RL solution. A hand-tuned PD baseline is a form of prior knowledge injection that obscures what the RL policy actually learned

The residual env code is preserved in `archive/old_envs/residual_tracking_env.py` for reference.

---

### Phase 3: Pure SAC — The Final Architecture

**Why SAC over PPO for this task:**

| Property | SAC | PPO |
|---|---|---|
| Algorithm type | Off-policy | On-policy |
| Replay buffer | ✅ 500k transitions | ❌ discarded each update |
| Sample efficiency | ~3–5× better | Baseline |
| Exploration | Entropy regularization (auto-tuned) | Entropy bonus (decays) |
| Continuous control | Purpose-built | Works but suboptimal |
| Hyperparameter sensitivity | Low (defaults work) | High (ε, GAE, LR all matter) |
| Steps to 90%+ success | ~100k | ~500k+ (estimated) |

SAC's key advantage is the **maximum entropy objective**:

$$\pi^* = \arg\max_\pi \sum_t \mathbb{E}\left[ r(s_t, a_t) + \alpha \cdot \mathcal{H}(\pi(\cdot | s_t)) \right]$$

where $\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a|s)]$ is the policy entropy and $\alpha$ is a temperature coefficient that is **auto-tuned** to maintain a target entropy $\mathcal{H}_{\text{target}} = -2.0$. This means exploration is maintained throughout training without manual scheduling — which is what allowed the policy to break through the 20% plateau that PPO was stuck at.

---

### The Observation Space (28D)

```
achieved_goal      (3,)   current EE position  x_ee ∈ ℝ³
desired_goal       (3,)   current target        x_d(t) ∈ ℝ³
observation        (10,)  FetchReach internal   [q, q̇] joint pos/vel
tracking_error     (3,)   e(t) = x_d(t) - x_ee
tracking_error_vel (3,)   ė(t) = (e(t) - e(t-1)) / Δt
target_vel         (3,)   ẋ_d(t) ≈ (x_d(t+ε) - x_d(t)) / ε
phase_encoding     (2,)   [sin(ωt), cos(ωt)]
prev_action        (4,)   a_{t-1}
```

The phase encoding $[\sin(\omega t),\, \cos(\omega t)]$ gives the policy a continuous, non-aliased representation of where it is in the trajectory cycle. Without it the policy can only react to the current error; with it, it can anticipate the next position and reduce phase lag.

The target velocity $\dot{x}_d(t)$ is computed via finite difference:

$$\dot{x}_d(t) \approx \frac{x_d(t + \varepsilon) - x_d(t)}{\varepsilon}, \quad \varepsilon = 10^{-5}$$

---

### The Reward Function

At each timestep the reward is:

$$r_t = -w_{\text{track}} \cdot \|e_t\|^2 \;-\; w_{\text{smooth}} \cdot \|\Delta a_t\|^2 \;-\; w_{\text{vel}} \cdot \|\dot{q}_t\|$$

where:
- $e_t = x_d(t) - x_{\text{ee}}(t)$ — 3D tracking error vector
- $\Delta a_t = a_t - a_{t-1}$ — action delta (smoothness)
- $\dot{q}_t$ — joint velocity vector from the simulator

Training weights: $w_{\text{track}}=1.0$,  $w_{\text{smooth}}=0.0025$,  $w_{\text{vel}}=0.2$

The squared tracking term $\|e_t\|^2$ gives dense gradients near the target and penalises large errors more than small ones. The smoothness penalty $\|\Delta a_t\|^2$ is critical but must be small — too large and the policy collapses to near-zero actions (zero smoothness penalty, poor tracking); too small and it learns jittery high-frequency control that achieves low mean error but fails the 3 cm step-success threshold.

---

### The Curriculum

The curriculum uses **performance-gated advancement** — the policy must sustain both a success rate gate and a mean error gate for 10 consecutive episodes before advancing. The advancement condition is:

$$\text{advance} \iff \underbrace{\bar{s}_{\text{recent}} \geq s^*}_{\text{success gate}} \;\wedge\; \underbrace{\bar{e}_{\text{recent}} \leq e^*}_{\text{error gate}} \;\wedge\; \underbrace{n_{\text{sustained}} \geq 10}_{\text{hysteresis}}$$

where $\bar{s}$ and $\bar{e}$ are rolling means over the last 50 episodes, and $n_{\text{sustained}}$ counts consecutive episodes where both gates are met (resets to 0 on any failure).

```
Stage 0: circle @ speed=0.6,  obs_noise=0.001, no action noise, no delay
Stage 1: circle @ speed=0.75, obs_noise=0.001
Stage 2: circle @ speed=0.9,  obs_noise=0.002
Stage 3: circle @ speed=1.1,  obs_noise=0.003, action_noise=0.004, delay=1
Stage 4: circle @ speed=1.3,  obs_noise=0.005, action_noise=0.007, delay=1
Stage 5: circle @ speed=1.5,  obs_noise=0.008, action_noise=0.010, delay=1
Stage 6: spline @ speed=0.6,  obs_noise=0.003, action_noise=0.005
Stage 7: spline @ speed=0.9,  obs_noise=0.005, action_noise=0.008, delay=1  ← stuck here
Stage 8: spline @ speed=1.2,  obs_noise=0.008, action_noise=0.010, delay=1
```

**What went wrong with the curriculum:**

The policy mastered circle stages 0–5 (98%+ step-success at speed=0.6) and advanced through them quickly. When it hit Stage 6 (spline), performance dropped sharply. By Stage 7 it was stuck at 7.6% success with 6.8 cm mean error.

This is **curriculum forgetting** — a well-known problem in curriculum RL. When the policy is forced onto a harder distribution (spline with noise and delay), gradient updates on the new task overwrite the weights that encoded circle tracking. The policy partially forgets what it learned.

The fix (not yet implemented) is **replay mixing**: train 80% on the current stage and 20% on randomly sampled previous stages. This keeps the earlier behavior alive while learning the new task.

---

### The Metric Confusion

The tensorboard `success_rate` metric was logged as:

```python
self.logger.record("train/success_rate", float(info["is_success"]))
```

This is called **every timestep** and SB3 averages it over the rollout buffer. So the 90%+ numbers in tensorboard mean: **90% of individual timesteps had tracking error < 3 cm** — not that 90% of episodes succeeded. This is a step-level metric, not an episode-level metric.

The eval suite was also using wrong parameters:
- `control_dt=0.03` (trained with `0.05`) — changes the physics the policy expects
- `max_episode_steps=400` (trained with `200`) — double the episode length
- `success_threshold=5cm` (trained with `3cm`)

These mismatches made the policy look much worse than it actually was. After fixing all parameters and using the correct checkpoint, the numbers match tensorboard.

---

### Why the Final 1M-Step Model Is Not the Best

Training ran for 1M steps total. The final saved `pure_sac_model.zip` is the policy at step 1,000,000 — which was stuck on Stage 7 (spline_medium) with 7.6% success and 6.8 cm mean error. This is **worse** than the policy at 175k steps.

The checkpoint scan confirms this clearly:

| Steps | Mean error | @3cm success |
|---|---|---|
| 100k | 1.20 cm | 98.0% |
| 175k | **0.74 cm** | **98.5%** ← best |
| 600k | 1.56 cm | 98.0% |
| 650k | 3.08 cm | 54.6% ← curriculum collapse |
| 700k | 4.25 cm | 32.8% |
| 1000k | ~6.8 cm | ~7.6% |

The best model is `best_model.zip` (175k steps), not `pure_sac_model.zip`.

---

### SAC vs PPO — Detailed Comparison

**Use SAC when:**
- Continuous action space (robotics, manipulation) ✅
- Dense rewards (tracking error every step) ✅
- Expensive simulation (MuJoCo at 12 FPS on M4) ✅
- Single environment (not parallelized) ✅
- Need exploration throughout training ✅
- Want minimal hyperparameter tuning ✅

**Use PPO when:**
- Discrete action space (games, text generation)
- Sparse rewards (win/lose)
- Fast simulation (can collect millions of steps cheaply)
- Parallel environments available (16–128 envs)
- Large-scale distributed training

For this task, SAC achieved 98.5% step-success at 175k steps. PPO would likely need 500k+ steps to reach comparable performance, and would require more careful tuning of ε, GAE λ, and learning rate.

---

### Generalization Analysis

The policy generalizes well within its training distribution:

- **Circle at trained speed (0.6)**: 98.5% — near-perfect
- **Figure-8 (never trained on)**: 94.8% — strong generalization
- **Spline (trained on, but at harder stages)**: 68.3% — decent
- **Obs noise, action noise, delay**: 92–98.5% — robust

The policy struggles with:
- **Sharp direction reversals** (zigzag, sharp corners) — never trained on discontinuous velocity
- **High-speed spline** — curriculum forgetting degraded this
- **Out-of-distribution trajectories** — RL learns statistical patterns, not universal control

For true generalization to arbitrary trajectories, the fix is either: (1) add more trajectory types to the curriculum, or (2) switch to a model-based approach that can plan over arbitrary paths.

### Trajectory Definitions

**Circle** — constant curvature, periodic:

$$x_d(t) = \begin{bmatrix} c_x + r\cos(\omega t) \\ c_y + r\sin(\omega t) \\ c_z \end{bmatrix}, \quad r=0.15\text{ m},\; \omega=0.6\text{ rad/s}$$

**Figure-8** — Lissajous curve, self-intersecting:

$$x_d(t) = \begin{bmatrix} c_x + r\sin(\omega t) \\ c_y + r\sin(2\omega t)/2 \\ c_z \end{bmatrix}$$

**Spline** — cubic spline through $N$ random waypoints sampled within a half-extent box, reparametrised to constant arc-length speed $v$.

---

## Work in Progress

These are active areas of exploration — not yet in the main codebase.

### Better Splines & Smoother Trajectories

The current spline implementation uses cubic interpolation through random waypoints, but the resulting path can have sharp curvature changes at knot points that the policy struggles with. The goal is to:

- Replace the waypoint spline with **minimum-jerk trajectories** — paths that minimise $\int \dddot{x}^2 \, dt$, producing the smoothest possible motion between waypoints
- Add **arc-length reparametrisation** so the target moves at truly constant speed regardless of curvature
- Investigate whether a smoother target trajectory directly improves tracking RMSE, or whether the policy can compensate for trajectory roughness on its own

The hypothesis is that the 3.53 cm RMSE on spline (vs 1.17 cm on circle) is partly due to the trajectory itself being harder to predict, not just the policy being undertrained on it.

### Reaching Unreachable Positions

FetchReach has a finite workspace — the arm physically cannot reach all points in 3D space. Currently the trajectory center and radius are hand-tuned to stay inside the reachable workspace. The next step is to:

- Map the actual reachable workspace boundary using forward kinematics
- Train the policy on trajectories that **graze the workspace boundary** — positions that are reachable but only barely, requiring the arm to operate near its joint limits
- Explore what happens when the target briefly exits the workspace: does the policy learn to track as close as possible (graceful degradation), or does it fail catastrophically?
- Potentially add a **workspace-aware reward** that gives partial credit for minimising distance to the nearest reachable point when the target is unreachable

This is relevant for real-world deployment where task trajectories are not always designed with robot kinematics in mind.

---

```bash
# Train with the pure SAC curriculum
python train/train_pure_sac.py --config experiments/pure_sac.yaml

# Resume from a checkpoint
python train/train_pure_sac.py \
  --config experiments/pure_sac.yaml \
  --resume checkpoints/pure_sac_175000_steps.zip

# Monitor training
tensorboard --logdir=./tensorboard/
```

**Key config parameters** (`experiments/pure_sac.yaml`):

```yaml
train:
  total_timesteps: 1_000_000
  device: mps                    # or cuda / cpu
  sac:
    learning_rate: 0.0003
    buffer_size: 500_000
    batch_size: 256
    gradient_steps: 2
    target_entropy: -2.0

env:
  control_dt: 0.05               # MUST match eval
  max_episode_steps: 200         # MUST match eval
  trajectory_kwargs:
    center: [1.34, 0.75, 0.53]
    radius: 0.15
    speed: 0.6

success_threshold: 0.03          # 3 cm — MUST match eval
```

---

## Evaluation

### Run the full eval suite (6 conditions)

```bash
python scripts/eval_suite.py \
  --model best_model.zip \
  --pure-sac \
  --no-track-orientation \
  --control-dt 0.05 \
  --max-episode-steps 200 \
  --max-steps 200 \
  --episodes 20 \
  --output-root outputs/my_eval \
  --record-videos
```

### Evaluate a single condition

```bash
python scripts/eval_policy.py \
  --model best_model.zip \
  --pure-sac \
  --control-dt 0.05 \
  --max-episode-steps 200 \
  --max-steps 200 \
  --trajectory circle \
  --trajectory-kwargs '{"center": [1.34, 0.75, 0.53], "radius": 0.15, "speed": 0.6}' \
  --episodes 20 \
  --save-dir outputs/single_eval
```

### Scan all checkpoints

```bash
python scripts/scan_checkpoints.py
# Outputs: outputs/checkpoint_scan.json + printed table
```

### Generate plots

```bash
python scripts/plot_best_eval.py
# Outputs: outputs/best_model_eval/plots/*.png
```

---

## Known Issues & Future Work

**Curriculum forgetting** — the biggest issue. The policy degrades on circle when pushed to spline stages. Fix: replay mixing (80% current stage + 20% random previous stages).

**Metric logging** — `success_rate` in tensorboard is step-level, not episode-level. This made training look better than it was during development. Fix: log `episode_success = mean_tracking_error < threshold` as the primary metric.

**Best model saving** — training only saved the final model, not the best. Fix: add `EvalCallback` with `best_model_save_path` to automatically save the checkpoint with lowest eval error.

**Spline generalization** — the policy achieves 68.3% on spline at slow speed. For better spline performance, the curriculum needs replay mixing and possibly a longer warm-up on spline_slow before introducing noise and delay.

---

## Dependencies

```
stable-baselines3==2.3.2
gymnasium==0.29.1
gymnasium-robotics @ git+https://github.com/Farama-Foundation/Gymnasium-Robotics.git@v1.4.2
numpy==1.26.4
matplotlib==3.8.4
imageio[ffmpeg]
pyyaml==6.0.1
torch  # MPS on Apple Silicon, CUDA on Linux
```

Python 3.11+ required.

---

## Hardware

Trained on Apple M4 MacBook Air (16GB unified memory).
- MuJoCo simulation: ~12–17 FPS
- 1M training steps: ~16 hours wall time
- Best checkpoint (175k steps): ~3 hours

---

*Best model: `best_model.zip` (175k steps) — 98.5% step-success @3cm on circle, 94.8% on figure-8.*
