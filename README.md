# Geometry Dash — Self-Learning AI

A visual reinforcement-learning agent that starts from random weights and learns Geometry Dash from experience it generates by playing.

It uses:
- no pretrained model;
- no human demonstrations;
- no pre-collected gameplay dataset;
- no player coordinates, obstacle coordinates, or memory reading from the game.

The baseline is **Recurrent PPO (CNN + LSTM)**. The agent sees live pixels from the Geometry Dash window, decides whether to hold/release jump, receives reward for staying alive, detects the end of a run visually, restarts automatically, and keeps training.

## Current pipeline

```text
Geometry Dash pixels
        ↓
      CNN
        ↓
      LSTM
        ↓
 Recurrent PPO policy
        ↓
release / hold jump
        ↓
next live frame + reward
        └──────────────→ learning update
```

All training experience is collected online while the agent plays.

## Requirements

- Windows 10/11
- Python 3.11 recommended
- Geometry Dash / Geometry Dash Lite
- Windowed or borderless-windowed mode
- NVIDIA GPU recommended, but CPU training is supported

## 1. Install

Clone the repository and run:

```powershell
.\setup.ps1
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Calibrate capture

Open Geometry Dash first, then:

```powershell
python -m gd_ai.calibrate
```

The script finds a visible window whose title contains `Geometry Dash`, captures only its client area, saves `runs/calibration.png`, and reports frame-motion statistics.

Optional keyboard-input check:

```powershell
python -m gd_ai.calibrate --test-input
```

If the game ignores synthetic input, run the terminal with the same privilege level as the game and make sure the game is focused.

## 3. Put the level at a retry-ready state

Open the level you want the AI to learn and leave it at a state where pressing Space starts/retries that same level.

This is environment setup only. It is not a demonstration and is never used as training data.

## 4. Train from zero

```powershell
python -m gd_ai.train
```

or:

```powershell
.\start-training.ps1
```

The default run starts from fresh random neural-network weights.

A short end-to-end validation run:

```powershell
python -m gd_ai.train --timesteps 20000
```

Resume only when you explicitly want to continue an existing policy:

```powershell
python -m gd_ai.train --resume models/checkpoints/gd_agent_100000_steps.zip
```

## What is saved automatically

- `runs/episodes.csv`: every episode's duration, reward, best survival time, termination reason, and motion score.
- `runs/monitor.csv`: Stable-Baselines episode monitor.
- `runs/tensorboard/`: learning curves and optimization metrics.
- `models/checkpoints/`: periodic model checkpoints.
- `models/best_model.zip`: longest-surviving policy seen so far.
- `models/final_model.zip`: model saved at normal completion.

Raw screenshots are consumed online by the agent but are not persisted by default, avoiding uncontrolled disk growth.

## Watch learning

In another terminal:

```powershell
.\.venv\Scripts\activate
tensorboard --logdir runs/tensorboard
```

## Evaluate without learning

```powershell
python -m gd_ai.evaluate models/best_model.zip --episodes 10
```

## Configuration

Edit `config.yaml` to change:
- screen resolution seen by the AI;
- training frequency;
- reward scale;
- visual death-detection threshold;
- PPO hyperparameters;
- checkpoint frequency.

The default observation is 84×84 grayscale pixels and the action space is only:

```text
0 = release jump
1 = hold jump
```

The LSTM learns temporal information such as motion and timing from consecutive observations.

For the architecture, termination detector, reward design, and later comparison plan (Rainbow-style DQN, NEAT, Dreamer/model-based RL), see `docs/DESIGN.md`.
