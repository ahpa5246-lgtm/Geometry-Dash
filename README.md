# Geometry Dash — Self-Learning AI

This repository trains an agent to play Geometry Dash from scratch using only live screen pixels, its own actions, and rewards generated while it plays.

No pretrained model, human demonstrations, or pre-collected gameplay dataset are required.

The first working baseline is **Recurrent PPO (CNN + LSTM)**. The agent captures the Geometry Dash window, chooses whether to hold/release jump, detects the end of a run from screen motion, restarts automatically, logs every episode, and continually improves from experience it generates itself.

## Requirements

- Windows 10/11
- Python 3.11 recommended
- Geometry Dash / Geometry Dash Lite running in windowed or borderless-windowed mode
- NVIDIA GPU recommended but not required

## Quick start

```powershell
git clone https://github.com/ahpa5246-lgtm/Geometry-Dash.git
cd Geometry-Dash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m gd_ai.calibrate
python -m gd_ai.train
```

During calibration, keep the game window visible. The script auto-detects a window whose title contains “Geometry Dash”, saves a screenshot, and verifies that frames are changing.

Training starts with randomly initialized weights. The AI receives only pixels and reward; it does not receive obstacle coordinates, player position, labels, or prior gameplay.

See `docs/DESIGN.md` for the learning design and `config.yaml` for settings.
