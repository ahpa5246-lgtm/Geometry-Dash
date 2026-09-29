# Design: zero-prior-data Geometry Dash learning

## Core requirement

The AI starts from random weights and does not receive human gameplay, labels, obstacle coordinates, player coordinates, or hidden game state.

There are now two learning loops.

## Loop A — live Recurrent PPO

```text
live pixels -> CNN -> LSTM -> PPO -> hold/release
      ^                              |
      |                              v
      +------- next pixels/reward ---+
```

Every live transition is also persisted to disk in compressed replay chunks.

This is the only stage that can discover truly new real-game states because the actual game is producing them.

## Loop B — offline CQL

When Geometry Dash is closed, a separate Q-network continues learning from the AI's own saved replay.

State:
- previous grayscale frame;
- current grayscale frame.

The two-frame stack supplies short-term motion information without reading game memory.

Action:
- release;
- hold.

Optimization:
- Double-DQN-style TD target;
- Huber TD loss;
- discrete Conservative Q-Learning penalty.

The CQL term discourages the policy from assigning unrealistically high value to actions unsupported by its stored experience, which is important when reusing a fixed offline dataset.

## What "learn while the game is closed" means

No algorithm can receive new real Geometry Dash observations while Geometry Dash is not running.

Offline learning can still:
- perform more gradient updates;
- extract better value estimates from stored experience;
- train a new policy from replay;
- reuse old transitions many times.

It cannot, by itself, discover a previously unseen obstacle configuration because that state has never been observed.

The next research stage is a learned world model. That model would predict future latent states and enable imagined rollouts while the real game is closed. It should only be introduced after enough real replay has been collected, otherwise the policy may exploit inaccuracies in the learned simulator.

## Persistent replay

Live PPO is wrapped with `PersistentReplayWrapper`.

Replay is stored under `runs/replay/` as compressed NPZ chunks containing:
- `obs`;
- `next_obs`;
- `actions`;
- `rewards`;
- `dones`;
- `episodes`;
- `episode_steps`.

Frames remain uint8 on disk.

## Calibration warning

The main menu is largely static. A motion median below the freeze threshold at the menu does not imply the detector is wrong.

Death detection must be calibrated while a level is actively scrolling, then compared with the retry/death state.

## Recommended cycle

1. Live exploration from random weights.
2. Accumulate replay.
3. Close the game.
4. Run offline CQL for many gradient steps.
5. Reopen the game.
6. Evaluate the offline policy.
7. Collect new experience.
8. Repeat.

Later:
9. Train a latent world model from the growing replay.
10. Add imagined rollouts.
11. Test generalization on unseen levels.
