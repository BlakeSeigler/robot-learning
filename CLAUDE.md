# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Robotics simulation sandbox for exploring robot learning approaches — training a manipulator arm (xArm6/xArm7) to pick up a block and place it at a target. Model families to explore: Action Chunking Transformer, Diffusion Policy, PPO/SAC, and VLA models (OpenVLA, SmolVLA, pi0-style).

## Environment Setup

Python 3.10 (pinned in `.python-version`). Use `uv` for dependency management:

```bash
uv sync                    # install deps into .venv
uv run python <script>     # run any script
```

Core deps in `pyproject.toml`: `numpy`, `pybullet`.

## Running the Demos

```bash
# OpenVLM-style policy demo (text-only, no window)
uv run python openvlm/demo.py

# Same demo with PyBullet GUI
uv run python openvlm/demo.py --gui

# Load xArm6 URDF in PyBullet GUI (holds open 100s)
uv run python pybullet/main.py

# Quick URDF check (closes after 1s)
uv run python pybullet/block.py
```

## Architecture

**`openvlm/demo.py`** — Self-contained VLM-policy loop. `OpenVLMSim.decide()` takes an observation dict `{block_pos, target_pos}` plus a language instruction string and returns an action string. Falls back to pure-Python text mode if `pybullet` is not importable.

**`pybullet/main.py` / `block.py`** — Minimal PyBullet harnesses that connect a GUI, load the xArm6 URDF, and exit. `block.py` exits after 1s; `main.py` holds for 100s.

**`pybullet/deps/uf-gym/`** — Git submodule providing:
- `urdf/xarm/` — URDF files for xArm6/xArm7 with grippers (used by path resolution in `pybullet/*.py`)
- `uf_gym/` — Gym environments (`XArm6Reach-v3`, `XArm7PickAndPlace-v3`, etc.)
- `train_xarm*.py` / `test_xarm*.py` — RL training/eval scripts (stable-baselines3, TQC/DDPG)

The uf-gym submodule requires `panda-gym`, `stable-baselines3`, `sb3-contrib`, and PyTorch — **not** in `pyproject.toml`, must be installed separately to use the RL scripts.

Scripts in `pybullet/` resolve the URDF path relative to `__file__`, so they work from any working directory.
