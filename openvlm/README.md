# openvlm

Mind-bogglingly small OpenVLM-style demo.

It uses a tiny simulated "VLM policy" instead of downloading a real model. If PyBullet is installed, it uses the sim. If not, it runs as plain text.

1. read an observation
2. combine it with a language instruction
3. choose an action
4. update the sim

Run it from the repo root:

```bash
python3 openvlm/demo.py
```

If your `uv` works, this also uses the project dependency:

```bash
uv run python openvlm/demo.py
```

To see the PyBullet sim window:

```bash
uv run python openvlm/demo.py --gui
```
