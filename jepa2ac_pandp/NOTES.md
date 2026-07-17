# V-JEPA2 / V-JEPA2-AC — Notes

## Is V-JEPA2 a VLM?

No. There is no text anywhere in it — no tokenizer, no language head, no
captions during training. It's a video encoder + predictor trained
self-supervised with the **JEPA objective**: mask out chunks of a video in
space/time, and train the predictor to guess the *latent representation* of
the masked chunks from the visible ones.

Compare to a VLM (LLaVA, GPT-4V): a vision encoder paired with a language
model, taking image+text in and producing text out. V-JEPA2 never produces
or consumes text — it's closer to a self-supervised encoder like DINO,
just for video.

## What makes V-JEPA2-AC a "world model"

In base V-JEPA2, the predictor guesses masked patches *within a video that
already happened* — filling in blanks, not simulating anything.

In the **AC** (action-conditioned) variant, the predictor is retrained so
its input is `(current latent state, action, robot pose)` and its target
is `(next latent state)`:

```
f(s_t, a_t) -> s_{t+1}
```

That's the textbook definition of a world model / transition function in
model-based RL: a function you can roll forward hypothetically, without
touching the real robot, to ask "if I did this, what would happen?"

This is exactly what `cem()` in `notebooks/utils/mpc_utils.py` does — it
rolls the predictor forward hundreds of times with different sampled
actions, purely in latent space, and picks whichever imagined future lands
closest to a goal latent. No pixels are ever decoded; the "imagination"
happens entirely in representation space.

## `vjepa2_demo.py` vs `energy_landscape_example.ipynb`

Two different models/purposes:

- **`vjepa2_demo.py`** — plain V-JEPA2 encoder (no action-conditioning).
  Runs a video through the encoder to get patch features, then through a
  supervised `AttentiveClassifier` probe to produce a classification label
  (e.g. an SSv2 action class like "pouring water into cup"). No robot, no
  actions, no planning — encoder -> probe -> label.

- **`energy_landscape_example.ipynb`** — V-JEPA2-AC encoder + action-
  conditioned predictor. Loads a real Franka arm trajectory
  (`franka_example_traj.npz`) and asks: "for the current frame, if I take
  action `a`, how far (in latent space) does the predicted next frame land
  from a goal frame?" That distance is the "energy." The notebook first
  visualizes the energy landscape over a grid of candidate actions, then
  shows the same energy minimized automatically via CEM
  (`world_model.infer_next_action`) to pick a good action.

## Where does the classifier label come from?

V-JEPA2 itself never produces labels — the encoder only outputs raw patch
features. `vjepa2_demo.py` bolts a separate small supervised head,
`AttentiveClassifier` (`src/models/attentive_pooler.py`), on top of the
*frozen* encoder output. That head is trained afterward, with labels, on
the Something-Something-v2 action-recognition dataset — it's the
`evals/ssv2-vitg-384-64x2x3.pt` checkpoint (see README probe table). The
human-readable class names come from `ssv2_classes.json`, a lookup table
for that dataset (not shipped in this repo — the demo script expects it
locally).

So the classification demo is really "how good are V-JEPA2's frozen
features when a tiny supervised probe is trained on top" — an eval
methodology, not a built-in capability of V-JEPA2 itself.
