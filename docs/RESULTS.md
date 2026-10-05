# Results log

Everything here was produced by the code in this repository on a 4-core CPU container with the
`nano` model (0.36M parameters) unless stated. **No result so far supports or refutes the
transfer hypothesis.** The entries below are what has been run and what was learned about the
experiment itself.

## 1. `pilot_cpu_tempo` (v1) — INVALID

Target: held-out `tempo` games; pretraining on same-family, other-family and random-reward
sets; 2 seeds, 96 runs, 98 minutes.

What the data showed, and why it cannot be used:

* **Bug:** the pretrained arms were never fine-tuned. Models rebuilt from saved inference
  weights with `Module::train()` do not track gradients, so `pre_*_full` showed *exactly* the
  same validation cross-entropy at every step, and `pre_*_backbone` equalled `pre_*_frozen`
  bit for bit. Fixed with `OmniAstra::enable_grad()` plus a regression test
  (`model_moved_from_inference_to_autodiff_trains_after_enable_grad`).
* **Metric floor:** even the scratch arm barely moved (CE 1.20 → 1.18).
* The strength column compared *pretrained, untrained-on-target* policies against the scratch
  arm's trained policy, so its differences are not transfer evidence either.

## 2. Learnability diagnostics (`omnia diagnose`, one tempo game, nano)

| expert targets | uniform CE | target entropy (floor) | learnable gap | nano after training |
|---|---|---|---|---|
| MCTS 16 sims, raw visits | 1.136 | 1.048 | **0.087** | train CE 1.097 (flat after ~100 steps) |
| MCTS 128 sims, raw visits | 1.064 | 0.967 | **0.096** | (not trained) |
| MCTS 64 sims, visits^(1/0.2) | 1.122 | 0.661 | **0.460** | train 0.93, val 1.007 after 500 steps (≈ 40% of the gap), val top-1 0.52 |

* UCT spreads visits across actions, so raw visit-count targets are nearly uniform whatever the
  budget; policy cross-entropy against them cannot discriminate between models. This is a
  property of the targets, not of the model.
* The value head does learn (val sign accuracy 0.44 → 0.64–0.73), so value metrics carry signal.
* With sharpened targets the nano model is under-fitted at 500 steps: it needs more updates or
  more capacity before differences between arms could show up.

Design consequences (implemented): `target_temp` sharpening in data generation; experiments
report the entropy-floor gap; value sign-accuracy is available next to policy CE.

## 3. `pilot_v2_tempo`

Config: `experiments/pilot_v2_tempo.json` (64-sim experts, target_temp 0.2, 1500 pretraining
steps, 300 fine-tune steps, 3 data budgets, 8 arms, 2 seeds). Status: **launched; results not
yet recorded here.** Interpretation rules are in `docs/EXPERIMENTS.md`: same-family transfer is
the sanity floor, other-family must beat both scratch and the random-reward control.

## 4. Measured throughput

| machine | nano train | tiny train | small train |
|---|---|---|---|
| 4-core CPU container | ~90 samples/s | ~25–30/s | not measured |
| user laptop (Ryzen AI 7 350), first run | ~250/s | — | — |
| same laptop, later run (likely power-throttled) | — | ~36/s | ~9/s |

Cross-machine reproducibility check: the `smoke.json` experiment produced identical validation
cross-entropy/top-1 values on the container and on the laptop.
