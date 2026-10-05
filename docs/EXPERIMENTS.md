# Experiment protocol

## The central test

For a held-out target game H and arms that differ **only** in how the network is initialised:

| arm | meaning |
|---|---|
| `scratch` | random init → H |
| `scratch_x3_steps` | random init → H with 3× the fine-tune steps (optimizer-update-matched baseline) |
| `pre_same_full` / `_backbone` / `_frozen` | pretrain on *other games of H's family*: whole network / backbone only (fresh adapter) / backbone frozen, adapter+heads trained |
| `pre_other_full` / `_backbone` | pretrain on games of *different* families (no shared skeleton) |
| `pre_rr_same_full` | control: pretrain on random-reward copies of the same-family games (identical inputs/pacing, outcomes carry no information) |

Reading the results:

* **same-family** transfer is the sanity floor. If it is absent, the setup (model size, data, adapter
  design) cannot detect transfer at all and nothing about cross-family transfer can be concluded.
* **other-families** transfer is the actual hypothesis. It must beat `scratch` *and*
  `pre_rr_same_full` (generic pretraining benefit) to count as evidence of reusable strategy.
* **backbone-only / frozen** arms tell you whether what transferred lives in the shared backbone
  rather than in game-specific adapters. A frozen backbone doing well is the strongest form.
* Compare at equal **target-game positions** (data efficiency) *and* report total compute
  (pretraining + fine-tuning FLOP estimates) — they answer different questions. `scratch_x3_steps`
  controls optimizer updates on H, not pretraining compute; read the table's compute column.

Metrics: held-out policy cross-entropy / top-1 agreement with the expert (MCTS) policy on the
target game (low variance), value MSE, and playing strength of the raw greedy policy against a
fixed opponent panel (random, MCTS-N) with Wilson CIs per match. The checkpoint is selected by
validation CE on held-out *games* of H. `summary.md` also reports a log-interpolated **effective-
data multiplier** (how many scratch positions the arm needs to match its CE).

## Discipline

* The config (arms, budgets, seeds, pretraining sets, opponents) is fixed before running and is
  copied into the output directory with a hash of every game used (`games.json`).
* The runner never adapts games, metrics, seeds or baselines to results. All runs are logged to
  `results.jsonl` (including the failures).
* Every arm of a seed starts from the *same* fresh initialisation (`B::seed` + materialised params).

## Further controls available (configs, not yet run)

* `reskin_of`: target = re-skinned copy of a pretraining game (does transfer survive id relabelling).
* `Variant::AltVictory` / `Inverted` groups: same surface, different optimal strategy
  (transfer should *drop* if it is strategic and not surface-level).
* Model ablations through `model_overrides` (`use_ids:false`, `use_desc:false`, `use_rel:false`,
  `use_history:false`, `use_registers:false`, `action_self_attn:false`, …).
* Pretraining-diversity sweeps: vary the number of pretraining games/families at fixed compute.

## Running

```
cargo build --release -p omnia-cli          # CPU (ndarray); add --features flex|cuda|wgpu
./target/release/omnia transfer experiments/smoke.json              # ~20 s pipeline check
./target/release/omnia transfer experiments/pilot_cpu_tempo.json    # ~1.5-2 h on 4 cores
```

Outputs: `runs/<name>/{config.json,games.json,results.jsonl,summary.md,pretrain/*}`.
`libopenblas-dev` is optional (`--features openblas`); it did not change throughput here.
