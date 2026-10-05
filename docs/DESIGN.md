# OmniAstra — design document

OmniAstra is an experimental platform for one question:

> Does training one shared Transformer across many *mechanically different* card/strategy games
> produce representations that transfer to an unseen game (less game-specific experience for the
> same strength)?

The platform is built so the answer can be **no**. Nothing in the generic engine or model knows
about any individual game; games are data (OmniDSL programs). This document records the design,
the decisions taken where the original specification was ambiguous or self-contradictory, and the
known limitations.

This is a **standalone** project; it does not use or modify the existing `astra/` code.

## 1. Status against the specification

| Spec area | State |
|---|---|
| OmniDSL (typed IR, validation, canonical hash, static analysis) | implemented (`omnia-dsl`) |
| OmniEngine (deterministic, events, stack/response windows, triggers, modifiers, delayed effects, cloning, replay, hidden-information views) | implemented (`omnia-engine`) |
| Three hand-written, mechanically different games | `games/` (skirmish, merchant, gambit) |
| OmniGen (families, controlled variants, validation, controls) | implemented, 5 families (`omnia-gen`) |
| Canonical variable-length observation + batching | implemented (`omnia-observation`) |
| OmniAstra model (relational Transformer, registers, action-query decoder), nano→medium presets, ablation switches | implemented (`omnia-model`) |
| Baseline agents: random, goal-heuristic, ISMCTS/PUCT | implemented (`omnia-search`) |
| Trainer (BC / search distillation), mixture sampling, checkpoints, compute accounting, data-parallel path | implemented (`omnia-training`); multi-device path parity-tested on CPU shards only |
| Evaluation, match CIs, transfer experiment runner, self-play + league | implemented (`omnia-eval`); self-play exercised only at toy scale |
| CLI | implemented (`omnia-cli`, binary `omnia`) |
| Rule tokens (effect AST fed to the model) | **not implemented**; v0 uses static *descriptors* (see §6) |
| Belief-state search for hidden information | **not implemented**; v0 uses determinization (see §7) |
| CUDA / multi-GPU runs | code paths written, **never executed** (no GPU in the dev container) |
| A transfer result | see `docs/RESULTS.md` |

## 2. Specification review: contradictions and decisions

1. **"No fixed token count" vs. fixed-width numeric features.** Token *counts* are variable
   (state, history and action sets are all variable-length; padding exists only in
   `omnia-observation::batch`). Per-token feature *width* is bounded (≤12 object attributes, ≤6
   resources, ≤16 numeric slots, ≤32 action defs). Games exceeding the bounds are rejected by
   the validator rather than silently truncated.
2. **"Games are data" vs. needing identity for cards.** A categorical template id is
   game-local and arbitrary. v0 therefore exposes *three* views of every entity, separately
   switchable in the model: (a) game-local ids (`use_ids`), (b) position-indexed numeric slots
   (`use_slot_pos`), (c) rule-derived descriptors (`use_desc`). Experiments can mask any
   subset; the transfer-relevant regime is "ids relabelled per game" (the generator does this by
   default, `relabel: true`).
3. **Value = P(win | observable state).** Not well defined in imperfect-information games
   (it depends on both players' policies and on belief). v0 defines the value target as the
   final payoff under the *data-generating policy* (expert self-play), per relative seat, in
   [-1, 1]. This is documented, not solved.
4. **Deterministic multi-GPU parity.** Bitwise determinism across thread/device schedules is not
   achievable; the parity test compares a single-batch update with a k-shard accumulated update
   under a numerical tolerance (1e-4), using *global* loss denominators so shard gradients sum
   exactly to the full-batch gradient.
5. **Reaction windows vs. "avoid Turing completeness".** Chains are modelled with an explicit
   LIFO stack and consecutive-pass resolution; effects are structured, loops bounded, step budget
   per resolution, trigger-generation cap, per-game turn/decision caps. Mid-resolution choices
   (a `SELECT` that suspends an effect) are *not* supported: all choices are made up front as
   action targets/parameters; `RandomSelect` is `Sel` with `SelOrder::Random`.
6. **Replacement effects** ("if practical"): not implemented. Continuous modifiers (attribute
   deltas), delayed effects, triggers and reactions are.
7. **Zero-shot from rules.** Not attempted in v0 (spec says so too). Rule *descriptors* are the
   stepping stone.
8. **"Equivalent reorderings should not change semantics" vs. deterministic action
   enumeration.** Enumeration order is deterministic (zone order, then position). Unordered
   zones are *canonically sorted* in the player view (insertion/draw order must not leak), and the
   model has no object positional encoding, so it is permutation-equivariant. Tested (see §9).

## 3. Crates and data flow

```
GameDef (RON/JSON)  ──omnia-dsl──►  validate / canonical hash / descriptors / relabel
        │
        ▼
 omnia-engine: Game (compiled) ──► State (true state, RNG, history) ──► PlayerView (information boundary)
        │                                   │ clone / apply(action index) / determinize(observer)
        ▼                                   ▼
 omnia-gen (families, controls)      omnia-observation: Tokenizer ──► Observation ──► HostBatch (padding only here)
 omnia-search (agents, ISMCTS)                                              │
        │                                                                   ▼
        └────────── datagen ───────────────► omnia-training (Trainer, MixSampler, checkpoints)
                                            omnia-model (Burn: OmniAstra<B>)
                                            omnia-eval (matches, transfer runner, self-play) ── omnia-cli
```

Simulation (`omnia-dsl`, `omnia-engine`, `omnia-search`, `omnia-gen`) has **no** dependency on the ML
stack.

## 4. OmniDSL / OmniEngine semantics (summary)

* **Players, resources, attributes, vars**: generic typed integers with min/max; resources can be
  public or owner-only. No resource has built-in meaning.
* **Zones**: per-player or shared; ordered/unordered; visibility `Public | Private | Hidden`;
  capacity; allowed object kinds. Zone *names* are labels only.
* **Objects**: template (kind, attrs, hooks, triggers, modifiers), owner, controller, zone,
  attachment, per-object reveal mask.
* **Effects** (bounded IR): `Seq If ForEach ForEachPlayer Repeat Move Create Copy Destroy
  Transform SetController SetAttr ModAttr SetRes ModRes SetVar ModVar Shuffle Reveal Hide Attach
  Detach Hook Emit Delay CancelStack Win Lose DrawGame EndPhase EndTurn`. Expressions/conditions
  are typed and side-effect free except `Rand`, which uses the explicit RNG.
* **Events** are first-class (`Event` with actor, source, targets, slot, before/after, zones,
  causal parent, per-observer visibility masks). Triggers/modifiers subscribe to them; the model
  sees the (redacted) history.
* **Actions** are structured (`Action{def, actor, source, targets}`), enumerated by the engine
  for the decision maker only; the model ranks them. Sources/targets are enumerated from rule
  filters; costs are checked and paid by the engine. Response-timing actions open reaction
  windows on a LIFO stack; players with no legal response are skipped automatically.
* **Termination**: executable per-player terminal rules (`Win`/`Lose`), effects, timeout rule
  (`Draw` or by resource), hard decision cap, and fault detection (step limit, trigger depth,
  object limit, stack overflow). Faults end the game as an invalid draw and are rejected by the
  generator's validator.
* **Determinism**: `(game hash, seed, action indices)` reproduces a game exactly
  (`Replay`); RNG is an in-crate xoshiro256** seeded through SplitMix64.
* **Cloning**: `State: Clone`; the history is persistent (`Arc` chunks) so search clones are
  cheap. ~375k clone+apply/s and ~380k random decisions/s per core on the dev machine. No
  copy-on-write object arena was needed at this scale.
* **Information boundary**: `State::view(player) -> PlayerView` is the only route to the model.
  Hidden zones contribute counts, not identities; events carry per-observer masks (e.g. an
  opponent's draw is visible as an event with the card redacted); card identities are linked
  from history to current entities only if they were visible at event time and still are.
  `State::determinize(observer)` resamples hidden identities (same owner, respecting allowed
  kinds); the leakage test asserts `view(s) == view(determinize(s))` at every decision point of
  random games in every manual game (and a negative control checks the resampler really changes
  hidden state).

## 5. OmniGen

Families are parameterised generators with a **shared strategic skeleton** and randomised
numbers/composition/rule toggles:

| family | strategic structure | notes |
|---|---|---|
| `tempo` | renewable energy curve, cheap vs. expensive creatures, board interaction, targeted/untargeted spells | hidden hands, high interaction |
| `engine` | shared market, invest in income vs. cash out for points | low interaction, long horizon |
| `stack` | points race with a response stack and counterspells | reaction mechanics |
| `tug` | shared signed track, push vs. anchor, optional stochastic pushes | opposite-signed objectives |
| `garden` | slow-maturing assets: yield vs. harvest | shares `invest_vs_cash` with `engine`, different surface mechanics |

Controls (spec §22): `reskin` (id permutation, mechanics identical), `random_reward`
(`Adjudication::Random`: the game ends as usual but the winner is a coin flip), `Variant::AltVictory`
(same entities, different objective), `Variant::Inverted` (cost/reward curve inverted: same
surface, different optimal strategy), and cross-family pairs sharing a tradeoff tag
(`invest_vs_cash`). Strategic labels in `meta.tags` are generator *labels*; measured properties
(length, branching, side balance, timeout rate, heuristic/MCTS-vs-random score) are in
`GameStats`, produced by `measure()` and enforced by rejection sampling (`generate_valid`).
Generated games get all ids randomly relabelled by default so there is no accidental slot
alignment between games.

Honest note: families are hand-built generators, i.e. "five strategic skeletons", not an open-ended
space. Statistical power on *diversity* needs far more families than this; see §11.

## 6. Observation and rule descriptors

One flat token list per decision: `[global | players | zones | objects | events | actions]`.
Each token carries: game-local ids (`cat`), game-independent semantic categoricals (`sem`:
relative seat, generic event kind, visibility class…), raw numeric slots (+mask), a static
descriptor `desc[16]`, and (events) an order index. Relations are typed directed edges
(owned-by, controlled-by, located-in, attached-to, event source/targets/actor/zone, causal
parent, action source/targets…) with inverses; the model turns them into additive attention
biases. `omnia-dsl::analysis` derives descriptors *from the rules*: per template/action an
effect-primitive histogram (moves, resource gain/loss, win/lose, loops, conditionals, randomness,
number of targets/costs…); per attribute/resource its role (read in costs, written by effects,
compared in terminal rules, public/private, range); per zone its structure and usage. This lets a
model ground otherwise arbitrary slot indices *without* a game id. It is **not** a full rule
encoder; the effect AST itself is not tokenised (future work).

## 7. Hidden information

The policy only ever sees `PlayerView`. Search uses single-observer ISMCTS with per-simulation
determinization: hidden identities are resampled among the same owner's hidden objects. Known
limitations: it assumes decklists are common knowledge; it ignores information the observer has
accumulated about specific hidden cards (e.g. a card shown earlier, then shuffled away); and
determinized search has the usual strategy-fusion / non-locality pathologies. Belief models,
public-belief-state search and an asymmetric critic are not implemented.

## 8. Model (`omnia-model`)

* Token embedding = class + semantic categoricals + (optional) id embeddings + position-indexed
  numeric slots + (optional) rule-descriptor features (including descriptor-conditioned numeric
  slots for attributes/resources). Everything up to here is the *input adapter*.
* Backbone: learned **register tokens** + state/history tokens in pre-norm blocks (RMSNorm,
  SwiGLU, multi-head attention) with **relation-aware additive attention bias** and a
  **relative temporal bias** for history–history pairs. Padding is masked; there are no object
  position encodings (permutation-equivariant by construction).
  *Deviation from the sketch*: the bias tensors are computed once per forward and shared across
  layers (Graphormer-style) with a learned per-head gain per layer. Measured reason: the
  per-layer (B×N×N) embedding gather dominated CPU time (~half of the forward pass).
* Value head reads the registers only (mean over registers → MLP → tanh, one output per
  relative seat; masked by player count).
* Action decoder: every legal action is a query token = its own embedding + pointer features
  (embeddings of its source / targets / target player gathered from the backbone output) →
  cross-attention to registers+state (with an action→entity relation bias) → action-to-action
  self-attention → one scalar logit; softmax only over legal actions. No policy vocabulary.
* Presets: `nano` 0.36M (CPU pilots), `tiny` 1.8M, `small` 9.2M, `medium` 46M parameters.
  Ablation switches in `ModelConfig`: `use_ids, use_slot_pos, use_desc, use_game_emb, use_rel,
  use_time_bias, use_history, use_registers, action_self_attn, use_action_links`.
* Burn modules initialise lazily; clones of an unmaterialised module draw *different* weights.
  `OmniAstra::new` materialises parameters so arms/replicas share an initialisation (found and
  fixed via the data-parallel parity test).

## 9. Test inventory

* DSL/engine: parse+validate manual games; 200 random games each terminate without faults;
  deterministic replay hash equality; clone independence; hidden-information leakage
  (determinization-invariance of views) + negative control; response stack (counters cancel
  spells); relabelled games are statistically equivalent.
* Observation/model: variable lengths; unordered-zone order invariance; padding invariance of
  logits; invariance to permuting object tokens (with relations remapped); overfit sanity.
* Training: data generation determinism and targets well-formed; mixture balance independent of
  dataset size; loss decreases and seeded runs are reproducible; checkpoint round trip, resume vs.
  weights-only; frozen backbone really frozen; 1-shard vs 3-shard update parity.
* Search: heuristic/MCTS beat random on all manual games.

## 10. Backend and performance (measured on the 4-core CPU dev container)

Burn 0.21 was chosen (CUDA via cubecl, portable wgpu, ndarray/flex CPU, `collective`/DDP
support, module/optimizer/record APIs) over candle (the candle backend panicked in autodiff
for this model). Only the **ndarray** and **flex** backends were runnable here (`cpu` backend needs
an LLVM download; CUDA/wgpu not available).

| item | measured |
|---|---|
| engine, random play | ~380k decisions/s/core |
| observation tokenization | ~34k/s (≈120 tokens) |
| nano training | ~90 samples/s |
| tiny training | ~25–30 samples/s (bs 32, ~60 state tokens) |
| nano single-observation inference | ~2.3 ms |

The tiny model reaches only ~13 GFLOPS here, far under CPU peak; `flex` made embedding gathers
10× faster but total throughput was unchanged. CPU runs are therefore *pilots*; the intended
scale-up path is GPU (`--features cuda|wgpu`), untested.

## 11. Known limitations

* Statistical power: few families, small models, small data on CPU. A null result at this scale
  does not falsify the hypothesis; a positive result needs the controls to be read, not the
  headline number.
* Expert targets come from low-budget rollout MCTS, whose own policy noise sets a CE floor.
* Neural MCTS inference is per-leaf (no batched evaluation / virtual loss yet).
* Self-play/league is implemented and unit-exercised only at toy scale.
* No rule tokens, no belief model, no replacement effects, 2-player families only (engine
  supports ≤4 players; value head has 4 seat outputs).
* Multi-device training uses host-side gradient reduction through Burn's accumulator; NCCL/Burn
  `collective` all-reduce is not wired, and no multi-GPU run was performed.

## 12. Experimental protocol

See `docs/EXPERIMENTS.md`.
