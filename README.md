# OmniAstra

An experimental platform for testing whether one Transformer, trained across many structurally
different card/strategy games, learns strategic representations that transfer to an unseen game.

* `docs/DESIGN.md` — architecture, decisions, limitations
* `docs/EXPERIMENTS.md` — transfer-experiment protocol and controls
* `docs/RESULTS.md` — what has actually been run, with numbers

Workspace (`crates/`): `omnia-dsl` (game IR), `omnia-engine` (deterministic simulator),
`omnia-gen` (procedural game families + controls), `omnia-observation` (variable-length tokens),
`omnia-model` (Burn Transformer), `omnia-search` (agents, ISMCTS), `omnia-training`,
`omnia-eval` (matches, transfer runner, self-play), `omnia-cli` (binary `omnia`).
Hand-written games live in `games/`, experiment configs in `experiments/`.

```
cargo test --release --workspace
cargo run --release -p omnia-cli -- game generate --family tempo --count 3 --out /tmp/g
cargo run --release -p omnia-cli -- play games/skirmish.ron --p1 mcts:64 --p2 random --games 20
cargo run --release -p omnia-cli -- benchmark --model nano
cargo run --release -p omnia-cli -- transfer experiments/smoke.json
```

This project is independent of the film/`astra/` code elsewhere in this repository.
