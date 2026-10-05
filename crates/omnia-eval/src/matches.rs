use omnia_engine::Game;
use omnia_search::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub type AgentFactory = Arc<dyn Fn() -> Box<dyn Agent> + Send + Sync>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MatchResult {
    pub games: u32,
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
    /// Mean score (win = 1, draw = .5) of agent A.
    pub score: f32,
    /// 95% Wilson interval of the score treating each game as a Bernoulli trial.
    pub ci_lo: f32,
    pub ci_hi: f32,
    /// Elo difference implied by `score` (A minus B).
    pub elo: f32,
}

pub fn wilson(score: f32, n: u32) -> (f32, f32) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.96f32;
    let n = n as f32;
    let d = 1.0 + z * z / n;
    let c = score + z * z / (2.0 * n);
    let m = z * (score * (1.0 - score) / n + z * z / (4.0 * n * n)).sqrt();
    (((c - m) / d).max(0.0), ((c + m) / d).min(1.0))
}

pub fn elo_from_score(s: f32) -> f32 {
    let s = s.clamp(0.005, 0.995);
    -400.0 * (1.0 / s - 1.0).log10()
}

/// A vs B over `games` games with alternating seats (2-player games). Seeds
/// are `seed0 + i`, so results are reproducible and identical across agents.
pub fn play_match(game: &Arc<Game>, a: &AgentFactory, b: &AgentFactory, games: u32, seed0: u64) -> MatchResult {
    let outcomes: Vec<f32> = (0..games)
        .into_par_iter()
        .map(|i| {
            let seat_a = (i % 2) as usize;
            let (mut aa, mut bb) = (a(), b());
            let mut agents: Vec<Box<dyn Agent>> = if seat_a == 0 {
                vec![std::mem::replace(&mut aa, Box::new(RandomAgent)), std::mem::replace(&mut bb, Box::new(RandomAgent))]
            } else {
                vec![std::mem::replace(&mut bb, Box::new(RandomAgent)), std::mem::replace(&mut aa, Box::new(RandomAgent))]
            };
            let (pay, _) = play_game(game, &mut agents, seed0 + i as u64);
            (pay[seat_a] + 1.0) / 2.0
        })
        .collect();
    let (mut w, mut d, mut l) = (0, 0, 0);
    for o in &outcomes {
        if *o > 0.75 {
            w += 1
        } else if *o < 0.25 {
            l += 1
        } else {
            d += 1
        }
    }
    let score = outcomes.iter().sum::<f32>() / games.max(1) as f32;
    let (lo, hi) = wilson(score, games);
    MatchResult { games, wins: w, draws: d, losses: l, score, ci_lo: lo, ci_hi: hi, elo: elo_from_score(score) }
}

pub fn random_factory() -> AgentFactory {
    Arc::new(|| Box::new(RandomAgent))
}

pub fn heuristic_factory(game: &Arc<Game>) -> AgentFactory {
    let def = game.def.clone();
    Arc::new(move || Box::new(HeuristicAgent::new(&def)))
}

pub fn mcts_factory(game: &Arc<Game>, sims: u32) -> AgentFactory {
    let def = game.def.clone();
    Arc::new(move || Box::new(MctsAgent::rollout(&def, sims)))
}
