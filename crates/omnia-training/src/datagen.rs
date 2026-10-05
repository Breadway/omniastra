//! Trajectory generation (stage A/B data): expert play with search-policy
//! targets and outcome value targets, stored in a game-independent format.

use omnia_dsl::MAX_PLAYERS;
use omnia_engine::{Game, Rng, State};
use omnia_observation::batch::Sample;
use omnia_observation::{Tokenizer, TokenizerConfig};
use omnia_search::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expert {
    Random,
    Heuristic,
    Mcts { sims: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSpec {
    pub expert: Expert,
    pub games: usize,
    /// Probability of overriding the played action with a uniformly random
    /// one (targets still come from the expert) to broaden state coverage.
    pub epsilon: f32,
    /// Sample played moves from the search policy for the first N decisions.
    pub temp_moves: u32,
    pub max_history: usize,
    pub seed: u64,
    /// Sharpen stored MCTS visit-count targets: pi^(1/t) renormalised (1.0 = raw
    /// visit distribution). UCT spreads visits, so raw targets are close to
    /// uniform; t < 1 makes the policy-imitation signal measurable.
    #[serde(default = "one")]
    pub target_temp: f32,
}

fn one() -> f32 {
    1.0
}

pub fn sharpen(pi: &[f32], t: f32) -> Vec<f32> {
    if (t - 1.0).abs() < 1e-6 {
        return pi.to_vec();
    }
    let w: Vec<f32> = pi.iter().map(|x| x.max(1e-9).powf(1.0 / t)).collect();
    let s: f32 = w.iter().sum();
    w.into_iter().map(|x| x / s).collect()
}

impl Default for DataSpec {
    fn default() -> Self {
        DataSpec { expert: Expert::Mcts { sims: 32 }, games: 100, epsilon: 0.1, temp_moves: 12, max_history: 32, seed: 1, target_temp: 1.0 }
    }
}

/// Generate samples from `spec.games` self-play games of the expert.
pub fn generate_samples(game: &Arc<Game>, game_idx: u32, family_idx: u32, spec: &DataSpec) -> Vec<Sample> {
    let per_game: Vec<Vec<Sample>> = (0..spec.games)
        .into_par_iter()
        .map(|g| play_and_record(game, game_idx, family_idx, spec, spec.seed.wrapping_mul(1_000_003).wrapping_add(g as u64)))
        .collect();
    per_game.into_iter().flatten().collect()
}

fn play_and_record(game: &Arc<Game>, game_idx: u32, family_idx: u32, spec: &DataSpec, seed: u64) -> Vec<Sample> {
    let tk = Tokenizer::new(TokenizerConfig { max_history: spec.max_history });
    let mut st = State::new(game, seed);
    let mut rng = Rng::new(seed ^ 0xDA7A);
    let mut mcts: Option<Mcts<RolloutEval>> = match spec.expert {
        Expert::Mcts { sims } => Some(Mcts::new(MctsConfig { sims, ..Default::default() }, RolloutEval::new(&game.def))),
        _ => None,
    };
    let mut heur = HeuristicAgent::new(&game.def);
    let n = game.n() as usize;
    let mut out: Vec<(Sample, u8)> = vec![];
    let mut decision = 0u32;
    while !st.is_terminal() {
        let p = st.decision_maker().unwrap();
        let na = st.legal_actions().len();
        // Expert target.
        let (pi, expert_best): (Vec<f32>, usize) = match &spec.expert {
            Expert::Random => (vec![1.0 / na as f32; na], rng.below(na as u64) as usize),
            Expert::Heuristic => {
                let a = heur.act(&mut st, &mut rng);
                let mut pi = vec![0.0; na];
                pi[a] = 1.0;
                (pi, a)
            }
            Expert::Mcts { .. } => {
                let r = mcts.as_mut().unwrap().search(&mut st, &mut rng);
                (r.policy, r.best)
            }
        };
        let obs = tk.observe(&mut st, p);
        // Acting policy.
        let a = if rng.f32() < spec.epsilon {
            rng.below(na as u64) as usize
        } else if decision < spec.temp_moves && matches!(spec.expert, Expert::Mcts { .. }) {
            sample_with_temperature(&pi, 1.0, &mut rng)
        } else {
            expert_best
        };
        let stored = if matches!(spec.expert, Expert::Mcts { .. }) { sharpen(&pi, spec.target_temp) } else { pi };
        out.push((Sample { obs, pi: stored, value: [0.0; MAX_PLAYERS], game_idx, family_idx }, p));
        st.apply(a);
        decision += 1;
    }
    let pay = st.payoffs().unwrap();
    out.into_iter()
        .map(|(mut s, p)| {
            for r in 0..n {
                s.value[r] = pay[(p as usize + r) % n];
            }
            s
        })
        .collect()
}
