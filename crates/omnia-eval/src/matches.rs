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

/// One agent A against `n - 1` copies of B in an `n`-player game. A's seat
/// rotates with the game index so no seat is favoured. Seeds are `seed0 + i`,
/// so results are reproducible and identical across agents.
///
/// Payoffs are zero-sum, so equal strength scores 0.5 per game
/// (`(payoff + 1) / 2`). A game counts as a win for A when its payoff is
/// positive and a loss when negative. With more than two players the score and
/// its interval are only indicative: a win is worth +1 but a loss only
/// `-1 / (n - 1)`.
pub fn play_match(game: &Arc<Game>, a: &AgentFactory, b: &AgentFactory, games: u32, seed0: u64) -> MatchResult {
    let n = game.n() as usize;
    let outcomes: Vec<(f32, f32)> = (0..games)
        .into_par_iter()
        .map(|i| {
            let seat_a = i as usize % n;
            let mut agents: Vec<Box<dyn Agent>> = (0..n).map(|s| if s == seat_a { a() } else { b() }).collect();
            let (pay, _) = play_game(game, &mut agents, seed0 + i as u64);
            (pay[seat_a], (pay[seat_a] + 1.0) / 2.0)
        })
        .collect();
    let (mut w, mut d, mut l) = (0, 0, 0);
    for (pay, _) in &outcomes {
        if *pay > 1e-6 {
            w += 1
        } else if *pay < -1e-6 {
            l += 1
        } else {
            d += 1
        }
    }
    let score = outcomes.iter().map(|o| o.1).sum::<f32>() / games.max(1) as f32;
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

#[cfg(test)]
mod tests {
    use super::*;
    use omnia_dsl::GameDef;

    fn melee() -> Arc<Game> {
        Game::new(GameDef::from_ron(include_str!("../../../games/melee.ron")).unwrap()).unwrap()
    }

    #[test]
    fn six_player_match_rotates_seats_and_classifies_by_payoff_sign() {
        let g = melee();
        // one agent under test against five copies of the opponent, seat rotating per game
        let r = play_match(&g, &mcts_factory(&g, 24), &random_factory(), 36, 100);
        assert_eq!(r.wins + r.draws + r.losses, 36);
        // with six seats an equal-strength agent wins 1 game in 6; clearly better must beat that by far
        assert!(r.wins as f32 / 36.0 > 0.35, "mcts won only {}/36 against five random players", r.wins);
        assert!(r.score > 0.5, "score {}", r.score);
    }

    #[test]
    fn equal_agents_win_about_one_game_in_six() {
        let g = melee();
        let r = play_match(&g, &random_factory(), &random_factory(), 300, 7);
        let rate = r.wins as f32 / 300.0;
        assert!((0.08..0.28).contains(&rate), "random vs random win rate {rate}");
        assert!((r.score - 0.5).abs() < 0.12, "score {}", r.score);
    }
}
