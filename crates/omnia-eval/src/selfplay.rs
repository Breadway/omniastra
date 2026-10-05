//! Search-enhanced self-play (stage D/E): neural PUCT/ISMCTS self-play with a
//! historical-checkpoint league, search-policy distillation back into the raw
//! network, and periodic evaluation against a fixed panel.

use crate::agent::*;
use crate::matches::*;
use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use omnia_dsl::MAX_PLAYERS;
use omnia_engine::{Game, Rng, State};
use omnia_model::*;
use omnia_observation::batch::Sample;
use omnia_observation::{Tokenizer, TokenizerConfig};
use omnia_search::*;
use omnia_training::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfPlayCfg {
    pub iterations: usize,
    pub games_per_iter: usize,
    pub sims: u32,
    pub c_explore: f32,
    pub temp_moves: u32,
    /// Probability of a uniformly random move (exploration); targets stay the search policy.
    pub explore_eps: f32,
    pub buffer_cap: usize,
    pub train_steps: usize,
    pub batch: usize,
    pub lr: f64,
    /// Snapshot the current network into the league every this many iterations.
    pub league_every: usize,
    pub league_size: usize,
    /// Probability that a self-play game pits the current net against a league member.
    pub p_league: f32,
    pub eval_games: u32,
    pub max_history: usize,
    pub seed: u64,
}

impl Default for SelfPlayCfg {
    fn default() -> Self {
        SelfPlayCfg { iterations: 10, games_per_iter: 16, sims: 16, c_explore: 1.5, temp_moves: 10, explore_eps: 0.05, buffer_cap: 20000, train_steps: 50, batch: 32, lr: 1e-3, league_every: 2, league_size: 6, p_league: 0.3, eval_games: 20, max_history: 16, seed: 1 }
    }
}

type Net<B> = NetHandle<<B as AutodiffBackend>::InnerBackend>;

fn play_one<B: AutodiffBackend>(game: &Arc<Game>, game_idx: u32, family_idx: u32, cfg: &SelfPlayCfg, current: &Net<B>, league: &Option<Net<B>>, seed: u64) -> Vec<Sample>
where
    OmniAstra<B::InnerBackend>: Send + Sync,
    B::Device: Send,
{
    let n = game.n() as usize;
    let tk = Tokenizer::new(TokenizerConfig { max_history: cfg.max_history });
    let mut rng = Rng::new(seed ^ 0xA11CE);
    let mcfg = MctsConfig { sims: cfg.sims, c_explore: cfg.c_explore, determinize: true, root_noise: 0.0 };
    // Seat 0/1 networks; the league member (if any) occupies a random seat.
    let league_seat = if league.is_some() { Some(rng.below(n as u64) as usize) } else { None };
    let mut mcts: Vec<Mcts<NeuralEvaluator<B::InnerBackend>>> = (0..n)
        .map(|s| {
            let net = if Some(s) == league_seat { league.clone().unwrap() } else { current.clone() };
            Mcts::new(mcfg.clone(), NeuralEvaluator::new(net))
        })
        .collect();
    let mut st = State::new(game, seed);
    let mut rec: Vec<(Sample, u8)> = vec![];
    let mut decision = 0u32;
    while !st.is_terminal() {
        let p = st.decision_maker().unwrap();
        let na = st.legal_actions().len();
        let (pi, best) = if na == 1 {
            (vec![1.0], 0)
        } else {
            let r = mcts[p as usize].search(&mut st, &mut rng);
            (r.policy, r.best)
        };
        if Some(p as usize) != league_seat && na > 1 {
            let obs = tk.observe(&mut st, p);
            rec.push((Sample { obs, pi: pi.clone(), value: [0.0; MAX_PLAYERS], game_idx, family_idx }, p));
        }
        let a = if rng.f32() < cfg.explore_eps {
            rng.below(na as u64) as usize
        } else if decision < cfg.temp_moves {
            sample_with_temperature(&pi, 1.0, &mut rng)
        } else {
            best
        };
        st.apply(a);
        decision += 1;
    }
    let pay = st.payoffs().unwrap();
    rec.into_iter()
        .map(|(mut s, p)| {
            for r in 0..n {
                s.value[r] = pay[(p as usize + r) % n];
            }
            s
        })
        .collect()
}

pub struct SelfPlayGame {
    pub game: Arc<Game>,
    pub game_idx: u32,
    pub family_idx: u32,
}

/// Run the loop; returns the trainer (final model) for further use.
pub fn run<B: AutodiffBackend>(games: &[SelfPlayGame], model: OmniAstra<B>, mcfg: &ModelConfig, cfg: &SelfPlayCfg, devices: Vec<B::Device>, out_dir: &Path) -> Result<Trainer<B>>
where
    OmniAstra<B::InnerBackend>: Send + Sync,
    B::Device: Send + Sync,
{
    std::fs::create_dir_all(out_dir)?;
    let mut log = RunLog::create(&out_dir.join("selfplay.jsonl"));
    let dev0 = devices[0].clone();
    let tcfg = TrainCfg { steps: cfg.iterations * cfg.train_steps, batch: cfg.batch, lr: cfg.lr, seed: cfg.seed, ..Default::default() };
    let mut trainer = Trainer::<B>::new(model, mcfg.clone(), tcfg, devices);
    let mut buffer: VecDeque<Sample> = VecDeque::new();
    let mut league: Vec<Net<B>> = vec![];
    let mut rng = Rng::new(cfg.seed);
    let mk_net = |m: OmniAstra<B::InnerBackend>, gi: u32| -> Net<B> { NetHandle { model: Arc::new(m), cfg: Arc::new(mcfg.clone()), dev: dev0.clone(), game_idx: gi, max_history: cfg.max_history } };

    for it in 0..cfg.iterations {
        let t0 = std::time::Instant::now();
        let current_model = trainer.valid();
        // Self-play.
        let jobs: Vec<(usize, u64, Option<usize>)> = (0..cfg.games_per_iter)
            .map(|g| {
                let gi = g % games.len();
                let use_league = !league.is_empty() && rng.f32() < cfg.p_league;
                (gi, cfg.seed.wrapping_mul(7919).wrapping_add((it * cfg.games_per_iter + g) as u64), if use_league { Some(rng.below(league.len() as u64) as usize) } else { None })
            })
            .collect();
        let results: Vec<Vec<Sample>> = jobs
            .par_iter()
            .map(|(gi, seed, lg)| {
                let sg = &games[*gi];
                let cur = mk_net(current_model.clone(), sg.game_idx);
                let opp = lg.map(|i| {
                    let mut n = league[i].clone();
                    n.game_idx = sg.game_idx;
                    n
                });
                play_one::<B>(&sg.game, sg.game_idx, sg.family_idx, cfg, &cur, &opp, *seed)
            })
            .collect();
        let new_samples: usize = results.iter().map(|r| r.len()).sum();
        for r in results {
            for s in r {
                buffer.push_back(s);
                if buffer.len() > cfg.buffer_cap {
                    buffer.pop_front();
                }
            }
        }
        let t_sp = t0.elapsed().as_secs_f32();
        // Train on the replay buffer.
        let flat: Vec<Sample> = buffer.iter().cloned().collect();
        let pools: Vec<Pool> = games
            .iter()
            .map(|g| Pool { game_idx: g.game_idx, family_idx: g.family_idx, samples: flat.iter().filter(|s| s.game_idx == g.game_idx).cloned().collect() })
            .filter(|p| !p.samples.is_empty())
            .collect();
        let mut last = StepStats::default();
        if !pools.is_empty() {
            let mut sampler = MixSampler::new(&pools, &MixSpec::UniformByGame, cfg.seed + it as u64);
            for _ in 0..cfg.train_steps {
                last = trainer.train_step(&sampler.next_batch(cfg.batch));
            }
        }
        if cfg.league_every > 0 && (it + 1) % cfg.league_every == 0 {
            league.push(mk_net(trainer.valid(), 0));
            if league.len() > cfg.league_size {
                league.remove(0);
            }
        }
        // Evaluation of the raw policy vs a fixed panel.
        let mut evals = vec![];
        let net_model = Arc::new(trainer.valid());
        for sg in games {
            for opp in ["random", "heuristic", "mcts:16"] {
                let fac: AgentFactory = match opp {
                    "random" => random_factory(),
                    "heuristic" => heuristic_factory(&sg.game),
                    _ => mcts_factory(&sg.game, 16),
                };
                let net = NetHandle { model: net_model.clone(), cfg: Arc::new(mcfg.clone()), dev: dev0.clone(), game_idx: sg.game_idx, max_history: cfg.max_history };
                let a: AgentFactory = Arc::new(move || Box::new(NeuralAgent::new(net.clone(), "net")));
                let r = play_match(&sg.game, &a, &fac, cfg.eval_games, 31337);
                evals.push(serde_json::json!({"game": sg.game.def.name, "opponent": opp, "score": r.score, "ci": [r.ci_lo, r.ci_hi]}));
            }
        }
        let row = serde_json::json!({
            "iteration": it, "new_samples": new_samples, "buffer": buffer.len(), "selfplay_secs": t_sp, "iter_secs": t0.elapsed().as_secs_f32(),
            "loss": last.loss, "policy_loss": last.policy, "value_loss": last.value, "league": league.len(), "counters": trainer.counters, "eval": evals,
        });
        println!("[selfplay] it {it}: +{new_samples} samples ({t_sp:.0}s), loss {:.3}, eval {}", last.loss, evals.iter().map(|e| format!("{}:{:.2}", e["opponent"].as_str().unwrap(), e["score"].as_f64().unwrap())).collect::<Vec<_>>().join(" "));
        log.write(&row);
    }
    Ok(trainer)
}
