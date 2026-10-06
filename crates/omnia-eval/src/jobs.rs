//! Stand-alone jobs: multi-game training from a config, checkpoint evaluation,
//! and self-play launched from JSON configs.

use crate::agent::*;
use crate::matches::*;
use crate::selfplay::{self, SelfPlayCfg, SelfPlayGame};
use crate::transfer::*;
use anyhow::{anyhow, Result};
use burn::module::Module;
use burn::tensor::backend::AutodiffBackend;
use omnia_model::*;
use omnia_observation::batch::Sample;
use omnia_training::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainJob {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub model_overrides: Option<serde_json::Value>,
    pub groups: Vec<GameGroup>,
    pub data: DataCfg,
    pub train: PretrainCfg,
    /// Held-out validation games per group (fresh episodes of the same games).
    pub val_games: usize,
    pub eval_every: usize,
    pub seed: u64,
    pub out_dir: String,
}

fn model_cfg(name: &str, ov: &Option<serde_json::Value>, n_games: usize) -> Result<ModelConfig> {
    let base = ModelConfig::by_name(name).ok_or_else(|| anyhow!("unknown model {name}"))?.with_n_games(n_games.max(8));
    match ov {
        None => Ok(base),
        Some(o) => {
            let mut v = serde_json::to_value(&base)?;
            for (k, x) in o.as_object().ok_or_else(|| anyhow!("model_overrides must be an object"))? {
                v.as_object_mut().unwrap().insert(k.clone(), x.clone());
            }
            Ok(serde_json::from_value(v)?)
        }
    }
}

/// Multi-game policy/value training from expert data; saves a checkpoint.
pub fn train_job<B: AutodiffBackend>(job: &TrainJob, games_dir: &Path, devices: Vec<B::Device>) -> Result<()>
where
    B::Device: Send + Sync,
    OmniAstra<B::InnerBackend>: Send + Sync,
{
    let out = PathBuf::from(&job.out_dir);
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("job.json"), serde_json::to_string_pretty(job)?)?;
    let tc = TransferConfig {
        name: job.name.clone(),
        model: job.model.clone(),
        model_overrides: None,
        seeds: vec![job.seed],
        groups: job.groups.clone(),
        target_group: String::new(),
        pretrain_sets: Default::default(),
        data: job.data.clone(),
        pretrain: job.train.clone(),
        finetune: FinetuneCfg { budgets: vec![], steps: 0, batch: 0, lr: 0.0, eval_every: 0, val_positions: 0 },
        arms: vec![],
        eval: EvalCfg { games: 0, opponents: vec![], batch: 64 },
        out_dir: job.out_dir.clone(),
    };
    let instances = build_instances(&tc, games_dir)?;
    let mcfg = model_cfg(&job.model, &job.model_overrides, instances.len())?;
    let mut pools = vec![];
    let mut val: Vec<Sample> = vec![];
    for (k, i) in instances.iter().enumerate() {
        let t = Instant::now();
        let spec = |games, seed| DataSpec { expert: Expert::Mcts { sims: job.data.sims }, games, epsilon: job.data.epsilon, temp_moves: job.data.temp_moves, max_history: job.data.max_history, seed, target_temp: job.data.target_temp };
        let tr = generate_samples(&i.game, i.game_idx, i.family_idx, &spec(job.data.pretrain_games_per_game, 100 + k as u64));
        val.extend(generate_samples(&i.game, i.game_idx, i.family_idx, &spec(job.val_games, 900 + k as u64)));
        println!("[{}] data {} : {} positions ({:.1}s)", job.name, i.id, tr.len(), t.elapsed().as_secs_f32());
        pools.push(Pool { game_idx: i.game_idx, family_idx: i.family_idx, samples: tr });
    }
    let dev0 = devices[0].clone();
    B::seed(&dev0, job.seed);
    let model = OmniAstra::<B>::new(&mcfg, &dev0);
    println!("[{}] model {} ({} params)", job.name, job.model, model.num_params());
    let tcfg = TrainCfg { steps: job.train.steps, batch: job.train.batch, lr: job.train.lr, seed: job.seed, ..Default::default() };
    let mut tr = Trainer::<B>::new(model, mcfg.clone(), tcfg.clone(), devices);
    let mut sampler = MixSampler::new(&pools, &job.train.mix, job.seed);
    let mut log = RunLog::create(&out.join("train.jsonl"));
    let val_refs: Vec<&Sample> = val.iter().collect();
    let t = Instant::now();
    for step in 1..=tcfg.steps {
        let s = tr.train_step(&sampler.next_batch(tcfg.batch));
        if step % job.eval_every.max(1) == 0 || step == tcfg.steps {
            let ev = evaluate_dataset::<B::InnerBackend>(&tr.valid(), &mcfg, &val_refs, 64, &dev0);
            println!("[{}] step {step}/{} loss {:.3} | val CE {:.3} top1 {:.3} vMSE {:.3} | {:.0}s", job.name, tcfg.steps, s.loss, ev.overall.policy_ce, ev.overall.top1, ev.overall.value_mse, t.elapsed().as_secs_f32());
            log.write(&serde_json::json!({"step": step, "train": s, "val": ev, "counters": tr.counters}));
        }
    }
    let games_meta: Vec<_> = instances.iter().map(|i| serde_json::json!({"id": i.id, "hash": i.def.hash_hex(), "game_idx": i.game_idx})).collect();
    save(&out.join("checkpoint"), &tr, serde_json::json!({"games": games_meta, "mix": job.train.mix, "seed": job.seed, "model": job.model}))?;
    println!("[{}] saved {}", job.name, out.join("checkpoint").display());
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SelfPlayJob {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub model_overrides: Option<serde_json::Value>,
    pub groups: Vec<GameGroup>,
    pub cfg: SelfPlayCfg,
    /// Optional weights-only initialisation (a `checkpoint` dir).
    #[serde(default)]
    pub init_from: Option<String>,
    pub out_dir: String,
}

pub fn selfplay_job<B: AutodiffBackend>(job: &SelfPlayJob, games_dir: &Path, devices: Vec<B::Device>) -> Result<()>
where
    B::Device: Send + Sync,
    OmniAstra<B::InnerBackend>: Send + Sync,
{
    let tc = TransferConfig {
        name: job.name.clone(),
        model: job.model.clone(),
        model_overrides: None,
        seeds: vec![job.cfg.seed],
        groups: job.groups.clone(),
        target_group: String::new(),
        pretrain_sets: Default::default(),
        data: DataCfg { sims: 0, pretrain_games_per_game: 0, target_train_games: 0, target_val_games: 0, epsilon: 0.0, temp_moves: 0, max_history: job.cfg.max_history, target_temp: 1.0 },
        pretrain: PretrainCfg { steps: 0, batch: 0, lr: 0.0, mix: MixSpec::UniformByGame },
        finetune: FinetuneCfg { budgets: vec![], steps: 0, batch: 0, lr: 0.0, eval_every: 0, val_positions: 0 },
        arms: vec![],
        eval: EvalCfg { games: 0, opponents: vec![], batch: 64 },
        out_dir: job.out_dir.clone(),
    };
    let instances = build_instances(&tc, games_dir)?;
    let mcfg = model_cfg(&job.model, &job.model_overrides, instances.len())?;
    let dev0 = devices[0].clone();
    B::seed(&dev0, job.cfg.seed);
    let model = match &job.init_from {
        Some(d) => load_weights::<B>(Path::new(d), &mcfg, &dev0)?,
        None => OmniAstra::<B>::new(&mcfg, &dev0),
    };
    let games: Vec<SelfPlayGame> = instances.iter().map(|i| SelfPlayGame { game: i.game.clone(), game_idx: i.game_idx, family_idx: i.family_idx }).collect();
    let out = PathBuf::from(&job.out_dir);
    let trainer = selfplay::run::<B>(&games, model, &mcfg, &job.cfg, devices, &out)?;
    save(&out.join("checkpoint"), &trainer, serde_json::json!({"job": job.name}))?;
    Ok(())
}

/// Evaluate a saved checkpoint's raw policy against a panel on a game.
pub fn eval_checkpoint<B: AutodiffBackend>(checkpoint: &Path, model: &str, overrides: &Option<serde_json::Value>, game: Arc<omnia_engine::Game>, game_idx: u32, opponents: &[String], games: u32, device: B::Device) -> Result<Vec<MatchResult>>
where
    OmniAstra<B::InnerBackend>: Send + Sync,
    B::Device: Send,
{
    let mcfg = model_cfg(model, overrides, 64)?;
    let m = load_weights::<B::InnerBackend>(checkpoint, &mcfg, &device)?;
    let net = NetHandle { model: Arc::new(m), cfg: Arc::new(mcfg), dev: device, game_idx, max_history: 16 };
    let mut out = vec![];
    for o in opponents {
        let fac = if o == "random" {
            random_factory()
        } else if o == "heuristic" {
            heuristic_factory(&game)
        } else if let Some(n) = o.strip_prefix("mcts:") {
            mcts_factory(&game, n.parse()?)
        } else {
            return Err(anyhow!("unknown opponent {o}"));
        };
        let n2 = net.clone();
        let a: AgentFactory = Arc::new(move || Box::new(NeuralAgent::new(n2.clone(), "net")));
        let r = play_match(&game, &a, &fac, games, 4242);
        println!("net vs {o}: score {:.3} [{:.3}, {:.3}] ({}-{}-{}), elo {:+.0}", r.score, r.ci_lo, r.ci_hi, r.wins, r.draws, r.losses, r.elo);
        out.push(r);
    }
    Ok(out)
}

/// Diagnose learnability on one game: target-entropy floor, uniform baseline,
/// and train/val fit over training. Use before trusting a transfer metric.
#[allow(clippy::too_many_arguments)]
pub fn diagnose<B: AutodiffBackend>(game: Arc<omnia_engine::Game>, model: &str, sims: u32, games: usize, steps: usize, batch: usize, lr: f64, target_temp: f32, devices: Vec<B::Device>) -> Result<()>
where
    B::Device: Send + Sync,
{
    let mcfg = model_cfg(model, &None, 8)?;
    let spec = |g, seed| DataSpec { expert: Expert::Mcts { sims }, games: g, epsilon: 0.1, temp_moves: 10, max_history: 12, seed, target_temp };
    let tr = generate_samples(&game, 0, 0, &spec(games, 11));
    let va = generate_samples(&game, 0, 0, &spec((games / 4).max(4), 99));
    let stat = |s: &[Sample]| -> (f32, f32, f32, f32) {
        let n = s.len() as f32;
        let unif: f32 = s.iter().map(|x| (x.pi.len() as f32).ln()).sum::<f32>() / n;
        let ent: f32 = s.iter().map(|x| -x.pi.iter().filter(|p| **p > 0.0).map(|p| p * p.ln()).sum::<f32>()).sum::<f32>() / n;
        let branch: f32 = s.iter().map(|x| x.pi.len() as f32).sum::<f32>() / n;
        let forced: f32 = s.iter().filter(|x| x.pi.len() == 1).count() as f32 / n;
        (unif, ent, branch, forced)
    };
    let (u, h, b, f) = stat(&tr);
    println!("data: {} train / {} val positions; mean legal actions {b:.1} ({:.0}% forced)", tr.len(), va.len(), f * 100.0);
    println!("train targets: uniform-policy CE {u:.3}, target entropy {h:.3} (CE floor for a perfect imitator; gap {:.3} = max learnable gain)", u - h);
    let (u2, h2, _, _) = stat(&va);
    println!("val   targets: uniform CE {u2:.3}, entropy {h2:.3}");
    let dev0 = devices[0].clone();
    B::seed(&dev0, 1);
    let m = OmniAstra::<B>::new(&mcfg, &dev0);
    let tcfg = TrainCfg { steps, batch, lr, ..Default::default() };
    let mut t = Trainer::<B>::new(m, mcfg.clone(), tcfg, devices);
    let pools = vec![Pool { game_idx: 0, family_idx: 0, samples: tr.clone() }];
    let mut sampler = MixSampler::new(&pools, &MixSpec::UniformByGame, 3);
    let trr: Vec<&Sample> = tr.iter().take(400).collect();
    let var: Vec<&Sample> = va.iter().take(400).collect();
    let every = (steps / 8).max(1);
    for step in 0..=steps {
        if step % every == 0 {
            let m = t.valid();
            let a = evaluate_dataset::<B::InnerBackend>(&m, &mcfg, &trr, 64, &dev0).overall;
            let v = evaluate_dataset::<B::InnerBackend>(&m, &mcfg, &var, 64, &dev0).overall;
            println!("step {step:>5}: train CE {:.3} top1 {:.3} | val CE {:.3} top1 {:.3} vMSE {:.3} vSignAcc {:.2}", a.policy_ce, a.top1, v.policy_ce, v.top1, v.value_mse, v.value_sign_acc);
        }
        if step < steps {
            t.train_step(&sampler.next_batch(batch));
        }
    }
    Ok(())
}
