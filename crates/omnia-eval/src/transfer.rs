//! The central transfer experiment.
//!
//! For a held-out target game H and a set of arms
//!   scratch | pretrained-on-S (full or backbone-only, frozen or not) | controls,
//! fine-tune on nested subsets of H's expert data and measure (a) held-out
//! policy/value loss, (b) playing strength, always with full compute accounting.
//! Pretraining sets, controls and all budgets/seeds are fixed in the config
//! *before* running; the runner never adapts them to results.

use crate::agent::*;
use crate::matches::*;
use anyhow::{anyhow, Result};
use burn::module::Module;
use burn::tensor::backend::AutodiffBackend;
use omnia_dsl::GameDef;
use omnia_engine::{Game, Rng};
use omnia_gen::*;
use omnia_model::*;
use omnia_observation::batch::Sample;
use omnia_training::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameGroup {
    pub id: String,
    /// Family name (see `Family::name`), or "manual:<file stem>" for games in `games/`.
    pub family: String,
    #[serde(default = "d_variant")]
    pub variant: String,
    pub n_games: usize,
    pub seed: u64,
    /// Randomly permute ids (default true: no accidental slot alignment between games).
    #[serde(default = "d_true")]
    pub relabel: bool,
    /// Control: outcome independent of play.
    #[serde(default)]
    pub random_reward: bool,
    /// Control: re-skin the games of another group (same mechanics, new ids).
    #[serde(default)]
    pub reskin_of: Option<String>,
    /// Control: random-reward copy of another group's games.
    #[serde(default)]
    pub random_reward_of: Option<String>,
}
fn d_variant() -> String {
    "normal".into()
}
fn d_true() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataCfg {
    pub sims: u32,
    pub pretrain_games_per_game: usize,
    pub target_train_games: usize,
    pub target_val_games: usize,
    pub epsilon: f32,
    pub temp_moves: u32,
    pub max_history: usize,
    /// Sharpen MCTS targets (see `DataSpec::target_temp`).
    #[serde(default = "d_one_f")]
    pub target_temp: f32,
}
fn d_one_f() -> f32 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PretrainCfg {
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub mix: MixSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinetuneCfg {
    /// Unique target-game training positions per run (nested subsets).
    pub budgets: Vec<usize>,
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub eval_every: usize,
    pub val_positions: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Reuse {
    /// Whole pretrained network including input adapter and heads.
    Full,
    /// Only the shared backbone; adapter (ids/slots/game embedding) and heads fresh.
    Backbone,
    /// Backbone and heads; adapter fresh.
    BackboneAndHeads,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum FreezeMode {
    None,
    Backbone,
    AllButAdapter,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Arm {
    pub name: String,
    /// Name of a pretrain set, or None for random initialisation.
    pub pretrain: Option<String>,
    pub reuse: Reuse,
    pub freeze: FreezeMode,
    /// Multiplier on fine-tune optimizer steps (compute-matched scratch baselines).
    #[serde(default = "d_one")]
    pub ft_steps_mult: f32,
}
fn d_one() -> f32 {
    1.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalCfg {
    pub games: u32,
    /// "random", "heuristic", "mcts:<sims>"
    pub opponents: Vec<String>,
    pub batch: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferConfig {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub model_overrides: Option<serde_json::Value>,
    pub seeds: Vec<u64>,
    pub groups: Vec<GameGroup>,
    pub target_group: String,
    pub pretrain_sets: BTreeMap<String, Vec<String>>,
    pub data: DataCfg,
    pub pretrain: PretrainCfg,
    pub finetune: FinetuneCfg,
    pub arms: Vec<Arm>,
    pub eval: EvalCfg,
    pub out_dir: String,
}

pub struct Instance {
    pub id: String,
    pub group: String,
    pub family_idx: u32,
    pub game_idx: u32,
    pub def: GameDef,
    pub game: Arc<Game>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrengthRow {
    pub opponent: String,
    pub score: f32,
    pub ci_lo: f32,
    pub ci_hi: f32,
    pub elo: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResultRow {
    pub experiment: String,
    pub seed: u64,
    pub target: String,
    pub arm: String,
    pub budget_positions: usize,
    pub pretrain_set: Option<String>,
    pub pretrain: Option<Counters>,
    pub finetune: Counters,
    pub ft_steps: usize,
    pub best_step: usize,
    pub val_ce: f32,
    pub val_top1: f32,
    pub val_value_mse: f32,
    pub strength: Vec<StrengthRow>,
    pub mean_strength_score: f32,
    pub curve: Vec<(usize, f32, f32)>,
    pub params: usize,
}

fn model_config(c: &TransferConfig, n_games: usize) -> Result<ModelConfig> {
    let base = ModelConfig::by_name(&c.model).ok_or_else(|| anyhow!("unknown model size {}", c.model))?.with_n_games(n_games.max(8));
    match &c.model_overrides {
        None => Ok(base),
        Some(o) => {
            let mut v = serde_json::to_value(&base)?;
            if let (Some(vm), Some(om)) = (v.as_object_mut(), o.as_object()) {
                for (k, x) in om {
                    vm.insert(k.clone(), x.clone());
                }
            }
            Ok(serde_json::from_value(v)?)
        }
    }
}

fn variant_of(s: &str) -> Variant {
    match s {
        "alt_victory" | "altvictory" => Variant::AltVictory,
        "inverted" => Variant::Inverted,
        _ => Variant::Normal,
    }
}

/// Resolve groups into concrete validated game instances.
pub fn build_instances(c: &TransferConfig, games_dir: &Path) -> Result<Vec<Instance>> {
    let mut out: Vec<Instance> = vec![];
    let mut by_group: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let fam_idx = |f: &str| -> u32 { Family::parse(f).map(|x| Family::ALL.iter().position(|y| *y == x).unwrap() as u32).unwrap_or(99) };
    let vcfg = ValidationConfig::default();
    for g in &c.groups {
        let mut defs: Vec<GameDef> = vec![];
        if let Some(src) = &g.reskin_of {
            for &i in by_group.get(src).ok_or_else(|| anyhow!("group {src} must precede {}", g.id))? {
                defs.push(reskin(&out[i].def, g.seed + defs.len() as u64));
            }
        } else if let Some(src) = &g.random_reward_of {
            for &i in by_group.get(src).ok_or_else(|| anyhow!("group {src} must precede {}", g.id))? {
                defs.push(random_reward(&out[i].def));
            }
        } else if let Some(stem) = g.family.strip_prefix("manual:") {
            let src = std::fs::read_to_string(games_dir.join(format!("{stem}.ron")))?;
            defs.push(GameDef::from_ron(&src)?);
        } else {
            let fam = Family::parse(&g.family).ok_or_else(|| anyhow!("unknown family {}", g.family))?;
            let v = variant_of(&g.variant);
            if !fam.supports(v) {
                return Err(anyhow!("family {} has no variant {}", g.family, g.variant));
            }
            for i in 0..g.n_games {
                let spec = GenSpec { family: fam, seed: g.seed + i as u64 * 7919, variant: v, relabel: g.relabel, random_reward: false };
                let (_, def, _, _) = generate_valid(&spec, &vcfg, 60).map_err(|e| anyhow!(e))?;
                defs.push(if g.random_reward { random_reward(&def) } else { def });
            }
        }
        for mut def in defs {
            def.name = format!("{}:{}", g.id, def.name);
            let game = Game::new(def.clone())?;
            let idx = out.len();
            by_group.entry(g.id.clone()).or_default().push(idx);
            out.push(Instance { id: def.name.clone(), group: g.id.clone(), family_idx: fam_idx(g.family.as_str().split(':').next().unwrap_or("")), game_idx: idx as u32, def, game });
        }
    }
    Ok(out)
}

fn data_spec(c: &DataCfg, games: usize, seed: u64) -> DataSpec {
    DataSpec { expert: Expert::Mcts { sims: c.sims }, games, epsilon: c.epsilon, temp_moves: c.temp_moves, max_history: c.max_history, seed, target_temp: c.target_temp }
}

fn opponent_factory(game: &Arc<Game>, name: &str) -> Result<AgentFactory> {
    if name == "random" {
        Ok(random_factory())
    } else if name == "heuristic" {
        Ok(heuristic_factory(game))
    } else if let Some(n) = name.strip_prefix("mcts:") {
        Ok(mcts_factory(game, n.parse()?))
    } else {
        Err(anyhow!("unknown opponent {name}"))
    }
}

/// Run the whole experiment. `B` is the autodiff backend; `devices` are the
/// training devices (one shard each).
pub fn run<B: AutodiffBackend>(cfgv: &TransferConfig, games_dir: &Path, devices: Vec<B::Device>) -> Result<Vec<ResultRow>>
where
    B::Device: Send + Sync,
    OmniAstra<B::InnerBackend>: Send + Sync,
{
    let out_dir = PathBuf::from(&cfgv.out_dir);
    std::fs::create_dir_all(&out_dir)?;
    std::fs::write(out_dir.join("config.json"), serde_json::to_string_pretty(cfgv)?)?;
    let mut log = RunLog::create(&out_dir.join("results.jsonl"));
    let t_start = Instant::now();

    let instances = build_instances(cfgv, games_dir)?;
    let mcfg = model_config(cfgv, instances.len())?;
    println!("[{}] {} game instances; model {} ({} games slots)", cfgv.name, instances.len(), cfgv.model, mcfg.n_games);
    let manifest: Vec<_> = instances.iter().map(|i| serde_json::json!({"id": i.id, "group": i.group, "hash": i.def.hash_hex(), "game_idx": i.game_idx, "tags": i.def.meta.tags})).collect();
    std::fs::write(out_dir.join("games.json"), serde_json::to_string_pretty(&manifest)?)?;

    let target_ids: Vec<usize> = instances.iter().enumerate().filter(|(_, i)| i.group == cfgv.target_group).map(|(k, _)| k).collect();
    if target_ids.is_empty() {
        return Err(anyhow!("target group {} is empty", cfgv.target_group));
    }

    // ---- data -------------------------------------------------------------
    let mut pretrain_data: BTreeMap<usize, Vec<Sample>> = BTreeMap::new();
    for (name, groups) in &cfgv.pretrain_sets {
        for k in 0..instances.len() {
            if groups.contains(&instances[k].group) && !pretrain_data.contains_key(&k) {
                let t = Instant::now();
                let s = generate_samples(&instances[k].game, instances[k].game_idx, instances[k].family_idx, &data_spec(&cfgv.data, cfgv.data.pretrain_games_per_game, 1000 + k as u64));
                println!("[{}] pretrain data {} ({} positions, {:.1}s) for set {}", cfgv.name, instances[k].id, s.len(), t.elapsed().as_secs_f32(), name);
                pretrain_data.insert(k, s);
            }
        }
    }
    let mut target_data: BTreeMap<usize, (Vec<Sample>, Vec<Sample>)> = BTreeMap::new();
    for &k in &target_ids {
        let tr = generate_samples(&instances[k].game, instances[k].game_idx, instances[k].family_idx, &data_spec(&cfgv.data, cfgv.data.target_train_games, 5000 + k as u64));
        let va = generate_samples(&instances[k].game, instances[k].game_idx, instances[k].family_idx, &data_spec(&cfgv.data, cfgv.data.target_val_games, 9000 + k as u64));
        println!("[{}] target {}: {} train / {} val positions", cfgv.name, instances[k].id, tr.len(), va.len());
        target_data.insert(k, (tr, va));
    }

    let dev0 = devices[0].clone();
    let mut rows: Vec<ResultRow> = vec![];

    for &seed in &cfgv.seeds {
        // ---- pretraining (once per set per seed) ---------------------------
        let mut pretrained: BTreeMap<String, (OmniAstra<B::InnerBackend>, Counters)> = BTreeMap::new();
        let used_sets: Vec<String> = cfgv.arms.iter().filter_map(|a| a.pretrain.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        for set in used_sets {
            let groups = cfgv.pretrain_sets.get(&set).ok_or_else(|| anyhow!("unknown pretrain set {set}"))?;
            let pools: Vec<Pool> = (0..instances.len())
                .filter(|k| groups.contains(&instances[*k].group))
                .map(|k| Pool { game_idx: instances[k].game_idx, family_idx: instances[k].family_idx, samples: pretrain_data[&k].clone() })
                .collect();
            B::seed(&dev0, seed * 7 + 1);
            let model = OmniAstra::<B>::new(&mcfg, &dev0);
            let tcfg = TrainCfg { steps: cfgv.pretrain.steps, batch: cfgv.pretrain.batch, lr: cfgv.pretrain.lr, seed, ..Default::default() };
            let mut tr = Trainer::<B>::new(model, mcfg.clone(), tcfg.clone(), devices.clone());
            let mut sampler = MixSampler::new(&pools, &cfgv.pretrain.mix, seed);
            let t = Instant::now();
            for step in 0..tcfg.steps {
                let b = sampler.next_batch(tcfg.batch);
                let st = tr.train_step(&b);
                if step % 50 == 0 || step + 1 == tcfg.steps {
                    println!("[{}] pretrain {set} seed {seed} step {step}/{} loss {:.3} (pol {:.3} val {:.3}) lr {:.2e} {:.0}s", cfgv.name, tcfg.steps, st.loss, st.policy, st.value, st.lr, t.elapsed().as_secs_f32());
                }
            }
            save_weights(&out_dir.join(format!("pretrain/{set}-seed{seed}")), &tr.valid(), &mcfg, serde_json::json!({"set": set, "seed": seed, "groups": groups, "counters": tr.counters}))?;
            pretrained.insert(set, (tr.valid(), tr.counters.clone()));
        }

        // ---- fine-tuning per target / arm / budget -------------------------
        for &tk in &target_ids {
            let target = &instances[tk];
            let (train_all, val_all) = &target_data[&tk];
            let mut order: Vec<usize> = (0..train_all.len()).collect();
            Rng::new(seed ^ 0xF17E).shuffle(&mut order);
            let val_refs: Vec<&Sample> = val_all.iter().take(cfgv.finetune.val_positions).collect();
            // fixed evaluation opponents/seeds across arms
            let opps: Vec<(String, AgentFactory)> = cfgv.eval.opponents.iter().map(|n| Ok((n.clone(), opponent_factory(&target.game, n)?))).collect::<Result<_>>()?;

            for arm in &cfgv.arms {
                for &budget in &cfgv.finetune.budgets {
                    let n = budget.min(train_all.len());
                    let train: Vec<&Sample> = order[..n].iter().map(|i| &train_all[*i]).collect();
                    let t_run = Instant::now();
                    // identical fresh initialisation for every arm of this seed
                    B::seed(&dev0, seed * 7 + 2);
                    let fresh = OmniAstra::<B>::new(&mcfg, &dev0);
                    let model: OmniAstra<B> = match (&arm.pretrain, &arm.reuse) {
                        (None, _) => fresh,
                        (Some(set), reuse) => {
                            let (src_inner, _) = &pretrained[set];
                            // `train()` yields a model that does not track gradients: re-enable.
                            let src: OmniAstra<B> = src_inner.clone().train::<B>().enable_grad();
                            match reuse {
                                Reuse::Full => src,
                                Reuse::Backbone => transplant_backbone(&src, fresh, false),
                                Reuse::BackboneAndHeads => transplant_backbone(&src, fresh, true),
                            }
                        }
                    };
                    let model = match arm.freeze {
                        FreezeMode::None => model,
                        FreezeMode::Backbone => model.freeze_backbone(),
                        FreezeMode::AllButAdapter => model.freeze_all_but_adapter(),
                    };
                    let steps = ((cfgv.finetune.steps as f32) * arm.ft_steps_mult).round() as usize;
                    let tcfg = TrainCfg { steps, batch: cfgv.finetune.batch, lr: cfgv.finetune.lr, seed, ..Default::default() };
                    let mut tr = Trainer::<B>::new(model, mcfg.clone(), tcfg.clone(), devices.clone());
                    let pool = vec![Pool { game_idx: target.game_idx, family_idx: target.family_idx, samples: train.iter().map(|s| (*s).clone()).collect() }];
                    let mut sampler = MixSampler::new(&pool, &MixSpec::UniformByGame, seed ^ 0xABCD);
                    let mut curve = vec![];
                    let mut best: Option<(f32, usize, OmniAstra<B::InnerBackend>, GroupMetrics)> = None;
                    let eval_every = cfgv.finetune.eval_every.max(1);
                    let idev = dev0.clone();
                    let check = |tr: &Trainer<B>, step: usize, curve: &mut Vec<(usize, f32, f32)>, best: &mut Option<(f32, usize, OmniAstra<B::InnerBackend>, GroupMetrics)>| {
                        let m = tr.valid();
                        let ev = evaluate_dataset::<B::InnerBackend>(&m, &mcfg, &val_refs, cfgv.eval.batch, &idev);
                        curve.push((step, ev.overall.policy_ce, ev.overall.top1));
                        if best.as_ref().map(|b| ev.overall.policy_ce < b.0).unwrap_or(true) {
                            *best = Some((ev.overall.policy_ce, step, m, ev.overall.clone()));
                        }
                    };
                    check(&tr, 0, &mut curve, &mut best);
                    for step in 1..=steps {
                        let b = sampler.next_batch(tcfg.batch);
                        tr.train_step(&b);
                        if step % eval_every == 0 || step == steps {
                            check(&tr, step, &mut curve, &mut best);
                        }
                    }
                    let (val_ce, best_step, best_model, bm) = best.unwrap();
                    // strength of the selected checkpoint
                    let net = NetHandle::<B::InnerBackend> { model: Arc::new(best_model), cfg: Arc::new(mcfg.clone()), dev: idev.clone(), game_idx: target.game_idx, max_history: cfgv.data.max_history };
                    let mut strength = vec![];
                    for (oi, (name, fac)) in opps.iter().enumerate() {
                        let netc = net.clone();
                        let a: AgentFactory = Arc::new(move || Box::new(NeuralAgent::new(netc.clone(), "net")));
                        let r = play_match(&target.game, &a, fac, cfgv.eval.games, 777_000 + oi as u64 * 100_000);
                        strength.push(StrengthRow { opponent: name.clone(), score: r.score, ci_lo: r.ci_lo, ci_hi: r.ci_hi, elo: r.elo });
                    }
                    let mean_strength = strength.iter().map(|s| s.score).sum::<f32>() / strength.len().max(1) as f32;
                    let row = ResultRow {
                        experiment: cfgv.name.clone(),
                        seed,
                        target: target.id.clone(),
                        arm: arm.name.clone(),
                        budget_positions: n,
                        pretrain_set: arm.pretrain.clone(),
                        pretrain: arm.pretrain.as_ref().map(|s| pretrained[s].1.clone()),
                        finetune: tr.counters.clone(),
                        ft_steps: steps,
                        best_step,
                        val_ce,
                        val_top1: bm.top1,
                        val_value_mse: bm.value_mse,
                        strength,
                        mean_strength_score: mean_strength,
                        curve,
                        params: tr.model.num_params(),
                    };
                    println!(
                        "[{}] seed {seed} target {} arm {:<24} N={:<6} val CE {:.3} top1 {:.3} strength {:.3} ({:.0}s)",
                        cfgv.name, target.id, arm.name, n, row.val_ce, row.val_top1, row.mean_strength_score, t_run.elapsed().as_secs_f32()
                    );
                    log.write(&row);
                    rows.push(row);
                }
            }
        }
    }
    println!("[{}] done in {:.1} min", cfgv.name, t_start.elapsed().as_secs_f32() / 60.0);
    write_summary(&out_dir, cfgv, &rows)?;
    Ok(rows)
}

/// Mean +- 95% CI (normal approx.) across seeds/targets.
fn mean_ci(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    if v.is_empty() {
        return (f32::NAN, 0.0);
    }
    let m = v.iter().sum::<f32>() / n;
    if v.len() < 2 {
        return (m, 0.0);
    }
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (n - 1.0);
    (m, 1.96 * (var / n).sqrt())
}

/// Effective-data multiplier: how many scratch positions are needed to match
/// `ce` (log-linear interpolation over the scratch learning curve).
fn equivalent_scratch_budget(scratch: &[(f32, f32)], ce: f32) -> Option<f32> {
    // scratch: (budget, mean val CE), ascending budget, CE roughly decreasing.
    if scratch.len() < 2 {
        return None;
    }
    if ce >= scratch[0].1 {
        return Some(scratch[0].0 * (scratch[0].1 / ce.max(1e-6)).min(1.0));
    }
    for w in scratch.windows(2) {
        let ((b0, c0), (b1, c1)) = (w[0], w[1]);
        if ce <= c0 && ce >= c1 && (c0 - c1).abs() > 1e-6 {
            let f = (c0 - ce) / (c0 - c1);
            return Some((b0.ln() + f * (b1.ln() - b0.ln())).exp());
        }
    }
    // better than the best scratch run: report lower bound
    Some(scratch.last().unwrap().0)
}

pub fn write_summary(dir: &Path, c: &TransferConfig, rows: &[ResultRow]) -> Result<()> {
    let mut s = String::new();
    s.push_str(&format!("# Transfer experiment `{}`\n\nmodel `{}`; seeds {:?}; target group `{}`; budgets {:?}.\n\n", c.name, c.model, c.seeds, c.target_group, c.finetune.budgets));
    s.push_str("Metrics are means over seeds x targets (+- 95% CI over those runs). `val CE` is policy cross-entropy against the expert (MCTS) policy on held-out positions of the target game; strength is mean score of the raw greedy policy against the opponent panel. `fwd FLOPs` = total estimated training FLOPs (pretraining + fine-tuning).\n\n");
    s.push_str("| arm | N | val CE | top1 | strength | data x vs scratch | ft updates | pretrain updates | total GFLOPs |\n|---|---|---|---|---|---|---|---|---|\n");
    let arms: Vec<String> = c.arms.iter().map(|a| a.name.clone()).collect();
    // scratch reference curve
    let scratch_name = c.arms.iter().find(|a| a.pretrain.is_none() && a.ft_steps_mult == 1.0).map(|a| a.name.clone());
    let mut scratch_curve: Vec<(f32, f32)> = vec![];
    if let Some(sn) = &scratch_name {
        for &b in &c.finetune.budgets {
            let v: Vec<f32> = rows.iter().filter(|r| &r.arm == sn && r.budget_positions == b.min(r.budget_positions.max(b))).map(|r| r.val_ce).collect();
            let v2: Vec<f32> = rows.iter().filter(|r| &r.arm == sn && budget_matches(r.budget_positions, b)).map(|r| r.val_ce).collect();
            let _ = v;
            if !v2.is_empty() {
                scratch_curve.push((b as f32, mean_ci(&v2).0));
            }
        }
    }
    for arm in &arms {
        for &b in &c.finetune.budgets {
            let rs: Vec<&ResultRow> = rows.iter().filter(|r| &r.arm == arm && budget_matches(r.budget_positions, b)).collect();
            if rs.is_empty() {
                continue;
            }
            let ce: Vec<f32> = rs.iter().map(|r| r.val_ce).collect();
            let t1: Vec<f32> = rs.iter().map(|r| r.val_top1).collect();
            let st: Vec<f32> = rs.iter().map(|r| r.mean_strength_score).collect();
            let (cem, cec) = mean_ci(&ce);
            let (t1m, t1c) = mean_ci(&t1);
            let (stm, stc) = mean_ci(&st);
            let mult = equivalent_scratch_budget(&scratch_curve, cem).map(|e| format!("{:.2}", e / b as f32)).unwrap_or("-".into());
            let ftu = rs.iter().map(|r| r.finetune.optimizer_updates as f64).sum::<f64>() / rs.len() as f64;
            let ptu = rs.iter().map(|r| r.pretrain.as_ref().map(|p| p.optimizer_updates as f64).unwrap_or(0.0)).sum::<f64>() / rs.len() as f64;
            let fl = rs.iter().map(|r| (r.finetune.flops_est + r.pretrain.as_ref().map(|p| p.flops_est).unwrap_or(0.0)) / 1e9).sum::<f64>() / rs.len() as f64;
            s.push_str(&format!("| {arm} | {b} | {cem:.3} +- {cec:.3} | {t1m:.3} +- {t1c:.3} | {stm:.3} +- {stc:.3} | {mult} | {ftu:.0} | {ptu:.0} | {fl:.0} |\n"));
        }
    }
    std::fs::write(dir.join("summary.md"), s)?;
    Ok(())
}

fn budget_matches(actual: usize, requested: usize) -> bool {
    actual == requested || actual < requested // clipped by pool size
}
