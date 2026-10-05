//! Evaluation metrics on labelled datasets + JSONL run logging.

use burn::tensor::backend::Backend;
use omnia_dsl::MAX_PLAYERS;
use omnia_model::*;
use omnia_observation::batch::{BatchOpts, HostBatch, Sample};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GroupMetrics {
    pub n: usize,
    pub policy_ce: f32,
    pub top1: f32,
    pub value_mse: f32,
    pub value_sign_acc: f32,
    pub entropy: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub overall: GroupMetrics,
    pub per_game: BTreeMap<u32, GroupMetrics>,
    pub per_family: BTreeMap<u32, GroupMetrics>,
}

#[derive(Default, Clone)]
struct Acc {
    n: usize,
    ce: f64,
    top1: f64,
    vmse: f64,
    vsign: f64,
    vsign_n: f64,
    ent: f64,
}

impl Acc {
    fn finish(&self) -> GroupMetrics {
        let n = self.n.max(1) as f64;
        GroupMetrics {
            n: self.n,
            policy_ce: (self.ce / n) as f32,
            top1: (self.top1 / n) as f32,
            value_mse: (self.vmse / n) as f32,
            value_sign_acc: (self.vsign / self.vsign_n.max(1.0)) as f32,
            entropy: (self.ent / n) as f32,
        }
    }
}

/// Evaluate `model` on `samples` (inference only).
pub fn evaluate_dataset<B: Backend>(model: &OmniAstra<B>, cfg: &ModelConfig, samples: &[&Sample], batch: usize, dev: &B::Device) -> EvalMetrics {
    let mut overall = Acc::default();
    let mut games: BTreeMap<u32, Acc> = BTreeMap::new();
    let mut fams: BTreeMap<u32, Acc> = BTreeMap::new();
    for chunk in samples.chunks(batch.max(1)) {
        let h = HostBatch::build(chunk, BatchOpts { n_reg: cfg.n_reg, mask_ids: !cfg.use_ids });
        let bt = Batch::<B>::from_host(&h, dev);
        let out = model.forward(&bt);
        let logits: Vec<f32> = out.logits.into_data().to_vec().unwrap();
        let values: Vec<f32> = out.value.into_data().to_vec().unwrap();
        for (i, s) in chunk.iter().enumerate() {
            let na = s.obs.n_actions();
            let l = &logits[i * h.na..i * h.na + na];
            let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + l.iter().map(|x| (x - mx).exp()).sum::<f32>().ln();
            let mut ce = 0.0f64;
            let mut ent = 0.0f64;
            for (a, x) in l.iter().enumerate() {
                let lp = (x - lse) as f64;
                ce -= s.pi[a] as f64 * lp;
                ent -= lp.exp() * lp;
            }
            let arg = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0);
            let top1 = (arg(l) == arg(&s.pi)) as u8 as f64;
            let n = s.obs.num_players as usize;
            let mut vm = 0.0;
            for r in 0..n {
                let d = (values[i * MAX_PLAYERS + r] - s.value[r]) as f64;
                vm += d * d;
            }
            vm /= n as f64;
            let sign_ok = if s.value[0].abs() > 0.5 { ((values[i * MAX_PLAYERS] > 0.0) == (s.value[0] > 0.0)) as u8 as f64 } else { 0.0 };
            let sign_n = (s.value[0].abs() > 0.5) as u8 as f64;
            for acc in [&mut overall, games.entry(s.game_idx).or_default(), fams.entry(s.family_idx).or_default()] {
                acc.n += 1;
                acc.ce += ce;
                acc.top1 += top1;
                acc.vmse += vm;
                acc.vsign += sign_ok;
                acc.vsign_n += sign_n;
                acc.ent += ent;
            }
        }
    }
    EvalMetrics {
        overall: overall.finish(),
        per_game: games.into_iter().map(|(k, v)| (k, v.finish())).collect(),
        per_family: fams.into_iter().map(|(k, v)| (k, v.finish())).collect(),
    }
}

/// Append-only JSON-lines run log.
pub struct RunLog {
    file: Option<std::fs::File>,
}

impl RunLog {
    pub fn create(path: &std::path::Path) -> RunLog {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).ok();
        }
        RunLog { file: std::fs::OpenOptions::new().create(true).append(true).open(path).ok() }
    }
    pub fn none() -> RunLog {
        RunLog { file: None }
    }
    pub fn write<T: Serialize>(&mut self, v: &T) {
        if let Some(f) = self.file.as_mut() {
            let _ = writeln!(f, "{}", serde_json::to_string(v).unwrap());
        }
    }
}
