//! Synchronous data-parallel trainer.
//!
//! One optimizer step = (optionally several) shards, each on its own device:
//! replicas run forward/backward with *global* loss denominators, gradients
//! are moved to the primary device and summed (equal to the full-batch
//! gradient), then a single optimizer update is applied. Single-device
//! training is the k = 1 case of the same code path.

use burn::module::{AutodiffModule, Module};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use omnia_model::*;
use omnia_observation::batch::{BatchOpts, HostBatch, Sample};
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainCfg {
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub warmup: usize,
    /// Final lr as a fraction of `lr` (cosine decay).
    pub min_lr_frac: f64,
    pub weight_decay: f32,
    pub value_weight: f64,
    pub grad_clip: f32,
    pub seed: u64,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg { steps: 500, batch: 32, lr: 1e-3, warmup: 20, min_lr_frac: 0.1, weight_decay: 0.01, value_weight: 1.0, grad_clip: 1.0, seed: 1 }
    }
}

impl TrainCfg {
    pub fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup {
            return self.lr * (step + 1) as f64 / self.warmup.max(1) as f64;
        }
        let t = (step - self.warmup) as f64 / (self.steps.saturating_sub(self.warmup)).max(1) as f64;
        let c = 0.5 * (1.0 + (std::f64::consts::PI * t.min(1.0)).cos());
        self.lr * (self.min_lr_frac + (1.0 - self.min_lr_frac) * c)
    }
}

/// Compute accounting (pretraining cost must be reported alongside transfer).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Counters {
    pub optimizer_updates: u64,
    pub positions: u64,
    pub tokens: u64,
    /// Rough training FLOPs: 6 * (non-embedding params) * tokens.
    pub flops_est: f64,
    pub wall_secs: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StepStats {
    pub loss: f32,
    pub policy: f32,
    pub value: f32,
    pub lr: f64,
}

pub struct Trainer<B: AutodiffBackend> {
    pub model: OmniAstra<B>,
    pub opt: OptimizerAdaptor<AdamW, OmniAstra<B>, B>,
    pub cfg: TrainCfg,
    pub model_cfg: ModelConfig,
    pub counters: Counters,
    pub devices: Vec<B::Device>,
    pub step_idx: usize,
    flops_per_token: f64,
}

pub fn make_optimizer<B: AutodiffBackend>(cfg: &TrainCfg) -> OptimizerAdaptor<AdamW, OmniAstra<B>, B> {
    AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay)
        .with_grad_clipping(Some(burn::grad_clipping::GradientClippingConfig::Norm(cfg.grad_clip)))
        .init::<B, OmniAstra<B>>()
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(model: OmniAstra<B>, model_cfg: ModelConfig, cfg: TrainCfg, devices: Vec<B::Device>) -> Self {
        use burn::module::Module;
        // Parameters excluding embedding tables dominate compute; approximate with all params
        // minus the id/game tables.
        let embed_params = model_cfg.cat_vocab * model_cfg.d_model * omnia_observation::NCAT + model_cfg.n_games * model_cfg.d_model;
        let compute_params = model.num_params().saturating_sub(embed_params) as f64;
        let opt = make_optimizer::<B>(&cfg);
        Trainer { model, opt, cfg, model_cfg, counters: Counters::default(), devices, step_idx: 0, flops_per_token: 6.0 * compute_params }
    }

    fn batch_opts(&self) -> BatchOpts {
        BatchOpts { n_reg: self.model_cfg.n_reg, mask_ids: !self.model_cfg.use_ids }
    }

    /// Gradients of the global mean loss over `samples`, computed over `k`
    /// shards (k = number of devices). Returns (grads, loss, policy, value).
    pub fn grads(&self, samples: &[&Sample], k: usize) -> (GradientsParams, f32, f32, f32) {
        let k = k.max(1).min(samples.len());
        let n_total = samples.len() as f64;
        let v_total: f64 = samples.iter().map(|s| s.obs.num_players as f64).sum();
        let chunk = samples.len().div_ceil(k);
        let shards: Vec<&[&Sample]> = samples.chunks(chunk).collect();
        let opts = self.batch_opts();
        let vw = self.cfg.value_weight;
        let primary = self.devices[0].clone();
        let run_shard = |shard: &[&Sample], dev: B::Device, model: &OmniAstra<B>, primary: &B::Device| {
            let host = HostBatch::build(shard, opts);
            let replica = if k == 1 { model.clone() } else { model.clone().fork(&dev) };
            let bt = Batch::<B>::from_host(&host, &dev);
            let out = replica.forward(&bt);
            let l = replica.losses_with(&bt, &out, vw, n_total, v_total);
            let (t, p, v): (f32, f32, f32) = (l.total.clone().into_scalar().elem::<f32>(), l.policy.into_scalar().elem::<f32>(), l.value.into_scalar().elem::<f32>());
            let mut grads = GradientsParams::from_grads(l.total.backward(), &replica);
            if k > 1 {
                grads = grads.to_device(primary, &replica);
            }
            (grads, t, p, v)
        };
        let results: Vec<(GradientsParams, f32, f32, f32)> = if shards.len() == 1 {
            // One shard: run on the calling thread. Backends such as CUDA keep a
            // stream and memory pool per thread, so a fresh thread every step leaks
            // a pool per step.
            vec![run_shard(shards[0], self.devices[0].clone(), &self.model, &primary)]
        } else {
            std::thread::scope(|sc| {
                let run_shard = &run_shard;
                let handles: Vec<_> = shards
                    .iter()
                    .enumerate()
                    .map(|(i, shard)| {
                        let dev = self.devices[i % self.devices.len()].clone();
                        let model = &self.model;
                        let primary = &primary;
                        sc.spawn(move || run_shard(shard, dev, model, primary))
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("shard thread")).collect()
            })
        };
        let mut acc = GradientsAccumulator::new();
        let (mut t, mut p, mut v) = (0.0, 0.0, 0.0);
        for (g, lt, lp, lv) in results {
            acc.accumulate(&self.model, g);
            t += lt;
            p += lp;
            v += lv;
        }
        (acc.grads(), t, p, v)
    }

    pub fn train_step(&mut self, samples: &[&Sample]) -> StepStats {
        let t0 = Instant::now();
        let lr = self.cfg.lr_at(self.step_idx);
        let k = self.devices.len();
        let (grads, loss, policy, value) = self.grads(samples, k);
        // Module clones are shallow (shared tensors); the optimizer consumes one handle.
        self.model = self.opt.step(lr, self.model.clone(), grads);
        self.step_idx += 1;
        let tokens: u64 = samples.iter().map(|s| s.obs.tokens.len() as u64).sum();
        self.counters.optimizer_updates += 1;
        self.counters.positions += samples.len() as u64;
        self.counters.tokens += tokens;
        self.counters.flops_est += self.flops_per_token * tokens as f64;
        self.counters.wall_secs += t0.elapsed().as_secs_f64();
        StepStats { loss, policy, value, lr }
    }

    /// Inference-only copy of the model on the primary device.
    pub fn valid(&self) -> OmniAstra<B::InnerBackend> {
        self.model.valid()
    }
}
