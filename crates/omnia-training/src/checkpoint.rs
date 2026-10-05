//! Checkpoints. Two explicitly different operations:
//!
//! * [`resume`]: model + optimizer state + step counters (continue training);
//! * [`load_weights`]: weights only, e.g. to initialise a transfer experiment
//!   from a pretrained model with a *fresh* optimizer.

use crate::trainer::*;
use burn::module::Module;
use burn::optim::Optimizer;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use omnia_model::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

type Rec = NamedMpkFileRecorder<FullPrecisionSettings>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub step: usize,
    pub counters: Counters,
    pub model_cfg: String, // serde_json of ModelConfig
    pub train_cfg: TrainCfg,
    /// Free-form experiment metadata: game-set identifiers/hashes, mixture, git commit, seeds.
    pub notes: serde_json::Value,
}

pub fn save<B: AutodiffBackend>(dir: &Path, t: &Trainer<B>, notes: serde_json::Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    let rec = Rec::new();
    t.model.clone().save_file(dir.join("model"), &rec).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    rec.record(t.opt.to_record(), dir.join("optim")).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let meta = CheckpointMeta { step: t.step_idx, counters: t.counters.clone(), model_cfg: serde_json::to_string(&t.model_cfg)?, train_cfg: t.cfg.clone(), notes };
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// Save just the model weights (+ meta) from an inference module.
pub fn save_weights<B: Backend>(dir: &Path, model: &OmniAstra<B>, model_cfg: &ModelConfig, notes: serde_json::Value) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir)?;
    model.clone().save_file(dir.join("model"), &Rec::new()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let meta = serde_json::json!({"model_cfg": serde_json::to_string(model_cfg)?, "notes": notes});
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    Ok(())
}

/// Weights-only load into a freshly constructed model of the same shape.
pub fn load_weights<B: Backend>(dir: &Path, cfg: &ModelConfig, dev: &B::Device) -> anyhow::Result<OmniAstra<B>> {
    let m = OmniAstra::<B>::new(cfg, dev);
    m.load_file(dir.join("model"), &Rec::new(), dev).map_err(|e| anyhow::anyhow!("{e:?}"))
}

/// Resume training: restores weights, optimizer state and counters.
pub fn resume<B: AutodiffBackend>(dir: &Path, cfg: &ModelConfig, tcfg: TrainCfg, devices: Vec<B::Device>) -> anyhow::Result<Trainer<B>> {
    let dev = devices[0].clone();
    let meta: CheckpointMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    let model = load_weights::<B>(dir, cfg, &dev)?;
    let mut t = Trainer::new(model, cfg.clone(), tcfg, devices);
    let rec = Rec::new();
    let record = rec.load(dir.join("optim"), &dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    t.opt = t.opt.load_record(record);
    t.step_idx = meta.step;
    t.counters = meta.counters;
    Ok(t)
}

/// Copy the shared backbone (registers, relational blocks, decoder, value/policy
/// heads optionally) from `src` into `dst`, leaving `dst`'s input adapter
/// (id/slot embeddings, game embedding) as it was.
pub fn transplant_backbone<B: Backend>(src: &OmniAstra<B>, mut dst: OmniAstra<B>, include_heads: bool) -> OmniAstra<B> {
    dst.reg = src.reg.clone();
    dst.blocks = src.blocks.clone();
    dst.final_norm = src.final_norm.clone();
    dst.link_proj = src.link_proj.clone();
    dst.dec = src.dec.clone();
    dst.dec_norm = src.dec_norm.clone();
    if include_heads {
        dst.policy_out = src.policy_out.clone();
        dst.value_in = src.value_in.clone();
        dst.value_out = src.value_out.clone();
    }
    dst
}
