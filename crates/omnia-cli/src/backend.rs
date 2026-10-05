//! Backend selection (compile-time features) and the generic commands that
//! need a tensor backend.

use anyhow::{anyhow, Result};
use burn::module::Module;
use burn::tensor::backend::Backend;
use omnia_dsl::MAX_PLAYERS;
use omnia_engine::*;
use omnia_gen::*;
use omnia_model::*;
use omnia_observation::batch::*;
use omnia_observation::*;
use omnia_training::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "cuda")]
pub type Inner = burn::backend::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
pub type Inner = burn::backend::Wgpu;
#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "flex"))]
pub type Inner = burn::backend::Flex;
#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), not(feature = "flex")))]
pub type Inner = burn::backend::NdArray<f32>;
pub type AB = burn::backend::Autodiff<Inner>;

pub fn backend_name() -> &'static str {
    std::any::type_name::<Inner>()
}

/// `n` devices for data-parallel training.
pub fn devices(n: usize) -> Vec<<AB as burn::tensor::backend::BackendTypes>::Device> {
    #[cfg(feature = "cuda")]
    {
        // NOTE: multi-GPU path is written against Burn's CUDA backend but was not
        // exercised in the CPU-only development container.
        return (0..n).map(burn::backend::cuda::CudaDevice::new).collect();
    }
    #[cfg(not(feature = "cuda"))]
    {
        (0..n).map(|_| Default::default()).collect()
    }
}

pub fn model_inspect(size: &str) -> Result<()> {
    let cfg = ModelConfig::by_name(size).ok_or_else(|| anyhow!("unknown size {size} (nano | tiny | small | medium)"))?;
    let m = OmniAstra::<Inner>::new(&cfg, &Default::default());
    println!("{size}: {} parameters (backend {})", m.num_params(), backend_name());
    println!("{cfg}");
    Ok(())
}

fn probe_samples(g: &Arc<Game>, n: usize, max_history: usize) -> Vec<Sample> {
    let tk = Tokenizer::new(TokenizerConfig { max_history });
    let mut out = vec![];
    let mut st = State::new(g, 1);
    let mut rng = Rng::new(1);
    while out.len() < n {
        if st.is_terminal() {
            st = State::new(g, rng.next_u64());
        }
        let p = st.decision_maker().unwrap();
        let obs = tk.observe(&mut st, p);
        let na = obs.n_actions();
        let mut pi = vec![0.0; na];
        let k = rng.below(na as u64) as usize;
        pi[k] = 1.0;
        let mut v = [0.0; MAX_PLAYERS];
        v[0] = if rng.f32() < 0.5 { 1.0 } else { -1.0 };
        v[1] = -v[0];
        out.push(Sample { obs, pi, value: v, game_idx: 0, family_idx: 0 });
        st.apply(k);
    }
    out
}

pub fn model_benchmark(g: &Arc<Game>, size: &str, batch: usize) -> Result<()> {
    let cfg = ModelConfig::by_name(size).ok_or_else(|| anyhow!("unknown size {size}"))?;
    let samples = probe_samples(g, batch, 32);
    let refs: Vec<&Sample> = samples.iter().collect();
    let t = Instant::now();
    let h = HostBatch::build(&refs, BatchOpts { n_reg: cfg.n_reg, mask_ids: false });
    println!("host batching ({batch} samples): {:.1} ms; padded state tokens {} actions {}", t.elapsed().as_secs_f64() * 1000.0, h.ns, h.na);
    let dev = devices(1).remove(0);
    let model = OmniAstra::<AB>::new(&cfg, &dev);
    println!("model {size}: {} params, backend {}", model.num_params(), backend_name());
    let mut tr = Trainer::<AB>::new(model, cfg.clone(), TrainCfg { batch, ..Default::default() }, vec![dev.clone()]);
    for i in 0..3 {
        let t = Instant::now();
        let s = tr.train_step(&refs);
        println!("train step {i}: {:.0} ms ({:.0} samples/s) loss {:.3}", t.elapsed().as_secs_f64() * 1000.0, batch as f64 / t.elapsed().as_secs_f64(), s.loss);
    }
    let inner = tr.valid();
    let t = Instant::now();
    let b = Batch::<Inner>::from_host(&h, &dev);
    let _ = inner.forward(&b).logits.into_data();
    println!("inference batch of {batch}: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let one = HostBatch::build(&refs[..1], BatchOpts { n_reg: cfg.n_reg, mask_ids: false });
    let t = Instant::now();
    for _ in 0..20 {
        let b = Batch::<Inner>::from_host(&one, &dev);
        let _ = inner.forward(&b).logits.into_data();
    }
    println!("inference single observation: {:.1} ms", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
    Ok(())
}

pub fn run_transfer(config: &Path, games_dir: &Path, n_devices: usize) -> Result<()> {
    let cfg: omnia_eval::transfer::TransferConfig = serde_json::from_str(&std::fs::read_to_string(config)?)?;
    println!("backend {} on {} device(s)", backend_name(), n_devices);
    omnia_eval::transfer::run::<AB>(&cfg, games_dir, devices(n_devices))?;
    Ok(())
}

pub fn run_train(config: &Path, games_dir: &Path, n_devices: usize) -> Result<()> {
    let job: omnia_eval::jobs::TrainJob = serde_json::from_str(&std::fs::read_to_string(config)?)?;
    println!("backend {} on {} device(s)", backend_name(), n_devices);
    omnia_eval::jobs::train_job::<AB>(&job, games_dir, devices(n_devices))
}

pub fn run_selfplay(config: &Path, games_dir: &Path, n_devices: usize) -> Result<()> {
    let job: omnia_eval::jobs::SelfPlayJob = serde_json::from_str(&std::fs::read_to_string(config)?)?;
    println!("backend {} on {} device(s)", backend_name(), n_devices);
    omnia_eval::jobs::selfplay_job::<AB>(&job, games_dir, devices(n_devices))
}

pub fn run_diagnose(game: Arc<Game>, model: &str, sims: u32, games: usize, steps: usize, batch: usize, lr: f64, target_temp: f32) -> Result<()> {
    omnia_eval::jobs::diagnose::<AB>(game, model, sims, games, steps, batch, lr, target_temp, devices(1))
}

pub fn run_eval(checkpoint: &Path, model: &str, game: Arc<Game>, game_idx: u32, opponents: &[String], games: u32) -> Result<()> {
    omnia_eval::jobs::eval_checkpoint::<AB>(checkpoint, model, &None, game, game_idx, opponents, games, devices(1).remove(0))?;
    Ok(())
}

/// Single- vs multi-shard gradient parity: one SGD step computed from the
/// full batch and from `shards` accumulated shards must give the same model.
pub fn parity_check(shards: usize) -> Result<()> {
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    let (_, def, _, _) = generate_valid(&GenSpec::new(Family::Stack, 3), &ValidationConfig { check_skill: false, ..Default::default() }, 30).map_err(|e| anyhow!(e))?;
    let g = Game::new(def)?;
    let samples = probe_samples(&g, 12, 16);
    let refs: Vec<&Sample> = samples.iter().collect();
    let cfg = ModelConfig::nano();
    let devs = devices(shards);
    AB::seed(&devs[0], 42);
    let base = OmniAstra::<AB>::new(&cfg, &devs[0]);
    let make = |k: usize| -> OmniAstra<AB> {
        let tr = Trainer::<AB>::new(base.clone(), cfg.clone(), TrainCfg::default(), devs.clone());
        let (grads, _, _, _) = tr.grads(&refs, k);
        let mut sgd = SgdConfig::new().init::<AB, OmniAstra<AB>>();
        sgd.step(0.05, base.clone(), grads)
    };
    let single = make(1);
    let multi = make(shards);
    let probe = HostBatch::build(&refs, BatchOpts { n_reg: cfg.n_reg, mask_ids: false });
    let bt = Batch::<AB>::from_host(&probe, &devs[0]);
    let (a, b) = (single.forward(&bt), multi.forward(&bt));
    let da: Vec<f32> = a.logits.into_data().to_vec().unwrap();
    let db: Vec<f32> = b.logits.into_data().to_vec().unwrap();
    let va: Vec<f32> = a.value.into_data().to_vec().unwrap();
    let vb: Vec<f32> = b.value.into_data().to_vec().unwrap();
    let md = da.iter().zip(&db).filter(|(x, _)| **x > -1e8).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let mv = va.iter().zip(&vb).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    println!("parity: max |logit diff| {md:.2e}, max |value diff| {mv:.2e} (1 vs {shards} shards)");
    let _ = GradientsParams::new();
    if md < 1e-4 && mv < 1e-4 {
        println!("PASS");
        Ok(())
    } else {
        Err(anyhow!("parity check FAILED"))
    }
}
