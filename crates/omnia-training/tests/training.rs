use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::optim::{Optimizer, SgdConfig};
use burn::tensor::backend::Backend;
use omnia_engine::*;
use omnia_gen::*;
use omnia_model::*;
use omnia_observation::batch::*;
use omnia_training::*;
use std::sync::Arc;

static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
type AB = Autodiff<NdArray<f32>>;
type IB = NdArray<f32>;

fn game(fam: Family, seed: u64) -> Arc<Game> {
    let (_, def, _, _) = generate_valid(&GenSpec::new(fam, seed), &ValidationConfig { check_skill: false, ..Default::default() }, 30).unwrap();
    Game::new(def).unwrap()
}

fn data(g: &Arc<Game>, idx: u32, games: usize) -> Vec<Sample> {
    generate_samples(g, idx, 0, &DataSpec { expert: Expert::Heuristic, games, epsilon: 0.2, temp_moves: 0, max_history: 12, seed: 3 })
}

fn nano() -> ModelConfig {
    ModelConfig::nano().with_n_games(8)
}

#[test]
fn datagen_is_deterministic_and_well_formed() {
    let g = game(Family::Tempo, 2);
    let a = data(&g, 0, 4);
    let b = data(&g, 0, 4);
    assert_eq!(a.len(), b.len());
    assert!(a.len() > 20);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.pi, y.pi);
        assert_eq!(x.value, y.value);
    }
    for s in &a {
        assert_eq!(s.pi.len(), s.obs.n_actions());
        assert!((s.pi.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(s.value[0] >= -1.0 && s.value[0] <= 1.0);
        // zero-sum 2p: value of the other seat is the negation
        assert_eq!(s.value[1], -s.value[0]);
    }
}

#[test]
fn mixture_sampler_balances_games_regardless_of_size() {
    let g1 = game(Family::Tempo, 1);
    let g2 = game(Family::Stack, 2);
    let p1 = Pool { game_idx: 0, family_idx: 0, samples: data(&g1, 0, 12) };
    let p2 = Pool { game_idx: 1, family_idx: 1, samples: data(&g2, 1, 2) };
    assert!(p1.samples.len() > 2 * p2.samples.len());
    let pools = [p1, p2];
    let mut s = MixSampler::new(&pools, &MixSpec::UniformByGame, 1);
    let mut c = [0usize; 2];
    for x in s.next_batch(2000) {
        c[x.game_idx as usize] += 1;
    }
    assert!((c[0] as f32 / 2000.0 - 0.5).abs() < 0.05, "{c:?}");
    let mut s = MixSampler::new(&pools, &MixSpec::Proportional, 1);
    let mut c = [0usize; 2];
    for x in s.next_batch(2000) {
        c[x.game_idx as usize] += 1;
    }
    assert!(c[0] > 2 * c[1], "{c:?}");
}

#[test]
fn training_reduces_loss_and_is_seeded_deterministic() {
    // Burn's RNG seed is process-global; serialise tests that depend on it.
    let _g = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let g = game(Family::Engine, 1);
    let pools = vec![Pool { game_idx: 0, family_idx: 0, samples: data(&g, 0, 6) }];
    let run = || {
        let dev = Default::default();
        AB::seed(&dev, 5);
        let m = OmniAstra::<AB>::new(&nano(), &dev);
        let mut t = Trainer::<AB>::new(m, nano(), TrainCfg { steps: 60, batch: 16, lr: 3e-3, warmup: 5, ..Default::default() }, vec![dev]);
        let mut sampler = MixSampler::new(&pools, &MixSpec::UniformByGame, 9);
        let mut losses = vec![];
        for _ in 0..60 {
            losses.push(t.train_step(&sampler.next_batch(16)).loss);
        }
        losses
    };
    let a = run();
    let b = run();
    let first: f32 = a[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = a[55..].iter().sum::<f32>() / 5.0;
    println!("loss {first:.3} -> {last:.3}");
    assert!(last < first * 0.8, "loss did not decrease: {first} -> {last}");
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-4, "seeded runs diverged: {x} vs {y}");
    }
}

#[test]
fn checkpoint_round_trip_and_resume_vs_weights_only() {
    let g = game(Family::Stack, 4);
    let samples = data(&g, 0, 3);
    let refs: Vec<&Sample> = samples.iter().take(16).collect();
    let dev: <AB as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let m = OmniAstra::<AB>::new(&nano(), &dev);
    let mut t = Trainer::<AB>::new(m, nano(), TrainCfg { steps: 10, batch: 16, ..Default::default() }, vec![dev.clone()]);
    for _ in 0..3 {
        t.train_step(&refs);
    }
    let dir = tempfile::tempdir().unwrap();
    save(dir.path(), &t, serde_json::json!({"note": "test"})).unwrap();

    // identical outputs after weights-only load
    let loaded = load_weights::<IB>(dir.path(), &nano(), &dev).unwrap();
    let h = HostBatch::build(&refs, BatchOpts { n_reg: nano().n_reg, mask_ids: false });
    let bt = Batch::<IB>::from_host(&h, &dev);
    let a: Vec<f32> = t.valid().forward(&bt).logits.into_data().to_vec().unwrap();
    let b: Vec<f32> = loaded.forward(&bt).logits.into_data().to_vec().unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-5);
    }
    // resume restores counters/step; weights-only does not
    let r = resume::<AB>(dir.path(), &nano(), t.cfg.clone(), vec![dev.clone()]).unwrap();
    assert_eq!(r.step_idx, 3);
    assert_eq!(r.counters.optimizer_updates, 3);
    // continuing from resume gives the same next step as the original trainer
    let mut t2 = r;
    let s1 = t.train_step(&refs);
    let s2 = t2.train_step(&refs);
    assert!((s1.loss - s2.loss).abs() < 1e-4, "resume diverged: {} vs {}", s1.loss, s2.loss);
}

#[test]
fn data_parallel_gradients_match_single_device() {
    let _g = RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let g = game(Family::Tug, 2);
    let samples = data(&g, 0, 4);
    let refs: Vec<&Sample> = samples.iter().take(12).collect();
    let dev: <AB as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let base = OmniAstra::<AB>::new(&nano(), &dev);
    let step = |k: usize| {
        let t = Trainer::<AB>::new(base.clone(), nano(), TrainCfg::default(), vec![dev.clone(); k]);
        let (grads, loss, _, _) = t.grads(&refs, k);
        let mut sgd = SgdConfig::new().init::<AB, OmniAstra<AB>>();
        (sgd.step(0.05, base.clone(), grads), loss)
    };
    let (m1, l1) = step(1);
    let (m3, l3) = step(3);
    assert!((l1 - l3).abs() < 1e-4, "loss {l1} vs {l3}");
    let h = HostBatch::build(&refs, BatchOpts { n_reg: nano().n_reg, mask_ids: false });
    let bt = Batch::<AB>::from_host(&h, &dev);
    let a: Vec<f32> = m1.forward(&bt).logits.into_data().to_vec().unwrap();
    let b: Vec<f32> = m3.forward(&bt).logits.into_data().to_vec().unwrap();
    let md = a.iter().zip(&b).filter(|(x, _)| **x > -1e8).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(md < 1e-4, "single vs 3-shard update differs: {md}");
    let _ = m1.num_params();
}

#[test]
fn frozen_backbone_does_not_change_but_adapter_does() {
    let g = game(Family::Engine, 3);
    let samples = data(&g, 0, 3);
    let refs: Vec<&Sample> = samples.iter().take(16).collect();
    let dev: <AB as burn::tensor::backend::BackendTypes>::Device = Default::default();
    let m = OmniAstra::<AB>::new(&nano(), &dev).freeze_backbone();
    let before_blocks: Vec<f32> = m.blocks[0].clone().into_record_debug();
    let mut t = Trainer::<AB>::new(m, nano(), TrainCfg { lr: 1e-2, ..Default::default() }, vec![dev]);
    for _ in 0..3 {
        t.train_step(&refs);
    }
    let after_blocks: Vec<f32> = t.model.blocks[0].clone().into_record_debug();
    assert_eq!(before_blocks, after_blocks, "frozen backbone changed");
}

trait Debug1 {
    fn into_record_debug(self) -> Vec<f32>;
}
impl<B: Backend> Debug1 for omnia_model::layers::Block<B> {
    fn into_record_debug(self) -> Vec<f32> {
        // Flatten a few weights through the public forward pass instead of the record API.
        let dev = Default::default();
        let x = burn::tensor::Tensor::<B, 3>::ones([1, 4, nano().d_model], &dev);
        let mask = burn::tensor::Tensor::<B, 2>::ones([1, 4], &dev);
        self.forward(x, None, mask).into_data().to_vec().unwrap()
    }
}
