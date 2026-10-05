use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use omnia_dsl::*;
use omnia_engine::*;
use omnia_gen::*;
use omnia_model::*;
use omnia_observation::batch::*;
use omnia_observation::*;
use std::sync::Arc;

type B = NdArray<f32>;
type AB = Autodiff<NdArray<f32>>;

fn game_for(fam: Family, seed: u64) -> Arc<Game> {
    let (_, def, _, _) = generate_valid(&GenSpec::new(fam, seed), &ValidationConfig { check_skill: false, ..Default::default() }, 30).unwrap();
    Game::new(def).unwrap()
}

fn samples(game: &Arc<Game>, game_idx: u32, n: usize, seed: u64) -> Vec<Sample> {
    let tk = Tokenizer::new(TokenizerConfig { max_history: 24 });
    let mut out = vec![];
    let mut rng = Rng::new(seed);
    let mut s = 0;
    while out.len() < n {
        let mut st = State::new(game, seed * 1000 + s);
        s += 1;
        while !st.is_terminal() && out.len() < n {
            let p = st.decision_maker().unwrap();
            let obs = tk.observe(&mut st, p);
            let na = obs.n_actions();
            let k = rng.below(na as u64) as usize;
            let mut pi = vec![0.0; na];
            pi[k] = 1.0;
            let mut value = [0.0; MAX_PLAYERS];
            value[0] = if rng.f32() < 0.5 { 1.0 } else { -1.0 };
            value[1] = -value[0];
            out.push(Sample { obs, pi, value, game_idx, family_idx: 0 });
            st.apply(k);
        }
    }
    out
}

fn host(ss: &[&Sample], cfg: &ModelConfig) -> HostBatch {
    HostBatch::build(ss, BatchOpts { n_reg: cfg.n_reg, mask_ids: !cfg.use_ids })
}

#[test]
fn forward_shapes_and_param_counts() {
    let dev = Default::default();
    let g = game_for(Family::Tempo, 1);
    let ss = samples(&g, 0, 6, 1);
    for (name, cfg) in [("tiny", ModelConfig::tiny()), ("small", ModelConfig::small()), ("medium", ModelConfig::medium())] {
        let m = OmniAstra::<B>::new(&cfg, &dev);
        println!("{name}: {} params", m.num_params());
        if name == "medium" {
            continue;
        }
        let refs: Vec<&Sample> = ss.iter().collect();
        let h = host(&refs, &cfg);
        let bt = Batch::<B>::from_host(&h, &dev);
        let out = m.forward(&bt);
        assert_eq!(out.logits.dims(), [6, h.na]);
        assert_eq!(out.value.dims(), [6, MAX_PLAYERS]);
        let l = m.losses(&bt, &out, 1.0);
        let v: f32 = l.total.into_scalar();
        assert!(v.is_finite());
    }
}

fn logits_of(m: &OmniAstra<B>, ss: &[&Sample], cfg: &ModelConfig, idx: usize) -> Vec<f32> {
    let dev = Default::default();
    let h = host(ss, cfg);
    let bt = Batch::<B>::from_host(&h, &dev);
    let out = m.forward(&bt);
    let na = ss[idx].obs.n_actions();
    let data: Vec<f32> = out.logits.into_data().to_vec().unwrap();
    data[idx * h.na..idx * h.na + na].to_vec()
}

#[test]
fn padding_does_not_change_outputs() {
    let dev = Default::default();
    let cfg = ModelConfig::tiny();
    let m = OmniAstra::<B>::new(&cfg, &dev);
    let g = game_for(Family::Tempo, 2);
    let ss = samples(&g, 0, 12, 3);
    // pick the largest sample as padding source
    let big = ss.iter().max_by_key(|s| s.obs.n_state() + s.obs.n_actions()).unwrap();
    let small = ss.iter().min_by_key(|s| s.obs.n_state() + s.obs.n_actions()).unwrap();
    assert!(big.obs.n_state() > small.obs.n_state());
    let alone = logits_of(&m, &[small], &cfg, 0);
    let padded = logits_of(&m, &[small, big], &cfg, 0);
    for (a, b) in alone.iter().zip(&padded) {
        assert!((a - b).abs() < 1e-3, "padding changed logits: {a} vs {b}");
    }
}

fn permute_objects(s: &Sample, perm: &[usize]) -> Sample {
    let o = &s.obs;
    let o0 = o.seg.objects.start;
    let n = o.seg.objects.len();
    assert_eq!(perm.len(), n);
    let mut out = s.clone();
    // new position i holds old object perm[i]
    let mut inv = vec![0usize; n];
    for (i, p) in perm.iter().enumerate() {
        inv[*p] = i;
        out.obs.tokens[o0 + i] = o.tokens[o0 + p].clone();
    }
    let remap = |t: u16| -> u16 {
        let t = t as usize;
        if t >= o0 && t < o0 + n {
            (o0 + inv[t - o0]) as u16
        } else {
            t as u16
        }
    };
    for r in out.obs.relations.iter_mut() {
        r.from = remap(r.from);
        r.to = remap(r.to);
    }
    out
}

#[test]
fn state_token_order_is_irrelevant() {
    let dev = Default::default();
    let cfg = ModelConfig::tiny();
    let m = OmniAstra::<B>::new(&cfg, &dev);
    let g = game_for(Family::Stack, 4);
    let ss = samples(&g, 0, 20, 5);
    let mut rng = Rng::new(9);
    let mut checked = 0;
    for s in ss.iter().filter(|s| s.obs.seg.objects.len() > 4) {
        let n = s.obs.seg.objects.len();
        let mut perm: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut perm);
        let p = permute_objects(s, &perm);
        let a = logits_of(&m, &[s], &cfg, 0);
        let b = logits_of(&m, &[&p], &cfg, 0);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-3, "object permutation changed logits: {x} vs {y}");
        }
        checked += 1;
    }
    assert!(checked >= 5);
}

#[test]
fn model_can_overfit_a_small_mixed_game_batch() {
    let dev = Default::default();
    let cfg = ModelConfig::tiny();
    let g1 = game_for(Family::Tempo, 1);
    let g2 = game_for(Family::Engine, 2);
    let mut ss = samples(&g1, 0, 8, 1);
    ss.extend(samples(&g2, 1, 8, 2));
    let refs: Vec<&Sample> = ss.iter().collect();
    let h = host(&refs, &cfg);
    let bt = Batch::<AB>::from_host(&h, &dev);
    let mut m = OmniAstra::<AB>::new(&cfg, &dev);
    let mut opt = AdamWConfig::new().init();
    let mut first = 0.0;
    let mut last = 0.0;
    for step in 0..250 {
        let out = m.forward(&bt);
        let l = m.losses(&bt, &out, 1.0);
        let v: f32 = l.total.clone().into_scalar();
        if step == 0 {
            first = v;
        }
        last = v;
        let grads = GradientsParams::from_grads(l.total.backward(), &m);
        m = opt.step(2e-3, m, grads);
    }
    println!("loss {first:.3} -> {last:.3}");
    assert!(last < first * 0.3, "loss did not fall enough: {first} -> {last}");
}

#[allow(dead_code)]
fn _assert_backend<T: Backend>() {}
