use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use omnia_dsl::*;
use omnia_engine::*;
use omnia_gen::*;
use omnia_model::*;
use omnia_observation::batch::*;
use omnia_observation::*;

type AB = Autodiff<NdArray<f32>>;

/// Smoke cover for the bench example path (`examples/bench.rs`): build a
/// small batch exactly the way the example does, then run one forward,
/// backward and optimizer step. Catches stale batch construction or
/// out-of-range gather ids without paying for the full benchmark.
#[test]
fn bench_path_smoke() {
    let cfg = ModelConfig::nano();
    let dev = Default::default();
    let (_, def, _, _) = generate_valid(
        &GenSpec::new(Family::Tempo, 1),
        &ValidationConfig {
            check_skill: false,
            ..Default::default()
        },
        30,
    )
    .unwrap();
    let game = Game::new(def).unwrap();
    let tk = Tokenizer::new(TokenizerConfig { max_history: 32 });
    let mut samples = vec![];
    let mut st = State::new(&game, 1);
    let mut rng = Rng::new(1);
    while samples.len() < 2 {
        if st.is_terminal() {
            st = State::new(&game, rng.next_u64());
        }
        let p = st.decision_maker().unwrap();
        let obs = tk.observe(&mut st, p);
        let na = obs.n_actions();
        let mut pi = vec![0.0; na];
        pi[0] = 1.0;
        samples.push(Sample {
            obs,
            pi,
            value: [0.0; MAX_PLAYERS],
            game_idx: 0,
            family_idx: 0,
        });
        let n = st.legal_actions().len();
        st.apply(rng.below(n as u64) as usize);
    }
    let refs: Vec<&Sample> = samples.iter().collect();
    // Same options the trainer uses, so the example and training cannot drift apart.
    let h = HostBatch::build(
        &refs,
        BatchOpts {
            n_reg: cfg.n_reg,
            mask_ids: !cfg.use_ids,
        },
    );
    // Every relation id must fit the model-side vocabulary, including the
    // register ids. This is the assertion that caught OMNI-28.
    assert!(h.rel_ss.iter().all(|id| (*id as usize) < rel::VOCAB));
    assert!(h.rel_as.iter().all(|id| (*id as usize) < rel::VOCAB));
    let bt = Batch::<AB>::from_host(&h, &dev);
    let mut m = OmniAstra::<AB>::new(&cfg, &dev);
    let mut opt = AdamWConfig::new().init();
    let out = m.forward(&bt);
    let l = m.losses(&bt, &out, 1.0);
    let v: f32 = l.total.clone().into_scalar();
    assert!(v.is_finite());
    let grads = GradientsParams::from_grads(l.total.backward(), &m);
    m = opt.step(1e-3, m, grads);
    let out = m.forward(&bt);
    let l = m.losses(&bt, &out, 1.0);
    let v: f32 = l.total.into_scalar();
    assert!(v.is_finite());
}
