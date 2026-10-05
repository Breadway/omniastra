//! Throughput micro-benchmark: `cargo run --release -p omnia-model --example bench [--features ...] -- tiny 32`
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use omnia_dsl::*;
use omnia_engine::*;
use omnia_gen::*;
use omnia_model::*;
use omnia_observation::batch::*;
use omnia_observation::*;
use std::time::Instant;

#[cfg(feature = "candle")]
type Inner = burn::backend::Candle<f32, i64>;
#[cfg(all(not(feature = "candle"), feature = "flex"))]
type Inner = burn::backend::Flex;
#[cfg(all(not(feature = "candle"), not(feature = "flex"), feature = "cpu"))]
type Inner = burn::backend::Cpu;
#[cfg(all(not(feature = "candle"), not(feature = "flex"), not(feature = "cpu")))]
type Inner = burn::backend::NdArray<f32>;
type AB = burn::backend::Autodiff<Inner>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size = args.get(1).map(|s| s.as_str()).unwrap_or("tiny");
    let bs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let mut cfg = ModelConfig::by_name(size).unwrap();
    if std::env::var("NOREL").is_ok() {
        cfg = cfg.with_use_rel(false);
    }
    let dev = Default::default();
    let (_, def, _, _) = generate_valid(&GenSpec::new(Family::Tempo, 1), &ValidationConfig { check_skill: false, ..Default::default() }, 30).unwrap();
    let game = Game::new(def).unwrap();
    let tk = Tokenizer::new(TokenizerConfig { max_history: 32 });
    let mut samples = vec![];
    let mut st = State::new(&game, 1);
    let mut rng = Rng::new(1);
    while samples.len() < bs {
        if st.is_terminal() {
            st = State::new(&game, rng.next_u64());
        }
        let p = st.decision_maker().unwrap();
        let obs = tk.observe(&mut st, p);
        let na = obs.n_actions();
        let mut pi = vec![0.0; na];
        pi[0] = 1.0;
        samples.push(Sample { obs, pi, value: [0.0; MAX_PLAYERS], game_idx: 0, family_idx: 0 });
        let n = st.legal_actions().len();
        st.apply(rng.below(n as u64) as usize);
    }
    let refs: Vec<&Sample> = samples.iter().collect();
    let h = HostBatch::build(&refs, BatchOpts { n_reg: cfg.n_reg, mask_ids: false });
    println!("{size} bs={bs} tokens/sample ~{} state, {} actions; backend={}", h.ns, h.na, std::any::type_name::<Inner>());
    let mut m = OmniAstra::<AB>::new(&cfg, &dev);
    println!("params: {}", m.num_params());
    let t = Instant::now();
    let bt = Batch::<AB>::from_host(&h, &dev);
    println!("host->tensor: {:?}", t.elapsed());
    {
        use burn::tensor::Tensor;
        let t = Instant::now();
        let xs = m.embed.forward(bt.s_class.clone(), bt.s_cat.clone(), bt.s_sem.clone(), bt.s_num.clone(), bt.s_nmask.clone(), bt.s_desc.clone(), bt.attr_desc.clone(), bt.res_desc.clone());
        let _ = xs.clone().sum().into_scalar();
        println!("embed fwd: {:?}", t.elapsed());
        let t = Instant::now();
        let a: Tensor<AB, 3> = Tensor::random([32, 71, 128], burn::tensor::Distribution::Default, &dev);
        let w: Tensor<AB, 2> = Tensor::random([128, 344], burn::tensor::Distribution::Default, &dev);
        for _ in 0..10 {
            let y = a.clone().matmul(w.clone().unsqueeze_dim::<3>(0));
            let _ = y.sum().into_scalar();
        }
        println!("10x matmul [32,71,128]x[128,344]: {:?}", t.elapsed());
        let t = Instant::now();
        let q: Tensor<AB, 4> = Tensor::random([32, 4, 71, 32], burn::tensor::Distribution::Default, &dev);
        for _ in 0..10 {
            let y = q.clone().matmul(q.clone().swap_dims(2, 3));
            let _ = y.sum().into_scalar();
        }
        {
            use burn::nn::{EmbeddingConfig};
            use burn::tensor::Int;
            let emb = EmbeddingConfig::new(40, 4).init::<AB>(&dev);
            let ids: Tensor<AB, 3, Int> = bt.rel_ss.clone();
            let t = Instant::now();
            for _ in 0..10 {
                let [b, n, k] = ids.dims();
                let y = emb.forward(ids.clone().reshape([b, n * k])).reshape([b, n, k, 4]).permute([0, 3, 1, 2]);
                let _ = y.sum().into_scalar();
            }
            println!("10x rel embed lookup {:?}: {:?}", ids.dims(), t.elapsed());
            let emb24 = EmbeddingConfig::new(40, 24).init::<AB>(&dev);
            let t = Instant::now();
            for _ in 0..10 {
                let [b, n, k] = ids.dims();
                let y = emb24.forward(ids.clone().reshape([b, n * k]));
                let _ = y.sum().into_scalar();
            }
            println!("10x wide(24) embed lookup: {:?}", t.elapsed());
            let t = Instant::now();
            for _ in 0..10 {
                let [b, n, k] = ids.dims();
                let flat = ids.clone().reshape([b * n * k]);
                let w = emb24.weight.val();
                let y = w.select(0, flat);
                let _ = y.sum().into_scalar();
            }
            println!("10x select(24): {:?}", t.elapsed());
            let t = Instant::now();
            for _ in 0..10 {
                let [b, n, k] = ids.dims();
                let flat = ids.clone().reshape([b * n * k]);
                let oh = flat.one_hot::<2>(40).float();
                let w = emb24.weight.val();
                let y = oh.matmul(w);
                let _ = y.sum().into_scalar();
            }
            println!("10x onehot-matmul(24): {:?}", t.elapsed());
            let t = Instant::now();
            for _ in 0..10 {
                let y: Tensor<AB, 4> = Tensor::random([32, 4, 71, 71], burn::tensor::Distribution::Default, &dev);
                let y = burn::tensor::activation::softmax(y, 3);
                let _ = y.sum().into_scalar();
            }
            println!("10x random+softmax [32,4,71,71]: {:?}", t.elapsed());
        }
        println!("10x attn qk [32,4,71,32]x[..,32,71]: {:?}", t.elapsed());
    }
    let mut opt = AdamWConfig::new().init();
    for i in 0..4 {
        let t = Instant::now();
        let out = m.forward(&bt);
        let l = m.losses(&bt, &out, 1.0);
        let _ = l.total.clone().into_scalar();
        let tf = t.elapsed();
        let grads = GradientsParams::from_grads(l.total.backward(), &m);
        let tb = t.elapsed();
        m = opt.step(1e-3, m, grads);
        println!("iter {i}: fwd {:?} fwd+bwd {:?} step {:?}  => {:.0} samples/s", tf, tb, t.elapsed(), bs as f64 / t.elapsed().as_secs_f64());
    }
}

#[allow(dead_code)]
fn unused() {}
