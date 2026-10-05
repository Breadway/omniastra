use burn::tensor::backend::Backend;
use omnia_dsl::MAX_PLAYERS;
use omnia_engine::{Rng, State};
use omnia_model::*;
use omnia_observation::batch::{BatchOpts, HostBatch, Sample};
use omnia_observation::{Observation, Tokenizer, TokenizerConfig};
use omnia_search::*;
use std::sync::Arc;

/// Shared inference model (cheap to clone).
pub struct NetHandle<B: Backend> {
    pub model: Arc<OmniAstra<B>>,
    pub cfg: Arc<ModelConfig>,
    pub dev: B::Device,
    pub game_idx: u32,
    pub max_history: usize,
}

impl<B: Backend> Clone for NetHandle<B> {
    fn clone(&self) -> Self {
        NetHandle { model: self.model.clone(), cfg: self.cfg.clone(), dev: self.dev.clone(), game_idx: self.game_idx, max_history: self.max_history }
    }
}

impl<B: Backend> NetHandle<B> {
    /// (logits over legal actions, value per relative seat) for one observation.
    pub fn infer(&self, obs: &Observation) -> (Vec<f32>, [f32; MAX_PLAYERS]) {
        let n = obs.n_actions();
        let sample = Sample { obs: obs.clone(), pi: vec![0.0; n], value: [0.0; MAX_PLAYERS], game_idx: self.game_idx, family_idx: 0 };
        let h = HostBatch::build(&[&sample], BatchOpts { n_reg: self.cfg.n_reg, mask_ids: !self.cfg.use_ids });
        let bt = Batch::<B>::from_host(&h, &self.dev);
        let out = self.model.forward(&bt);
        let logits: Vec<f32> = out.logits.into_data().to_vec().unwrap();
        let value: Vec<f32> = out.value.into_data().to_vec().unwrap();
        let mut v = [0.0; MAX_PLAYERS];
        v.copy_from_slice(&value[..MAX_PLAYERS]);
        (logits[..n].to_vec(), v)
    }
}

pub fn softmax(l: &[f32]) -> Vec<f32> {
    let mx = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = l.iter().map(|x| (x - mx).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

/// Raw policy network (no search).
pub struct NeuralAgent<B: Backend> {
    pub net: NetHandle<B>,
    pub tk: Tokenizer,
    pub label: String,
    /// 0 = greedy.
    pub temperature: f32,
}

impl<B: Backend> NeuralAgent<B> {
    pub fn new(net: NetHandle<B>, label: &str) -> Self {
        let tk = Tokenizer::new(TokenizerConfig { max_history: net.max_history });
        NeuralAgent { net, tk, label: label.into(), temperature: 0.0 }
    }
}

impl<B: Backend> Agent for NeuralAgent<B>
where
    B::Device: Send,
    OmniAstra<B>: Send + Sync,
{
    fn name(&self) -> String {
        self.label.clone()
    }
    fn act(&mut self, st: &mut State, rng: &mut Rng) -> usize {
        if st.legal_actions().len() == 1 {
            return 0;
        }
        let me = st.decision_maker().unwrap();
        let obs = self.tk.observe(st, me);
        let (logits, _) = self.net.infer(&obs);
        if self.temperature <= 0.0 {
            logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0)
        } else {
            sample_with_temperature(&softmax(&logits), self.temperature, rng)
        }
    }
}

/// Network as an MCTS evaluator (priors from the policy, values from the value head).
pub struct NeuralEvaluator<B: Backend> {
    pub net: NetHandle<B>,
    pub tk: Tokenizer,
}

impl<B: Backend> NeuralEvaluator<B> {
    pub fn new(net: NetHandle<B>) -> Self {
        let tk = Tokenizer::new(TokenizerConfig { max_history: net.max_history });
        NeuralEvaluator { net, tk }
    }
}

impl<B: Backend> Evaluator for NeuralEvaluator<B>
where
    B::Device: Send,
    OmniAstra<B>: Send + Sync,
{
    fn evaluate(&mut self, st: &mut State, _rng: &mut Rng) -> Eval {
        let me = st.decision_maker().unwrap();
        let n = st.num_players() as usize;
        let obs = self.tk.observe(st, me);
        let (logits, v) = self.net.infer(&obs);
        let mut values = [0.0; MAX_PLAYERS];
        for r in 0..n {
            values[(me as usize + r) % n] = v[r];
        }
        Eval { priors: Some(softmax(&logits)), values }
    }
}
