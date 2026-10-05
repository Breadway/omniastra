use crate::config::ModelConfig;
use crate::embed::Embedder;
use crate::layers::*;
use crate::tensors::Batch;
use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{log_softmax, silu};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor};
use omnia_dsl::MAX_PLAYERS;
use omnia_observation::batch::{N_LINKS, TB_VOCAB};
use omnia_observation::rel;

#[derive(Module, Debug)]
pub struct OmniAstra<B: Backend> {
    /// Input adapter (game-specific ids/slots -> shared space).
    pub embed: Embedder<B>,
    pub game_emb: Embedding<B>,
    /// Shared strategic backbone.
    pub reg: Param<Tensor<B, 2>>,
    /// Shared relational / temporal attention-bias tables (state tokens) and
    /// the action->state table used by the decoder.
    pub rel_tab: Option<BiasTable<B>>,
    pub tb_tab: Option<BiasTable<B>>,
    pub rel_as_tab: Option<BiasTable<B>>,
    pub blocks: Vec<Block<B>>,
    pub final_norm: RmsNorm<B>,
    /// Action-query decoder (shared scorer).
    pub link_proj: Linear<B>,
    pub dec: Vec<DecBlock<B>>,
    pub dec_norm: RmsNorm<B>,
    pub policy_out: Linear<B>,
    pub value_in: Linear<B>,
    pub value_out: Linear<B>,
    d_model: usize,
    n_reg: usize,
    use_history: bool,
    use_registers: bool,
    use_action_links: bool,
    use_game_emb: bool,
}

pub struct Output<B: Backend> {
    /// [B, Na] masked logits (padding = -1e9).
    pub logits: Tensor<B, 2>,
    /// [B, MAX_PLAYERS] expected payoff per relative seat in (-1, 1).
    pub value: Tensor<B, 2>,
    /// [B, R, D] final register states (for probing / analysis).
    pub regs: Tensor<B, 3>,
}

pub struct Losses<B: Backend> {
    pub total: Tensor<B, 1>,
    pub policy: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
}

/// Burn initialises parameters lazily; clones of a not-yet-materialised
/// module would each draw *different* random weights. Force materialisation so
/// that clones (arms of an experiment, data-parallel replicas) share weights.
struct Materialize;
impl<B: Backend> ModuleVisitor<B> for Materialize {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let _ = param.val();
    }
}

impl<B: Backend> OmniAstra<B> {
    pub fn new(c: &ModelConfig, dev: &B::Device) -> Self {
        let m = Self::new_lazy(c, dev);
        m.visit(&mut Materialize);
        m
    }

    fn new_lazy(c: &ModelConfig, dev: &B::Device) -> Self {
        let d = c.d_model;
        let h = c.n_heads;
        OmniAstra {
            embed: Embedder::new(c, dev),
            game_emb: EmbeddingConfig::new(c.n_games, d).with_initializer(burn::module::Initializer::Normal { mean: 0.0, std: 0.05 }).init(dev),
            reg: burn::module::Initializer::Normal { mean: 0.0, std: 0.05 }.init([c.n_reg, d], dev),
            rel_tab: if c.use_rel { Some(BiasTable::new(rel::VOCAB, h, dev)) } else { None },
            tb_tab: if c.use_rel && c.use_time_bias { Some(BiasTable::new(TB_VOCAB, h, dev)) } else { None },
            rel_as_tab: if c.use_rel { Some(BiasTable::new(rel::VOCAB, h, dev)) } else { None },
            blocks: (0..c.n_layers).map(|_| Block::new(d, c.n_heads, c.d_ff, c.use_rel, dev)).collect(),
            final_norm: RmsNormConfig::new(d).init(dev),
            link_proj: LinearConfig::new(N_LINKS * d, d).with_bias(false).init(dev),
            dec: (0..c.n_dec_layers).map(|_| DecBlock::new(d, c.n_heads, c.d_ff, c.use_rel, c.action_self_attn, dev)).collect(),
            dec_norm: RmsNormConfig::new(d).init(dev),
            policy_out: LinearConfig::new(d, 1).init(dev),
            value_in: LinearConfig::new(d, d).init(dev),
            value_out: LinearConfig::new(d, MAX_PLAYERS).init(dev),
            d_model: d,
            n_reg: c.n_reg,
            use_history: c.use_history,
            use_registers: c.use_registers,
            use_action_links: c.use_action_links,
            use_game_emb: c.use_game_emb,
        }
    }

    pub fn forward(&self, bt: &Batch<B>) -> Output<B> {
        let trace = std::env::var("OMNIA_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let (b, ns, na, r, d) = (bt.b, bt.ns, bt.na, self.n_reg, self.d_model);
        let dev = bt.s_mask.device();
        let xs = self.embed.forward(bt.s_class.clone(), bt.s_cat.clone(), bt.s_sem.clone(), bt.s_num.clone(), bt.s_nmask.clone(), bt.s_desc.clone(), bt.attr_desc.clone(), bt.res_desc.clone());

        if trace { eprintln!("  embed {:?}", t0.elapsed()); }
        // Registers.
        let mut regs = self.reg.val().unsqueeze_dim::<3>(0).expand([b, r, d]);
        if self.use_game_emb {
            let g = self.game_emb.forward(bt.game_idx.clone().reshape([b, 1])); // [B,1,D]
            regs = regs + g;
        }
        let x = Tensor::cat(vec![regs, xs.clone()], 1); // [B, R+Ns, D]

        // Key masks.
        let mut s_mask = bt.s_mask.clone();
        if !self.use_history {
            let ev = bt.s_class.clone().slice([0..b, 0..ns, 4..5]).reshape([b, ns]);
            s_mask = s_mask * (ev * -1.0 + 1.0);
        }
        let reg_mask = if self.use_registers { Tensor::<B, 2>::ones([b, r], &dev) } else { Tensor::<B, 2>::zeros([b, r], &dev) };
        let key_mask = Tensor::cat(vec![reg_mask, s_mask.clone()], 1);

        // Shared bias, computed once.
        let bias_ss: Option<Tensor<B, 4>> = self.rel_tab.as_ref().map(|t| {
            let mut bias = t.forward(bt.rel_ss.clone());
            if let Some(tb) = &self.tb_tab {
                bias = bias + tb.forward(bt.tb_ss.clone());
            }
            bias
        });
        if trace { eprintln!("  bias {:?}", t0.elapsed()); }
        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x, bias_ss.clone(), key_mask.clone());
        }
        if trace { let _ = x.clone().sum().into_scalar(); eprintln!("  blocks {:?}", t0.elapsed()); }
        let x = self.final_norm.forward(x);
        let regs_out = x.clone().slice([0..b, 0..r, 0..d]);

        // Value from the global workspace (registers) or, in the ablation, pooled tokens.
        let pooled = if self.use_registers {
            regs_out.clone().mean_dim(1).reshape([b, d])
        } else {
            let m = s_mask.clone().unsqueeze_dim::<3>(2);
            let st = x.clone().slice([0..b, r..r + ns, 0..d]);
            (st * m.clone()).sum_dim(1).reshape([b, d]) / (m.sum_dim(1).reshape([b, 1]) + 1e-6)
        };
        let value = self.value_out.forward(silu(self.value_in.forward(pooled))).tanh();

        // Action queries.
        let mut a = self.embed.forward(bt.a_class.clone(), bt.a_cat.clone(), bt.a_sem.clone(), bt.a_num.clone(), bt.a_nmask.clone(), bt.a_desc.clone(), bt.attr_desc.clone(), bt.res_desc.clone());
        if self.use_action_links {
            let x_pad = Tensor::cat(vec![x.clone(), Tensor::<B, 3>::zeros([b, 1, d], &dev)], 1);
            let idx = bt.a_links.clone().reshape([b, na * N_LINKS]).unsqueeze_dim::<3>(2).expand([b, na * N_LINKS, d]);
            let g = x_pad.gather(1, idx).reshape([b, na, N_LINKS * d]);
            a = a + self.link_proj.forward(g);
        }
        let bias_as: Option<Tensor<B, 4>> = self.rel_as_tab.as_ref().map(|t| t.forward(bt.rel_as.clone()));
        for blk in &self.dec {
            a = blk.forward(a, x.clone(), bias_as.clone(), key_mask.clone(), bt.a_mask.clone());
        }
        if trace { let _ = a.clone().sum().into_scalar(); eprintln!("  decoder {:?}", t0.elapsed()); }
        let logits = self.policy_out.forward(self.dec_norm.forward(a)).reshape([b, na]);
        let logits = logits + (bt.a_mask.clone() - 1.0) * 1.0e9;
        Output { logits, value, regs: regs_out }
    }

    /// Policy cross-entropy against (soft) targets + masked value MSE, both
    /// normalised by this batch's own sample / seat counts.
    pub fn losses(&self, bt: &Batch<B>, out: &Output<B>, value_weight: f64) -> Losses<B> {
        let n = bt.b as f64;
        let v = bt.vmask.clone().sum().into_scalar().elem::<f64>();
        self.losses_with(bt, out, value_weight, n, v)
    }

    /// Like [`losses`](Self::losses) but with explicit global denominators, so
    /// that gradients of shards sum exactly to the full-batch gradient
    /// (data-parallel training).
    pub fn losses_with(&self, bt: &Batch<B>, out: &Output<B>, value_weight: f64, n_total: f64, vmask_total: f64) -> Losses<B> {
        let logp = log_softmax(out.logits.clone(), 1);
        let policy = (bt.pi.clone() * logp).sum().neg().reshape([1]) / n_total;
        let diff = out.value.clone() - bt.value.clone();
        let value = (diff.clone() * diff * bt.vmask.clone()).sum().reshape([1]) / (vmask_total + 1e-6);
        let total = policy.clone() + value.clone() * value_weight;
        Losses { total, policy, value }
    }

    /// Freeze the shared backbone (registers, relational blocks, decoder);
    /// the input adapter and output heads stay trainable.
    pub fn freeze_backbone(mut self) -> Self {
        self.reg = self.reg.set_require_grad(false);
        self.blocks = self.blocks.no_grad();
        self.final_norm = self.final_norm.no_grad();
        self.link_proj = self.link_proj.no_grad();
        self.dec = self.dec.no_grad();
        self.dec_norm = self.dec_norm.no_grad();
        self
    }

    /// Freeze everything except the input adapter (embedder + game embedding).
    pub fn freeze_all_but_adapter(self) -> Self {
        let mut m = self.freeze_backbone();
        m.policy_out = m.policy_out.no_grad();
        m.value_in = m.value_in.no_grad();
        m.value_out = m.value_out.no_grad();
        m
    }
}
