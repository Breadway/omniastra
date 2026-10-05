use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Multi-head attention with an optional *shared* additive bias.
///
/// The relational / temporal bias tensor is computed once per forward pass
/// (Graphormer-style, shared across layers); each layer only owns a per-head
/// gain on it. This keeps the cost of the (B x N x N) gather out of every layer.
#[derive(Module, Debug)]
pub struct MhAttn<B: Backend> {
    wq: Linear<B>,
    wk: Linear<B>,
    wv: Linear<B>,
    wo: Linear<B>,
    gain: Option<Param<Tensor<B, 1>>>,
    n_heads: usize,
}

impl<B: Backend> MhAttn<B> {
    pub fn new(d: usize, n_heads: usize, with_bias: bool, dev: &B::Device) -> Self {
        let lin = || LinearConfig::new(d, d).with_bias(false).init(dev);
        MhAttn {
            wq: lin(),
            wk: lin(),
            wv: lin(),
            wo: lin(),
            gain: if with_bias { Some(burn::module::Initializer::Ones.init([n_heads], dev)) } else { None },
            n_heads,
        }
    }

    /// `xq`: [B,Nq,D], `xkv`: [B,Nk,D]; `bias`: [B,H,Nq,Nk]; `key_mask`: [B,Nk] (1 = real).
    pub fn forward(&self, xq: Tensor<B, 3>, xkv: Tensor<B, 3>, bias: Option<Tensor<B, 4>>, key_mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let [b, nq, d] = xq.dims();
        let nk = xkv.dims()[1];
        let h = self.n_heads;
        let dh = d / h;
        let q = self.wq.forward(xq).reshape([b, nq, h, dh]).swap_dims(1, 2);
        let k = self.wk.forward(xkv.clone()).reshape([b, nk, h, dh]).swap_dims(1, 2);
        let v = self.wv.forward(xkv).reshape([b, nk, h, dh]).swap_dims(1, 2);
        let mut scores = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt();
        if let (Some(g), Some(bias)) = (&self.gain, bias) {
            scores = scores + bias * g.val().reshape([1, h, 1, 1]);
        }
        let neg = (key_mask.reshape([b, 1, 1, nk]) - 1.0) * 1.0e9;
        scores = scores + neg;
        let attn = softmax(scores, 3);
        let out = attn.matmul(v).swap_dims(1, 2).reshape([b, nq, d]);
        self.wo.forward(out)
    }
}

/// Lookup of per-head attention biases from relation / time-bucket ids.
#[derive(Module, Debug)]
pub struct BiasTable<B: Backend> {
    emb: Embedding<B>,
    n_heads: usize,
}

impl<B: Backend> BiasTable<B> {
    pub fn new(vocab: usize, n_heads: usize, dev: &B::Device) -> Self {
        let emb = EmbeddingConfig::new(vocab, n_heads).with_initializer(burn::module::Initializer::Normal { mean: 0.0, std: 0.02 }).init(dev);
        BiasTable { emb, n_heads }
    }
    /// ids [B,Nq,Nk] -> [B,H,Nq,Nk]
    pub fn forward(&self, ids: Tensor<B, 3, Int>) -> Tensor<B, 4> {
        let [b, nq, nk] = ids.dims();
        self.emb.forward(ids.reshape([b, nq * nk])).reshape([b, nq, nk, self.n_heads]).permute([0, 3, 1, 2])
    }
}

#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    w1: Linear<B>,
    w3: Linear<B>,
    w2: Linear<B>,
}

impl<B: Backend> SwiGlu<B> {
    pub fn new(d: usize, ff: usize, dev: &B::Device) -> Self {
        SwiGlu {
            w1: LinearConfig::new(d, ff).with_bias(false).init(dev),
            w3: LinearConfig::new(d, ff).with_bias(false).init(dev),
            w2: LinearConfig::new(ff, d).with_bias(false).init(dev),
        }
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2.forward(silu(self.w1.forward(x.clone())) * self.w3.forward(x))
    }
}

/// Pre-norm self-attention block over registers + state tokens.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    n1: RmsNorm<B>,
    attn: MhAttn<B>,
    n2: RmsNorm<B>,
    ff: SwiGlu<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(d: usize, heads: usize, ff: usize, with_bias: bool, dev: &B::Device) -> Self {
        Block {
            n1: RmsNormConfig::new(d).init(dev),
            attn: MhAttn::new(d, heads, with_bias, dev),
            n2: RmsNormConfig::new(d).init(dev),
            ff: SwiGlu::new(d, ff, dev),
        }
    }
    pub fn forward(&self, x: Tensor<B, 3>, bias: Option<Tensor<B, 4>>, key_mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let h = self.n1.forward(x.clone());
        let x = x + self.attn.forward(h.clone(), h, bias, key_mask);
        let h = self.n2.forward(x.clone());
        x + self.ff.forward(h)
    }
}

/// Action-query decoder block: cross-attend to state, optionally self-attend
/// among actions, then FFN.
#[derive(Module, Debug)]
pub struct DecBlock<B: Backend> {
    n1: RmsNorm<B>,
    cross: MhAttn<B>,
    n2: RmsNorm<B>,
    self_attn: Option<MhAttn<B>>,
    n3: RmsNorm<B>,
    ff: SwiGlu<B>,
}

impl<B: Backend> DecBlock<B> {
    pub fn new(d: usize, heads: usize, ff: usize, with_bias: bool, self_attn: bool, dev: &B::Device) -> Self {
        DecBlock {
            n1: RmsNormConfig::new(d).init(dev),
            cross: MhAttn::new(d, heads, with_bias, dev),
            n2: RmsNormConfig::new(d).init(dev),
            self_attn: if self_attn { Some(MhAttn::new(d, heads, false, dev)) } else { None },
            n3: RmsNormConfig::new(d).init(dev),
            ff: SwiGlu::new(d, ff, dev),
        }
    }

    pub fn forward(&self, a: Tensor<B, 3>, mem: Tensor<B, 3>, bias_as: Option<Tensor<B, 4>>, mem_mask: Tensor<B, 2>, a_mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let h = self.n1.forward(a.clone());
        let mut a = a + self.cross.forward(h, mem, bias_as, mem_mask);
        if let Some(sa) = &self.self_attn {
            let h = self.n2.forward(a.clone());
            a = a + sa.forward(h.clone(), h, None, a_mask);
        }
        let h = self.n3.forward(a.clone());
        a + self.ff.forward(h)
    }
}
