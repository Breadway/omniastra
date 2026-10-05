//! Token embedding ("input adapter"). Everything in here is the part of the
//! network that maps a game's arbitrary ids/slots into the shared space; the
//! transfer experiments can therefore re-initialise or train it separately
//! from the shared backbone.

use crate::config::ModelConfig;
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use omnia_dsl::{DESC_DIM, MAX_ATTRS, MAX_RESOURCES};
use omnia_observation::{NCAT, NNUM, NSEM, NUM_CLASSES};

const NF: usize = 2; // numeric features per slot value

#[derive(Module, Debug)]
pub struct Embedder<B: Backend> {
    class_proj: Linear<B>,
    cat_emb: Vec<Embedding<B>>,
    sem_emb: Vec<Embedding<B>>,
    num_pos: Linear<B>,
    desc_proj: Linear<B>,
    attr_h1: Linear<B>,
    attr_h1b: Linear<B>,
    attr_h2: Linear<B>,
    attr_h2b: Linear<B>,
    res_h1: Linear<B>,
    res_h1b: Linear<B>,
    res_h2: Linear<B>,
    res_h2b: Linear<B>,
    use_ids: bool,
    use_slot_pos: bool,
    use_desc: bool,
}

fn small_emb<B: Backend>(n: usize, d: usize, dev: &B::Device) -> Embedding<B> {
    EmbeddingConfig::new(n, d).with_initializer(burn::module::Initializer::Normal { mean: 0.0, std: 0.05 }).init(dev)
}

fn lin<B: Backend>(i: usize, o: usize, bias: bool, dev: &B::Device) -> Linear<B> {
    LinearConfig::new(i, o).with_bias(bias).init(dev)
}

impl<B: Backend> Embedder<B> {
    pub fn new(c: &ModelConfig, dev: &B::Device) -> Self {
        let d = c.d_model;
        Embedder {
            class_proj: lin(NUM_CLASSES, d, false, dev),
            cat_emb: (0..NCAT).map(|_| small_emb(c.cat_vocab, d, dev)).collect(),
            sem_emb: (0..NSEM).map(|_| small_emb(c.sem_vocab, d, dev)).collect(),
            num_pos: lin(NUM_CLASSES * NNUM * NF, d, false, dev),
            desc_proj: lin(DESC_DIM, d, true, dev),
            attr_h1: lin(DESC_DIM, d, true, dev),
            attr_h1b: lin(d, d, false, dev),
            attr_h2: lin(DESC_DIM, d, true, dev),
            attr_h2b: lin(d, d, false, dev),
            res_h1: lin(DESC_DIM, d, true, dev),
            res_h1b: lin(d, d, false, dev),
            res_h2: lin(DESC_DIM, d, true, dev),
            res_h2b: lin(d, d, false, dev),
            use_ids: c.use_ids,
            use_slot_pos: c.use_slot_pos,
            use_desc: c.use_desc,
        }
    }

    fn lookup(embs: &[Embedding<B>], ids: Tensor<B, 3, Int>) -> Tensor<B, 3> {
        let [b, n, k] = ids.dims();
        let mut out: Option<Tensor<B, 3>> = None;
        for (j, e) in embs.iter().enumerate().take(k) {
            let col = ids.clone().slice([0..b, 0..n, j..j + 1]).reshape([b, n]);
            let present = col.clone().greater_elem(0).float().unsqueeze_dim::<3>(2);
            let v = e.forward(col) * present;
            out = Some(match out {
                None => v,
                Some(o) => o + v,
            });
        }
        out.unwrap()
    }

    fn slot_mlp(h: &Linear<B>, hb: &Linear<B>, desc: Tensor<B, 3>) -> Tensor<B, 3> {
        hb.forward(silu(h.forward(desc)))
    }

    /// `class`: [B,N,C] one-hot; `cat`: [B,N,NCAT]; `sem`: [B,N,NSEM]; `num`,`nmask`: [B,N,NNUM];
    /// `desc`: [B,N,DESC]; `attr_desc`: [B,MAX_ATTRS,DESC]; `res_desc`: [B,MAX_RESOURCES,DESC].
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        class: Tensor<B, 3>,
        cat: Tensor<B, 3, Int>,
        sem: Tensor<B, 3, Int>,
        num: Tensor<B, 3>,
        nmask: Tensor<B, 3>,
        desc: Tensor<B, 3>,
        attr_desc: Tensor<B, 3>,
        res_desc: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, n, _] = class.dims();
        let mut x = self.class_proj.forward(class.clone()) + Self::lookup(&self.sem_emb, sem);
        if self.use_ids {
            x = x + Self::lookup(&self.cat_emb, cat);
        }
        // Numeric features.
        let sym = num.clone().sign() * (num.clone().abs() + 1.0).log() * nmask.clone();
        let lin_f = num.clamp(-32.0, 32.0) / 8.0 * nmask;
        if self.use_slot_pos {
            let feat = Tensor::cat(vec![sym.clone().unsqueeze_dim::<4>(3), lin_f.clone().unsqueeze_dim::<4>(3)], 3).reshape([b, n, NNUM * NF]);
            let expanded = class.clone().unsqueeze_dim::<4>(3) * feat.unsqueeze_dim::<4>(2); // [B,N,C,NNUM*NF]
            x = x + self.num_pos.forward(expanded.reshape([b, n, NUM_CLASSES * NNUM * NF]));
        }
        if self.use_desc {
            x = x + self.desc_proj.forward(desc);
            // Descriptor-grounded numeric slots: object attrs and player resources.
            let obj = class.clone().slice([0..b, 0..n, 3..4]);
            let ply = class.slice([0..b, 0..n, 1..2]);
            let s_a = sym.clone().slice([0..b, 0..n, 0..MAX_ATTRS]);
            let l_a = lin_f.clone().slice([0..b, 0..n, 0..MAX_ATTRS]);
            let h1 = Self::slot_mlp(&self.attr_h1, &self.attr_h1b, attr_desc.clone());
            let h2 = Self::slot_mlp(&self.attr_h2, &self.attr_h2b, attr_desc);
            x = x + (s_a.matmul(h1) + l_a.matmul(h2)) * obj;
            let s_r = sym.slice([0..b, 0..n, 0..MAX_RESOURCES]);
            let l_r = lin_f.slice([0..b, 0..n, 0..MAX_RESOURCES]);
            let g1 = Self::slot_mlp(&self.res_h1, &self.res_h1b, res_desc.clone());
            let g2 = Self::slot_mlp(&self.res_h2, &self.res_h2b, res_desc);
            x = x + (s_r.matmul(g1) + l_r.matmul(g2)) * ply;
        }
        x
    }
}
