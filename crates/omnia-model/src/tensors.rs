use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use omnia_dsl::{DESC_DIM, MAX_ATTRS, MAX_PLAYERS, MAX_RESOURCES};
use omnia_observation::batch::{HostBatch, N_LINKS};
use omnia_observation::{NCAT, NNUM, NSEM, NUM_CLASSES};

/// Device-resident batch.
#[derive(Clone)]
pub struct Batch<B: Backend> {
    pub b: usize,
    pub ns: usize,
    pub na: usize,
    pub n_reg: usize,
    pub s_class: Tensor<B, 3>,
    pub s_cat: Tensor<B, 3, Int>,
    pub s_sem: Tensor<B, 3, Int>,
    pub s_num: Tensor<B, 3>,
    pub s_nmask: Tensor<B, 3>,
    pub s_desc: Tensor<B, 3>,
    pub s_mask: Tensor<B, 2>,
    pub a_class: Tensor<B, 3>,
    pub a_cat: Tensor<B, 3, Int>,
    pub a_sem: Tensor<B, 3, Int>,
    pub a_num: Tensor<B, 3>,
    pub a_nmask: Tensor<B, 3>,
    pub a_desc: Tensor<B, 3>,
    pub a_mask: Tensor<B, 2>,
    pub rel_ss: Tensor<B, 3, Int>,
    pub tb_ss: Tensor<B, 3, Int>,
    pub rel_as: Tensor<B, 3, Int>,
    pub a_links: Tensor<B, 3, Int>,
    pub attr_desc: Tensor<B, 3>,
    pub res_desc: Tensor<B, 3>,
    pub game_idx: Tensor<B, 1, Int>,
    pub pi: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub vmask: Tensor<B, 2>,
}

fn f<B: Backend, const D: usize>(v: &[f32], shape: [usize; D], d: &B::Device) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d)
}

fn i<B: Backend, const D: usize>(v: &[i32], shape: [usize; D], d: &B::Device) -> Tensor<B, D, Int> {
    Tensor::from_data(TensorData::new(v.to_vec(), shape), d)
}

impl<B: Backend> Batch<B> {
    pub fn from_host(h: &HostBatch, d: &B::Device) -> Batch<B> {
        let (b, ns, na, kn) = (h.b, h.ns, h.na, h.kn());
        Batch {
            b,
            ns,
            na,
            n_reg: h.n_reg,
            s_class: f(&h.s_class, [b, ns, NUM_CLASSES], d),
            s_cat: i(&h.s_cat, [b, ns, NCAT], d),
            s_sem: i(&h.s_sem, [b, ns, NSEM], d),
            s_num: f(&h.s_num, [b, ns, NNUM], d),
            s_nmask: f(&h.s_nmask, [b, ns, NNUM], d),
            s_desc: f(&h.s_desc, [b, ns, DESC_DIM], d),
            s_mask: f(&h.s_mask, [b, ns], d),
            a_class: f(&h.a_class, [b, na, NUM_CLASSES], d),
            a_cat: i(&h.a_cat, [b, na, NCAT], d),
            a_sem: i(&h.a_sem, [b, na, NSEM], d),
            a_num: f(&h.a_num, [b, na, NNUM], d),
            a_nmask: f(&h.a_nmask, [b, na, NNUM], d),
            a_desc: f(&h.a_desc, [b, na, DESC_DIM], d),
            a_mask: f(&h.a_mask, [b, na], d),
            rel_ss: i(&h.rel_ss, [b, kn, kn], d),
            tb_ss: i(&h.tb_ss, [b, kn, kn], d),
            rel_as: i(&h.rel_as, [b, na, kn], d),
            a_links: i(&h.a_links, [b, na, N_LINKS], d),
            attr_desc: f(&h.attr_desc, [b, MAX_ATTRS, DESC_DIM], d),
            res_desc: f(&h.res_desc, [b, MAX_RESOURCES, DESC_DIM], d),
            game_idx: i(&h.game_idx, [b], d),
            pi: f(&h.pi, [b, na], d),
            value: f(&h.value, [b, MAX_PLAYERS], d),
            vmask: f(&h.vmask, [b, MAX_PLAYERS], d),
        }
    }
}
