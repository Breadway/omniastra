//! Host-side batching: variable-length observations -> padded flat arrays.
//!
//! Padding exists only here. Registers (learned global tokens, owned by the
//! model) are accounted for in the relation matrices: they occupy positions
//! `0..n_reg` of the "key" axis, followed by the state tokens.

use crate::*;
use serde::{Deserialize, Serialize};

pub const TB_VOCAB: usize = 18;
pub const N_LINKS: usize = 5;

/// One training example: a decision point with search/expert targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub obs: Observation,
    /// Target distribution over the legal actions (aligned with action tokens).
    pub pi: Vec<f32>,
    /// Final payoff per *relative* seat (0 = decision maker).
    pub value: [f32; MAX_PLAYERS],
    pub game_idx: u32,
    /// Index of the game within its family-level grouping (for per-family stats).
    pub family_idx: u32,
}

#[derive(Clone, Debug, Default)]
pub struct HostBatch {
    pub b: usize,
    pub n_reg: usize,
    /// State tokens per sample (excluding registers), padded.
    pub ns: usize,
    pub na: usize,
    pub s_class: Vec<f32>, // [b, ns, NUM_CLASSES] one-hot
    pub s_cat: Vec<i32>,   // [b, ns, NCAT]
    pub s_sem: Vec<i32>,   // [b, ns, NSEM]
    pub s_num: Vec<f32>,   // [b, ns, NNUM]
    pub s_nmask: Vec<f32>, // [b, ns, NNUM]
    pub s_desc: Vec<f32>,  // [b, ns, DESC_DIM]
    pub s_mask: Vec<f32>,  // [b, ns]
    pub a_class: Vec<f32>,
    pub a_cat: Vec<i32>,
    pub a_sem: Vec<i32>,
    pub a_num: Vec<f32>,
    pub a_nmask: Vec<f32>,
    pub a_desc: Vec<f32>,
    pub a_mask: Vec<f32>, // [b, na]
    /// Relation ids among (registers + state tokens): [b, R+ns, R+ns].
    pub rel_ss: Vec<i32>,
    /// Time-bucket ids among (registers + state tokens): [b, R+ns, R+ns].
    pub tb_ss: Vec<i32>,
    /// Relation ids action -> (registers + state tokens): [b, na, R+ns].
    pub rel_as: Vec<i32>,
    /// Links from actions to state-token positions in (registers + state)
    /// space; `R+ns` means "none" (zero vector).
    pub a_links: Vec<i32>, // [b, na, N_LINKS]
    /// Slot descriptors: [b, MAX_ATTRS, DESC] and [b, MAX_RESOURCES, DESC].
    pub attr_desc: Vec<f32>,
    pub res_desc: Vec<f32>,
    pub game_idx: Vec<i32>,
    pub pi: Vec<f32>,      // [b, na]
    pub value: Vec<f32>,   // [b, MAX_PLAYERS]
    pub vmask: Vec<f32>,   // [b, MAX_PLAYERS]
    pub family_idx: Vec<i32>,
}

fn time_bucket(dt: i64) -> i32 {
    // dt = t_key - t_query
    let mag = |x: i64| -> i32 { (63 - (x as u64).leading_zeros() as i32).clamp(0, 7) };
    if dt < 0 {
        1 + mag(-dt)
    } else if dt == 0 {
        9
    } else {
        10 + mag(dt)
    }
}

/// How ID-like features are exposed (ablations / transfer regimes).
#[derive(Clone, Copy, Debug)]
pub struct BatchOpts {
    pub n_reg: usize,
    /// Zero all game-local categorical ids (`cat`), keeping only semantic features.
    pub mask_ids: bool,
}

impl HostBatch {
    pub fn build(samples: &[&Sample], opts: BatchOpts) -> HostBatch {
        let b = samples.len();
        let r = opts.n_reg;
        let ns = samples.iter().map(|s| s.obs.n_state()).max().unwrap_or(1);
        let na = samples.iter().map(|s| s.obs.n_actions()).max().unwrap_or(1).max(1);
        let kn = r + ns;
        let mut hb = HostBatch { b, n_reg: r, ns, na, ..Default::default() };
        hb.s_class = vec![0.0; b * ns * NUM_CLASSES];
        hb.s_cat = vec![0; b * ns * NCAT];
        hb.s_sem = vec![0; b * ns * NSEM];
        hb.s_num = vec![0.0; b * ns * NNUM];
        hb.s_nmask = vec![0.0; b * ns * NNUM];
        hb.s_desc = vec![0.0; b * ns * DESC_DIM];
        hb.s_mask = vec![0.0; b * ns];
        hb.a_class = vec![0.0; b * na * NUM_CLASSES];
        hb.a_cat = vec![0; b * na * NCAT];
        hb.a_sem = vec![0; b * na * NSEM];
        hb.a_num = vec![0.0; b * na * NNUM];
        hb.a_nmask = vec![0.0; b * na * NNUM];
        hb.a_desc = vec![0.0; b * na * DESC_DIM];
        hb.a_mask = vec![0.0; b * na];
        hb.rel_ss = vec![0; b * kn * kn];
        hb.tb_ss = vec![0; b * kn * kn];
        hb.rel_as = vec![0; b * na * kn];
        hb.a_links = vec![kn as i32; b * na * N_LINKS];
        hb.attr_desc = vec![0.0; b * MAX_ATTRS * DESC_DIM];
        hb.res_desc = vec![0.0; b * MAX_RESOURCES * DESC_DIM];
        hb.game_idx = vec![0; b];
        hb.pi = vec![0.0; b * na];
        hb.value = vec![0.0; b * MAX_PLAYERS];
        hb.vmask = vec![0.0; b * MAX_PLAYERS];
        hb.family_idx = vec![0; b];

        for (bi, s) in samples.iter().enumerate() {
            let o = &s.obs;
            let nst = o.n_state();
            let put = |tok: &Token, cls: &mut [f32], cat: &mut [i32], sem: &mut [i32], num: &mut [f32], nm: &mut [f32], desc: &mut [f32]| {
                cls[tok.class as usize] = 1.0;
                for k in 0..NCAT {
                    cat[k] = if opts.mask_ids { 0 } else { tok.cat[k] as i32 };
                }
                for k in 0..NSEM {
                    sem[k] = tok.sem[k] as i32;
                }
                for k in 0..NNUM {
                    if tok.num_mask & (1 << k) != 0 {
                        num[k] = tok.num[k];
                        nm[k] = 1.0;
                    }
                }
                desc.copy_from_slice(&tok.desc);
            };
            for (i, tok) in o.tokens[..nst].iter().enumerate() {
                let base = bi * ns + i;
                put(
                    tok,
                    &mut hb.s_class[base * NUM_CLASSES..(base + 1) * NUM_CLASSES],
                    &mut hb.s_cat[base * NCAT..(base + 1) * NCAT],
                    &mut hb.s_sem[base * NSEM..(base + 1) * NSEM],
                    &mut hb.s_num[base * NNUM..(base + 1) * NNUM],
                    &mut hb.s_nmask[base * NNUM..(base + 1) * NNUM],
                    &mut hb.s_desc[base * DESC_DIM..(base + 1) * DESC_DIM],
                );
                hb.s_mask[base] = 1.0;
            }
            for (i, tok) in o.tokens[nst..].iter().enumerate() {
                let base = bi * na + i;
                put(
                    tok,
                    &mut hb.a_class[base * NUM_CLASSES..(base + 1) * NUM_CLASSES],
                    &mut hb.a_cat[base * NCAT..(base + 1) * NCAT],
                    &mut hb.a_sem[base * NSEM..(base + 1) * NSEM],
                    &mut hb.a_num[base * NNUM..(base + 1) * NNUM],
                    &mut hb.a_nmask[base * NNUM..(base + 1) * NNUM],
                    &mut hb.a_desc[base * DESC_DIM..(base + 1) * DESC_DIM],
                );
                hb.a_mask[base] = 1.0;
            }
            // Register relations.
            for i in 0..r {
                for j in 0..kn {
                    let id = if j < r { rel::REG_REG } else { rel::REG_TO_TOK };
                    hb.rel_ss[(bi * kn + i) * kn + j] = id as i32;
                }
                for j in r..kn {
                    // token query i' attends register j'
                    hb.rel_ss[(bi * kn + j) * kn + i] = rel::TOK_TO_REG as i32;
                }
            }
            // State relations (+ inverses).
            for rl in &o.relations {
                let (f, t) = (rl.from as usize, rl.to as usize);
                if f < nst && t < nst {
                    hb.rel_ss[(bi * kn + r + f) * kn + r + t] = rl.kind as i32;
                    hb.rel_ss[(bi * kn + r + t) * kn + r + f] = (rl.kind + rel::INV) as i32;
                } else if f >= nst && t < nst {
                    let a = f - nst;
                    hb.rel_as[(bi * na + a) * kn + r + t] = rl.kind as i32;
                    let slot = match rl.kind {
                        rel::ACT_SOURCE => Some(0),
                        rel::ACT_TARGET0 => Some(1),
                        rel::ACT_TARGET1 => Some(2),
                        rel::ACT_TARGET2 => Some(3),
                        rel::ACT_TARGET_PLAYER => Some(4),
                        _ => None,
                    };
                    if let Some(sl) = slot {
                        let li = (bi * na + a) * N_LINKS + sl;
                        if hb.a_links[li] == kn as i32 {
                            hb.a_links[li] = (r + t) as i32;
                        }
                    }
                }
            }
            // Register rows for action keys.
            for a in 0..o.n_actions() {
                for j in 0..r {
                    hb.rel_as[(bi * na + a) * kn + j] = rel::TOK_TO_REG as i32;
                }
            }
            // Time buckets among events.
            let ev = o.seg.events.clone();
            for i in ev.clone() {
                for j in ev.clone() {
                    let dt = o.tokens[j].time as i64 - o.tokens[i].time as i64;
                    hb.tb_ss[(bi * kn + r + i) * kn + r + j] = time_bucket(dt);
                }
            }
            for (i, d) in o.attr_desc.iter().enumerate() {
                hb.attr_desc[(bi * MAX_ATTRS + i) * DESC_DIM..(bi * MAX_ATTRS + i + 1) * DESC_DIM].copy_from_slice(d);
            }
            for (i, d) in o.res_desc.iter().enumerate() {
                hb.res_desc[(bi * MAX_RESOURCES + i) * DESC_DIM..(bi * MAX_RESOURCES + i + 1) * DESC_DIM].copy_from_slice(d);
            }
            hb.game_idx[bi] = s.game_idx as i32;
            hb.family_idx[bi] = s.family_idx as i32;
            for (a, p) in s.pi.iter().enumerate() {
                hb.pi[bi * na + a] = *p;
            }
            for k in 0..MAX_PLAYERS {
                hb.value[bi * MAX_PLAYERS + k] = s.value[k];
                hb.vmask[bi * MAX_PLAYERS + k] = (k < o.num_players as usize) as u8 as f32;
            }
        }
        hb
    }

    /// Total key axis length (registers + state tokens).
    pub fn kn(&self) -> usize {
        self.n_reg + self.ns
    }
}
