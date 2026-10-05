//! Mixture sampling across games: prevents games that produce more decisions
//! from silently dominating optimisation.

use omnia_engine::Rng;
use omnia_observation::batch::Sample;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MixSpec {
    /// Every game equally likely per drawn sample.
    UniformByGame,
    /// Every family equally likely, then uniform over its games.
    UniformByFamily,
    /// Proportional to dataset size.
    Proportional,
    /// Explicit weight per pool (same order as pools).
    Weighted(Vec<f32>),
}

pub struct Pool {
    pub game_idx: u32,
    pub family_idx: u32,
    pub samples: Vec<Sample>,
}

pub struct MixSampler<'a> {
    pools: &'a [Pool],
    probs: Vec<f64>,
    orders: Vec<Vec<usize>>,
    cursors: Vec<usize>,
    rng: Rng,
}

impl<'a> MixSampler<'a> {
    pub fn new(pools: &'a [Pool], spec: &MixSpec, seed: u64) -> MixSampler<'a> {
        let pools_nonempty: Vec<usize> = (0..pools.len()).filter(|i| !pools[*i].samples.is_empty()).collect();
        let mut probs = vec![0.0f64; pools.len()];
        match spec {
            MixSpec::UniformByGame => pools_nonempty.iter().for_each(|i| probs[*i] = 1.0),
            MixSpec::Proportional => pools_nonempty.iter().for_each(|i| probs[*i] = pools[*i].samples.len() as f64),
            MixSpec::Weighted(w) => pools_nonempty.iter().for_each(|i| probs[*i] = w.get(*i).copied().unwrap_or(0.0) as f64),
            MixSpec::UniformByFamily => {
                let mut fam_count = std::collections::HashMap::new();
                for i in &pools_nonempty {
                    *fam_count.entry(pools[*i].family_idx).or_insert(0usize) += 1;
                }
                for i in &pools_nonempty {
                    probs[*i] = 1.0 / fam_count[&pools[*i].family_idx] as f64;
                }
            }
        }
        let tot: f64 = probs.iter().sum();
        assert!(tot > 0.0, "empty mixture");
        probs.iter_mut().for_each(|p| *p /= tot);
        let mut rng = Rng::new(seed);
        let orders = pools
            .iter()
            .map(|p| {
                let mut o: Vec<usize> = (0..p.samples.len()).collect();
                rng.shuffle(&mut o);
                o
            })
            .collect();
        MixSampler { pools, probs, orders, cursors: vec![0; pools.len()], rng }
    }

    pub fn probabilities(&self) -> &[f64] {
        &self.probs
    }

    /// Draw `n` samples (pool chosen by the mixture, then epoch-shuffled within the pool).
    pub fn next_batch(&mut self, n: usize) -> Vec<&'a Sample> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut u = self.rng.f32() as f64;
            let mut k = self.probs.len() - 1;
            for (i, p) in self.probs.iter().enumerate() {
                if u < *p {
                    k = i;
                    break;
                }
                u -= p;
            }
            if self.probs[k] == 0.0 {
                k = self.probs.iter().position(|p| *p > 0.0).unwrap();
            }
            if self.cursors[k] >= self.orders[k].len() {
                let mut o = std::mem::take(&mut self.orders[k]);
                self.rng.shuffle(&mut o);
                self.orders[k] = o;
                self.cursors[k] = 0;
            }
            let idx = self.orders[k][self.cursors[k]];
            self.cursors[k] += 1;
            out.push(&self.pools[k].samples[idx]);
        }
        out
    }
}
