//! Generic agents: random, goal-heuristic (1-ply), and information-set MCTS.
//! Nothing here knows about any particular game.

use omnia_dsl::*;
use omnia_engine::*;
use std::collections::HashMap;

pub trait Agent: Send {
    fn name(&self) -> String;
    /// Choose an index into `st.legal_actions()`.
    fn act(&mut self, st: &mut State, rng: &mut Rng) -> usize;
}

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn name(&self) -> String {
        "random".into()
    }
    fn act(&mut self, st: &mut State, rng: &mut Rng) -> usize {
        rng.below(st.legal_actions().len() as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Goal model: which resources matter, derived from the game's terminal rules
// ---------------------------------------------------------------------------

/// Per-resource signed weights inferred from terminal rules / timeout rule.
#[derive(Clone, Debug)]
pub struct GoalModel {
    pub weights: [f32; MAX_RESOURCES],
    pub scale: [f32; MAX_RESOURCES],
}

impl GoalModel {
    pub fn from_game(def: &GameDef) -> GoalModel {
        let mut w = [0f32; MAX_RESOURCES];
        let mut sc = [1f32; MAX_RESOURCES];
        fn walk(c: &Cond, result: PlayerResult, w: &mut [f32; MAX_RESOURCES], sc: &mut [f32; MAX_RESOURCES]) {
            match c {
                Cond::And(v) | Cond::Or(v) => v.iter().for_each(|x| walk(x, result, w, sc)),
                Cond::Not(_) => {}
                Cond::Cmp(a, op, b) => {
                    if let (Expr::Res(PRef::Me, r), Expr::Const(k)) = (&**a, &**b) {
                        let up = matches!(op, CmpOp::Ge | CmpOp::Gt);
                        let good_up = match result {
                            PlayerResult::Win => up,
                            PlayerResult::Lose => !up,
                        };
                        let i = *r as usize;
                        w[i] += if good_up { 1.0 } else { -1.0 };
                        sc[i] = (k.abs() as f32).max(1.0);
                    }
                }
                _ => {}
            }
        }
        for r in &def.terminal {
            walk(&r.cond, r.result, &mut w, &mut sc);
        }
        if let Timeout::ByResource(r) = def.timeout {
            if w[r as usize] == 0.0 {
                w[r as usize] = 1.0;
            }
        }
        GoalModel { weights: w, scale: sc }
    }

    /// Score of `p` minus mean score of the others, read from true state.
    pub fn margin(&self, st: &State, p: u8) -> f32 {
        let n = st.num_players() as usize;
        let score = |q: usize| -> f32 { (0..MAX_RESOURCES).map(|r| self.weights[r] * st.resource(q as u8, r) as f32 / self.scale[r]).sum() };
        let me = score(p as usize);
        let others: f32 = (0..n).filter(|q| *q != p as usize).map(score).sum::<f32>() / (n - 1) as f32;
        me - others
    }
}

pub fn squash(x: f32) -> f32 {
    (x * 0.5).tanh()
}

// ---------------------------------------------------------------------------
// 1-ply heuristic
// ---------------------------------------------------------------------------

pub struct HeuristicAgent {
    goal: GoalModel,
    pub epsilon: f32,
}

impl HeuristicAgent {
    pub fn new(def: &GameDef) -> HeuristicAgent {
        HeuristicAgent { goal: GoalModel::from_game(def), epsilon: 0.05 }
    }
}

impl Agent for HeuristicAgent {
    fn name(&self) -> String {
        "heuristic".into()
    }
    fn act(&mut self, st: &mut State, rng: &mut Rng) -> usize {
        let n = st.legal_actions().len();
        if rng.f32() < self.epsilon {
            return rng.below(n as u64) as usize;
        }
        let me = st.decision_maker().unwrap();
        let mut best = vec![];
        let mut best_v = f32::NEG_INFINITY;
        for i in 0..n {
            let mut c = st.clone();
            c.apply(i);
            let v = match c.payoffs() {
                Some(p) => p[me as usize] * 100.0,
                None => self.goal.margin(&c, me),
            };
            if v > best_v + 1e-6 {
                best_v = v;
                best.clear();
                best.push(i);
            } else if (v - best_v).abs() <= 1e-6 {
                best.push(i);
            }
        }
        best[rng.below(best.len() as u64) as usize]
    }
}

// ---------------------------------------------------------------------------
// Evaluators
// ---------------------------------------------------------------------------

pub struct Eval {
    /// Optional policy prior aligned with `st.legal_actions()`.
    pub priors: Option<Vec<f32>>,
    /// Estimated payoff per absolute seat in [-1, 1].
    pub values: [f32; MAX_PLAYERS],
}

pub trait Evaluator: Send {
    fn evaluate(&mut self, st: &mut State, rng: &mut Rng) -> Eval;
}

/// Uniform-random rollouts with a goal-margin cutoff.
pub struct RolloutEval {
    pub goal: GoalModel,
    pub max_steps: u32,
}

impl RolloutEval {
    pub fn new(def: &GameDef) -> RolloutEval {
        RolloutEval { goal: GoalModel::from_game(def), max_steps: 120 }
    }
}

impl Evaluator for RolloutEval {
    fn evaluate(&mut self, st: &mut State, rng: &mut Rng) -> Eval {
        let mut s = st.clone();
        let mut steps = 0;
        while !s.is_terminal() && steps < self.max_steps {
            let n = s.legal_actions().len();
            s.apply(rng.below(n as u64) as usize);
            steps += 1;
        }
        let mut values = [0f32; MAX_PLAYERS];
        match s.payoffs() {
            Some(p) => values = p,
            None => {
                for p in 0..s.num_players() {
                    values[p as usize] = squash(self.goal.margin(&s, p));
                }
            }
        }
        Eval { priors: None, values }
    }
}

// ---------------------------------------------------------------------------
// Single-observer information-set MCTS (determinized) with optional priors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct MctsConfig {
    pub sims: u32,
    pub c_explore: f32,
    /// Resample hidden information at every simulation.
    pub determinize: bool,
    /// Dirichlet-free root exploration noise mix (0 = off).
    pub root_noise: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig { sims: 200, c_explore: 1.2, determinize: true, root_noise: 0.0 }
    }
}

struct Edge {
    action: Action,
    child: usize,
    visits: f32,
    avail: f32,
    total: f32,
    prior: f32,
}

struct Node {
    edges: Vec<Edge>,
}

pub struct MctsResult {
    /// Visit fraction per root legal action (aligned with the legal list).
    pub policy: Vec<f32>,
    /// Mean value for the root player.
    pub value: f32,
    pub best: usize,
}

pub struct Mcts<E: Evaluator> {
    pub cfg: MctsConfig,
    pub eval: E,
}

impl<E: Evaluator> Mcts<E> {
    pub fn new(cfg: MctsConfig, eval: E) -> Self {
        Mcts { cfg, eval }
    }

    pub fn search(&mut self, root: &mut State, rng: &mut Rng) -> MctsResult {
        let me = root.decision_maker().expect("search at a decision point");
        let n_root = root.legal_actions().len();
        if n_root == 1 {
            return MctsResult { policy: vec![1.0], value: 0.0, best: 0 };
        }
        let mut nodes: Vec<Node> = vec![Node { edges: vec![] }];
        let mut root_value = 0.0f32;
        let mut root_n = 0.0f32;
        for _ in 0..self.cfg.sims {
            let mut st = if self.cfg.determinize { root.determinize(me, rng) } else { root.clone() };
            // Path of (node, edge) pairs for backprop; remember decider at each.
            let mut path: Vec<(usize, usize, u8)> = vec![];
            let mut node = 0usize;
            let mut leaf_eval: Option<Eval> = None;
            loop {
                if st.is_terminal() {
                    break;
                }
                let decider = st.decision_maker().unwrap();
                let legal: Vec<Action> = st.legal_actions().to_vec();
                // Edges available in this determinization.
                let mut avail_idx = vec![];
                for (ei, e) in nodes[node].edges.iter().enumerate() {
                    if legal.contains(&e.action) {
                        avail_idx.push(ei);
                    }
                }
                // Expansion: any legal action without an edge?
                let untried: Vec<usize> = (0..legal.len()).filter(|li| !nodes[node].edges.iter().any(|e| e.action == legal[*li])).collect();
                if !untried.is_empty() {
                    // Evaluate node once to get priors for new edges.
                    let ev = self.eval.evaluate(&mut st, rng);
                    let k = untried[rng.below(untried.len() as u64) as usize];
                    for &li in &untried {
                        let prior = ev.priors.as_ref().map(|p| p[li]).unwrap_or(0.0);
                        let child = nodes.len();
                        nodes.push(Node { edges: vec![] });
                        nodes[node].edges.push(Edge { action: legal[li].clone(), child, visits: 0.0, avail: 0.0, total: 0.0, prior });
                    }
                    // Choose the sampled untried edge (or highest prior when available).
                    let ei = if ev.priors.is_some() {
                        let mut best = untried[0];
                        for &li in &untried {
                            if ev.priors.as_ref().unwrap()[li] > ev.priors.as_ref().unwrap()[best] {
                                best = li;
                            }
                        }
                        nodes[node].edges.iter().position(|e| e.action == legal[best]).unwrap()
                    } else {
                        nodes[node].edges.iter().position(|e| e.action == legal[k]).unwrap()
                    };
                    for ai in nodes[node].edges.iter_mut() {
                        if legal.contains(&ai.action) {
                            ai.avail += 1.0;
                        }
                    }
                    path.push((node, ei, decider));
                    let a = nodes[node].edges[ei].action.clone();
                    let li = st.find_action(&a).unwrap();
                    st.apply(li);
                    leaf_eval = Some(if st.is_terminal() { Eval { priors: None, values: st.payoffs().unwrap() } } else { self.eval.evaluate(&mut st, rng) });
                    break;
                }
                // Selection (UCB1 / PUCT with availability counts).
                let total_avail: f32 = avail_idx.iter().map(|i| nodes[node].edges[*i].avail.max(1.0)).sum::<f32>().max(1.0);
                let mut best = avail_idx[0];
                let mut best_s = f32::NEG_INFINITY;
                let has_priors = avail_idx.iter().any(|i| nodes[node].edges[*i].prior > 0.0);
                for &ei in &avail_idx {
                    let e = &nodes[node].edges[ei];
                    let q = if e.visits > 0.0 { e.total / e.visits } else { 0.0 };
                    let s = if has_priors {
                        q + self.cfg.c_explore * e.prior * total_avail.sqrt() / (1.0 + e.visits)
                    } else {
                        q + self.cfg.c_explore * (e.avail.max(1.0).ln() / e.visits.max(1e-6)).sqrt()
                    };
                    if s > best_s {
                        best_s = s;
                        best = ei;
                    }
                }
                for &ei in &avail_idx {
                    nodes[node].edges[ei].avail += 1.0;
                }
                path.push((node, best, decider));
                let a = nodes[node].edges[best].action.clone();
                let child = nodes[node].edges[best].child;
                let li = st.find_action(&a).unwrap();
                st.apply(li);
                node = child;
            }
            let values = match leaf_eval {
                Some(e) => e.values,
                None => st.payoffs().unwrap_or([0.0; MAX_PLAYERS]),
            };
            for (nd, ei, decider) in path {
                let e = &mut nodes[nd].edges[ei];
                e.visits += 1.0;
                e.total += values[decider as usize];
            }
            root_value += values[me as usize];
            root_n += 1.0;
        }
        // Visit distribution over the root's *true* legal actions.
        let legal: Vec<Action> = root.legal_actions().to_vec();
        let mut visits = vec![0f32; legal.len()];
        for e in &nodes[0].edges {
            if let Some(i) = legal.iter().position(|a| *a == e.action) {
                visits[i] += e.visits;
            }
        }
        let tot: f32 = visits.iter().sum();
        let best = visits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|x| x.0).unwrap_or(0);
        let policy = if tot > 0.0 { visits.iter().map(|v| v / tot).collect() } else { vec![1.0 / legal.len() as f32; legal.len()] };
        MctsResult { policy, value: root_value / root_n.max(1.0), best }
    }
}

pub struct MctsAgent<E: Evaluator> {
    pub mcts: Mcts<E>,
    pub label: String,
    pub temperature: f32,
}

impl MctsAgent<RolloutEval> {
    pub fn rollout(def: &GameDef, sims: u32) -> Self {
        MctsAgent { mcts: Mcts::new(MctsConfig { sims, ..Default::default() }, RolloutEval::new(def)), label: format!("mcts{}", sims), temperature: 0.0 }
    }
}

impl<E: Evaluator> Agent for MctsAgent<E> {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn act(&mut self, st: &mut State, rng: &mut Rng) -> usize {
        let r = self.mcts.search(st, rng);
        if self.temperature <= 0.0 {
            r.best
        } else {
            sample_with_temperature(&r.policy, self.temperature, rng)
        }
    }
}

pub fn sample_with_temperature(p: &[f32], t: f32, rng: &mut Rng) -> usize {
    let w: Vec<f32> = p.iter().map(|x| x.max(1e-9).powf(1.0 / t)).collect();
    let s: f32 = w.iter().sum();
    let mut u = rng.f32() * s;
    for (i, x) in w.iter().enumerate() {
        if u < *x {
            return i;
        }
        u -= x;
    }
    w.len() - 1
}

/// Play one game; returns final payoffs per absolute seat.
pub fn play_game(game: &std::sync::Arc<Game>, agents: &mut [Box<dyn Agent>], seed: u64) -> ([f32; MAX_PLAYERS], u32) {
    let mut st = State::new(game, seed);
    let mut rng = Rng::new(seed ^ 0x5EED);
    while !st.is_terminal() {
        let p = st.decision_maker().unwrap() as usize;
        let i = agents[p].act(&mut st, &mut rng);
        st.apply(i);
    }
    (st.payoffs().unwrap(), st.decisions())
}

#[allow(dead_code)]
type _Unused = HashMap<(), ()>;
