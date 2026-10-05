//! OmniGen: procedural generation of valid, strategically structured games.
//!
//! Games are produced by *families*: parameterised generators that share an
//! underlying strategic structure (e.g. "renewable energy + creature tempo",
//! "invest in income vs cash out for points") while varying numbers, object
//! composition, rule toggles and arbitrary ids. Families also expose controlled
//! variants used as experimental controls (re-skin, random reward, altered
//! victory, inverted cost curve).
//!
//! This crate is *tooling*: the engine and model never see family names. A
//! generated game is an ordinary [`GameDef`] with generator metadata in
//! `meta.tags` (labels, not measurements — measurements live in [`GameStats`]).

pub mod build;
mod engine;
mod garden;
mod stack;
mod tempo;
mod tug;

use omnia_dsl::*;
use omnia_engine::{Game, Rng, State};
use omnia_search::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Variant {
    Normal,
    /// Same entities and mechanics, different victory condition.
    AltVictory,
    /// Cost/reward curve inverted: cheap things are the efficient ones.
    Inverted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Family {
    Tempo,
    Engine,
    Stack,
    Tug,
    Garden,
}

impl Family {
    pub const ALL: [Family; 5] = [Family::Tempo, Family::Engine, Family::Stack, Family::Tug, Family::Garden];
    pub fn name(&self) -> &'static str {
        match self {
            Family::Tempo => "tempo",
            Family::Engine => "engine",
            Family::Stack => "stack",
            Family::Tug => "tug",
            Family::Garden => "garden",
        }
    }
    pub fn parse(s: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.name() == s)
    }
    /// Whether the family implements the given control variant.
    pub fn supports(&self, v: Variant) -> bool {
        !(matches!(self, Family::Tug) && v == Variant::AltVictory)
    }
    /// The abstract strategic tradeoff this family is built around.
    pub fn tradeoff(&self) -> &'static str {
        match self {
            Family::Tempo => "tempo_vs_value",
            Family::Engine | Family::Garden => "invest_vs_cash",
            Family::Stack => "counter_vs_commit",
            Family::Tug => "push_vs_anchor",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenSpec {
    pub family: Family,
    pub seed: u64,
    pub variant: Variant,
    /// Randomly permute all ids (resources, attrs, zones, templates, hooks,
    /// action order, class labels) so no two games share accidental slot
    /// semantics.
    pub relabel: bool,
    /// Control: outcome independent of play (see `Adjudication::Random`).
    pub random_reward: bool,
}

impl GenSpec {
    pub fn new(family: Family, seed: u64) -> GenSpec {
        GenSpec { family, seed, variant: Variant::Normal, relabel: true, random_reward: false }
    }
}

pub fn generate(spec: &GenSpec) -> GameDef {
    let salt = match spec.family {
        Family::Tempo => 0x11,
        Family::Engine => 0x22,
        Family::Stack => 0x33,
        Family::Tug => 0x44,
        Family::Garden => 0x55,
    };
    let mut rng = Rng::new(spec.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt);
    let mut def = match spec.family {
        Family::Tempo => tempo::generate(&mut rng, spec.variant),
        Family::Engine => engine::generate(&mut rng, spec.variant),
        Family::Stack => stack::generate(&mut rng, spec.variant),
        Family::Tug => tug::generate(&mut rng, spec.variant),
        Family::Garden => garden::generate(&mut rng, spec.variant),
    };
    def.meta.seed = spec.seed;
    def.meta.tags.push(("variant".into(), format!("{:?}", spec.variant).to_lowercase()));
    if spec.relabel {
        let mut rr = Rng::new(spec.seed ^ 0xABCD_EF01);
        def = Relabel::random(&def, |n| rr.below(n)).apply(&def);
        def.meta.tags.push(("ids".into(), "relabelled".into()));
    } else {
        def.meta.tags.push(("ids".into(), "canonical".into()));
    }
    if spec.random_reward {
        def.adjudication = Adjudication::Random;
        def.meta.tags.push(("control".into(), "random_reward".into()));
    }
    def.name = format!("{}-{:?}-{}{}", spec.family.name(), spec.variant, spec.seed, if spec.random_reward { "-rr" } else { "" }).to_lowercase();
    def
}

/// Mechanically identical game with all ids re-drawn (re-skin control).
pub fn reskin(def: &GameDef, seed: u64) -> GameDef {
    let mut rr = Rng::new(seed ^ 0x5EED_5EED);
    let mut out = Relabel::random(def, |n| rr.below(n)).apply(def);
    out.name = format!("{}-reskin{}", def.name, seed);
    out.meta.tags.push(("control".into(), "reskin".into()));
    out
}

/// Same game, outcome replaced by a coin flip (random-reward control).
pub fn random_reward(def: &GameDef) -> GameDef {
    let mut out = def.clone();
    out.adjudication = Adjudication::Random;
    out.name = format!("{}-rr", def.name);
    out.meta.tags.push(("control".into(), "random_reward".into()));
    out
}

// ---------------------------------------------------------------------------
// Validation and measured statistics
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GameStats {
    pub random_games: u32,
    pub avg_decisions: f32,
    pub avg_branching: f32,
    pub p0_win_rate: f32,
    pub p1_win_rate: f32,
    pub draw_rate: f32,
    pub timeout_rate: f32,
    pub fault_rate: f32,
    /// Win score (win=1, draw=.5) of the 1-ply goal heuristic vs random.
    pub heuristic_vs_random: Option<f32>,
    /// Win score of low-budget MCTS vs random: a coarse "skill depth" proxy.
    pub mcts_vs_random: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct ValidationConfig {
    pub random_games: u32,
    pub skill_games: u32,
    pub mcts_sims: u32,
    pub min_decisions: f32,
    pub max_timeout_rate: f32,
    pub min_side_win_rate: f32,
    pub min_skill: f32,
    pub check_skill: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        ValidationConfig { random_games: 60, skill_games: 10, mcts_sims: 24, min_decisions: 8.0, max_timeout_rate: 0.3, min_side_win_rate: 0.15, min_skill: 0.6, check_skill: true }
    }
}

fn score(p: f32) -> f32 {
    (p + 1.0) / 2.0
}

pub fn measure(def: &GameDef, cfg: &ValidationConfig) -> Result<GameStats, String> {
    def.check().map_err(|e| e.to_string())?;
    let game = Game::new(def.clone()).map_err(|e| e.to_string())?;
    let mut st_ = GameStats { random_games: cfg.random_games, ..Default::default() };
    let (mut dec, mut branch, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let (mut w0, mut w1, mut dr, mut to, mut fa) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for seed in 0..cfg.random_games as u64 {
        let mut rng = Rng::new(seed ^ 0xBEEF);
        let mut st = State::new(&game, 1000 + seed);
        while !st.is_terminal() {
            let n = st.legal_actions().len();
            if n == 0 {
                return Err("non-terminal state with no legal actions".into());
            }
            branch += n as f64;
            nb += 1.0;
            st.apply(rng.below(n as u64) as usize);
        }
        let o = st.outcome().unwrap();
        match o.reason {
            omnia_engine::EndReason::Fault(f) => {
                fa += 1;
                if fa > 0 {
                    return Err(format!("engine fault {:?}", f));
                }
            }
            omnia_engine::EndReason::Timeout => to += 1,
            _ => {}
        }
        dec += st.decisions() as f64;
        if o.payoffs[0] > 0.0 {
            w0 += 1
        } else if o.payoffs[1] > 0.0 {
            w1 += 1
        } else {
            dr += 1
        }
    }
    let n = cfg.random_games as f32;
    st_.avg_decisions = (dec / n as f64) as f32;
    st_.avg_branching = (branch / nb.max(1.0)) as f32;
    st_.p0_win_rate = w0 as f32 / n;
    st_.p1_win_rate = w1 as f32 / n;
    st_.draw_rate = dr as f32 / n;
    st_.timeout_rate = to as f32 / n;
    st_.fault_rate = fa as f32 / n;

    if st_.avg_decisions < cfg.min_decisions {
        return Err(format!("games too short (avg {:.1} decisions)", st_.avg_decisions));
    }
    if st_.timeout_rate > cfg.max_timeout_rate {
        return Err(format!("too many timeouts ({:.2})", st_.timeout_rate));
    }
    let random_outcome = def.adjudication == Adjudication::Random;
    if !random_outcome && (st_.p0_win_rate < cfg.min_side_win_rate || st_.p1_win_rate < cfg.min_side_win_rate) {
        return Err(format!("degenerate side balance under random play (p0 {:.2} p1 {:.2})", st_.p0_win_rate, st_.p1_win_rate));
    }
    if cfg.check_skill && !random_outcome {
        let mut mc = 0.0;
        let mut he = 0.0;
        for s in 0..cfg.skill_games as u64 {
            let seat = (s % 2) as usize;
            let mk = |agent: Box<dyn Agent>| -> Vec<Box<dyn Agent>> {
                if seat == 0 {
                    vec![agent, Box::new(RandomAgent)]
                } else {
                    vec![Box::new(RandomAgent), agent]
                }
            };
            let (p, _) = play_game(&game, &mut mk(Box::new(MctsAgent::rollout(def, cfg.mcts_sims))), 5000 + s);
            mc += score(p[seat]);
            let (p, _) = play_game(&game, &mut mk(Box::new(HeuristicAgent::new(def))), 6000 + s);
            he += score(p[seat]);
        }
        st_.mcts_vs_random = Some(mc / cfg.skill_games as f32);
        st_.heuristic_vs_random = Some(he / cfg.skill_games as f32);
        if mc / (cfg.skill_games as f32) < cfg.min_skill {
            return Err(format!("no skill gradient (mcts vs random {:.2})", mc / cfg.skill_games as f32));
        }
    }
    Ok(st_)
}

/// Generate with rejection sampling: tries successive derived seeds until a
/// game passes validation. Returns the accepted spec (with its final seed),
/// the game and its measured statistics.
pub fn generate_valid(spec: &GenSpec, cfg: &ValidationConfig, max_tries: u32) -> Result<(GenSpec, GameDef, GameStats, u32), String> {
    let mut last = String::new();
    for k in 0..max_tries {
        let mut s = spec.clone();
        s.seed = spec.seed.wrapping_add(k as u64 * 1_000_003);
        let def = generate(&s);
        match measure(&def, cfg) {
            Ok(stats) => return Ok((s, def, stats, k + 1)),
            Err(e) => last = e,
        }
    }
    Err(format!("no valid {} game after {} tries (last: {})", spec.family.name(), max_tries, last))
}

// ---------------------------------------------------------------------------
// Game sets
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameEntry {
    pub id: String,
    pub family: String,
    pub tradeoff: String,
    pub variant: String,
    pub relabel: bool,
    pub random_reward: bool,
    pub seed: u64,
    pub tries: u32,
    pub hash: String,
    pub stats: GameStats,
    pub path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GameSet {
    pub name: String,
    pub games: Vec<GameEntry>,
}

/// Generate `count` valid games of a family into `dir` (RON files) and return
/// manifest entries. Deterministic given `base_seed`.
pub fn generate_set(dir: &std::path::Path, family: Family, variant: Variant, relabel: bool, random_reward_flag: bool, base_seed: u64, count: usize, cfg: &ValidationConfig) -> Vec<GameEntry> {
    std::fs::create_dir_all(dir).expect("create dir");
    let results: Vec<_> = (0..count)
        .into_par_iter()
        .filter_map(|i| {
            let spec = GenSpec { family, seed: base_seed + i as u64 * 7919, variant, relabel, random_reward: false };
            // Validate the honest game first; the random-reward control is derived from it.
            let (s, def, stats, tries) = generate_valid(&spec, cfg, 40).ok()?;
            let def = if random_reward_flag { random_reward(&def) } else { def };
            Some((s, def, stats, tries))
        })
        .collect();
    let mut entries = vec![];
    for (s, def, stats, tries) in results {
        let path = dir.join(format!("{}.ron", def.name));
        std::fs::write(&path, def.to_ron()).expect("write game");
        entries.push(GameEntry {
            id: def.name.clone(),
            family: family.name().into(),
            tradeoff: family.tradeoff().into(),
            variant: format!("{:?}", variant).to_lowercase(),
            relabel,
            random_reward: random_reward_flag,
            seed: s.seed,
            tries,
            hash: def.hash_hex(),
            stats,
            path: path.to_string_lossy().into(),
        });
    }
    entries.sort_by(|a, b| a.id.cmp(&b.id));
    entries
}

pub fn load_game(path: &std::path::Path) -> Result<Arc<Game>, String> {
    let src = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let def = GameDef::from_ron(&src).map_err(|e| format!("{}: {}", path.display(), e))?;
    Game::new(def).map_err(|e| e.to_string())
}
