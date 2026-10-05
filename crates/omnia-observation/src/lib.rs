//! Canonical, game-independent, variable-length ML observation.
//!
//! Built exclusively from [`PlayerView`] (the information boundary), never
//! from simulator internals. Token counts are variable; padding is a concern
//! of the batcher only.
//!
//! Layout (one flat token list, indices are global within the observation):
//!
//! ```text
//! [ global | players | zones | objects | events ]  [ actions ]
//!   \________________ state tokens _____________/    query tokens
//! ```
//!
//! Relations are typed directed edges between tokens; the model adds inverse
//! types and turns them into attention biases.

use omnia_dsl::*;
use omnia_engine::*;
use serde::{Deserialize, Serialize};
use std::ops::Range;

pub const NCAT: usize = 6;
pub const NNUM: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TokClass {
    Global = 0,
    Player = 1,
    Zone = 2,
    Object = 3,
    Event = 4,
    Action = 5,
}
pub const NUM_CLASSES: usize = 6;

/// Relation types (directed, from -> to). 0 means "none".
pub mod rel {
    pub const OWNED_BY: u8 = 1; // object -> player
    pub const CONTROLLED_BY: u8 = 2; // object -> player
    pub const LOCATED_IN: u8 = 3; // object -> zone
    pub const ZONE_OWNER: u8 = 4; // zone -> player
    pub const ATTACHED_TO: u8 = 5; // object -> object
    pub const EV_SOURCE: u8 = 6; // event -> object
    pub const EV_TARGET0: u8 = 7;
    pub const EV_TARGET1: u8 = 8;
    pub const EV_ACTOR: u8 = 9; // event -> player
    pub const EV_FROM: u8 = 10; // event -> zone
    pub const EV_TO: u8 = 11; // event -> zone
    pub const CAUSED_BY: u8 = 12; // event -> event
    pub const ACT_SOURCE: u8 = 13; // action -> object
    pub const ACT_TARGET0: u8 = 14;
    pub const ACT_TARGET1: u8 = 15;
    pub const ACT_TARGET2: u8 = 16;
    pub const ACT_TARGET_PLAYER: u8 = 17; // action -> player
    pub const STACK_ITEM: u8 = 18; // global -> object (source of a pending stack item)
    pub const NUM_BASE: usize = 19;
    /// Total relation ids including inverses (0 = none).
    pub const NUM_TOTAL: usize = 1 + 2 * (NUM_BASE - 1) + 1;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Token {
    pub class: TokClass,
    /// Categorical features; 0 = absent/padding.
    pub cat: [u16; NCAT],
    /// Numeric features (raw, unscaled); valid where `num_mask` bit is set.
    pub num: [f32; NNUM],
    pub num_mask: u16,
    /// Static rule-derived descriptor of the thing this token represents.
    pub desc: [f32; DESC_DIM],
    /// Position in visible event order (events only).
    pub time: u32,
}

impl Token {
    fn new(class: TokClass) -> Token {
        Token { class, cat: [0; NCAT], num: [0.0; NNUM], num_mask: 0, desc: [0.0; DESC_DIM], time: 0 }
    }
    fn set(&mut self, slot: usize, v: f32) {
        self.num[slot] = v;
        self.num_mask |= 1 << slot;
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Relation {
    pub from: u16,
    pub to: u16,
    pub kind: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segments {
    pub global: usize,
    pub players: Range<usize>,
    pub zones: Range<usize>,
    pub objects: Range<usize>,
    pub events: Range<usize>,
    pub actions: Range<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub game_hash: u64,
    pub num_players: u8,
    /// All tokens: state tokens first, then one query token per legal action.
    pub tokens: Vec<Token>,
    pub relations: Vec<Relation>,
    pub seg: Segments,
    pub history_total: usize,
    pub history_truncated: bool,
    /// Static rule-derived descriptors of each numeric slot: object attribute
    /// slots 0..MAX_ATTRS and player resource slots 0..MAX_RESOURCES. Zero for
    /// undefined slots. Lets a model ground otherwise arbitrary slot indices.
    pub attr_desc: Vec<[f32; DESC_DIM]>,
    pub res_desc: Vec<[f32; DESC_DIM]>,
}

impl Observation {
    pub fn n_state(&self) -> usize {
        self.seg.events.end
    }
    pub fn n_actions(&self) -> usize {
        self.seg.actions.len()
    }
    pub fn state_tokens(&self) -> &[Token] {
        &self.tokens[..self.n_state()]
    }
    pub fn action_tokens(&self) -> &[Token] {
        &self.tokens[self.seg.actions.clone()]
    }

    /// Structural self-check used by tests.
    pub fn check(&self) -> Result<(), String> {
        let n = self.tokens.len();
        for r in &self.relations {
            if r.from as usize >= n || r.to as usize >= n {
                return Err(format!("relation out of range: {:?}", r));
            }
        }
        if self.seg.actions.end != n {
            return Err("action segment must end the token list".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    /// Most recent visible events included (older ones are truncated; the
    /// interface allows a future hierarchical summary without API change).
    pub max_history: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        TokenizerConfig { max_history: 96 }
    }
}

fn event_kind_id(k: EventKind) -> u16 {
    match k {
        EventKind::TurnStart => 1,
        EventKind::TurnEnd => 2,
        EventKind::PhaseStart => 3,
        EventKind::PhaseEnd => 4,
        EventKind::ActionTaken => 5,
        EventKind::Moved => 6,
        EventKind::Created => 7,
        EventKind::Destroyed => 8,
        EventKind::ResourceChanged => 9,
        EventKind::AttrChanged => 10,
        EventKind::Revealed => 11,
        EventKind::Attached => 12,
        EventKind::Shuffled => 13,
        EventKind::ZoneEmpty => 14,
        EventKind::StackResolved => 15,
        EventKind::StackCancelled => 16,
        EventKind::GameEnd => 17,
        EventKind::Custom(k) => 18 + (k as u16).min(13),
    }
}
pub const NUM_EVENT_KINDS: usize = 32;

fn clampu(v: i32, hi: i32) -> u16 {
    v.clamp(0, hi) as u16
}

pub struct Tokenizer {
    pub cfg: TokenizerConfig,
}

impl Tokenizer {
    pub fn new(cfg: TokenizerConfig) -> Tokenizer {
        Tokenizer { cfg }
    }

    /// Observe `st` from the point of view of `player`.
    pub fn observe(&self, st: &mut State, player: u8) -> Observation {
        let game = st.game().clone();
        let view = st.view(player);
        self.from_view(&view, &game)
    }

    pub fn from_view(&self, v: &PlayerView, game: &Game) -> Observation {
        let d = &game.desc;
        let n = v.num_players as usize;
        let mut toks: Vec<Token> = vec![];
        let mut rels: Vec<Relation> = vec![];
        let mut add_rel = |from: usize, to: usize, kind: u8| rels.push(Relation { from: from as u16, to: to as u16, kind });

        // Global.
        let mut g = Token::new(TokClass::Global);
        g.cat[0] = 1;
        g.cat[1] = v.phase as u16 + 1;
        g.cat[2] = v.active_rel as u16 + 1;
        g.cat[3] = if v.decider_rel == NO_REL { 0 } else { v.decider_rel as u16 + 1 };
        g.set(0, v.turn as f32);
        g.set(1, v.stack.len() as f32);
        g.set(2, n as f32);
        for (i, x) in v.vars.iter().take(game.def.vars.len()).enumerate() {
            g.set(3 + i, *x as f32);
        }
        toks.push(g);

        // Players.
        let p0 = toks.len();
        for p in &v.players {
            let mut t = Token::new(TokClass::Player);
            t.cat[0] = p.rel as u16 + 1;
            for (i, r) in p.resources.iter().enumerate().take(game.def.resources.len()) {
                if let Some(x) = r {
                    t.set(i, *x as f32);
                }
            }
            t.set(6, (v.decider_rel == p.rel) as u8 as f32);
            t.set(7, (v.active_rel == p.rel) as u8 as f32);
            toks.push(t);
        }
        let p1 = toks.len();

        // Zones (one token per zone instance, in view order).
        let z0 = toks.len();
        for (zi, z) in v.zones.iter().enumerate() {
            let mut t = Token::new(TokClass::Zone);
            t.cat[0] = z.def as u16 + 1;
            t.cat[2] = if z.owner_rel == NO_REL { 0 } else { z.owner_rel as u16 + 1 };
            t.set(0, z.size as f32);
            t.set(1, z.visible.len() as f32);
            t.desc = d.zones[z.def as usize];
            toks.push(t);
            if z.owner_rel != NO_REL {
                add_rel(z0 + zi, p0 + z.owner_rel as usize, rel::ZONE_OWNER);
            }
        }
        let z1 = toks.len();
        let zone_tok = |def: u8, owner_rel: u8| -> Option<usize> {
            v.zones.iter().position(|z| z.def == def && z.owner_rel == owner_rel).map(|i| z0 + i)
        };

        // Objects.
        let o0 = toks.len();
        for (oi, o) in v.objects.iter().enumerate() {
            let mut t = Token::new(TokClass::Object);
            t.cat[0] = o.template + 1;
            t.cat[1] = o.kind as u16 + 1;
            t.cat[2] = o.owner_rel as u16 + 1;
            t.cat[3] = o.controller_rel as u16 + 1;
            let zdef = v.zones[o.zone as usize].def;
            t.cat[4] = zdef as u16 + 1;
            for a in 0..game.def.attrs.len() {
                t.set(a, o.attrs[a] as f32);
            }
            if game.def.zones[zdef as usize].ordered {
                t.set(12, o.pos as f32);
            }
            t.set(13, v.zones[o.zone as usize].size as f32);
            t.desc = d.templates[o.template as usize];
            toks.push(t);
            add_rel(o0 + oi, p0 + o.owner_rel as usize, rel::OWNED_BY);
            add_rel(o0 + oi, p0 + o.controller_rel as usize, rel::CONTROLLED_BY);
            add_rel(o0 + oi, z0 + o.zone as usize, rel::LOCATED_IN);
            if let Some(par) = o.attached_to {
                add_rel(o0 + oi, o0 + par as usize, rel::ATTACHED_TO);
            }
        }
        let o1 = toks.len();
        for (i, s) in v.stack.iter().enumerate() {
            let _ = i;
            if let Some(src) = s.source {
                add_rel(0, o0 + src as usize, rel::STACK_ITEM);
            }
        }

        // History (most recent `max_history`).
        let total = v.history.len();
        let start = total.saturating_sub(self.cfg.max_history);
        let e0 = toks.len();
        let seq_to_tok: std::collections::HashMap<u32, usize> = v.history[start..].iter().enumerate().map(|(i, e)| (e.seq, e0 + i)).collect();
        for (i, e) in v.history[start..].iter().enumerate() {
            let mut t = Token::new(TokClass::Event);
            t.time = i as u32;
            t.cat[0] = event_kind_id(e.kind);
            t.cat[1] = clampu(e.slot + 1, 63);
            t.cat[2] = if e.actor_rel == NO_REL { 0 } else { e.actor_rel as u16 + 1 };
            t.cat[3] = e.source.template.map(|x| x + 1).unwrap_or(0);
            t.cat[4] = e.targets[0].template.map(|x| x + 1).unwrap_or(0);
            t.cat[5] = e.targets[1].template.map(|x| x + 1).unwrap_or(0);
            if let Some(b) = e.before {
                t.set(0, b as f32);
            }
            if let Some(a) = e.after {
                t.set(1, a as f32);
            }
            if let (Some(b), Some(a)) = (e.before, e.after) {
                t.set(5, (a - b) as f32);
            }
            t.set(2, v.turn.saturating_sub(e.turn) as f32);
            t.set(3, (total - start - i) as f32);
            t.set(4, e.phase as f32);
            // Rule-derived descriptors for slot-bearing events.
            match e.kind {
                EventKind::ResourceChanged => {
                    if let Some(x) = d.resources.get(e.slot as usize) {
                        t.desc = *x;
                    }
                }
                EventKind::AttrChanged => {
                    if let Some(x) = d.attrs.get(e.slot as usize) {
                        t.desc = *x;
                    }
                }
                EventKind::ActionTaken | EventKind::StackResolved | EventKind::StackCancelled => {
                    if let Some(x) = d.actions.get(e.slot as usize) {
                        t.desc = *x;
                    }
                }
                _ => {}
            }
            toks.push(t);
            let me = e0 + i;
            if e.actor_rel != NO_REL && (e.actor_rel as usize) < n {
                add_rel(me, p0 + e.actor_rel as usize, rel::EV_ACTOR);
            }
            if let Some(l) = e.source.local {
                add_rel(me, o0 + l as usize, rel::EV_SOURCE);
            }
            if let Some(l) = e.targets[0].local {
                add_rel(me, o0 + l as usize, rel::EV_TARGET0);
            }
            if let Some(l) = e.targets[1].local {
                add_rel(me, o0 + l as usize, rel::EV_TARGET1);
            }
            if let Some((zd, zo)) = e.from_zone {
                if let Some(zt) = zone_tok(zd, zo) {
                    add_rel(me, zt, rel::EV_FROM);
                }
            }
            if let Some((zd, zo)) = e.to_zone {
                if let Some(zt) = zone_tok(zd, zo) {
                    add_rel(me, zt, rel::EV_TO);
                }
            }
            if let Some(p) = e.parent {
                if let Some(pt) = seq_to_tok.get(&p) {
                    add_rel(me, *pt, rel::CAUSED_BY);
                }
            }
        }
        let e1 = toks.len();

        // Action queries.
        let a0 = toks.len();
        for (ai, a) in v.actions.iter().enumerate() {
            let mut t = Token::new(TokClass::Action);
            let me = a0 + ai;
            if a.def == PASS_DEF {
                t.cat[0] = 1;
            } else {
                let def = &game.def.actions[a.def as usize];
                t.cat[0] = a.def + 2;
                t.cat[1] = def.class as u16 + 1;
                t.desc = d.actions[a.def as usize];
            }
            if let Some(s) = a.source {
                t.cat[2] = v.objects[s as usize].template + 1;
                add_rel(me, o0 + s as usize, rel::ACT_SOURCE);
            }
            t.cat[3] = a.targets.len() as u16 + 1;
            let mut numi = 6;
            for (k, tg) in a.targets.iter().enumerate() {
                match tg {
                    TargetView::Obj(l) => {
                        if k < 2 {
                            t.cat[4 + k] = v.objects[*l as usize].template + 1;
                        }
                        let kind = [rel::ACT_TARGET0, rel::ACT_TARGET1, rel::ACT_TARGET2][k.min(2)];
                        add_rel(me, o0 + *l as usize, kind);
                    }
                    TargetView::Player(r) => {
                        if k < 2 {
                            t.cat[4 + k] = 100 + *r as u16;
                        }
                        add_rel(me, p0 + *r as usize, rel::ACT_TARGET_PLAYER);
                    }
                    TargetView::Num(x) => {
                        if numi < NNUM {
                            t.set(numi, *x as f32);
                            numi += 1;
                        }
                        if k < 2 {
                            t.cat[4 + k] = 200;
                        }
                    }
                    TargetView::HiddenObj => {
                        if k < 2 {
                            t.cat[4 + k] = 201;
                        }
                    }
                }
            }
            for (r, amt) in &a.costs {
                if (*r as usize) < 6 {
                    t.set(*r as usize, *amt as f32);
                }
            }
            toks.push(t);
        }
        let a1 = toks.len();

        let mut attr_desc = vec![[0.0; DESC_DIM]; MAX_ATTRS];
        for (i, x) in d.attrs.iter().enumerate() {
            attr_desc[i] = *x;
        }
        let mut res_desc = vec![[0.0; DESC_DIM]; MAX_RESOURCES];
        for (i, x) in d.resources.iter().enumerate() {
            res_desc[i] = *x;
        }
        Observation {
            attr_desc,
            res_desc,
            game_hash: v.game_hash,
            num_players: v.num_players,
            tokens: toks,
            relations: rels,
            seg: Segments { global: 0, players: p0..p1, zones: z0..z1, objects: o0..o1, events: e0..e1, actions: a0..a1 },
            history_total: total,
            history_truncated: start > 0,
        }
    }
}
