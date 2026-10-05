//! The information boundary: [`PlayerView`] is everything a player may legally
//! know. All ML observation building starts from this struct, never from
//! [`State`] internals.

use crate::rng::Rng;
use crate::state::*;
use crate::types::*;
use omnia_dsl::*;
use serde::{Deserialize, Serialize};

pub const NO_REL: u8 = 255;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlayerView {
    pub me: u8,
    pub num_players: u8,
    pub game_hash: u64,
    pub turn: u32,
    pub phase: u8,
    /// Seat of the active player relative to `me` (0 = me).
    pub active_rel: u8,
    /// Seat holding the decision relative to `me` (NO_REL if none).
    pub decider_rel: u8,
    pub terminal: bool,
    pub stack: Vec<StackView>,
    pub players: Vec<PlayerInfo>,
    pub vars: [i32; MAX_VARS],
    pub zones: Vec<ZoneView>,
    pub objects: Vec<ObjView>,
    pub history: Vec<EventView>,
    /// Legal actions (only populated when `me` is the decision maker).
    pub actions: Vec<ActionView>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlayerInfo {
    pub rel: u8,
    pub resources: [Option<i32>; MAX_RESOURCES],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ZoneView {
    pub def: u8,
    pub owner_rel: u8,
    /// Total number of objects (hidden ones are counted, not described).
    pub size: u16,
    /// Local indices of the objects whose identity is visible.
    pub visible: Vec<u16>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObjView {
    pub template: u16,
    pub kind: u8,
    pub owner_rel: u8,
    pub controller_rel: u8,
    /// Index into `PlayerView::zones`.
    pub zone: u16,
    pub pos: u16,
    pub attrs: [i32; MAX_ATTRS],
    pub attached_to: Option<u16>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct ObjRef {
    /// Present iff the identity was visible to the observer at event time.
    pub template: Option<u16>,
    pub kind: Option<u8>,
    /// Present iff it was visible then *and* the object is still visible now.
    pub local: Option<u16>,
    pub present: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EventView {
    pub seq: u32,
    pub kind: EventKind,
    pub actor_rel: u8,
    pub source: ObjRef,
    pub targets: [ObjRef; 2],
    pub slot: i32,
    pub before: Option<i32>,
    pub after: Option<i32>,
    /// (zone def, owner_rel)
    pub from_zone: Option<(u8, u8)>,
    pub to_zone: Option<(u8, u8)>,
    pub turn: u32,
    pub phase: u8,
    pub parent: Option<u32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TargetView {
    Obj(u16),
    HiddenObj,
    Player(u8),
    Num(i32),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ActionView {
    pub def: u16,
    pub source: Option<u16>,
    pub targets: Vec<TargetView>,
    pub costs: Vec<(u8, i32)>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StackView {
    pub def: u16,
    pub actor_rel: u8,
    pub source: Option<u16>,
}

impl State {
    fn rel(&self, me: u8, p: u8) -> u8 {
        if p == NO_PLAYER {
            NO_REL
        } else {
            (p + self.game.n() - me) % self.game.n()
        }
    }

    fn zone_view_ref(&self, me: u8, inst: u16) -> Option<(u8, u8)> {
        if inst == NO_ZONE {
            None
        } else {
            let (z, o) = self.game.inst[inst as usize];
            Some((z, self.rel(me, o)))
        }
    }

    /// Build the observation-boundary view for player `me`.
    ///
    /// Takes `&mut self` only because attribute evaluation shares the
    /// interpreter; evaluation is pure (no RNG use, no mutation of game state).
    pub fn view(&mut self, me: u8) -> PlayerView {
        let g = self.game.clone();
        let n = g.n();
        let bit = 1u8 << me;

        // Local object indexing: zone-instance order, then position.
        let mut local = vec![u16::MAX; self.objs.len()];
        let mut order: Vec<ObjId> = vec![];
        let mut zones = vec![];
        for (inst, members) in self.zones.iter().enumerate() {
            let (zdef, owner) = g.inst[inst];
            let mut visible = vec![];
            for o in members {
                if self.obj_mask(*o) & bit != 0 {
                    local[*o as usize] = order.len() as u16;
                    visible.push(order.len() as u16);
                    order.push(*o);
                }
            }
            zones.push(ZoneView { def: zdef, owner_rel: self.rel(me, owner), size: members.len() as u16, visible });
        }
        let mut objects = vec![];
        for o in &order {
            let ob = self.objs[*o as usize].clone();
            let pos = self.zones[ob.zone as usize].iter().position(|x| x == o).unwrap_or(0) as u16;
            let mut attrs = [0; MAX_ATTRS];
            for a in 0..g.def.attrs.len() {
                attrs[a] = self.attr_eff(*o, a as u8, false);
            }
            let attached_to = if ob.parent != NONE_OBJ && local[ob.parent as usize] != u16::MAX { Some(local[ob.parent as usize]) } else { None };
            objects.push(ObjView {
                template: ob.template,
                kind: g.def.templates[ob.template as usize].kind,
                owner_rel: self.rel(me, ob.owner),
                controller_rel: self.rel(me, ob.controller),
                zone: ob.zone,
                pos,
                attrs,
                attached_to,
            });
        }

        let mut players = vec![];
        for r in 0..n {
            let p = (me + r) % n;
            let mut resources = [None; MAX_RESOURCES];
            for (i, rd) in g.def.resources.iter().enumerate() {
                if rd.public || p == me {
                    resources[i] = Some(self.res[p as usize][i]);
                }
            }
            players.push(PlayerInfo { rel: r, resources });
        }

        // History with redaction.
        let mk_ref = |o: ObjId, tpl: u16, vis: Mask| -> ObjRef {
            if o == NONE_OBJ {
                return ObjRef::default();
            }
            let seen = vis & bit != 0;
            ObjRef {
                template: if seen { Some(tpl) } else { None },
                kind: if seen { Some(g.def.templates[tpl as usize].kind) } else { None },
                local: if seen && local[o as usize] != u16::MAX { Some(local[o as usize]) } else { None },
                present: true,
            }
        };
        let mut history = Vec::with_capacity(self.hist.len());
        for e in self.hist.iter() {
            let source = mk_ref(e.source, e.src_tpl, e.obj_vis);
            let t0 = mk_ref(e.targets[0], e.tgt_tpl[0], e.obj_vis);
            let t1 = mk_ref(e.targets[1], e.tgt_tpl[1], e.obj_vis);
            let vv = e.val_vis & bit != 0;
            history.push(EventView {
                seq: e.seq,
                kind: e.kind,
                actor_rel: self.rel(me, e.actor),
                source,
                targets: [t0, t1],
                slot: e.slot,
                before: if vv { Some(e.before) } else { None },
                after: if vv { Some(e.after) } else { None },
                from_zone: self.zone_view_ref(me, e.from_zone),
                to_zone: self.zone_view_ref(me, e.to_zone),
                turn: e.turn,
                phase: e.phase,
                parent: if e.parent == NO_SEQ { None } else { Some(e.parent) },
            });
        }

        let tgt_view = |t: &Target| -> TargetView {
            match t {
                Target::Obj(o) => {
                    if local[*o as usize] != u16::MAX {
                        TargetView::Obj(local[*o as usize])
                    } else {
                        TargetView::HiddenObj
                    }
                }
                Target::Player(p) => TargetView::Player((p + n - me) % n),
                Target::Num(v) => TargetView::Num(*v),
            }
        };
        let mut actions = vec![];
        if self.decision_maker() == Some(me) {
            let legal = self.legal.clone();
            for a in &legal {
                let mut costs = vec![];
                if !a.is_pass() {
                    let d = &g.def.actions[a.def as usize];
                    let mut c = Ctx::new(a.actor);
                    c.pure = true;
                    c.source = a.source;
                    c.targets = a.targets.clone();
                    for (r, x) in &d.costs {
                        costs.push((*r, self.eval(x, &mut c)));
                    }
                }
                actions.push(ActionView {
                    def: a.def,
                    source: if a.source == NONE_OBJ || local[a.source as usize] == u16::MAX { None } else { Some(local[a.source as usize]) },
                    targets: a.targets.iter().map(&tgt_view).collect(),
                    costs,
                });
            }
        }
        let stack = self
            .stack
            .iter()
            .map(|s| StackView {
                def: s.def,
                actor_rel: self.rel(me, s.actor),
                source: if s.source != NONE_OBJ && local[s.source as usize] != u16::MAX { Some(local[s.source as usize]) } else { None },
            })
            .collect();

        PlayerView {
            me,
            num_players: n,
            game_hash: g.hash64,
            turn: self.turn,
            phase: self.phase,
            active_rel: self.rel(me, self.active),
            decider_rel: match self.decision_maker() {
                Some(d) => self.rel(me, d),
                None => NO_REL,
            },
            terminal: self.is_terminal(),
            stack,
            players,
            vars: self.vars,
            zones,
            objects,
            history,
            actions,
        }
    }

    /// Resample everything `observer` cannot see, consistently with the
    /// public structure (zone sizes, owners, allowed kinds). Object ids,
    /// positions and zones are preserved; only hidden identities are
    /// permuted among hidden objects of the same owner. The returned state
    /// is a valid simulator state for search.
    ///
    /// Limitation (v0): it does not condition on knowledge the observer has
    /// accumulated about hidden cards from history (e.g. a card seen earlier
    /// and then returned face-down).
    pub fn determinize(&self, observer: u8, rng: &mut Rng) -> State {
        let mut st = self.clone();
        let g = self.game.clone();
        let bit = 1u8 << observer;
        for owner in 0..g.n() {
            let hidden: Vec<ObjId> = (0..st.objs.len() as ObjId)
                .filter(|o| {
                    let ob = &st.objs[*o as usize];
                    ob.alive && ob.owner == owner && st.obj_mask(*o) & bit == 0
                })
                .collect();
            if hidden.len() < 2 {
                continue;
            }
            let contents: Vec<(u16, [i32; MAX_ATTRS])> = hidden.iter().map(|o| (st.objs[*o as usize].template, st.objs[*o as usize].attrs)).collect();
            let mut perm: Vec<usize> = (0..hidden.len()).collect();
            let mut ok = false;
            for _ in 0..24 {
                rng.shuffle(&mut perm);
                ok = hidden.iter().enumerate().all(|(i, o)| {
                    let zd = st.zone_def_of(st.objs[*o as usize].zone);
                    match &zd.allowed_kinds {
                        Some(k) => k.contains(&g.def.templates[contents[perm[i]].0 as usize].kind),
                        None => true,
                    }
                });
                if ok {
                    break;
                }
            }
            if ok {
                for (i, o) in hidden.iter().enumerate() {
                    let (t, a) = contents[perm[i]];
                    let ob = &mut st.objs[*o as usize];
                    ob.template = t;
                    ob.attrs = a;
                }
            }
        }
        st.rng = rng.fork();
        st.legal.clear();
        st.advance();
        st
    }

    /// Index of `a` in the current legal-action list.
    pub fn find_action(&self, a: &Action) -> Option<usize> {
        self.legal.iter().position(|x| x == a)
    }
}
