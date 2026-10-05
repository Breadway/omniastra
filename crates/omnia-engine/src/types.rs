use omnia_dsl::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::sync::Arc;

pub type ObjId = u32;
pub const NONE_OBJ: ObjId = u32::MAX;
pub const NO_ZONE: u16 = u16::MAX;
pub const NO_PLAYER: u8 = 255;
pub const PASS_DEF: u16 = u16::MAX;
pub const NO_SEQ: u32 = u32::MAX;

/// Bitmask over players (bit p = player p).
pub type Mask = u8;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Target {
    Obj(ObjId),
    Player(u8),
    Num(i32),
}

pub type Targets = SmallVec<[Target; 3]>;

/// A concrete, fully-specified legal action.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Action {
    pub def: u16,
    pub actor: u8,
    pub source: ObjId,
    pub targets: Targets,
}

impl Action {
    pub fn is_pass(&self) -> bool {
        self.def == PASS_DEF
    }
}

#[derive(Clone, Debug)]
pub struct Obj {
    pub template: u16,
    pub owner: u8,
    pub controller: u8,
    /// Zone *instance* index, or `NO_ZONE` if destroyed.
    pub zone: u16,
    pub attrs: [i32; MAX_ATTRS],
    pub parent: ObjId,
    /// Players who have been explicitly shown this object's identity.
    pub reveal: Mask,
    pub alive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub seq: u32,
    pub kind: EventKind,
    pub actor: u8,
    pub source: ObjId,
    pub targets: [ObjId; 2],
    /// attr / resource / action-def / phase id depending on `kind`.
    pub slot: i32,
    pub before: i32,
    pub after: i32,
    pub from_zone: u16,
    pub to_zone: u16,
    pub turn: u32,
    pub phase: u8,
    /// Seq of the causing event (e.g. the `ActionTaken` that led to this).
    pub parent: u32,
    /// Players who may see identities of the objects named in this event.
    pub obj_vis: Mask,
    /// Players who may see the numeric values (before/after).
    pub val_vis: Mask,
    /// Template/kind snapshots of source and targets at event time.
    pub src_tpl: u16,
    pub tgt_tpl: [u16; 2],
}

/// Persistent append-only history (cheap clone for search).
#[derive(Clone, Debug, Default)]
pub struct History {
    chunks: Vec<Arc<Vec<Event>>>,
    tail: Vec<Event>,
}

const CHUNK: usize = 64;

impl History {
    pub fn push(&mut self, e: Event) {
        self.tail.push(e);
        if self.tail.len() == CHUNK {
            let t = std::mem::take(&mut self.tail);
            self.chunks.push(Arc::new(t));
        }
    }
    pub fn len(&self) -> usize {
        self.chunks.len() * CHUNK + self.tail.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, i: usize) -> Option<&Event> {
        let c = i / CHUNK;
        if c < self.chunks.len() {
            self.chunks[c].get(i % CHUNK)
        } else {
            self.tail.get(i - self.chunks.len() * CHUNK)
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.chunks.iter().flat_map(|c| c.iter()).chain(self.tail.iter())
    }
    pub fn iter_from(&self, start: usize) -> impl Iterator<Item = &Event> {
        self.iter().skip(start)
    }
}

#[derive(Clone, Debug)]
pub struct StackItem {
    pub def: u16,
    pub actor: u8,
    pub source: ObjId,
    pub targets: Targets,
    pub cancelled: bool,
    pub cause_seq: u32,
}

#[derive(Clone, Debug)]
pub struct Delayed {
    pub fire_turn: u32,
    pub at: Moment,
    pub owner: u8,
    pub source: ObjId,
    pub effect: Effect,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Fault {
    StepLimit,
    TriggerDepth,
    TooManyActions,
    ObjectLimit,
    StackOverflow,
    NoProgress,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum EndReason {
    Rules,
    Timeout,
    Fault(Fault),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Outcome {
    /// Payoff per seat in [-1, 1].
    pub payoffs: [f32; MAX_PLAYERS],
    pub reason: EndReason,
}

/// Compiled, immutable game: definition plus lookup tables.
#[derive(Debug)]
pub struct Game {
    pub def: GameDef,
    /// First zone-instance index of each zone def.
    pub zone_base: Vec<u16>,
    pub n_instances: usize,
    /// (zone def, owner or NO_PLAYER) per instance.
    pub inst: Vec<(ZoneId, u8)>,
    pub desc: GameDescriptors,
    pub hash64: u64,
    /// For each attr: does any template define a modifier on it?
    pub attr_modified: [bool; MAX_ATTRS],
    pub template_hooks: Vec<[Option<u16>; 16]>,
    /// Number of template-level triggers (to skip scanning if none).
    pub any_template_triggers: bool,
}

impl Game {
    pub fn new(def: GameDef) -> Result<Arc<Game>, DslError> {
        def.check()?;
        let n = def.num_players as usize;
        let mut zone_base = vec![];
        let mut inst = vec![];
        for (zi, z) in def.zones.iter().enumerate() {
            zone_base.push(inst.len() as u16);
            if z.per_player {
                for p in 0..n {
                    inst.push((zi as ZoneId, p as u8));
                }
            } else {
                inst.push((zi as ZoneId, NO_PLAYER));
            }
        }
        let mut attr_modified = [false; MAX_ATTRS];
        let mut template_hooks = vec![];
        let mut any_template_triggers = false;
        for t in &def.templates {
            for m in &t.modifiers {
                attr_modified[m.attr as usize] = true;
            }
            let mut h = [None; 16];
            for (i, (hid, _)) in t.hooks.iter().enumerate() {
                h[*hid as usize] = Some(i as u16);
            }
            template_hooks.push(h);
            any_template_triggers |= !t.triggers.is_empty();
        }
        let desc = analyze(&def);
        let hash64 = def.hash64();
        Ok(Arc::new(Game { def, zone_base, n_instances: inst.len(), inst, desc, hash64, attr_modified, template_hooks, any_template_triggers }))
    }

    #[inline]
    pub fn n(&self) -> u8 {
        self.def.num_players
    }
}
