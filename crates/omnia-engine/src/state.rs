use crate::rng::Rng;
use crate::types::*;
use omnia_dsl::*;
use smallvec::SmallVec;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transition {
    None,
    EndPhase,
    EndTurn,
}

/// Complete (true) simulator state. Hidden information lives here; players
/// only ever see it through [`State::view`](crate::view).
#[derive(Clone)]
pub struct State {
    pub(crate) game: Arc<Game>,
    pub(crate) objs: Vec<Obj>,
    pub(crate) zones: Vec<Vec<ObjId>>,
    pub(crate) res: [[i32; MAX_RESOURCES]; MAX_PLAYERS],
    pub(crate) vars: [i32; MAX_VARS],
    pub(crate) turn: u32,
    pub(crate) phase: u8,
    pub(crate) active: u8,
    pub(crate) stack: Vec<StackItem>,
    pub(crate) priority: u8,
    pub(crate) passes: u8,
    pub(crate) hist: History,
    pub(crate) rng: Rng,
    pub(crate) delayed: Vec<Delayed>,
    pub(crate) outcome: Option<Outcome>,
    pub(crate) win_mask: Mask,
    pub(crate) lose_mask: Mask,
    pub(crate) draw_flag: bool,
    pub(crate) legal: Vec<Action>,
    pub(crate) decider: u8,
    pub(crate) fault: Option<Fault>,
    pub(crate) steps: u32,
    pub(crate) pending: Vec<(Event, u32)>,
    pub(crate) cur_gen: u32,
    pub(crate) decisions: u32,
    pub(crate) next_seq: u32,
    pub(crate) transition: Transition,
    pub(crate) cause: u32,
}

/// Execution context for expressions/effects.
#[derive(Clone, Debug)]
pub(crate) struct Ctx {
    pub me: u8,
    pub source: ObjId,
    pub targets: Targets,
    pub iters: SmallVec<[ObjId; 4]>,
    pub last: ObjId,
    pub ev: Option<Event>,
    /// No RNG consumption / no side effects (legality checks).
    pub pure: bool,
    /// Ignore continuous modifiers when reading attributes.
    pub raw: bool,
}

impl Ctx {
    pub fn new(me: u8) -> Ctx {
        Ctx { me, source: NONE_OBJ, targets: Targets::new(), iters: SmallVec::new(), last: NONE_OBJ, ev: None, pure: false, raw: false }
    }
}

impl State {
    pub fn new(game: &Arc<Game>, seed: u64) -> State {
        let g = game.clone();
        let mut res = [[0; MAX_RESOURCES]; MAX_PLAYERS];
        for p in 0..g.n() as usize {
            for (i, r) in g.def.resources.iter().enumerate() {
                res[p][i] = r.initial;
            }
        }
        let mut vars = [0; MAX_VARS];
        for (i, v) in g.def.vars.iter().enumerate() {
            vars[i] = v.initial;
        }
        let mut st = State {
            game: g.clone(),
            objs: Vec::with_capacity(64),
            zones: vec![vec![]; g.n_instances],
            res,
            vars,
            turn: 1,
            phase: 0,
            active: 0,
            stack: vec![],
            priority: 0,
            passes: 0,
            hist: History::default(),
            rng: Rng::new(seed),
            delayed: vec![],
            outcome: None,
            win_mask: 0,
            lose_mask: 0,
            draw_flag: false,
            legal: vec![],
            decider: 0,
            fault: None,
            steps: 0,
            pending: vec![],
            cur_gen: 0,
            decisions: 0,
            next_seq: 0,
            transition: Transition::None,
            cause: NO_SEQ,
        };
        for p in 0..g.n() {
            for e in &g.def.player_setup {
                for _ in 0..e.count {
                    let inst = g.zone_base[e.zone as usize] + p as u16;
                    st.spawn(e.template, p, inst, false);
                }
            }
        }
        for e in &g.def.shared_setup {
            for _ in 0..e.count {
                let inst = g.zone_base[e.zone as usize];
                st.spawn(e.template, 0, inst, false);
            }
        }
        let ctx = Ctx::new(0);
        let setup = g.def.setup.clone();
        st.run_top(&setup, ctx);
        st.check_terminal();
        if st.outcome.is_none() {
            st.begin_turn();
        }
        st.advance();
        st
    }

    // ---- basic accessors --------------------------------------------------

    pub fn game(&self) -> &Arc<Game> {
        &self.game
    }
    pub fn num_players(&self) -> u8 {
        self.game.n()
    }
    pub fn turn(&self) -> u32 {
        self.turn
    }
    pub fn phase(&self) -> u8 {
        self.phase
    }
    pub fn active_player(&self) -> u8 {
        self.active
    }
    pub fn is_terminal(&self) -> bool {
        self.outcome.is_some()
    }
    pub fn outcome(&self) -> Option<&Outcome> {
        self.outcome.as_ref()
    }
    pub fn payoffs(&self) -> Option<[f32; MAX_PLAYERS]> {
        self.outcome.as_ref().map(|o| o.payoffs)
    }
    pub fn fault(&self) -> Option<Fault> {
        self.fault
    }
    pub fn decision_maker(&self) -> Option<u8> {
        if self.outcome.is_some() || self.legal.is_empty() {
            None
        } else {
            Some(self.decider)
        }
    }
    pub fn legal_actions(&self) -> &[Action] {
        &self.legal
    }
    pub fn history(&self) -> &History {
        &self.hist
    }
    pub fn decisions(&self) -> u32 {
        self.decisions
    }
    pub fn stack_len(&self) -> usize {
        self.stack.len()
    }
    pub fn resource(&self, p: u8, r: usize) -> i32 {
        self.res[p as usize][r]
    }
    pub fn num_objects(&self) -> usize {
        self.objs.len()
    }
    pub fn object(&self, id: ObjId) -> &Obj {
        &self.objs[id as usize]
    }
    pub fn zone_contents(&self, inst: usize) -> &[ObjId] {
        &self.zones[inst]
    }

    /// Hash over the full true state (determinism / replay tests).
    pub fn state_hash(&self) -> u64 {
        let mut h = blake3::Hasher::new();
        for o in &self.objs {
            h.update(&o.template.to_le_bytes());
            h.update(&[o.owner, o.controller, o.reveal, o.alive as u8]);
            h.update(&o.zone.to_le_bytes());
            h.update(&o.parent.to_le_bytes());
            for a in &o.attrs {
                h.update(&a.to_le_bytes());
            }
        }
        for z in &self.zones {
            h.update(&(z.len() as u32).to_le_bytes());
            for o in z {
                h.update(&o.to_le_bytes());
            }
        }
        for p in &self.res {
            for r in p {
                h.update(&r.to_le_bytes());
            }
        }
        for v in &self.vars {
            h.update(&v.to_le_bytes());
        }
        h.update(&self.turn.to_le_bytes());
        h.update(&[self.phase, self.active, self.priority, self.passes]);
        h.update(&(self.stack.len() as u32).to_le_bytes());
        h.update(&(self.hist.len() as u32).to_le_bytes());
        h.update(&serde_json::to_vec(&self.rng).unwrap());
        u64::from_le_bytes(h.finalize().as_bytes()[..8].try_into().unwrap())
    }

    // ---- object / zone helpers -------------------------------------------

    pub(crate) fn all_mask(&self) -> Mask {
        ((1u16 << self.game.n()) - 1) as Mask
    }

    pub(crate) fn zone_def_of(&self, inst: u16) -> &ZoneDef {
        &self.game.def.zones[self.game.inst[inst as usize].0 as usize]
    }

    pub(crate) fn zone_mask(&self, inst: u16) -> Mask {
        if inst == NO_ZONE {
            return 0;
        }
        let zd = self.zone_def_of(inst);
        match zd.vis {
            Visibility::Public => self.all_mask(),
            Visibility::Private => {
                let owner = self.game.inst[inst as usize].1;
                if owner == NO_PLAYER {
                    0
                } else {
                    1 << owner
                }
            }
            Visibility::Hidden => 0,
        }
    }

    /// Players who can currently see the identity of `o`.
    pub fn obj_mask(&self, o: ObjId) -> Mask {
        if o == NONE_OBJ {
            return self.all_mask();
        }
        let ob = &self.objs[o as usize];
        self.zone_mask(ob.zone) | ob.reveal
    }

    pub(crate) fn spawn(&mut self, template: u16, owner: u8, inst: u16, emit: bool) -> ObjId {
        let g = self.game.clone();
        let t = &g.def.templates[template as usize];
        let mut attrs = [0; MAX_ATTRS];
        for (i, a) in g.def.attrs.iter().enumerate() {
            attrs[i] = a.default;
        }
        for (a, v) in &t.attrs {
            attrs[*a as usize] = *v;
        }
        let controller = if g.inst[inst as usize].1 != NO_PLAYER { g.inst[inst as usize].1 } else { owner };
        let id = self.objs.len() as ObjId;
        self.objs.push(Obj { template, owner, controller, zone: inst, attrs, parent: NONE_OBJ, reveal: 0, alive: true });
        self.zones[inst as usize].push(id);
        if emit {
            let m = self.obj_mask(id);
            self.emit(EventKind::Created, owner, id, [NONE_OBJ; 2], 0, 0, 0, NO_ZONE, inst, m, self.all_mask());
        }
        id
    }

    // ---- events -----------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit(
        &mut self,
        kind: EventKind,
        actor: u8,
        source: ObjId,
        targets: [ObjId; 2],
        slot: i32,
        before: i32,
        after: i32,
        from_zone: u16,
        to_zone: u16,
        obj_vis: Mask,
        val_vis: Mask,
    ) -> u32 {
        let tpl = |s: &State, o: ObjId| if o == NONE_OBJ { u16::MAX } else { s.objs[o as usize].template };
        let ev = Event {
            seq: self.next_seq,
            kind,
            actor,
            source,
            targets,
            slot,
            before,
            after,
            from_zone,
            to_zone,
            turn: self.turn,
            phase: self.phase,
            parent: self.cause,
            obj_vis,
            val_vis,
            src_tpl: tpl(self, source),
            tgt_tpl: [tpl(self, targets[0]), tpl(self, targets[1])],
        };
        self.next_seq += 1;
        let seq = ev.seq;
        let has_triggers = !self.game.def.triggers.is_empty() || self.game.any_template_triggers;
        if has_triggers {
            self.pending.push((ev.clone(), self.cur_gen + 1));
        }
        self.hist.push(ev);
        seq
    }

    pub(crate) fn tick(&mut self) -> Result<(), ()> {
        if self.steps == 0 {
            self.fault.get_or_insert(Fault::StepLimit);
            return Err(());
        }
        self.steps -= 1;
        Ok(())
    }
}
