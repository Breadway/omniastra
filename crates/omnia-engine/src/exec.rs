//! Interpreter for expressions, conditions, selections and effects.

use crate::state::*;
use crate::types::*;
use omnia_dsl::*;
use smallvec::SmallVec;

type R = Result<(), ()>;
pub(crate) type ObjList = SmallVec<[ObjId; 16]>;

fn subject_of(ev: &Event) -> ObjId {
    match ev.kind {
        EventKind::Moved | EventKind::Created | EventKind::Destroyed | EventKind::AttrChanged | EventKind::Revealed | EventKind::Attached => ev.source,
        _ => {
            if ev.targets[0] != NONE_OBJ {
                ev.targets[0]
            } else {
                ev.source
            }
        }
    }
}

impl State {
    // ---- reference resolution --------------------------------------------

    pub(crate) fn oref(&self, o: ORef, c: &Ctx) -> ObjId {
        match o {
            ORef::Source => c.source,
            ORef::Target(i) => match c.targets.get(i as usize) {
                Some(Target::Obj(x)) => *x,
                _ => NONE_OBJ,
            },
            ORef::Iter => c.iters.last().copied().unwrap_or(NONE_OBJ),
            ORef::Outer => {
                if c.iters.len() >= 2 {
                    c.iters[c.iters.len() - 2]
                } else {
                    NONE_OBJ
                }
            }
            ORef::Last => c.last,
            ORef::EventObj => c.ev.as_ref().map(subject_of).unwrap_or(NONE_OBJ),
        }
    }

    pub(crate) fn pref(&self, p: PRef, c: &Ctx) -> u8 {
        let n = self.game.n();
        match p {
            PRef::Me => c.me,
            PRef::Opp => (c.me + 1) % n,
            PRef::Active => self.active,
            PRef::Seat(i) => i.min(n - 1),
            PRef::OwnerOf(o) => {
                let x = self.oref(o, c);
                if x == NONE_OBJ {
                    c.me
                } else {
                    self.objs[x as usize].owner
                }
            }
            PRef::ControllerOf(o) => {
                let x = self.oref(o, c);
                if x == NONE_OBJ {
                    c.me
                } else {
                    self.objs[x as usize].controller
                }
            }
            PRef::EventActor => match &c.ev {
                Some(e) if e.actor != NO_PLAYER => e.actor,
                _ => c.me,
            },
        }
    }

    pub(crate) fn zinsts(&self, z: ZRef, c: &Ctx) -> SmallVec<[u16; 4]> {
        let base = self.game.zone_base[z.zone as usize];
        let per_player = self.game.def.zones[z.zone as usize].per_player;
        let mut out = SmallVec::new();
        if !per_player {
            out.push(base);
            return out;
        }
        match z.who {
            ZWho::Each => {
                for p in 0..self.game.n() as u16 {
                    out.push(base + p);
                }
            }
            ZWho::Shared => out.push(base),
            ZWho::Me => out.push(base + c.me as u16),
            ZWho::Opp => out.push(base + ((c.me + 1) % self.game.n()) as u16),
            ZWho::Active => out.push(base + self.active as u16),
            ZWho::OwnerOf(o) => {
                let x = self.oref(o, c);
                let owner = if x == NONE_OBJ { c.me } else { self.objs[x as usize].owner };
                out.push(base + owner as u16)
            }
        }
        out
    }

    // ---- attributes -------------------------------------------------------

    pub(crate) fn attr_eff(&mut self, o: ObjId, a: AttrId, raw: bool) -> i32 {
        if o == NONE_OBJ {
            return 0;
        }
        let g = self.game.clone();
        let mut v = self.objs[o as usize].attrs[a as usize];
        if !raw && g.attr_modified[a as usize] {
            for h in 0..self.objs.len() {
                let (alive, htpl, hzone, hctrl) = {
                    let host = &self.objs[h];
                    (host.alive, host.template as usize, host.zone, host.controller)
                };
                if !alive {
                    continue;
                }
                let t = &g.def.templates[htpl];
                if t.modifiers.is_empty() {
                    continue;
                }
                let host_zone_def = g.inst[hzone as usize].0;
                for m in &t.modifiers {
                    if m.attr != a {
                        continue;
                    }
                    if !m.active_in.is_empty() && !m.active_in.contains(&host_zone_def) {
                        continue;
                    }
                    let mut c = Ctx::new(hctrl);
                    c.source = h as ObjId;
                    c.pure = true;
                    c.raw = true;
                    let sel = self.select(&m.affects, &mut c);
                    if sel.contains(&o) {
                        c.iters.push(o);
                        v += self.eval(&m.delta, &mut c);
                    }
                }
            }
        }
        let d = &g.def.attrs[a as usize];
        v.clamp(d.min, d.max)
    }

    // ---- selections -------------------------------------------------------

    pub(crate) fn select(&mut self, sel: &Sel, c: &mut Ctx) -> ObjList {
        match sel {
            Sel::One(o) => {
                let x = self.oref(*o, c);
                let mut v = ObjList::new();
                if x != NONE_OBJ && self.objs[x as usize].alive {
                    v.push(x);
                }
                v
            }
            Sel::Zones { zones, filter, order } => {
                let mut insts: SmallVec<[u16; 4]> = SmallVec::new();
                for z in zones {
                    insts.extend(self.zinsts(*z, c));
                }
                let mut out = ObjList::new();
                let trivial = matches!(filter, Cond::True);
                for inst in insts {
                    let members = self.zones[inst as usize].clone();
                    for o in members {
                        if trivial {
                            out.push(o);
                        } else {
                            c.iters.push(o);
                            let ok = self.cond(filter, c);
                            c.iters.pop();
                            if ok {
                                out.push(o);
                            }
                        }
                    }
                }
                match order {
                    SelOrder::All => out,
                    SelOrder::Top(n) => {
                        let n = self.eval(n, c).max(0) as usize;
                        out.truncate(n);
                        out
                    }
                    SelOrder::Bottom(n) => {
                        let n = self.eval(n, c).max(0) as usize;
                        if out.len() > n {
                            let k = out.len() - n;
                            out.drain(0..k);
                        }
                        out
                    }
                    SelOrder::Random(n) => {
                        let n = (self.eval(n, c).max(0) as usize).min(out.len());
                        if c.pure {
                            out.truncate(n);
                            return out;
                        }
                        // partial Fisher-Yates
                        for i in 0..n {
                            let j = i + self.rng.below((out.len() - i) as u64) as usize;
                            out.swap(i, j);
                        }
                        out.truncate(n);
                        out
                    }
                }
            }
        }
    }

    // ---- expressions & conditions ----------------------------------------

    pub(crate) fn eval(&mut self, e: &Expr, c: &mut Ctx) -> i32 {
        match e {
            Expr::Const(v) => *v,
            Expr::Attr(o, a) => {
                let x = self.oref(*o, c);
                self.attr_eff(x, *a, c.raw)
            }
            Expr::Res(p, r) => {
                let p = self.pref(*p, c);
                self.res[p as usize][*r as usize]
            }
            Expr::Var(v) => self.vars[*v as usize],
            Expr::Count(s) => self.select(s, c).len() as i32,
            Expr::Param(i) => match c.targets.get(*i as usize) {
                Some(Target::Num(n)) => *n,
                _ => 0,
            },
            Expr::Turn => self.turn as i32,
            Expr::Seat(p) => self.pref(*p, c) as i32,
            Expr::Kind(o) => {
                let x = self.oref(*o, c);
                if x == NONE_OBJ {
                    -1
                } else {
                    self.game.def.templates[self.objs[x as usize].template as usize].kind as i32
                }
            }
            Expr::EventValue => match &c.ev {
                Some(ev) => match ev.kind {
                    EventKind::ResourceChanged | EventKind::AttrChanged => ev.after - ev.before,
                    _ => ev.after,
                },
                None => 0,
            },
            Expr::EventSlot => c.ev.as_ref().map(|e| e.slot).unwrap_or(-1),
            Expr::Neg(a) => -self.eval(a, c),
            Expr::Add(a, b) => self.eval(a, c).saturating_add(self.eval(b, c)),
            Expr::Sub(a, b) => self.eval(a, c).saturating_sub(self.eval(b, c)),
            Expr::Mul(a, b) => self.eval(a, c).saturating_mul(self.eval(b, c)),
            Expr::Div(a, b) => {
                let d = self.eval(b, c);
                let n = self.eval(a, c);
                if d == 0 {
                    0
                } else {
                    n / d
                }
            }
            Expr::Min(a, b) => self.eval(a, c).min(self.eval(b, c)),
            Expr::Max(a, b) => self.eval(a, c).max(self.eval(b, c)),
            Expr::Rand(lo, hi) => {
                if c.pure {
                    *lo
                } else {
                    self.rng.range_i32(*lo, *hi)
                }
            }
            Expr::IfElse(cd, a, b) => {
                if self.cond(cd, c) {
                    self.eval(a, c)
                } else {
                    self.eval(b, c)
                }
            }
        }
    }

    pub(crate) fn cond(&mut self, cd: &Cond, c: &mut Ctx) -> bool {
        match cd {
            Cond::True => true,
            Cond::Not(a) => !self.cond(a, c),
            Cond::And(v) => {
                for x in v {
                    if !self.cond(x, c) {
                        return false;
                    }
                }
                true
            }
            Cond::Or(v) => {
                for x in v {
                    if self.cond(x, c) {
                        return true;
                    }
                }
                false
            }
            Cond::Cmp(a, op, b) => {
                let x = self.eval(a, c);
                let y = self.eval(b, c);
                match op {
                    CmpOp::Lt => x < y,
                    CmpOp::Le => x <= y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                    CmpOp::Ge => x >= y,
                    CmpOp::Gt => x > y,
                }
            }
            Cond::InZone(o, z) => {
                let x = self.oref(*o, c);
                if x == NONE_OBJ {
                    return false;
                }
                let zi = self.zinsts(*z, c);
                zi.contains(&self.objs[x as usize].zone)
            }
            Cond::IsTemplate(o, t) => {
                let x = self.oref(*o, c);
                x != NONE_OBJ && self.objs[x as usize].template == *t
            }
            Cond::IsKind(o, k) => {
                let x = self.oref(*o, c);
                x != NONE_OBJ && self.game.def.templates[self.objs[x as usize].template as usize].kind == *k
            }
            Cond::SamePlayer(a, b) => self.pref(*a, c) == self.pref(*b, c),
            Cond::Exists(s) => !self.select(s, c).is_empty(),
            Cond::Attached(o) => {
                let x = self.oref(*o, c);
                x != NONE_OBJ && self.objs[x as usize].parent != NONE_OBJ
            }
            Cond::HasRoom(z) => {
                let zi = self.zinsts(*z, c);
                zi.iter().any(|i| match self.zone_def_of(*i).capacity {
                    Some(cap) => (self.zones[*i as usize].len() as u16) < cap,
                    None => true,
                })
            }
            Cond::IsActive(p) => self.pref(*p, c) == self.active,
            Cond::InPhase(p) => self.phase == *p,
            Cond::EventToZone(z) => c.ev.as_ref().map(|e| e.to_zone != NO_ZONE && self.game.inst[e.to_zone as usize].0 == *z).unwrap_or(false),
            Cond::EventFromZone(z) => c.ev.as_ref().map(|e| e.from_zone != NO_ZONE && self.game.inst[e.from_zone as usize].0 == *z).unwrap_or(false),
        }
    }

    // ---- effects ----------------------------------------------------------

    /// Run `eff` as one top-level resolution: fresh step budget, then flush
    /// triggers.
    pub(crate) fn run_top(&mut self, eff: &Effect, mut c: Ctx) {
        self.steps = self.game.def.limits.max_steps;
        self.cur_gen = 0;
        let _ = self.exec(eff, &mut c);
        let _ = self.drain_triggers();
    }

    pub(crate) fn exec(&mut self, eff: &Effect, c: &mut Ctx) -> R {
        self.tick()?;
        match eff {
            Effect::NoOp => {}
            Effect::Seq(v) => {
                for x in v {
                    self.exec(x, c)?;
                }
            }
            Effect::If(cd, a, b) => {
                if self.cond(cd, c) {
                    self.exec(a, c)?
                } else {
                    self.exec(b, c)?
                }
            }
            Effect::ForEach(sel, body) => {
                let objs = self.select(sel, c);
                for o in objs {
                    self.tick()?;
                    if !self.objs[o as usize].alive {
                        continue;
                    }
                    c.iters.push(o);
                    let r = self.exec(body, c);
                    c.iters.pop();
                    r?;
                }
            }
            Effect::ForEachPlayer(ps, body) => {
                let saved = c.me;
                let n = self.game.n();
                let players: SmallVec<[u8; 4]> = (0..n).filter(|p| matches!(ps, PSel::All) || *p != saved).collect();
                for p in players {
                    self.tick()?;
                    c.me = p;
                    let r = self.exec(body, c);
                    c.me = saved;
                    r?;
                }
            }
            Effect::Repeat(n, body) => {
                let n = self.eval(n, c).clamp(0, 64);
                for _ in 0..n {
                    self.tick()?;
                    self.exec(body, c)?;
                }
            }
            Effect::Move { what, to, pos } => {
                let objs = self.select(what, c);
                let dst = self.zinsts(*to, c)[0];
                for o in objs {
                    self.tick()?;
                    self.move_obj(o, dst, *pos, c);
                }
            }
            Effect::Create { template, to, owner } => {
                let dst = self.zinsts(*to, c)[0];
                let owner = self.pref(*owner, c);
                if self.objs.len() as u32 >= self.game.def.limits.max_objects {
                    self.fault.get_or_insert(Fault::ObjectLimit);
                    return Err(());
                }
                if self.has_room(dst) && self.kind_allowed(dst, *template) {
                    c.last = self.spawn(*template, owner, dst, true);
                }
            }
            Effect::Copy { what, to } => {
                let src = self.oref(*what, c);
                let dst = self.zinsts(*to, c)[0];
                if src != NONE_OBJ && self.objs[src as usize].alive {
                    if self.objs.len() as u32 >= self.game.def.limits.max_objects {
                        self.fault.get_or_insert(Fault::ObjectLimit);
                        return Err(());
                    }
                    let (tpl, owner, attrs) = {
                        let s = &self.objs[src as usize];
                        (s.template, c.me, s.attrs)
                    };
                    if self.has_room(dst) && self.kind_allowed(dst, tpl) {
                        let id = self.spawn(tpl, owner, dst, true);
                        self.objs[id as usize].attrs = attrs;
                        c.last = id;
                    }
                }
            }
            Effect::Destroy(sel) => {
                for o in self.select(sel, c) {
                    self.tick()?;
                    self.destroy_obj(o, c.me);
                }
            }
            Effect::Transform(sel, t) => {
                let g = self.game.clone();
                for o in self.select(sel, c) {
                    let mut attrs = [0; MAX_ATTRS];
                    for (i, a) in g.def.attrs.iter().enumerate() {
                        attrs[i] = a.default;
                    }
                    for (a, v) in &g.def.templates[*t as usize].attrs {
                        attrs[*a as usize] = *v;
                    }
                    let ob = &mut self.objs[o as usize];
                    ob.template = *t;
                    ob.attrs = attrs;
                }
            }
            Effect::SetController(sel, p) => {
                let p = self.pref(*p, c);
                for o in self.select(sel, c) {
                    self.objs[o as usize].controller = p;
                }
            }
            Effect::SetAttr(sel, a, x) | Effect::ModAttr(sel, a, x) => {
                let is_set = matches!(eff, Effect::SetAttr(..));
                let d = self.game.def.attrs[*a as usize].clone();
                for o in self.select(sel, c) {
                    self.tick()?;
                    c.iters.push(o);
                    let v = self.eval(x, c);
                    c.iters.pop();
                    let before = self.objs[o as usize].attrs[*a as usize];
                    let after = if is_set { v } else { before.saturating_add(v) }.clamp(d.min, d.max);
                    if after != before {
                        self.objs[o as usize].attrs[*a as usize] = after;
                        let m = self.obj_mask(o);
                        self.emit(EventKind::AttrChanged, c.me, o, [NONE_OBJ; 2], *a as i32, before, after, NO_ZONE, NO_ZONE, m, m);
                    }
                }
            }
            Effect::SetRes(p, r, x) | Effect::ModRes(p, r, x) => {
                let is_set = matches!(eff, Effect::SetRes(..));
                let p = self.pref(*p, c);
                let v = self.eval(x, c);
                self.change_res(p, *r, v, is_set, c.me, c.source);
            }
            Effect::SetVar(v, x) => {
                let x = self.eval(x, c);
                self.vars[*v as usize] = x;
            }
            Effect::ModVar(v, x) => {
                let x = self.eval(x, c);
                self.vars[*v as usize] = self.vars[*v as usize].saturating_add(x);
            }
            Effect::Shuffle(z) => {
                if !c.pure {
                    for inst in self.zinsts(*z, c) {
                        let mut v = std::mem::take(&mut self.zones[inst as usize]);
                        self.rng.shuffle(&mut v);
                        self.zones[inst as usize] = v;
                        let zdef = self.game.inst[inst as usize].0 as i32;
                        self.emit(EventKind::Shuffled, c.me, NONE_OBJ, [NONE_OBJ; 2], zdef, 0, 0, inst, inst, 0, self.all_mask());
                    }
                }
            }
            Effect::Reveal(sel, ps) => {
                let all = self.all_mask();
                let mask = match ps {
                    PSel::All => all,
                    PSel::Others => all & !(1 << c.me),
                };
                for o in self.select(sel, c) {
                    self.objs[o as usize].reveal |= mask;
                    let m = self.obj_mask(o);
                    self.emit(EventKind::Revealed, c.me, o, [NONE_OBJ; 2], 0, 0, 0, NO_ZONE, NO_ZONE, m, all);
                }
            }
            Effect::Hide(sel) => {
                for o in self.select(sel, c) {
                    self.objs[o as usize].reveal = 0;
                }
            }
            Effect::Attach(sel, parent) => {
                let p = self.oref(*parent, c);
                if p != NONE_OBJ {
                    for o in self.select(sel, c) {
                        if o != p {
                            self.objs[o as usize].parent = p;
                            let m = self.obj_mask(o) | self.obj_mask(p);
                            let all = self.all_mask();
                            self.emit(EventKind::Attached, c.me, o, [p, NONE_OBJ], 0, 0, 0, NO_ZONE, NO_ZONE, m, all);
                        }
                    }
                }
            }
            Effect::Detach(sel) => {
                for o in self.select(sel, c) {
                    self.objs[o as usize].parent = NONE_OBJ;
                }
            }
            Effect::Hook(o, h) => {
                let host = self.oref(*o, c);
                if host != NONE_OBJ {
                    let tpl = self.objs[host as usize].template as usize;
                    let g = self.game.clone();
                    if let Some(idx) = g.template_hooks[tpl][*h as usize] {
                        let body = &g.def.templates[tpl].hooks[idx as usize].1;
                        let mut c2 = c.clone();
                        c2.source = host;
                        c2.iters.clear();
                        self.exec(body, &mut c2)?;
                        c.last = c2.last;
                    }
                }
            }
            Effect::Emit(k, x) => {
                let v = self.eval(x, c);
                let m = self.obj_mask(c.source);
                let all = self.all_mask();
                self.emit(EventKind::Custom(*k), c.me, c.source, [NONE_OBJ; 2], *k as i32, 0, v, NO_ZONE, NO_ZONE, m, all);
            }
            Effect::Delay { turns, at, effect } => {
                self.delayed.push(Delayed { fire_turn: self.turn + *turns as u32, at: *at, owner: c.me, source: c.source, effect: (**effect).clone() });
            }
            Effect::CancelStack => {
                if let Some(top) = self.stack.last_mut() {
                    top.cancelled = true;
                }
            }
            Effect::Win(p) => {
                let p = self.pref(*p, c);
                self.win_mask |= 1 << p;
            }
            Effect::Lose(p) => {
                let p = self.pref(*p, c);
                self.lose_mask |= 1 << p;
            }
            Effect::DrawGame => self.draw_flag = true,
            Effect::EndPhase => {
                if self.transition != Transition::EndTurn {
                    self.transition = Transition::EndPhase;
                }
            }
            Effect::EndTurn => self.transition = Transition::EndTurn,
        }
        Ok(())
    }

    pub(crate) fn change_res(&mut self, p: u8, r: u8, v: i32, set: bool, actor: u8, source: ObjId) {
        let d = &self.game.def.resources[r as usize];
        let (min, max, public) = (d.min, d.max, d.public);
        let before = self.res[p as usize][r as usize];
        let after = if set { v } else { before.saturating_add(v) }.clamp(min, max);
        if after != before {
            self.res[p as usize][r as usize] = after;
            let vis = if public { self.all_mask() } else { 1 << p };
            let _ = actor;
            let m = self.obj_mask(source);
            self.emit(EventKind::ResourceChanged, p, source, [NONE_OBJ; 2], r as i32, before, after, NO_ZONE, NO_ZONE, m, vis);
        }
    }

    fn has_room(&self, inst: u16) -> bool {
        match self.zone_def_of(inst).capacity {
            Some(cap) => (self.zones[inst as usize].len() as u16) < cap,
            None => true,
        }
    }

    fn kind_allowed(&self, inst: u16, template: u16) -> bool {
        match &self.zone_def_of(inst).allowed_kinds {
            Some(k) => k.contains(&self.game.def.templates[template as usize].kind),
            None => true,
        }
    }

    pub(crate) fn move_obj(&mut self, o: ObjId, dst: u16, pos: Pos, c: &Ctx) {
        let src = self.objs[o as usize].zone;
        if !self.objs[o as usize].alive || src == dst && matches!(pos, Pos::Top) && self.zones[dst as usize].first() == Some(&o) {
            return;
        }
        if src != dst && !self.has_room(dst) {
            return;
        }
        let tpl = self.objs[o as usize].template;
        if !self.kind_allowed(dst, tpl) {
            return;
        }
        let before = self.obj_mask(o);
        if src != NO_ZONE {
            let z = &mut self.zones[src as usize];
            if let Some(i) = z.iter().position(|x| *x == o) {
                z.remove(i);
            }
        }
        for ch in 0..self.objs.len() {
            if self.objs[ch].parent == o {
                self.objs[ch].parent = NONE_OBJ;
            }
        }
        let owner_of_zone = self.game.inst[dst as usize].1;
        {
            let ob = &mut self.objs[o as usize];
            ob.parent = NONE_OBJ;
            ob.zone = dst;
            ob.reveal = 0;
            if owner_of_zone != NO_PLAYER {
                ob.controller = owner_of_zone;
            }
        }
        let len = self.zones[dst as usize].len();
        match pos {
            Pos::Top => self.zones[dst as usize].insert(0, o),
            Pos::Bottom => self.zones[dst as usize].push(o),
            Pos::Random => {
                let i = if c.pure { len } else { self.rng.below(len as u64 + 1) as usize };
                self.zones[dst as usize].insert(i, o)
            }
        }
        let after = self.obj_mask(o);
        let all = self.all_mask();
        self.emit(EventKind::Moved, c.me, o, [NONE_OBJ; 2], 0, 0, 0, src, dst, before | after, all);
        if src != NO_ZONE && src != dst && self.zones[src as usize].is_empty() {
            let zdef = self.game.inst[src as usize].0 as i32;
            self.emit(EventKind::ZoneEmpty, c.me, NONE_OBJ, [NONE_OBJ; 2], zdef, 0, 0, src, src, 0, all);
        }
    }

    pub(crate) fn destroy_obj(&mut self, o: ObjId, actor: u8) {
        if !self.objs[o as usize].alive {
            return;
        }
        let before = self.obj_mask(o);
        let src = self.objs[o as usize].zone;
        if src != NO_ZONE {
            let z = &mut self.zones[src as usize];
            if let Some(i) = z.iter().position(|x| *x == o) {
                z.remove(i);
            }
        }
        for ch in 0..self.objs.len() {
            if self.objs[ch].parent == o {
                self.objs[ch].parent = NONE_OBJ;
            }
        }
        // Emit while still identifiable (mask captured up front).
        let all = self.all_mask();
        self.emit(EventKind::Destroyed, actor, o, [NONE_OBJ; 2], 0, 0, 0, src, NO_ZONE, before, all);
        let ob = &mut self.objs[o as usize];
        ob.alive = false;
        ob.zone = NO_ZONE;
        ob.parent = NONE_OBJ;
    }

    // ---- triggers ---------------------------------------------------------

    pub(crate) fn drain_triggers(&mut self) -> R {
        let g = self.game.clone();
        let max_gen = g.def.limits.max_trigger_depth;
        let mut i = 0;
        while i < self.pending.len() {
            let (ev, gen) = self.pending[i].clone();
            i += 1;
            if gen > max_gen {
                self.fault.get_or_insert(Fault::TriggerDepth);
                self.pending.clear();
                return Err(());
            }
            self.cur_gen = gen;
            let prev_cause = self.cause;
            self.cause = ev.seq;
            // Global triggers.
            for tr in &g.def.triggers {
                if tr.on != ev.kind {
                    continue;
                }
                let me = if ev.actor != NO_PLAYER { ev.actor } else { self.active };
                let mut c = Ctx::new(me);
                c.ev = Some(ev.clone());
                if self.cond(&tr.cond, &mut c) {
                    if self.exec(&tr.effect, &mut c).is_err() {
                        self.pending.clear();
                        self.cause = prev_cause;
                        return Err(());
                    }
                }
            }
            if g.any_template_triggers {
                let n = self.objs.len();
                for h in 0..n {
                    let (alive, tpl, zone, ctrl) = {
                        let o = &self.objs[h];
                        (o.alive, o.template as usize, o.zone, o.controller)
                    };
                    if !alive {
                        continue;
                    }
                    let t = &g.def.templates[tpl];
                    if t.triggers.is_empty() {
                        continue;
                    }
                    let zdef = g.inst[zone as usize].0;
                    for tr in &t.triggers {
                        if tr.on != ev.kind || (!tr.active_in.is_empty() && !tr.active_in.contains(&zdef)) {
                            continue;
                        }
                        if !self.objs[h].alive {
                            break;
                        }
                        let mut c = Ctx::new(ctrl);
                        c.source = h as ObjId;
                        c.ev = Some(ev.clone());
                        if self.cond(&tr.cond, &mut c) && self.exec(&tr.effect, &mut c).is_err() {
                            self.pending.clear();
                            self.cause = prev_cause;
                            return Err(());
                        }
                    }
                }
            }
            self.cause = prev_cause;
        }
        self.pending.clear();
        self.cur_gen = 0;
        Ok(())
    }
}
