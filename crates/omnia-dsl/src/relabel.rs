//! Id relabelling: rewrite a game with permuted resource / attribute / var /
//! zone / template / hook ids, action order and action-class labels, leaving
//! the mechanics identical. Used for re-skin controls and to remove
//! accidental ID alignment between generated games.

use crate::ir::*;
use std::cell::Cell;

/// Which id space `Expr::EventSlot` indexes for the trigger being rewritten.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotSpace {
    None,
    Attr,
    Res,
    Action,
    Zone,
}

fn slot_space(on: EventKind) -> SlotSpace {
    match on {
        EventKind::AttrChanged => SlotSpace::Attr,
        EventKind::ResourceChanged => SlotSpace::Res,
        EventKind::ActionTaken | EventKind::StackResolved | EventKind::StackCancelled => SlotSpace::Action,
        EventKind::Shuffled | EventKind::ZoneEmpty => SlotSpace::Zone,
        _ => SlotSpace::None,
    }
}

#[derive(Clone, Debug)]
pub struct Relabel {
    pub res: Vec<u8>,
    pub attr: Vec<u8>,
    pub var: Vec<u8>,
    pub zone: Vec<u8>,
    pub template: Vec<u16>,
    pub hook: Vec<u8>,
    pub action: Vec<u16>,
    pub class: Vec<u8>,
    /// Id space of `EventSlot` for the trigger currently being rewritten.
    slot: Cell<SlotSpace>,
}

fn identity8(n: usize) -> Vec<u8> {
    (0..n).map(|i| i as u8).collect()
}

impl Relabel {
    pub fn identity(g: &GameDef) -> Relabel {
        Relabel {
            res: identity8(g.resources.len()),
            attr: identity8(g.attrs.len()),
            var: identity8(g.vars.len()),
            zone: identity8(g.zones.len()),
            template: (0..g.templates.len() as u16).collect(),
            hook: identity8(g.hooks.len()),
            action: (0..g.actions.len() as u16).collect(),
            class: identity8(256),
            slot: Cell::new(SlotSpace::None),
        }
    }

    /// Fully random relabelling from a deterministic shuffle function.
    pub fn random(g: &GameDef, mut below: impl FnMut(u64) -> u64) -> Relabel {
        fn shuf<T: Copy>(v: &mut Vec<T>, below: &mut impl FnMut(u64) -> u64) {
            for i in (1..v.len()).rev() {
                let j = below(i as u64 + 1) as usize;
                v.swap(i, j);
            }
        }
        let mut r = Relabel::identity(g);
        shuf(&mut r.res, &mut below);
        shuf(&mut r.attr, &mut below);
        shuf(&mut r.var, &mut below);
        shuf(&mut r.zone, &mut below);
        shuf(&mut r.template, &mut below);
        shuf(&mut r.hook, &mut below);
        shuf(&mut r.action, &mut below);
        let mut c: Vec<u8> = (0..16).collect();
        shuf(&mut c, &mut below);
        for (i, x) in c.into_iter().enumerate() {
            r.class[i] = x;
        }
        r
    }

    fn z(&self, z: &ZRef) -> ZRef {
        ZRef { who: self.zw(&z.who), zone: self.zone[z.zone as usize] }
    }
    fn zw(&self, w: &ZWho) -> ZWho {
        match w {
            ZWho::OwnerOf(o) => ZWho::OwnerOf(*o),
            x => *x,
        }
    }

    fn sel(&self, s: &Sel) -> Sel {
        match s {
            Sel::One(o) => Sel::One(*o),
            Sel::Zones { zones, filter, order } => Sel::Zones {
                zones: zones.iter().map(|z| self.z(z)).collect(),
                filter: self.cond(filter),
                order: match order {
                    SelOrder::All => SelOrder::All,
                    SelOrder::Top(e) => SelOrder::Top(Box::new(self.expr(e))),
                    SelOrder::Bottom(e) => SelOrder::Bottom(Box::new(self.expr(e))),
                    SelOrder::Random(e) => SelOrder::Random(Box::new(self.expr(e))),
                },
            },
        }
    }

    fn expr(&self, e: &Expr) -> Expr {
        let b = |x: &Expr| Box::new(self.expr(x));
        match e {
            Expr::Attr(o, a) => Expr::Attr(*o, self.attr[*a as usize]),
            Expr::Res(p, r) => Expr::Res(*p, self.res[*r as usize]),
            Expr::Var(v) => Expr::Var(self.var[*v as usize]),
            Expr::Count(s) => Expr::Count(self.sel(s)),
            Expr::Neg(a) => Expr::Neg(b(a)),
            Expr::Add(a, c) => Expr::Add(b(a), b(c)),
            Expr::Sub(a, c) => Expr::Sub(b(a), b(c)),
            Expr::Mul(a, c) => Expr::Mul(b(a), b(c)),
            Expr::Div(a, c) => Expr::Div(b(a), b(c)),
            Expr::Min(a, c) => Expr::Min(b(a), b(c)),
            Expr::Max(a, c) => Expr::Max(b(a), b(c)),
            Expr::IfElse(c, a, d) => Expr::IfElse(Box::new(self.cond(c)), b(a), b(d)),
            other => other.clone(),
        }
    }

    /// Remap a constant compared against `EventSlot` into the new id space.
    fn slot_const(&self, k: i32) -> i32 {
        let at = |m: &[u8]| m.get(k as usize).map(|x| *x as i32);
        let mapped = if k < 0 {
            None
        } else {
            match self.slot.get() {
                SlotSpace::None => None,
                SlotSpace::Attr => at(&self.attr),
                SlotSpace::Res => at(&self.res),
                SlotSpace::Zone => at(&self.zone),
                SlotSpace::Action => self.action.get(k as usize).map(|x| *x as i32),
            }
        };
        mapped.unwrap_or(k)
    }

    /// Only equality tests against a constant are permutation-safe; ordering
    /// tests on slot ids are left untouched.
    fn cmp(&self, a: &Expr, op: CmpOp, b: &Expr) -> Cond {
        let eq = matches!(op, CmpOp::Eq | CmpOp::Ne);
        match (a, b) {
            (Expr::EventSlot, Expr::Const(k)) if eq => Cond::Cmp(Box::new(Expr::EventSlot), op, Box::new(Expr::Const(self.slot_const(*k)))),
            (Expr::Const(k), Expr::EventSlot) if eq => Cond::Cmp(Box::new(Expr::Const(self.slot_const(*k))), op, Box::new(Expr::EventSlot)),
            _ => Cond::Cmp(Box::new(self.expr(a)), op, Box::new(self.expr(b))),
        }
    }

    fn cond(&self, c: &Cond) -> Cond {
        match c {
            Cond::Not(a) => Cond::Not(Box::new(self.cond(a))),
            Cond::And(v) => Cond::And(v.iter().map(|x| self.cond(x)).collect()),
            Cond::Or(v) => Cond::Or(v.iter().map(|x| self.cond(x)).collect()),
            Cond::Cmp(a, op, b) => self.cmp(a, *op, b),
            Cond::InZone(o, z) => Cond::InZone(*o, self.z(z)),
            Cond::IsTemplate(o, t) => Cond::IsTemplate(*o, self.template[*t as usize]),
            Cond::Exists(s) => Cond::Exists(Box::new(self.sel(s))),
            Cond::HasRoom(z) => Cond::HasRoom(self.z(z)),
            Cond::EventToZone(z) => Cond::EventToZone(self.zone[*z as usize]),
            Cond::EventFromZone(z) => Cond::EventFromZone(self.zone[*z as usize]),
            other => other.clone(),
        }
    }

    fn effect(&self, e: &Effect) -> Effect {
        let bx = |x: &Effect| Box::new(self.effect(x));
        match e {
            Effect::Seq(v) => Effect::Seq(v.iter().map(|x| self.effect(x)).collect()),
            Effect::If(c, a, b) => Effect::If(self.cond(c), bx(a), bx(b)),
            Effect::ForEach(s, b) => Effect::ForEach(self.sel(s), bx(b)),
            Effect::ForEachPlayer(p, b) => Effect::ForEachPlayer(*p, bx(b)),
            Effect::Repeat(n, b) => Effect::Repeat(self.expr(n), bx(b)),
            Effect::Move { what, to, pos } => Effect::Move { what: self.sel(what), to: self.z(to), pos: *pos },
            Effect::Create { template, to, owner } => Effect::Create { template: self.template[*template as usize], to: self.z(to), owner: *owner },
            Effect::Copy { what, to } => Effect::Copy { what: *what, to: self.z(to) },
            Effect::Destroy(s) => Effect::Destroy(self.sel(s)),
            Effect::Transform(s, t) => Effect::Transform(self.sel(s), self.template[*t as usize]),
            Effect::SetController(s, p) => Effect::SetController(self.sel(s), *p),
            Effect::SetAttr(s, a, x) => Effect::SetAttr(self.sel(s), self.attr[*a as usize], self.expr(x)),
            Effect::ModAttr(s, a, x) => Effect::ModAttr(self.sel(s), self.attr[*a as usize], self.expr(x)),
            Effect::SetRes(p, r, x) => Effect::SetRes(*p, self.res[*r as usize], self.expr(x)),
            Effect::ModRes(p, r, x) => Effect::ModRes(*p, self.res[*r as usize], self.expr(x)),
            Effect::SetVar(v, x) => Effect::SetVar(self.var[*v as usize], self.expr(x)),
            Effect::ModVar(v, x) => Effect::ModVar(self.var[*v as usize], self.expr(x)),
            Effect::Shuffle(z) => Effect::Shuffle(self.z(z)),
            Effect::Reveal(s, p) => Effect::Reveal(self.sel(s), *p),
            Effect::Hide(s) => Effect::Hide(self.sel(s)),
            Effect::Attach(s, o) => Effect::Attach(self.sel(s), *o),
            Effect::Detach(s) => Effect::Detach(self.sel(s)),
            Effect::Hook(o, h) => Effect::Hook(*o, self.hook[*h as usize]),
            Effect::Emit(k, x) => Effect::Emit(*k, self.expr(x)),
            Effect::Delay { turns, at, effect } => Effect::Delay { turns: *turns, at: *at, effect: bx(effect) },
            other => other.clone(),
        }
    }

    fn trigger(&self, t: &TriggerDef) -> TriggerDef {
        self.slot.set(slot_space(t.on));
        let out = TriggerDef { on: t.on, active_in: t.active_in.iter().map(|z| self.zone[*z as usize]).collect(), cond: self.cond(&t.cond), effect: self.effect(&t.effect) };
        self.slot.set(SlotSpace::None);
        out
    }

    pub fn apply(&self, g: &GameDef) -> GameDef {
        let mut out = g.clone();
        fn place<T: Clone>(old: &[T], perm: &[impl Copy + Into<usize>]) -> Vec<T> {
            let mut v: Vec<Option<T>> = vec![None; old.len()];
            for (i, x) in old.iter().enumerate() {
                v[perm[i].into()] = Some(x.clone());
            }
            v.into_iter().map(|x| x.unwrap()).collect()
        }
        out.resources = place(&g.resources, &self.res);
        out.attrs = place(&g.attrs, &self.attr);
        out.vars = place(&g.vars, &self.var);
        out.hooks = place(&g.hooks, &self.hook);
        out.zones = place(&g.zones, &self.zone);
        let templates: Vec<TemplateDef> = g
            .templates
            .iter()
            .map(|t| TemplateDef {
                name: t.name.clone(),
                kind: t.kind,
                attrs: t.attrs.iter().map(|(a, v)| (self.attr[*a as usize], *v)).collect(),
                hooks: t.hooks.iter().map(|(h, e)| (self.hook[*h as usize], self.effect(e))).collect(),
                triggers: t.triggers.iter().map(|x| self.trigger(x)).collect(),
                modifiers: t
                    .modifiers
                    .iter()
                    .map(|m| ModifierDef {
                        active_in: m.active_in.iter().map(|z| self.zone[*z as usize]).collect(),
                        affects: self.sel(&m.affects),
                        attr: self.attr[m.attr as usize],
                        delta: self.expr(&m.delta),
                    })
                    .collect(),
            })
            .collect();
        out.templates = place(&templates, &self.template);
        let setup = |v: &[SetupEntry]| -> Vec<SetupEntry> {
            v.iter().map(|e| SetupEntry { zone: self.zone[e.zone as usize], template: self.template[e.template as usize], count: e.count }).collect()
        };
        out.player_setup = setup(&g.player_setup);
        out.shared_setup = setup(&g.shared_setup);
        out.setup = self.effect(&g.setup);
        for p in out.phases.iter_mut() {
            p.on_enter = self.effect(&p.on_enter);
        }
        let actions: Vec<ActionDef> = g
            .actions
            .iter()
            .map(|a| ActionDef {
                name: a.name.clone(),
                class: self.class[a.class as usize],
                phases: a.phases.clone(),
                timing: a.timing,
                source: a.source.as_ref().map(|s| SourceSpec { zones: s.zones.iter().map(|z| self.z(z)).collect(), filter: self.cond(&s.filter) }),
                targets: a
                    .targets
                    .iter()
                    .map(|t| match t {
                        TargetSpec::Object(s) => TargetSpec::Object(self.sel(s)),
                        TargetSpec::Player(p) => TargetSpec::Player(*p),
                        TargetSpec::Number { lo, hi } => TargetSpec::Number { lo: self.expr(lo), hi: self.expr(hi) },
                    })
                    .collect(),
                costs: a.costs.iter().map(|(r, x)| (self.res[*r as usize], self.expr(x))).collect(),
                require: self.cond(&a.require),
                on_use: self.effect(&a.on_use),
                effect: self.effect(&a.effect),
                on_cancel: self.effect(&a.on_cancel),
                stack: a.stack,
            })
            .collect();
        out.actions = place(&actions, &self.action);
        out.triggers = g.triggers.iter().map(|t| self.trigger(t)).collect();
        out.terminal = g.terminal.iter().map(|t| TerminalRule { cond: self.cond(&t.cond), result: t.result }).collect();
        if let Timeout::ByResource(r) = g.timeout {
            out.timeout = Timeout::ByResource(self.res[r as usize]);
        }
        out
    }
}
