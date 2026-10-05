//! Static validation of a [`GameDef`]: id ranges, scoping of references,
//! structural limits and information-visibility rules for targets.

use crate::ir::*;

#[derive(Clone, Copy, Default)]
struct Scope {
    source: bool,
    ntargets: usize,
    iter_depth: usize,
    event: bool,
}

struct V<'a> {
    g: &'a GameDef,
    errs: Vec<String>,
    ctx: String,
    /// Which target slots are Number targets (for `Param`).
    number_targets: Vec<bool>,
}

impl<'a> V<'a> {
    fn err(&mut self, m: impl Into<String>) {
        self.errs.push(format!("{}: {}", self.ctx, m.into()));
    }

    fn zref(&mut self, z: &ZRef) {
        match self.g.zones.get(z.zone as usize) {
            None => self.err(format!("zone id {} out of range", z.zone)),
            Some(zd) => match z.who {
                ZWho::Shared if zd.per_player => self.err(format!("zone '{}' is per-player but referenced as Shared", zd.name)),
                ZWho::Each | ZWho::Me | ZWho::Opp | ZWho::Active | ZWho::OwnerOf(_) if !zd.per_player => {
                    self.err(format!("zone '{}' is shared but referenced per-player", zd.name))
                }
                _ => {}
            },
        }
        if let ZWho::OwnerOf(o) = z.who {
            self.oref_ok(o, &Scope { source: true, ntargets: 3, iter_depth: 2, event: true }, false);
        }
    }

    fn oref_ok(&mut self, o: ORef, s: &Scope, report: bool) {
        let bad = match o {
            ORef::Source => !s.source,
            ORef::Target(i) => (i as usize) >= s.ntargets,
            ORef::Iter => s.iter_depth < 1,
            ORef::Outer => s.iter_depth < 2,
            ORef::Last => false,
            ORef::EventObj => !s.event,
        };
        if bad && report {
            self.err(format!("reference {:?} not in scope", o));
        }
    }

    fn oref(&mut self, o: ORef, s: &Scope) {
        self.oref_ok(o, s, true)
    }

    fn pref(&mut self, p: &PRef, s: &Scope) {
        match p {
            PRef::OwnerOf(o) | PRef::ControllerOf(o) => self.oref(*o, s),
            PRef::EventActor if !s.event => self.err("EventActor not in scope"),
            PRef::Seat(i) if *i >= self.g.num_players => self.err("seat out of range"),
            _ => {}
        }
    }

    fn sel(&mut self, sel: &Sel, s: &Scope) {
        match sel {
            Sel::One(o) => self.oref(*o, s),
            Sel::Zones { zones, filter, order } => {
                for z in zones {
                    self.zref(z);
                }
                let mut s2 = *s;
                s2.iter_depth += 1;
                self.cond(filter, &s2);
                match order {
                    SelOrder::All => {}
                    SelOrder::Top(e) | SelOrder::Bottom(e) | SelOrder::Random(e) => self.expr(e, s),
                }
            }
        }
    }

    fn expr(&mut self, e: &Expr, s: &Scope) {
        match e {
            Expr::Const(_) | Expr::Turn | Expr::Rand(..) => {}
            Expr::EventValue | Expr::EventSlot => {
                if !s.event {
                    self.err("event expression not in scope")
                }
            }
            Expr::Attr(o, a) => {
                self.oref(*o, s);
                if (*a as usize) >= self.g.attrs.len() {
                    self.err("attr id out of range");
                }
            }
            Expr::Res(p, r) => {
                self.pref(p, s);
                if (*r as usize) >= self.g.resources.len() {
                    self.err("resource id out of range");
                }
            }
            Expr::Var(v) => {
                if (*v as usize) >= self.g.vars.len() {
                    self.err("var id out of range");
                }
            }
            Expr::Count(sel) => self.sel(sel, s),
            Expr::Param(i) => {
                if !self.number_targets.get(*i as usize).copied().unwrap_or(false) {
                    self.err(format!("Param({}) does not refer to a Number target", i));
                }
            }
            Expr::Seat(p) => self.pref(p, s),
            Expr::Kind(o) => self.oref(*o, s),
            Expr::Neg(a) => self.expr(a, s),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Min(a, b) | Expr::Max(a, b) => {
                self.expr(a, s);
                self.expr(b, s);
            }
            Expr::IfElse(c, a, b) => {
                self.cond(c, s);
                self.expr(a, s);
                self.expr(b, s);
            }
        }
    }

    fn cond(&mut self, c: &Cond, s: &Scope) {
        match c {
            Cond::True => {}
            Cond::Not(a) => self.cond(a, s),
            Cond::And(v) | Cond::Or(v) => {
                for x in v {
                    self.cond(x, s)
                }
            }
            Cond::Cmp(a, _, b) => {
                self.expr(a, s);
                self.expr(b, s);
            }
            Cond::InZone(o, z) => {
                self.oref(*o, s);
                self.zref(z);
            }
            Cond::IsTemplate(o, t) => {
                self.oref(*o, s);
                if (*t as usize) >= self.g.templates.len() {
                    self.err("template id out of range");
                }
            }
            Cond::IsKind(o, _) | Cond::Attached(o) => self.oref(*o, s),
            Cond::SamePlayer(a, b) => {
                self.pref(a, s);
                self.pref(b, s);
            }
            Cond::Exists(sel) => self.sel(sel, s),
            Cond::HasRoom(z) => self.zref(z),
            Cond::IsActive(p) => self.pref(p, s),
            Cond::EventToZone(z) | Cond::EventFromZone(z) => {
                if !s.event {
                    self.err("event condition not in scope")
                }
                if (*z as usize) >= self.g.zones.len() {
                    self.err("zone out of range")
                }
            }
            Cond::InPhase(p) => {
                if (*p as usize) >= self.g.phases.len() {
                    self.err("phase id out of range")
                }
            }
        }
    }

    fn effect(&mut self, e: &Effect, s: &Scope) {
        match e {
            Effect::Seq(v) => {
                for x in v {
                    self.effect(x, s)
                }
            }
            Effect::If(c, a, b) => {
                self.cond(c, s);
                self.effect(a, s);
                self.effect(b, s);
            }
            Effect::ForEach(sel, body) => {
                self.sel(sel, s);
                let mut s2 = *s;
                s2.iter_depth += 1;
                self.effect(body, &s2);
            }
            Effect::ForEachPlayer(_, body) => self.effect(body, s),
            Effect::Repeat(n, body) => {
                self.expr(n, s);
                self.effect(body, s);
            }
            Effect::Move { what, to, .. } => {
                self.sel(what, s);
                self.zref(to);
            }
            Effect::Create { template, to, owner } => {
                if (*template as usize) >= self.g.templates.len() {
                    self.err("template id out of range");
                }
                self.zref(to);
                self.pref(owner, s);
            }
            Effect::Copy { what, to } => {
                self.oref(*what, s);
                self.zref(to);
            }
            Effect::Destroy(sel) => self.sel(sel, s),
            Effect::Transform(sel, t) => {
                self.sel(sel, s);
                if (*t as usize) >= self.g.templates.len() {
                    self.err("template id out of range");
                }
            }
            Effect::SetController(sel, p) => {
                self.sel(sel, s);
                self.pref(p, s);
            }
            Effect::SetAttr(sel, a, x) | Effect::ModAttr(sel, a, x) => {
                self.sel(sel, s);
                if (*a as usize) >= self.g.attrs.len() {
                    self.err("attr id out of range");
                }
                // The value expression is evaluated once per selected object
                // with `Iter` bound to it.
                let mut s2 = *s;
                s2.iter_depth += 1;
                self.expr(x, &s2);
            }
            Effect::SetRes(p, r, x) | Effect::ModRes(p, r, x) => {
                self.pref(p, s);
                if (*r as usize) >= self.g.resources.len() {
                    self.err("resource id out of range");
                }
                self.expr(x, s);
            }
            Effect::SetVar(v, x) | Effect::ModVar(v, x) => {
                if (*v as usize) >= self.g.vars.len() {
                    self.err("var id out of range");
                }
                self.expr(x, s);
            }
            Effect::Shuffle(z) => self.zref(z),
            Effect::Reveal(sel, _) | Effect::Hide(sel) | Effect::Detach(sel) => self.sel(sel, s),
            Effect::Attach(sel, o) => {
                self.sel(sel, s);
                self.oref(*o, s);
            }
            Effect::Hook(o, h) => {
                self.oref(*o, s);
                if (*h as usize) >= self.g.hooks.len() {
                    self.err("hook id out of range");
                }
            }
            Effect::Emit(_, x) => self.expr(x, s),
            Effect::Delay { effect, .. } => {
                let mut s2 = *s;
                s2.iter_depth = 0;
                self.effect(effect, &s2);
            }
            Effect::Win(p) | Effect::Lose(p) => self.pref(p, s),
            Effect::CancelStack | Effect::DrawGame | Effect::EndPhase | Effect::EndTurn | Effect::NoOp => {}
        }
    }

    /// Object targets must be visible to the actor, otherwise the action
    /// would have to expose hidden identities.
    fn target_visibility(&mut self, sel: &Sel) {
        if let Sel::Zones { zones, .. } = sel {
            for z in zones {
                if let Some(zd) = self.g.zones.get(z.zone as usize) {
                    let ok = match zd.vis {
                        Visibility::Public => true,
                        Visibility::Private => matches!(z.who, ZWho::Me | ZWho::Active),
                        Visibility::Hidden => false,
                    };
                    if !ok {
                        self.err(format!("target selector uses zone '{}' whose contents are not visible to the actor", zd.name));
                    }
                }
            }
        } else {
            self.err("object targets must use a zone selector");
        }
    }
}

/// Returns all validation errors (empty = valid).
pub fn validate(g: &GameDef) -> Vec<String> {
    let mut v = V { g, errs: vec![], ctx: String::new(), number_targets: vec![] };
    macro_rules! limit {
        ($cond:expr, $msg:expr) => {
            if !$cond {
                v.errs.push($msg.to_string());
            }
        };
    }
    limit!((2..=MAX_PLAYERS as u8).contains(&g.num_players), "num_players must be 2..=4");
    limit!(g.resources.len() <= MAX_RESOURCES, "too many resources");
    limit!(g.attrs.len() <= MAX_ATTRS, "too many attrs");
    limit!(g.vars.len() <= MAX_VARS, "too many vars");
    limit!(g.zones.len() <= MAX_ZONES && !g.zones.is_empty(), "zone count out of range");
    limit!(g.templates.len() <= MAX_TEMPLATES && !g.templates.is_empty(), "template count out of range");
    limit!(g.actions.len() <= MAX_ACTION_DEFS && !g.actions.is_empty(), "action def count out of range");
    limit!(!g.phases.is_empty() && g.phases.len() <= 16, "phase count out of range");
    limit!(g.hooks.len() <= 16, "too many hooks");
    limit!(g.limits.max_turns >= 1, "max_turns must be >= 1");
    for r in &g.resources {
        limit!(r.min <= r.initial && r.initial <= r.max, format!("resource '{}' initial outside [min,max]", r.name));
    }
    for a in &g.attrs {
        limit!(a.min <= a.default && a.default <= a.max, format!("attr '{}' default outside [min,max]", a.name));
    }
    if let Timeout::ByResource(r) = g.timeout {
        limit!((r as usize) < g.resources.len(), "timeout resource out of range");
    }

    // Setup entries.
    for (what, list, per_player) in [("player_setup", &g.player_setup, true), ("shared_setup", &g.shared_setup, false)] {
        for e in list.iter() {
            match g.zones.get(e.zone as usize) {
                None => v.errs.push(format!("{}: zone out of range", what)),
                Some(z) if z.per_player != per_player => v.errs.push(format!("{}: zone '{}' has wrong ownership kind", what, z.name)),
                _ => {}
            }
            limit!((e.template as usize) < g.templates.len(), format!("{}: template out of range", what));
        }
    }

    let base = Scope::default();
    v.ctx = "setup".into();
    v.effect(&g.setup, &base);

    for (i, p) in g.phases.iter().enumerate() {
        v.ctx = format!("phase {} ({})", i, p.name);
        v.effect(&p.on_enter, &base);
    }

    for (ti, t) in g.templates.iter().enumerate() {
        for (a, _) in &t.attrs {
            limit!((*a as usize) < g.attrs.len(), format!("template {}: attr out of range", ti));
        }
        let hs = Scope { source: true, ..Default::default() };
        for (h, e) in &t.hooks {
            v.ctx = format!("template {} ({}) hook {}", ti, t.name, h);
            if (*h as usize) >= g.hooks.len() {
                v.err("hook id out of range");
            }
            v.effect(e, &hs);
        }
        let es = Scope { source: true, event: true, ..Default::default() };
        for (k, tr) in t.triggers.iter().enumerate() {
            v.ctx = format!("template {} ({}) trigger {}", ti, t.name, k);
            for z in &tr.active_in {
                limit!((*z as usize) < g.zones.len(), "trigger zone out of range");
            }
            v.cond(&tr.cond, &es);
            v.effect(&tr.effect, &es);
        }
        for (k, m) in t.modifiers.iter().enumerate() {
            v.ctx = format!("template {} ({}) modifier {}", ti, t.name, k);
            v.sel(&m.affects, &hs);
            let ds = Scope { source: true, iter_depth: 1, ..Default::default() };
            v.expr(&m.delta, &ds);
            limit!((m.attr as usize) < g.attrs.len(), "modifier attr out of range");
        }
    }

    let gs = Scope { event: true, ..Default::default() };
    for (k, tr) in g.triggers.iter().enumerate() {
        v.ctx = format!("global trigger {}", k);
        v.cond(&tr.cond, &gs);
        v.effect(&tr.effect, &gs);
    }

    for (k, r) in g.terminal.iter().enumerate() {
        v.ctx = format!("terminal rule {}", k);
        v.cond(&r.cond, &base);
    }

    for (ai, a) in g.actions.iter().enumerate() {
        v.ctx = format!("action {} ({})", ai, a.name);
        for p in &a.phases {
            if (*p as usize) >= g.phases.len() {
                v.err("phase out of range");
            }
        }
        if a.targets.len() > MAX_TARGETS {
            v.err("too many targets");
        }
        v.number_targets = a.targets.iter().map(|t| matches!(t, TargetSpec::Number { .. })).collect();
        let mut sc = Scope { source: a.source.is_some(), ..Default::default() };
        if let Some(src) = &a.source {
            for z in &src.zones {
                v.zref(z);
            }
            let mut s2 = sc;
            s2.iter_depth = 1;
            v.cond(&src.filter, &s2);
        }
        // Targets are evaluated with earlier targets in scope.
        for (i, t) in a.targets.iter().enumerate() {
            let mut st = sc;
            st.ntargets = i;
            st.iter_depth = 1;
            match t {
                TargetSpec::Object(sel) => {
                    v.target_visibility(sel);
                    v.sel(sel, &st);
                }
                TargetSpec::Player(_) => {}
                TargetSpec::Number { lo, hi } => {
                    st.iter_depth = 0;
                    v.expr(lo, &st);
                    v.expr(hi, &st);
                }
            }
        }
        sc.ntargets = a.targets.len();
        for (r, x) in &a.costs {
            limit!((*r as usize) < g.resources.len(), "cost resource out of range");
            v.expr(x, &sc);
        }
        v.cond(&a.require, &sc);
        v.effect(&a.on_use, &sc);
        v.effect(&a.effect, &sc);
        if a.stack && a.timing == Timing::Response && a.source.is_none() && a.targets.is_empty() {
            // allowed, just noting nothing to check
        }
    }
    v.errs
}
