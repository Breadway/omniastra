//! Static analysis producing game-independent *descriptor* vectors.
//!
//! Descriptors summarise what a template / action / attribute / resource /
//! zone *does in the rules* (e.g. "this action moves objects, spends
//! resource-like quantities and ends the phase") without naming it. They are
//! the v0 stand-in for full rule-token encoding: the ML side may consume them
//! to ground otherwise arbitrary per-game slot indices.

use crate::ir::*;

pub const DESC_DIM: usize = 16;
pub type Desc = [f32; DESC_DIM];

#[derive(Clone, Debug, Default)]
pub struct GameDescriptors {
    pub templates: Vec<Desc>,
    pub actions: Vec<Desc>,
    pub attrs: Vec<Desc>,
    pub resources: Vec<Desc>,
    pub zones: Vec<Desc>,
}

fn symlog(v: f32) -> f32 {
    v.signum() * (1.0 + v.abs()).ln()
}

#[derive(Default, Clone)]
struct Tally {
    eff: [f32; 12],
    consts: f32,
    random: f32,
    conds: f32,
    // Attr / resource usage.
    attr_read: [f32; MAX_ATTRS],
    attr_write: [f32; MAX_ATTRS],
    attr_cmp_res: [f32; MAX_ATTRS],
    res_read: [f32; MAX_RESOURCES],
    res_gain: [f32; MAX_RESOURCES],
    res_loss: [f32; MAX_RESOURCES],
    res_set: [f32; MAX_RESOURCES],
    zone_dst: [f32; MAX_ZONES],
    zone_src: [f32; MAX_ZONES],
}

impl Tally {
    fn expr(&mut self, e: &Expr) {
        match e {
            Expr::Const(c) => self.consts += c.abs() as f32,
            Expr::Attr(_, a) => self.attr_read[*a as usize % MAX_ATTRS] += 1.0,
            Expr::Res(_, r) => self.res_read[*r as usize % MAX_RESOURCES] += 1.0,
            Expr::Count(s) => self.sel(s),
            Expr::Rand(..) => self.random += 1.0,
            Expr::Neg(a) => self.expr(a),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Min(a, b) | Expr::Max(a, b) => {
                self.expr(a);
                self.expr(b)
            }
            Expr::IfElse(c, a, b) => {
                self.cond(c);
                self.expr(a);
                self.expr(b)
            }
            _ => {}
        }
    }

    fn zref(&mut self, z: &ZRef, dst: bool) {
        let i = z.zone as usize % MAX_ZONES;
        if dst {
            self.zone_dst[i] += 1.0
        } else {
            self.zone_src[i] += 1.0
        }
    }

    fn sel(&mut self, s: &Sel) {
        if let Sel::Zones { zones, filter, order } = s {
            for z in zones {
                self.zref(z, false);
            }
            self.cond(filter);
            match order {
                SelOrder::All => {}
                SelOrder::Top(e) | SelOrder::Bottom(e) => self.expr(e),
                SelOrder::Random(e) => {
                    self.random += 1.0;
                    self.expr(e)
                }
            }
        }
    }

    fn cond(&mut self, c: &Cond) {
        self.conds += 1.0;
        match c {
            Cond::Not(a) => self.cond(a),
            Cond::And(v) | Cond::Or(v) => v.iter().for_each(|x| self.cond(x)),
            Cond::Cmp(a, _, b) => {
                // Attribute compared against a resource = "affordability" pattern.
                let (mut a_attr, mut b_res) = (vec![], false);
                collect_attrs(a, &mut a_attr);
                b_res |= has_res(b);
                let (mut b_attr, mut a_res) = (vec![], false);
                collect_attrs(b, &mut b_attr);
                a_res |= has_res(a);
                if b_res {
                    a_attr.iter().for_each(|i| self.attr_cmp_res[*i % MAX_ATTRS] += 1.0);
                }
                if a_res {
                    b_attr.iter().for_each(|i| self.attr_cmp_res[*i % MAX_ATTRS] += 1.0);
                }
                self.expr(a);
                self.expr(b);
            }
            Cond::Exists(s) => self.sel(s),
            Cond::HasRoom(z) => self.zref(z, true),
            _ => {}
        }
    }

    fn effect(&mut self, e: &Effect) {
        match e {
            Effect::Seq(v) => v.iter().for_each(|x| self.effect(x)),
            Effect::If(c, a, b) => {
                self.eff[10] += 1.0;
                self.cond(c);
                self.effect(a);
                self.effect(b)
            }
            Effect::ForEach(s, b) => {
                self.eff[9] += 1.0;
                self.sel(s);
                self.effect(b)
            }
            Effect::ForEachPlayer(_, b) => {
                self.eff[9] += 1.0;
                self.effect(b)
            }
            Effect::Repeat(n, b) => {
                self.eff[9] += 1.0;
                self.expr(n);
                self.effect(b)
            }
            Effect::Move { what, to, pos } => {
                self.eff[0] += 1.0;
                if matches!(pos, Pos::Random) {
                    self.random += 1.0;
                }
                self.sel(what);
                self.zref(to, true)
            }
            Effect::Create { to, .. } | Effect::Copy { to, .. } => {
                self.eff[1] += 1.0;
                self.zref(to, true)
            }
            Effect::Transform(s, _) => {
                self.eff[1] += 1.0;
                self.sel(s)
            }
            Effect::Destroy(s) => {
                self.eff[2] += 1.0;
                self.sel(s)
            }
            Effect::SetController(s, _) | Effect::Attach(s, _) | Effect::Detach(s) => {
                self.eff[12 - 1] += 1.0;
                self.sel(s)
            }
            Effect::SetAttr(s, a, x) | Effect::ModAttr(s, a, x) => {
                self.eff[3] += 1.0;
                self.attr_write[*a as usize % MAX_ATTRS] += 1.0;
                self.sel(s);
                self.expr(x)
            }
            Effect::SetRes(_, r, x) => {
                self.eff[4] += 1.0;
                self.res_set[*r as usize % MAX_RESOURCES] += 1.0;
                self.expr(x)
            }
            Effect::ModRes(_, r, x) => {
                let neg = matches!(x, Expr::Neg(_)) || matches!(x, Expr::Const(c) if *c < 0);
                let i = *r as usize % MAX_RESOURCES;
                if neg {
                    self.eff[5] += 1.0;
                    self.res_loss[i] += 1.0;
                } else {
                    self.eff[4] += 1.0;
                    self.res_gain[i] += 1.0;
                }
                self.expr(x)
            }
            Effect::SetVar(_, x) | Effect::ModVar(_, x) | Effect::Emit(_, x) => {
                self.eff[11] += 1.0;
                self.expr(x)
            }
            Effect::Shuffle(_) | Effect::Reveal(..) | Effect::Hide(_) => self.eff[6] += 1.0,
            Effect::Win(_) | Effect::Lose(_) | Effect::DrawGame => self.eff[7] += 1.0,
            Effect::EndPhase | Effect::EndTurn => self.eff[8] += 1.0,
            Effect::Delay { effect, .. } => {
                self.eff[11] += 1.0;
                self.effect(effect)
            }
            Effect::CancelStack => self.eff[11] += 1.0,
            Effect::Hook(..) | Effect::NoOp => {}
        }
    }

    fn eff_desc(&self, ntargets: usize, ncosts: usize) -> Desc {
        let mut d = [0.0; DESC_DIM];
        for i in 0..12 {
            d[i] = (1.0 + self.eff[i]).ln();
        }
        d[12] = ntargets as f32;
        d[13] = ncosts as f32;
        d[14] = symlog(self.consts) * 0.25;
        d[15] = (self.random > 0.0) as u8 as f32;
        d
    }
}

fn collect_attrs(e: &Expr, out: &mut Vec<usize>) {
    match e {
        Expr::Attr(_, a) => out.push(*a as usize),
        Expr::Neg(a) => collect_attrs(a, out),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Min(a, b) | Expr::Max(a, b) => {
            collect_attrs(a, out);
            collect_attrs(b, out)
        }
        _ => {}
    }
}

fn has_res(e: &Expr) -> bool {
    match e {
        Expr::Res(..) => true,
        Expr::Neg(a) => has_res(a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Min(a, b) | Expr::Max(a, b) => has_res(a) || has_res(b),
        _ => false,
    }
}

pub fn analyze(g: &GameDef) -> GameDescriptors {
    let mut out = GameDescriptors::default();

    // Global usage tallies (actions + templates + phases + triggers + terminal).
    let mut all = Tally::default();
    let mut cost_attr = [0f32; MAX_ATTRS];
    let mut cost_res = [0f32; MAX_RESOURCES];
    let mut term_attr = [0f32; MAX_ATTRS];
    let mut term_res = [0f32; MAX_RESOURCES];
    let mut act_src_zone = [0f32; MAX_ZONES];
    let mut trig_zone = [0f32; MAX_ZONES];

    for a in &g.actions {
        let mut t = Tally::default();
        t.effect(&a.effect);
        t.effect(&a.on_use);
        t.cond(&a.require);
        for (r, x) in &a.costs {
            t.expr(x);
            t.eff[5] += 1.0;
            cost_res[*r as usize % MAX_RESOURCES] += 1.0;
            let mut v = vec![];
            collect_attrs(x, &mut v);
            v.iter().for_each(|i| cost_attr[*i % MAX_ATTRS] += 1.0);
        }
        if let Some(s) = &a.source {
            for z in &s.zones {
                act_src_zone[z.zone as usize % MAX_ZONES] += 1.0;
            }
        }
        let mut d = t.eff_desc(a.targets.len(), a.costs.len());
        // Action descriptors reuse slot 15 for "uses stack" instead of random.
        d[15] = (t.random > 0.0) as u8 as f32 * 0.5 + a.stack as u8 as f32 * 0.5;
        out.actions.push(d);
        merge(&mut all, &t);
    }
    for tp in &g.templates {
        let mut t = Tally::default();
        for (_, e) in &tp.hooks {
            t.effect(e);
        }
        for tr in &tp.triggers {
            t.effect(&tr.effect);
            t.cond(&tr.cond);
            for z in &tr.active_in {
                trig_zone[*z as usize % MAX_ZONES] += 1.0;
            }
        }
        for m in &tp.modifiers {
            t.expr(&m.delta);
            t.sel(&m.affects);
            t.eff[3] += 1.0;
            t.attr_write[m.attr as usize % MAX_ATTRS] += 1.0;
        }
        out.templates.push(t.eff_desc(0, 0));
        merge(&mut all, &t);
    }
    for p in &g.phases {
        let mut t = Tally::default();
        t.effect(&p.on_enter);
        merge(&mut all, &t);
    }
    for tr in &g.triggers {
        let mut t = Tally::default();
        t.effect(&tr.effect);
        t.cond(&tr.cond);
        merge(&mut all, &t);
    }
    for r in &g.terminal {
        let mut t = Tally::default();
        t.cond(&r.cond);
        // Terminal usage.
        fn walk_cond(c: &Cond, ta: &mut [f32; MAX_ATTRS], tr: &mut [f32; MAX_RESOURCES]) {
            match c {
                Cond::Not(a) => walk_cond(a, ta, tr),
                Cond::And(v) | Cond::Or(v) => v.iter().for_each(|x| walk_cond(x, ta, tr)),
                Cond::Cmp(a, _, b) => {
                    for e in [a, b] {
                        walk_expr(e, ta, tr)
                    }
                }
                _ => {}
            }
        }
        fn walk_expr(e: &Expr, ta: &mut [f32; MAX_ATTRS], tr: &mut [f32; MAX_RESOURCES]) {
            match e {
                Expr::Attr(_, a) => ta[*a as usize % MAX_ATTRS] += 1.0,
                Expr::Res(_, r) => tr[*r as usize % MAX_RESOURCES] += 1.0,
                Expr::Neg(a) => walk_expr(a, ta, tr),
                Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Min(a, b) | Expr::Max(a, b) => {
                    walk_expr(a, ta, tr);
                    walk_expr(b, ta, tr)
                }
                _ => {}
            }
        }
        walk_cond(&r.cond, &mut term_attr, &mut term_res);
        merge(&mut all, &t);
    }

    for (i, a) in g.attrs.iter().enumerate() {
        let mut d = [0.0; DESC_DIM];
        d[0] = (1.0 + all.attr_read[i]).ln();
        d[1] = (1.0 + all.attr_write[i]).ln();
        d[2] = (1.0 + cost_attr[i]).ln();
        d[3] = (1.0 + term_attr[i]).ln();
        d[4] = (1.0 + all.attr_cmp_res[i]).ln();
        d[5] = symlog(a.default as f32) * 0.3;
        d[6] = symlog(a.min as f32) * 0.3;
        d[7] = symlog(a.max as f32) * 0.3;
        out.attrs.push(d);
    }
    for (i, r) in g.resources.iter().enumerate() {
        let mut d = [0.0; DESC_DIM];
        d[0] = (1.0 + all.res_gain[i]).ln();
        d[1] = (1.0 + all.res_loss[i]).ln();
        d[2] = (1.0 + cost_res[i]).ln();
        d[3] = (1.0 + all.res_read[i]).ln();
        d[4] = (1.0 + term_res[i]).ln();
        d[5] = (1.0 + all.res_set[i]).ln();
        d[6] = r.public as u8 as f32;
        d[7] = symlog(r.initial as f32) * 0.3;
        d[8] = symlog(r.min as f32) * 0.3;
        d[9] = symlog(r.max as f32) * 0.3;
        out.resources.push(d);
    }
    let mut init_size = [0f32; MAX_ZONES];
    for e in g.player_setup.iter().chain(g.shared_setup.iter()) {
        init_size[e.zone as usize % MAX_ZONES] += e.count as f32;
    }
    for (i, z) in g.zones.iter().enumerate() {
        let mut d = [0.0; DESC_DIM];
        d[0] = z.per_player as u8 as f32;
        d[1] = z.ordered as u8 as f32;
        d[2] = (z.vis == Visibility::Public) as u8 as f32;
        d[3] = (z.vis == Visibility::Private) as u8 as f32;
        d[4] = (z.vis == Visibility::Hidden) as u8 as f32;
        d[5] = symlog(z.capacity.unwrap_or(0) as f32) * 0.4;
        d[6] = (1.0 + act_src_zone[i]).ln();
        d[7] = (1.0 + all.zone_dst[i]).ln();
        d[8] = (1.0 + all.zone_src[i]).ln();
        d[9] = symlog(init_size[i]) * 0.4;
        d[10] = (1.0 + trig_zone[i]).ln();
        out.zones.push(d);
    }
    out
}

fn merge(a: &mut Tally, b: &Tally) {
    for i in 0..12 {
        a.eff[i] += b.eff[i];
    }
    for i in 0..MAX_ATTRS {
        a.attr_read[i] += b.attr_read[i];
        a.attr_write[i] += b.attr_write[i];
        a.attr_cmp_res[i] += b.attr_cmp_res[i];
    }
    for i in 0..MAX_RESOURCES {
        a.res_read[i] += b.res_read[i];
        a.res_gain[i] += b.res_gain[i];
        a.res_loss[i] += b.res_loss[i];
        a.res_set[i] += b.res_set[i];
    }
    for i in 0..MAX_ZONES {
        a.zone_dst[i] += b.zone_dst[i];
        a.zone_src[i] += b.zone_src[i];
    }
    a.consts += b.consts;
    a.random += b.random;
    a.conds += b.conds;
}
