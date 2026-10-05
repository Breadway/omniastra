//! Terse constructors for assembling OmniDSL programs in the generator.

use omnia_dsl::*;
use omnia_engine::Rng;

pub fn zr(who: ZWho, zone: u8) -> ZRef {
    ZRef { who, zone }
}
pub fn me(zone: u8) -> ZRef {
    zr(ZWho::Me, zone)
}
pub fn opp(zone: u8) -> ZRef {
    zr(ZWho::Opp, zone)
}
pub fn shared(zone: u8) -> ZRef {
    zr(ZWho::Shared, zone)
}
pub fn owner_of(o: ORef, zone: u8) -> ZRef {
    zr(ZWho::OwnerOf(o), zone)
}

pub fn c(v: i32) -> Expr {
    Expr::Const(v)
}
pub fn attr(o: ORef, a: u8) -> Expr {
    Expr::Attr(o, a)
}
pub fn res(p: PRef, r: u8) -> Expr {
    Expr::Res(p, r)
}
pub fn add(a: Expr, b: Expr) -> Expr {
    Expr::Add(Box::new(a), Box::new(b))
}
pub fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Sub(Box::new(a), Box::new(b))
}
pub fn neg(a: Expr) -> Expr {
    Expr::Neg(Box::new(a))
}
pub fn cmp(a: Expr, op: CmpOp, b: Expr) -> Cond {
    Cond::Cmp(Box::new(a), op, Box::new(b))
}
pub fn and(v: Vec<Cond>) -> Cond {
    Cond::And(v)
}

pub fn zones_all(zones: Vec<ZRef>) -> Sel {
    Sel::Zones { zones, filter: Cond::True, order: SelOrder::All }
}
pub fn zones_where(zones: Vec<ZRef>, filter: Cond) -> Sel {
    Sel::Zones { zones, filter, order: SelOrder::All }
}
pub fn top_n(zone: ZRef, n: i32) -> Sel {
    Sel::Zones { zones: vec![zone], filter: Cond::True, order: SelOrder::Top(Box::new(c(n))) }
}
pub fn one(o: ORef) -> Sel {
    Sel::One(o)
}

pub fn mv(what: Sel, to: ZRef, pos: Pos) -> Effect {
    Effect::Move { what, to, pos }
}
pub fn mv_one(o: ORef, to: ZRef, pos: Pos) -> Effect {
    mv(Sel::One(o), to, pos)
}
pub fn seq(v: Vec<Effect>) -> Effect {
    Effect::Seq(v)
}
pub fn if_(c: Cond, a: Effect, b: Effect) -> Effect {
    Effect::If(c, Box::new(a), Box::new(b))
}
pub fn foreach(s: Sel, e: Effect) -> Effect {
    Effect::ForEach(s, Box::new(e))
}
pub fn for_players(e: Effect) -> Effect {
    Effect::ForEachPlayer(PSel::All, Box::new(e))
}

/// Draw `n` from `deck` into `hand` for `Me`; lose if the deck is empty.
pub fn draw_or_lose(deck: u8, hand: u8, n: i32) -> Effect {
    if_(Cond::Exists(Box::new(zones_all(vec![me(deck)]))), mv(top_n(me(deck), n), me(hand), Pos::Top), Effect::Lose(PRef::Me))
}

pub fn action(name: &str, class: u8, phases: Vec<u8>, timing: Timing, effect: Effect) -> ActionDef {
    ActionDef {
        name: name.into(),
        class,
        phases,
        timing,
        source: None,
        targets: vec![],
        costs: vec![],
        require: Cond::True,
        on_use: Effect::NoOp,
        effect,
        stack: false,
    }
}

pub fn template(name: &str, kind: u8, attrs: Vec<(u8, i32)>) -> TemplateDef {
    TemplateDef { name: name.into(), kind, attrs, hooks: vec![], triggers: vec![], modifiers: vec![] }
}

pub fn zone(name: &str, per_player: bool, ordered: bool, vis: Visibility, capacity: Option<u16>) -> ZoneDef {
    ZoneDef { name: name.into(), per_player, ordered, vis, capacity, allowed_kinds: None }
}

pub fn resource(name: &str, initial: i32, min: i32, max: i32) -> ResourceDef {
    ResourceDef { name: name.into(), initial, min, max, public: true }
}

pub fn attr_def(name: &str, default: i32, min: i32, max: i32) -> AttrDef {
    AttrDef { name: name.into(), default, min, max }
}

// ---- randomness helpers -----------------------------------------------------

pub fn range(rng: &mut Rng, lo: i32, hi: i32) -> i32 {
    rng.range_i32(lo, hi)
}
pub fn chance(rng: &mut Rng, p: f32) -> bool {
    rng.f32() < p
}
pub fn pick<T: Copy>(rng: &mut Rng, xs: &[T]) -> T {
    xs[rng.below(xs.len() as u64) as usize]
}

pub fn tags(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
}
