use anyhow::Result;
use omnia_dsl::*;
use omnia_engine::*;

fn obj_name(st: &State, o: ObjId) -> String {
    if o == NONE_OBJ {
        return "-".into();
    }
    let ob = st.object(o);
    format!("{}#{}", st.game().def.templates[ob.template as usize].name, o)
}

/// Debug description of an action using the true state (CLI is omniscient).
pub fn action(st: &State, a: &Action) -> String {
    if a.is_pass() {
        return "pass".into();
    }
    let def = &st.game().def.actions[a.def as usize];
    let mut s = def.name.clone();
    if a.source != NONE_OBJ {
        s += &format!(" {}", obj_name(st, a.source));
    }
    for t in &a.targets {
        s += &match t {
            Target::Obj(o) => format!(" -> {}", obj_name(st, *o)),
            Target::Player(p) => format!(" -> player {p}"),
            Target::Num(n) => format!(" = {n}"),
        };
    }
    s
}

pub fn state(st: &State) {
    let g = st.game();
    println!("turn {} phase {} active {} stack {}", st.turn(), g.def.phases[st.phase() as usize].name, st.active_player(), st.stack_len());
    for p in 0..st.num_players() {
        let r: Vec<String> = g.def.resources.iter().enumerate().map(|(i, d)| format!("{}={}", d.name, st.resource(p, i))).collect();
        println!("  P{p}: {}", r.join(" "));
    }
    for inst in 0..g.n_instances {
        let (zd, owner) = g.inst[inst];
        let objs = st.zone_contents(inst);
        if objs.is_empty() {
            continue;
        }
        let names: Vec<String> = objs.iter().map(|o| obj_name(st, *o)).collect();
        let who = if owner == NO_PLAYER { "shared".to_string() } else { format!("P{owner}") };
        println!("  {} {}: {}", who, g.def.zones[zd as usize].name, names.join(", "));
    }
}

pub fn game(d: &GameDef) -> Result<()> {
    println!("game {} (hash {})  players {}", d.name, d.hash_hex(), d.num_players);
    println!("family {:?} seed {} tags {:?}", d.meta.family, d.meta.seed, d.meta.tags);
    println!("resources: {}", d.resources.iter().map(|r| format!("{}[{}..{}]", r.name, r.min, r.max)).collect::<Vec<_>>().join(", "));
    println!("attrs: {}", d.attrs.iter().map(|a| a.name.clone()).collect::<Vec<_>>().join(", "));
    println!("zones: {}", d.zones.iter().map(|z| format!("{}({}{}{:?})", z.name, if z.per_player { "per-player " } else { "shared " }, if z.ordered { "ordered " } else { "" }, z.vis)).collect::<Vec<_>>().join(", "));
    println!("templates: {}  phases: {}  actions: {}  triggers: {}", d.templates.len(), d.phases.len(), d.actions.len(), d.triggers.len());
    for a in &d.actions {
        println!("  action {:<18} class {} timing {:?} stack {} targets {} costs {}", a.name, a.class, a.timing, a.stack, a.targets.len(), a.costs.len());
    }
    let desc = analyze(d);
    println!("action descriptors (effect histogram; first 12 dims):");
    for (a, x) in d.actions.iter().zip(&desc.actions) {
        println!("  {:<18} {}", a.name, x[..12].iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(" "));
    }
    println!("adjudication {:?}, timeout {:?}", d.adjudication, d.timeout);
    Ok(())
}
