//! Family "stack": points race with a response stack and counters.
//! Strategic structure: commit resources to proactive plays vs hold them for
//! reactions; hidden information about the opponent's answers.

use crate::build::*;
use crate::Variant;
use omnia_dsl::*;
use omnia_engine::Rng;

const DECK: u8 = 0;
const HAND: u8 = 1;
const DISCARD: u8 = 2;
const LIMBO: u8 = 3;
const POINTS: u8 = 0;
const FOCUS: u8 = 1;
const CASTS: u8 = 2;
const COST: u8 = 0;

pub fn generate(rng: &mut Rng, variant: Variant) -> GameDef {
    let alt = variant == Variant::AltVictory;
    let inverted = variant == Variant::Inverted;
    let target = range(rng, 6, 10);
    let focus_per_turn = range(rng, 2, 4);
    let start_hand = range(rng, 4, 6);
    let cast_target = range(rng, 5, 8);
    let n_counters = if chance(rng, 0.15) { 0 } else { range(rng, 4, 9) };
    let counter_cost = range(rng, 1, 2);
    let ratio = 0.8 + 0.6 * rng.f32();

    let mut templates: Vec<TemplateDef> = vec![];
    let mut counts: Vec<u16> = vec![];
    let pts = |cost: i32| -> i32 {
        let base = (cost as f32 * ratio).round() as i32 + 0;
        let p = if inverted { (5 - base).max(1) } else { base.max(1) };
        p.min(6)
    };
    // proactive point cards
    for cost in [1, 2, 3, 4] {
        if cost == 4 && chance(rng, 0.5) {
            continue;
        }
        let p = pts(cost);
        let mut t = template(&format!("gain{}", cost), 0, vec![(COST, cost)]);
        t.hooks.push((0, Effect::ModRes(PRef::Me, POINTS, c(p))));
        templates.push(t);
        counts.push(range(rng, 3, 6) as u16);
    }
    // drain
    {
        let cost = range(rng, 2, 3);
        let mut t = template("drain", 0, vec![(COST, cost)]);
        t.hooks.push((0, seq(vec![Effect::ModRes(PRef::Me, POINTS, c(1)), Effect::ModRes(PRef::Opp, POINTS, c(-range(rng, 1, 2)))])));
        templates.push(t);
        counts.push(range(rng, 3, 5) as u16);
    }
    // draw
    {
        let mut t = template("study", 0, vec![(COST, 1)]);
        t.hooks.push((0, mv(top_n(me(DECK), range(rng, 1, 2)), me(HAND), Pos::Top)));
        templates.push(t);
        counts.push(range(rng, 3, 5) as u16);
    }
    // tax
    if chance(rng, 0.5) {
        let mut t = template("tax", 0, vec![(COST, 1)]);
        t.hooks.push((0, Effect::ModRes(PRef::Opp, FOCUS, c(-range(rng, 1, 2)))));
        templates.push(t);
        counts.push(range(rng, 2, 4) as u16);
    }
    let n_action_templates = templates.len();
    let _ = n_action_templates;
    if n_counters > 0 {
        templates.push(template("counter", 1, vec![(COST, counter_cost)]));
        counts.push(n_counters as u16);
    }
    let player_setup: Vec<SetupEntry> = counts.iter().enumerate().map(|(i, n)| SetupEntry { zone: DECK, template: i as u16, count: *n }).collect();

    let mut actions = vec![];
    let mut cast = action("cast", 0, vec![1], Timing::Main, seq(vec![Effect::Hook(ORef::Source, 0), mv_one(ORef::Source, me(DISCARD), Pos::Top)]));
    cast.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::IsKind(ORef::Iter, 0) });
    cast.costs = vec![(FOCUS, attr(ORef::Source, COST))];
    cast.stack = true;
    cast.on_use = seq({
        let mut v = vec![mv_one(ORef::Source, shared(LIMBO), Pos::Bottom)];
        if alt {
            v.push(Effect::ModRes(PRef::Me, CASTS, c(1)));
        }
        v
    });
    actions.push(cast);
    if n_counters > 0 {
        let mut ctr = action("counter", 1, vec![], Timing::Response, seq(vec![Effect::CancelStack, mv_one(ORef::Source, me(DISCARD), Pos::Top)]));
        ctr.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::IsKind(ORef::Iter, 1) });
        ctr.costs = vec![(FOCUS, attr(ORef::Source, COST))];
        ctr.stack = true;
        ctr.on_use = mv_one(ORef::Source, shared(LIMBO), Pos::Bottom);
        actions.push(ctr);
    }
    actions.push(action("end_turn", 2, vec![1], Timing::Main, Effect::EndTurn));

    let mut resources = vec![resource("points", 0, 0, 30), resource("focus", 0, 0, 8)];
    if alt {
        resources.push(resource("casts", 0, 0, 30));
    }
    let terminal = if alt {
        vec![TerminalRule { cond: cmp(res(PRef::Me, CASTS), CmpOp::Ge, c(cast_target)), result: PlayerResult::Win }]
    } else {
        vec![TerminalRule { cond: cmp(res(PRef::Me, POINTS), CmpOp::Ge, c(target)), result: PlayerResult::Win }]
    };

    GameDef {
        name: "stack".into(),
        num_players: 2,
        resources,
        attrs: vec![attr_def("cost", 0, 0, 6)],
        vars: vec![],
        zones: vec![
            zone("deck", true, true, Visibility::Hidden, None),
            zone("hand", true, false, Visibility::Private, Some(8)),
            zone("discard", true, true, Visibility::Public, None),
            zone("limbo", false, true, Visibility::Public, None),
        ],
        templates,
        hooks: vec!["on_resolve".into()],
        player_setup,
        shared_setup: vec![],
        setup: for_players(seq(vec![Effect::Shuffle(me(DECK)), mv(top_n(me(DECK), start_hand), me(HAND), Pos::Top)])),
        phases: vec![
            PhaseDef { name: "draw".into(), on_enter: seq(vec![Effect::SetRes(PRef::Me, FOCUS, c(focus_per_turn)), draw_or_lose(DECK, HAND, 1)]), auto_end: true },
            PhaseDef { name: "main".into(), on_enter: Effect::NoOp, auto_end: false },
        ],
        actions,
        triggers: vec![],
        terminal,
        timeout: Timeout::ByResource(if alt { CASTS } else { POINTS }),
        adjudication: Adjudication::Rules,
        limits: Limits { max_turns: 30, ..Default::default() },
        meta: Meta {
            family: "stack".into(),
            seed: 0,
            tags: tags(&[
                ("information", "hidden"),
                ("resource_pressure", "medium"),
                ("interaction", "high"),
                ("horizon", "short"),
                ("randomness", "low"),
                ("tempo_importance", "medium"),
                ("resource_conservation", "high"),
                ("engine_building", "none"),
                ("reaction_system", if n_counters > 0 { "stack_counters" } else { "none" }),
                ("tradeoff", "counter_vs_commit"),
            ]),
        },
    }
}
