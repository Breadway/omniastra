//! Family "tempo": creature combat on a renewable-energy curve.
//! Strategic structure: scarce renewable resource, cheap low-impact vs
//! expensive high-impact objects, board interaction, hidden hands.

use crate::build::*;
use crate::Variant;
use omnia_dsl::*;
use omnia_engine::Rng;

// ids
const DECK: u8 = 0;
const HAND: u8 = 1;
const BOARD: u8 = 2;
const DISCARD: u8 = 3;
const LIFE: u8 = 0;
const ENERGY: u8 = 1;
const MAXE: u8 = 2;
const KILLS: u8 = 3;
const COST: u8 = 0;
const POWER: u8 = 1;
const HEALTH: u8 = 2;
const EXH: u8 = 3;
const FRESH: u8 = 4;

pub fn generate(rng: &mut Rng, variant: Variant) -> GameDef {
    let life0 = range(rng, 12, 28);
    let e_cap = range(rng, 5, 8);
    let hand_cap = range(rng, 5, 8) as u16;
    let board_cap = range(rng, 3, 6) as u16;
    let start_hand = range(rng, 3, 5);
    let haste = chance(rng, 0.25);
    let taunt = chance(rng, 0.6);
    let max_cost = range(rng, 4, 6);
    let alt = variant == Variant::AltVictory;
    let inverted = variant == Variant::Inverted;
    let kills_to_win = range(rng, 3, 5);

    let mut attrs = vec![attr_def("cost", 0, 0, 10), attr_def("power", 0, 0, 12), attr_def("health", 0, -50, 30), attr_def("exhausted", 0, 0, 1)];
    if !haste {
        attrs.push(attr_def("fresh", 0, 0, 1));
    }
    let mut resources = vec![resource("life", life0, -50, 60), resource("energy", 0, 0, 12), resource("max_energy", 0, 0, e_cap)];
    if alt {
        resources.push(resource("kills", 0, 0, 20));
    }

    // Creature curve.
    let n_creatures = range(rng, 4, 6);
    let mut templates: Vec<TemplateDef> = vec![];
    let mut weights: Vec<f32> = vec![];
    let mut cost_pool: Vec<i32> = (1..=max_cost).collect();
    rng.shuffle(&mut cost_pool);
    let mut costs: Vec<i32> = cost_pool.into_iter().take(n_creatures as usize).collect();
    costs.sort();
    for (i, cst) in costs.iter().enumerate() {
        let eff_cost = if inverted { max_cost + 1 - cst } else { *cst };
        let bonus = range(rng, 0, 1);
        let total = 2 * eff_cost + 1 + bonus;
        let split = 0.3 + 0.4 * rng.f32();
        let power = ((total as f32 * split).round() as i32).clamp(1, 10);
        let health = (total - power).max(1);
        templates.push(template(&format!("unit{}", i), 0, vec![(COST, *cst), (POWER, power), (HEALTH, health)]));
        weights.push(1.0 / (*cst as f32).sqrt() * (0.6 + rng.f32()));
    }
    let n_creature_templates = templates.len();

    // Spells.
    let mut hooks_used = 0;
    if !alt && chance(rng, 0.7) {
        let d = range(rng, 2, 4) * if inverted { 1 } else { 1 };
        let mut t = template("burn", 1, vec![(COST, range(rng, 2, 4))]);
        t.hooks.push((0, Effect::ModRes(PRef::Opp, LIFE, c(-d))));
        templates.push(t);
        weights.push(0.5 + rng.f32());
        hooks_used += 1;
    }
    let has_rally = chance(rng, 0.6);
    if has_rally {
        let mut t = template("rally", 1, vec![(COST, range(rng, 1, 3))]);
        t.hooks.push((0, foreach(zones_all(vec![me(BOARD)]), Effect::ModAttr(one(ORef::Iter), POWER, c(1)))));
        templates.push(t);
        weights.push(0.5 + rng.f32());
        hooks_used += 1;
    }
    let has_zap = chance(rng, 0.7);
    if has_zap {
        let d = range(rng, 2, 4);
        let mut t = template("zap", 2, vec![(COST, range(rng, 1, 3))]);
        t.hooks.push((0, Effect::ModAttr(one(ORef::Target(0)), HEALTH, c(-d))));
        templates.push(t);
        weights.push(0.6 + rng.f32());
        hooks_used += 1;
    }
    let _ = hooks_used;
    let has_untargeted = templates.iter().any(|t| t.kind == 1);

    // Deck composition.
    let deck_size = range(rng, 24, 30);
    let mut counts = vec![1u16; templates.len()];
    let wsum: f32 = weights.iter().sum();
    let mut remaining = deck_size - templates.len() as i32;
    while remaining > 0 {
        let mut u = rng.f32() * wsum;
        let mut k = 0;
        for (i, w) in weights.iter().enumerate() {
            if u < *w {
                k = i;
                break;
            }
            u -= w;
            k = i;
        }
        counts[k] += 1;
        remaining -= 1;
    }
    // Guarantee the deck is at least ~50% creatures.
    let creatures: u16 = counts[..n_creature_templates].iter().sum();
    if (creatures as i32) * 2 < deck_size {
        counts[0] += (deck_size as u16 / 2).saturating_sub(creatures);
    }
    let player_setup: Vec<SetupEntry> = counts.iter().enumerate().map(|(i, n)| SetupEntry { zone: DECK, template: i as u16, count: *n }).collect();

    // Actions.
    let ready = {
        let mut conds = vec![cmp(attr(ORef::Iter, EXH), CmpOp::Eq, c(0))];
        if !haste {
            conds.push(cmp(attr(ORef::Iter, FRESH), CmpOp::Eq, c(0)));
        }
        and(conds)
    };
    let mut actions = vec![];
    let mut play = action(
        "play_creature",
        0,
        vec![1],
        Timing::Main,
        seq({
            let mut v = vec![mv_one(ORef::Source, me(BOARD), Pos::Bottom)];
            if !haste {
                v.push(Effect::SetAttr(one(ORef::Source), FRESH, c(1)));
            }
            v
        }),
    );
    play.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::IsKind(ORef::Iter, 0) });
    play.costs = vec![(ENERGY, attr(ORef::Source, COST))];
    play.require = Cond::HasRoom(me(BOARD));
    actions.push(play);
    if has_untargeted {
        let mut a = action("cast_spell", 1, vec![1], Timing::Main, seq(vec![Effect::Hook(ORef::Source, 0), mv_one(ORef::Source, me(DISCARD), Pos::Top)]));
        a.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::IsKind(ORef::Iter, 1) });
        a.costs = vec![(ENERGY, attr(ORef::Source, COST))];
        actions.push(a);
    }
    if has_zap {
        let mut a = action("cast_targeted", 2, vec![1], Timing::Main, seq(vec![Effect::Hook(ORef::Source, 0), mv_one(ORef::Source, me(DISCARD), Pos::Top)]));
        a.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::IsKind(ORef::Iter, 2) });
        a.targets = vec![TargetSpec::Object(zones_all(vec![opp(BOARD)]))];
        a.costs = vec![(ENERGY, attr(ORef::Source, COST))];
        actions.push(a);
    }
    let mut atk = action(
        "attack_creature",
        3,
        vec![1],
        Timing::Main,
        seq(vec![
            Effect::ModAttr(one(ORef::Target(0)), HEALTH, neg(attr(ORef::Source, POWER))),
            Effect::ModAttr(one(ORef::Source), HEALTH, neg(attr(ORef::Target(0), POWER))),
            Effect::SetAttr(one(ORef::Source), EXH, c(1)),
        ]),
    );
    atk.source = Some(SourceSpec { zones: vec![me(BOARD)], filter: ready.clone() });
    atk.targets = vec![TargetSpec::Object(zones_all(vec![opp(BOARD)]))];
    actions.push(atk);
    if !alt {
        let mut face = action(
            "attack_face",
            4,
            vec![1],
            Timing::Main,
            seq(vec![Effect::ModRes(PRef::Opp, LIFE, neg(attr(ORef::Source, POWER))), Effect::SetAttr(one(ORef::Source), EXH, c(1))]),
        );
        face.source = Some(SourceSpec { zones: vec![me(BOARD)], filter: ready });
        face.targets = vec![TargetSpec::Player(PSel::Others)];
        if taunt {
            face.require = Cond::Not(Box::new(Cond::Exists(Box::new(zones_all(vec![opp(BOARD)])))));
        }
        actions.push(face);
    }
    actions.push(action("end_turn", 5, vec![1], Timing::Main, Effect::EndTurn));

    // Phases.
    let mut refresh = vec![
        Effect::ModRes(PRef::Me, MAXE, c(1)),
        Effect::SetRes(PRef::Me, ENERGY, res(PRef::Me, MAXE)),
    ];
    let mut reset = vec![Effect::SetAttr(one(ORef::Iter), EXH, c(0))];
    if !haste {
        reset.push(Effect::SetAttr(one(ORef::Iter), FRESH, c(0)));
    }
    refresh.push(foreach(zones_all(vec![me(BOARD)]), seq(reset)));
    refresh.push(draw_or_lose(DECK, HAND, 1));
    let phases = vec![
        PhaseDef { name: "refresh".into(), on_enter: seq(refresh), auto_end: true },
        PhaseDef { name: "main".into(), on_enter: Effect::NoOp, auto_end: false },
    ];

    // Death trigger (+ kill counting in the alternate-victory variant).
    let mut death = vec![mv_one(ORef::EventObj, owner_of(ORef::EventObj, DISCARD), Pos::Top)];
    if alt {
        death.push(if_(
            Cond::Not(Box::new(Cond::SamePlayer(PRef::OwnerOf(ORef::EventObj), PRef::EventActor))),
            Effect::ModRes(PRef::EventActor, KILLS, c(1)),
            Effect::NoOp,
        ));
    }
    let triggers = vec![TriggerDef {
        on: EventKind::AttrChanged,
        active_in: vec![],
        cond: and(vec![cmp(Expr::EventSlot, CmpOp::Eq, c(HEALTH as i32)), cmp(attr(ORef::EventObj, HEALTH), CmpOp::Le, c(0))]),
        effect: seq(death),
    }];

    let terminal = if alt {
        vec![TerminalRule { cond: cmp(res(PRef::Me, KILLS), CmpOp::Ge, c(kills_to_win)), result: PlayerResult::Win }]
    } else {
        vec![TerminalRule { cond: cmp(res(PRef::Me, LIFE), CmpOp::Le, c(0)), result: PlayerResult::Lose }]
    };

    GameDef {
        name: "tempo".into(),
        num_players: 2,
        resources,
        attrs,
        vars: vec![],
        zones: vec![
            zone("deck", true, true, Visibility::Hidden, None),
            zone("hand", true, false, Visibility::Private, Some(hand_cap)),
            zone("board", true, true, Visibility::Public, Some(board_cap)),
            zone("discard", true, true, Visibility::Public, None),
        ],
        templates,
        hooks: vec!["on_play".into()],
        player_setup,
        shared_setup: vec![],
        setup: for_players(seq(vec![Effect::Shuffle(me(DECK)), mv(top_n(me(DECK), start_hand), me(HAND), Pos::Top)])),
        phases,
        actions,
        triggers,
        terminal,
        timeout: Timeout::ByResource(if alt { KILLS } else { LIFE }),
        adjudication: Adjudication::Rules,
        limits: Limits { max_turns: 40, ..Default::default() },
        meta: Meta {
            family: "tempo".into(),
            seed: 0,
            tags: tags(&[
                ("information", "hidden"),
                ("resource_pressure", "high"),
                ("interaction", "high"),
                ("horizon", "short"),
                ("randomness", "low"),
                ("tempo_importance", "high"),
                ("resource_conservation", "medium"),
                ("engine_building", "none"),
                ("reaction_system", "none"),
                ("tradeoff", "tempo_vs_value"),
            ]),
        },
    }
}
