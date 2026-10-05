//! Family "garden": plant slow-maturing assets that yield income, harvest for
//! points. Strategic structure: invest vs cash out (shares the
//! `invest_vs_cash` tradeoff with "engine" while having entirely different
//! surface mechanics: hands, plots, ageing; no shared market).

use crate::build::*;
use crate::Variant;
use omnia_dsl::*;
use omnia_engine::Rng;

const DECK: u8 = 0;
const HAND: u8 = 1;
const PLOT: u8 = 2;
const DISCARD: u8 = 3;
const GOLD: u8 = 0;
const VP: u8 = 1;
const HARVESTS: u8 = 2;
const COST: u8 = 0;
const YIELD: u8 = 1;
const VALUE: u8 = 2;
const AGE: u8 = 3;
const MATURE: u8 = 4;

pub fn generate(rng: &mut Rng, variant: Variant) -> GameDef {
    let alt = variant == Variant::AltVictory;
    let inverted = variant == Variant::Inverted;
    let vp_target = range(rng, 5, 9);
    let harvest_target = range(rng, 2, 4);
    let plot_cap = range(rng, 3, 6) as u16;
    let hand_cap = range(rng, 5, 7) as u16;
    let gold0 = range(rng, 2, 4);
    let start_hand = range(rng, 3, 4);

    let n_types = range(rng, 4, 7);
    let mut templates = vec![];
    let mut counts = vec![];
    for i in 0..n_types {
        let mature = range(rng, 1, 4);
        let yld = range(rng, 0, 3);
        let value = range(rng, 2, 3) + mature / 2;
        let raw_cost = ((value + yld * mature) as f32 * 0.7).round() as i32 + range(rng, 0, 1);
        let cost = if inverted { (7 - raw_cost).clamp(1, 7) } else { raw_cost.clamp(1, 7) };
        templates.push(template(&format!("plant{}", i), 0, vec![(COST, cost), (YIELD, yld), (VALUE, value.min(8)), (MATURE, mature)]));
        counts.push(range(rng, 3, 5) as u16);
    }
    let player_setup: Vec<SetupEntry> = counts.iter().enumerate().map(|(i, n)| SetupEntry { zone: DECK, template: i as u16, count: *n }).collect();

    let mut plant = action("plant", 0, vec![1], Timing::Main, mv_one(ORef::Source, me(PLOT), Pos::Bottom));
    plant.source = Some(SourceSpec { zones: vec![me(HAND)], filter: Cond::True });
    plant.costs = vec![(GOLD, attr(ORef::Source, COST))];
    plant.require = Cond::HasRoom(me(PLOT));
    let mut harvest_fx = vec![Effect::ModRes(PRef::Me, VP, attr(ORef::Source, VALUE))];
    if alt {
        harvest_fx.push(Effect::ModRes(PRef::Me, HARVESTS, c(1)));
    }
    harvest_fx.push(mv_one(ORef::Source, me(DISCARD), Pos::Top));
    let mut harvest = action("harvest", 1, vec![1], Timing::Main, seq(harvest_fx));
    harvest.source = Some(SourceSpec { zones: vec![me(PLOT)], filter: cmp(attr(ORef::Iter, AGE), CmpOp::Ge, attr(ORef::Iter, MATURE)) });
    let actions = vec![plant, harvest, action("end_turn", 2, vec![1], Timing::Main, Effect::EndTurn)];

    let mut resources = vec![resource("gold", gold0, 0, 40), resource("vp", 0, 0, 60)];
    if alt {
        resources.push(resource("harvests", 0, 0, 30));
    }
    let terminal = if alt {
        vec![TerminalRule { cond: cmp(res(PRef::Me, HARVESTS), CmpOp::Ge, c(harvest_target)), result: PlayerResult::Win }]
    } else {
        vec![TerminalRule { cond: cmp(res(PRef::Me, VP), CmpOp::Ge, c(vp_target)), result: PlayerResult::Win }]
    };

    GameDef {
        name: "garden".into(),
        num_players: 2,
        resources,
        attrs: vec![
            attr_def("cost", 0, 0, 10),
            attr_def("yield", 0, 0, 6),
            attr_def("value", 0, 0, 10),
            attr_def("age", 0, 0, 12),
            attr_def("mature", 0, 0, 6),
        ],
        vars: vec![],
        zones: vec![
            zone("deck", true, true, Visibility::Hidden, None),
            zone("hand", true, false, Visibility::Private, Some(hand_cap)),
            zone("plot", true, true, Visibility::Public, Some(plot_cap)),
            zone("discard", true, true, Visibility::Public, None),
        ],
        templates,
        hooks: vec![],
        player_setup,
        shared_setup: vec![],
        setup: for_players(seq(vec![Effect::Shuffle(me(DECK)), mv(top_n(me(DECK), start_hand), me(HAND), Pos::Top)])),
        phases: vec![
            PhaseDef {
                name: "grow".into(),
                on_enter: seq(vec![
                    Effect::ModRes(PRef::Me, GOLD, c(1)),
                    foreach(
                        zones_all(vec![me(PLOT)]),
                        seq(vec![Effect::ModRes(PRef::Me, GOLD, attr(ORef::Iter, YIELD)), Effect::ModAttr(one(ORef::Iter), AGE, c(1))]),
                    ),
                    draw_or_lose(DECK, HAND, 1),
                ]),
                auto_end: true,
            },
            PhaseDef { name: "main".into(), on_enter: Effect::NoOp, auto_end: false },
        ],
        actions,
        triggers: vec![],
        terminal,
        timeout: Timeout::ByResource(if alt { HARVESTS } else { VP }),
        adjudication: Adjudication::Rules,
        limits: Limits { max_turns: 60, max_decisions: 900, ..Default::default() },
        meta: Meta {
            family: "garden".into(),
            seed: 0,
            tags: tags(&[
                ("information", "hidden"),
                ("resource_pressure", "medium"),
                ("interaction", "low"),
                ("horizon", "long"),
                ("randomness", "low"),
                ("tempo_importance", "medium"),
                ("resource_conservation", "medium"),
                ("engine_building", "high"),
                ("reaction_system", "none"),
                ("tradeoff", "invest_vs_cash"),
            ]),
        },
    }
}
