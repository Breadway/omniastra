//! Family "engine": shared-market engine building.
//! Strategic structure: invest in income now vs cash out for points later;
//! contested shared market; no direct attacks.

use crate::build::*;
use crate::Variant;
use omnia_dsl::*;
use omnia_engine::Rng;

const MARKET: u8 = 0;
const SUPPLY: u8 = 1;
const TABLEAU: u8 = 2;
const VP: u8 = 0;
const GOLD: u8 = 1;
const INCOME: u8 = 2;
const COST: u8 = 0;
const PTS: u8 = 1;
const INC: u8 = 2;

pub fn generate(rng: &mut Rng, variant: Variant) -> GameDef {
    let alt = variant == Variant::AltVictory;
    let inverted = variant == Variant::Inverted;
    let market_size = range(rng, 3, 5);
    let vp_target = range(rng, 6, 11);
    let take_gold = range(rng, 1, 3);
    let refresh_cost = pick(rng, &[Some(1), Some(2), None]);
    let multi_buy = chance(rng, 0.4);
    let income0 = range(rng, 1, 2);
    let inc_target = range(rng, 4, 6);

    let n_types = range(rng, 5, 8);
    let mut templates = vec![];
    let mut counts = vec![];
    for i in 0..n_types {
        let arche = pick(rng, &[0, 0, 1, 1, 2]); // engine, points, hybrid
        let (vp, inc) = match arche {
            0 => (0, range(rng, 1, 3)),
            1 => (range(rng, 1, 4), 0),
            _ => (range(rng, 1, 2), 1),
        };
        let value = vp * 2 + inc * 3;
        let base = ((value as f32) * 0.95).round() as i32 + range(rng, 0, 1);
        let cost = if inverted { (10 - base).clamp(1, 9) } else { base.clamp(1, 9) };
        templates.push(template(&format!("card{}", i), 0, vec![(COST, cost), (PTS, vp), (INC, inc)]));
        counts.push(range(rng, 2, 5) as u16);
    }
    let shared_setup: Vec<SetupEntry> = counts.iter().enumerate().map(|(i, n)| SetupEntry { zone: SUPPLY, template: i as u16, count: *n }).collect();

    let mut actions = vec![];
    let mut buy_effects = vec![
        mv_one(ORef::Source, me(TABLEAU), Pos::Bottom),
        Effect::ModRes(PRef::Me, VP, attr(ORef::Source, PTS)),
        Effect::ModRes(PRef::Me, INCOME, attr(ORef::Source, INC)),
        mv(top_n(shared(SUPPLY), 1), shared(MARKET), Pos::Bottom),
    ];
    if !multi_buy {
        buy_effects.push(Effect::EndTurn);
    }
    let mut buy = action("buy", 0, vec![1], Timing::Main, seq(buy_effects));
    buy.source = Some(SourceSpec { zones: vec![shared(MARKET)], filter: Cond::True });
    buy.costs = vec![(GOLD, attr(ORef::Source, COST))];
    actions.push(buy);
    actions.push(action("take_gold", 1, vec![1], Timing::Main, seq(vec![Effect::ModRes(PRef::Me, GOLD, c(take_gold)), Effect::EndTurn])));
    if let Some(rc) = refresh_cost {
        let mut a = action(
            "refresh_market",
            2,
            vec![1],
            Timing::Main,
            seq(vec![
                foreach(zones_all(vec![shared(MARKET)]), mv_one(ORef::Iter, shared(SUPPLY), Pos::Bottom)),
                mv(top_n(shared(SUPPLY), market_size), shared(MARKET), Pos::Bottom),
            ]),
        );
        a.costs = vec![(GOLD, c(rc))];
        a.require = Cond::Exists(Box::new(zones_all(vec![shared(MARKET)])));
        actions.push(a);
    }
    if multi_buy {
        actions.push(action("end_turn", 3, vec![1], Timing::Main, Effect::EndTurn));
    }

    let terminal = if alt {
        vec![TerminalRule { cond: cmp(res(PRef::Me, INCOME), CmpOp::Ge, c(inc_target)), result: PlayerResult::Win }]
    } else {
        vec![TerminalRule { cond: cmp(res(PRef::Me, VP), CmpOp::Ge, c(vp_target)), result: PlayerResult::Win }]
    };

    GameDef {
        name: "engine".into(),
        num_players: 2,
        resources: vec![resource("vp", 0, 0, 60), resource("gold", 0, 0, 40), resource("income", income0, 0, 14)],
        attrs: vec![attr_def("cost", 0, 0, 10), attr_def("vp", 0, 0, 10), attr_def("income", 0, 0, 10)],
        vars: vec![],
        zones: vec![
            zone("market", false, true, Visibility::Public, Some(market_size as u16)),
            zone("supply", false, true, Visibility::Hidden, None),
            zone("tableau", true, true, Visibility::Public, None),
        ],
        templates,
        hooks: vec![],
        player_setup: vec![],
        shared_setup,
        setup: seq(vec![Effect::Shuffle(shared(SUPPLY)), mv(top_n(shared(SUPPLY), market_size), shared(MARKET), Pos::Bottom)]),
        phases: vec![
            PhaseDef { name: "income".into(), on_enter: Effect::ModRes(PRef::Me, GOLD, res(PRef::Me, INCOME)), auto_end: true },
            PhaseDef { name: "main".into(), on_enter: Effect::NoOp, auto_end: false },
        ],
        actions,
        triggers: vec![],
        terminal,
        timeout: Timeout::ByResource(if alt { INCOME } else { VP }),
        adjudication: Adjudication::Rules,
        limits: Limits { max_turns: 60, max_decisions: 800, ..Default::default() },
        meta: Meta {
            family: "engine".into(),
            seed: 0,
            tags: tags(&[
                ("information", "hidden"),
                ("resource_pressure", "medium"),
                ("interaction", "low"),
                ("horizon", "long"),
                ("randomness", "low"),
                ("tempo_importance", "medium"),
                ("resource_conservation", "high"),
                ("engine_building", "high"),
                ("reaction_system", "none"),
                ("tradeoff", "invest_vs_cash"),
            ]),
        },
    }
}
